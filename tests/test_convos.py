"""Claude Code session parsing, grouping, and incremental ingest."""
import json

import pytest

from cloxy import convos, memory

SID = "26266c06-3364-41e7-b488-7359c836290c"
CWD = "/Users/someone/projects/rmbr"


def _line(**d) -> bytes:
    return (json.dumps(d) + "\n").encode()


def _user(text, ts, cwd=CWD, **extra):
    return _line(type="user", message={"role": "user", "content": text}, timestamp=ts,
                 cwd=cwd, sessionId=SID, gitBranch="main", **extra)


def _assistant(text, ts, cwd=CWD, **extra):
    content = [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": text}]
    return _line(type="assistant", message={"role": "assistant", "content": content},
                 timestamp=ts, cwd=cwd, sessionId=SID, gitBranch="main", **extra)


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def test_message_text_strips_harness_blocks():
    raw = ("<system-reminder>secret</system-reminder>real question "
           "<command-name>/model</command-name><local-command-stdout>noise</local-command-stdout>")
    assert convos.message_text(raw) == "real question /model"
    assert convos.message_text([{"type": "tool_result", "content": "x"},
                                {"type": "text", "text": " hi "}]) == "hi"
    assert convos.message_text(42) == ""


def test_parse_messages_reads_complete_lines_and_meta(tmp_path):
    p = tmp_path / f"{SID}.jsonl"
    p.write_bytes(
        _line(type="mode", mode="default", sessionId=SID)
        + _line(type="ai-title", aiTitle="Board colors", sessionId=SID)
        + _user("first", "2026-09-10T10:00:00Z")
        + _user("meta", "2026-09-10T10:00:01Z", isMeta=True)
        + _user("side", "2026-09-10T10:00:02Z", isSidechain=True)
        + _assistant("answer", "2026-09-10T10:00:03Z")
        + b'{"type": "user", "message": {"role": "user", "content": "partial'   # no newline
    )
    meta = convos.SessionMeta(session_id="from-filename")
    msgs, meta, end = convos.parse_messages(str(p), 0, meta)
    assert [(m.role, m.text) for m in msgs] == [("user", "first"), ("assistant", "answer")]
    assert meta.session_id == SID and meta.project == CWD
    assert meta.git_branch == "main" and meta.title == "Board colors"
    assert end == msgs[-1].end_offset == len(p.read_bytes()) - len(b'{"type": "user", "message": {"role": "user", "content": "partial')


def test_group_messages_packs_whole_messages_and_splits_giants():
    m = lambda i, n: convos.Msg("user", f"m{i} " + "x" * n, None, i)
    groups = convos.group_messages([m(1, 600), m(2, 600), m(3, 600), m(4, 100)], size=1500)
    assert [[x.text[:2] for x in g] for g in groups] == [["m1", "m2"], ["m3", "m4"]]
    giant = convos.group_messages([m(1, 10), m(2, 5000), m(3, 10)], size=1500)
    assert [len(g) for g in giant] == [1] + [1] * (len(giant) - 2) + [1]
    assert giant[0][0].text.startswith("m1") and giant[-1][0].text.startswith("m3")
    assert all(len(g[0].text) <= 1500 + 5 for g in giant[1:-1])


def test_render_group_header():
    meta = convos.SessionMeta(session_id=SID, project=CWD, title="Board colors")
    text = convos.render_group([convos.Msg("user", "q", "2026-09-10T10:00:00Z", 1),
                                convos.Msg("assistant", "a", "2026-09-10T10:00:03Z", 2)], meta)
    assert text.startswith("[2026-09-10 10:00 · rmbr · Board colors]\nUSER: q\n\nASSISTANT: a")
    bare = convos.render_group([convos.Msg("user", "q", None, 1)],
                               convos.SessionMeta(session_id=SID))
    assert bare.startswith(f"[{SID[:8]}]\nUSER: q")


# ---------------------------------------------------------------------------
# incremental ingest through the running app
# ---------------------------------------------------------------------------

def _msg_pair(i, n=560):
    ts_u = f"2026-09-1{i % 9}T10:{i:02d}:00Z"
    ts_a = f"2026-09-1{i % 9}T10:{i:02d}:30Z"
    return _user(f"question {i} " + f"alpha{i} " * (n // 7), ts_u) + \
        _assistant(f"answer {i} " + f"beta{i} " * (n // 6), ts_a)


def _session_chunks(client, sid=SID):
    return client.post("/recall", json={"query": "question", "top_k": 100,
                                        "mode": "keyword"}).json()["results"]


def test_incremental_ingest_replaces_tail_and_keeps_metadata(client, tmp_path):
    d = tmp_path / "projects" / "-Users-someone-projects-rmbr"
    d.mkdir(parents=True)
    p = d / f"{SID}.jsonl"

    # A legacy (pre-v5) chunk for this session should be replaced on first ingest.
    legacy = client.post("/ingest_text", json={"text": "old style chunk", "source": f"convo:{SID}"}).json()
    assert legacy["chunks_stored"] == 1

    p.write_bytes(_line(type="ai-title", aiTitle="Board colors", sessionId=SID)
                  + b"".join(_msg_pair(i) for i in range(1, 4)))     # 6 msgs → 3 groups
    rep = client.post("/ingest_convos", json={"convo_dir": str(tmp_path / "projects")}).json()
    assert rep["files_found"] == 1 and rep["files_processed"] == 1
    assert rep["chunks_stored"] == 3 and not rep["errors"]

    hits = client.post("/recall", json={"query": "alpha2 question", "project": "rmbr"}).json()["results"]
    assert hits and hits[0]["session_id"] == SID
    assert hits[0]["project"] == CWD and hits[0]["ts_start"].startswith("2026-09-1")
    assert hits[0]["metadata"]["title"] == "Board colors"
    assert hits[0]["metadata"]["git_branch"] == "main"
    assert hits[0]["content"].startswith("[2026-09-1")
    assert client.post("/recall", json={"query": "old style chunk", "mode": "keyword"}).json()["results"] == []

    # Unchanged file → skipped.
    rep = client.post("/ingest_convos", json={"convo_dir": str(tmp_path / "projects")}).json()
    assert rep["files_processed"] == 0 and rep["files_skipped"] == 1

    # Append two more pairs. Re-reading from the committed offset regroups the
    # old tail [u3,a3] as final, adds [u4,a4], and a new tail [u5,a5]: 3 stored,
    # 1 replaced → 5 chunks for 10 messages, no duplicates.
    with open(p, "ab") as f:
        f.write(b"".join(_msg_pair(i) for i in range(4, 6)))
    rep = client.post("/ingest_convos", json={"convo_dir": str(tmp_path / "projects")}).json()
    assert rep["files_processed"] == 1 and rep["chunks_stored"] == 3
    sources = sorted(h["source"] for h in _session_chunks(client))
    assert sources == [f"convo:{SID}:{i}" for i in range(5)]

    status = client.get("/ingest_status").json()
    assert status["files_tracked"] == 1

    # Force re-ingest from scratch lands on the same chunk set.
    rep = client.post("/ingest_convos", json={"convo_dir": str(tmp_path / "projects"),
                                              "force": True}).json()
    assert rep["chunks_stored"] == 5
    assert sorted(h["source"] for h in _session_chunks(client)) == sources

    # Date filter across sessions: everything here is in Sept 2026.
    assert client.post("/recall", json={"query": "question", "since": "2026-10-01"}).json()["results"] == []
    assert len(client.post("/recall", json={"query": "question", "until": "2026-09-30",
                                            "top_k": 10}).json()["results"]) == 5

    # Cleanup for other tests in this module's DB.
    assert client.post("/forget", json={"source_prefix": f"convo:{SID}"}).json()["deleted"] == 5


def test_ingest_convos_bad_dir(client):
    assert client.post("/ingest_convos", json={"convo_dir": "/definitely/not/here"}).status_code == 400
