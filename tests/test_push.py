"""`cloxy push` client ↔ /push endpoint, and ingest of the mirrored files."""
import base64
import json

import pytest

from cloxy import config, push

SID = "7d3f0a52-1111-4222-8333-944455556666"


def _line(**d) -> bytes:
    return (json.dumps(d) + "\n").encode()


def _user(text, ts):
    return _line(type="user", message={"role": "user", "content": text}, timestamp=ts,
                 cwd="/Users/someone/projects/peace", sessionId=SID)


def _poster(client):
    def post(payload):
        r = client.post("/push", json=payload)
        return r.status_code, r.json()
    return post


@pytest.fixture
def local(tmp_path):
    root = tmp_path / "claude-projects"
    f = root / "-Users-someone-projects-peace" / f"{SID}.jsonl"
    f.parent.mkdir(parents=True)
    return root, f


def test_mirror_path_rejects_escapes():
    for host, agent, rel in [("a/b", "claude", "x.jsonl"), ("box", "gpt", "x.jsonl"),
                             ("box", "claude", "../x.jsonl"), ("box", "claude", "/etc/x.jsonl"),
                             ("box", "claude", "x.txt"), ("box", "claude", "a//x.jsonl")]:
        with pytest.raises(push.PushError) as e:
            push.mirror_path(host, agent, rel)
        assert e.value.status == 400


def test_complete_slice_stops_at_last_newline(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_bytes(b'{"a":1}\n{"b":2}\n{"c":')
    assert push.complete_slice(p, 0, 1000) == b'{"a":1}\n{"b":2}\n'
    assert push.complete_slice(p, 0, 10) == b'{"a":1}\n'
    assert push.complete_slice(p, 16, 1000) == b""
    p.write_bytes(b"x" * 50 + b"\n")
    assert push.complete_slice(p, 0, 10) == b"x" * 50 + b"\n"   # one giant line


def test_push_mirrors_incrementally_and_ingests(client, local):
    root, f = local
    post = _poster(client)
    sources = [("claude", str(root))]
    f.write_bytes(_user("deploy the mesh to node ten", "2026-11-02T10:00:00Z")
                  + b'{"type": "progress"')                          # unfinished line
    s = push.push_once(post, host="mbp", sources=sources, max_bytes=40)
    assert s.files_pushed == 1 and not s.errors
    mirror = push.mirror_path("mbp", "claude", f.relative_to(root).as_posix())
    assert mirror.read_bytes() == _user("deploy the mesh to node ten", "2026-11-02T10:00:00Z")

    # Nothing new → nothing sent.
    assert push.push_once(post, host="mbp", sources=sources).bytes_sent == 0

    # Finish the line and add another; only the new bytes go.
    with open(f, "ab") as fh:
        fh.write(b', "n": 1}\n' + _user("rotate the api key on the vps", "2026-11-02T11:00:00Z"))
    s = push.push_once(post, host="mbp", sources=sources)
    assert not s.errors and mirror.read_bytes() == f.read_bytes()

    rep = client.post("/ingest_convos", json={"convo_dir": config.inbox_dir()}).json()
    assert rep["chunks_stored"] >= 1 and not rep["errors"]
    hits = client.post("/recall", json={"query": "rotate the api key", "project": "peace"}).json()["results"]
    assert hits and hits[0]["session_id"] == SID
    assert client.post("/forget", json={"source_prefix": f"convo:{SID}"}).json()["deleted"] >= 1


def test_lost_client_state_resyncs_from_server(client, local):
    root, f = local
    post = _poster(client)
    sources = [("claude", str(root))]
    f.write_bytes(_user("one", "2026-11-03T10:00:00Z") + _user("two", "2026-11-03T10:01:00Z"))
    push.push_once(post, host="laptop", sources=sources)
    push.save_state({})                                   # client forgets everything
    with open(f, "ab") as fh:
        fh.write(_user("three", "2026-11-03T10:02:00Z"))
    s = push.push_once(post, host="laptop", sources=sources)
    assert s.resyncs == 1 and not s.errors
    mirror = push.mirror_path("laptop", "claude", f.relative_to(root).as_posix())
    assert mirror.read_bytes() == f.read_bytes()          # no duplicated lines


def test_push_endpoint_validation(client):
    good = base64.b64encode(b'{"x":1}\n').decode()
    r = client.post("/push", json={"host": "h", "path": "a/b.jsonl", "offset": 5, "data": good})
    assert r.status_code == 409 and r.json()["size"] == 0
    r = client.post("/push", json={"host": "h", "path": "a/b.jsonl", "offset": 0,
                                   "data": base64.b64encode(b"no newline").decode()})
    assert r.status_code == 400
    r = client.post("/push", json={"host": "h", "path": "../../b.jsonl", "offset": 0, "data": good})
    assert r.status_code == 400
    r = client.post("/push", json={"host": "h", "path": "a/b.jsonl", "offset": 0, "data": "!!"})
    assert r.status_code == 400


def test_loopback_detection():
    assert push.is_loopback_url("http://127.0.0.1:9055")
    assert push.is_loopback_url("http://localhost:9055")
    assert not push.is_loopback_url("http://192.168.1.20:9055")


def test_local_rewrite_replaces_mirror(client, local):
    root, f = local
    post = _poster(client)
    sources = [("claude", str(root))]
    f.write_bytes(_user("long " * 20, "2026-11-04T10:00:00Z") * 3)
    push.push_once(post, host="desk", sources=sources)
    f.write_bytes(_user("short", "2026-11-04T11:00:00Z"))       # file shrank locally
    s = push.push_once(post, host="desk", sources=sources)
    assert not s.errors
    mirror = push.mirror_path("desk", "claude", f.relative_to(root).as_posix())
    assert mirror.read_bytes() == f.read_bytes()
