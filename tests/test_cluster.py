"""MLX cluster: upstream backend through the chat endpoint, and mlx.launch argv."""
import json

import httpx
import pytest

from cloxy import cli, cluster, config
from cloxy.backends import openai_backend

SEEN = []


def upstream(request: httpx.Request) -> httpx.Response:
    SEEN.append(request)
    if request.url.path == "/v1/models":
        return httpx.Response(200, json={"data": [{"id": "cached/other-model"}]})
    body = json.loads(request.content)
    if body.get("stream"):
        lines = [
            {"choices": [{"index": 0, "delta": {"content": "Hel"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2,
                                      "total_tokens": 7}},
        ]
        sse = "".join(f"data: {json.dumps(x)}\n\n" for x in lines) + "data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=sse.encode())
    return httpx.Response(200, json={
        "id": "x", "object": "chat.completion", "model": "default_model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi there"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    })


@pytest.fixture
def cluster_llm(monkeypatch, tmp_path):
    SEEN.clear()
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"model_hf_id": "mlx-community/Qwen3-32B-4bit"}))
    monkeypatch.setattr(config, "CONFIG_PATH", cfg)
    monkeypatch.setattr(config, "LLM_URL", "http://rank0.lan:9056")
    monkeypatch.setattr(config, "LLM_API_KEY", "sekrit")
    monkeypatch.setattr(openai_backend, "_transport", httpx.MockTransport(upstream))
    monkeypatch.setattr(openai_backend, "_model_id", None)
    yield


def test_chat_forwards_to_upstream_without_model_field(client, cluster_llm):
    r = client.post("/v1/chat/completions", json={
        "model": "anything", "messages": [{"role": "developer", "content": "be brief"},
                                          {"role": "user", "content": "hi"}],
        "max_tokens": 16})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["choices"][0]["message"]["content"] == "hi there"
    assert out["model"] == "mlx-community/Qwen3-32B-4bit"     # configured id, not /v1/models
    assert out["cloxy_meta"]["upstream"] == "http://rank0.lan:9056"

    post = [q for q in SEEN if q.method == "POST"][-1]
    assert post.headers["authorization"] == "Bearer sekrit"
    sent = json.loads(post.content)
    assert "model" not in sent                       # rank 0 must not try to load another model
    assert sent["messages"][0] == {"role": "system", "content": "be brief"}
    assert sent["max_tokens"] == 16

    models = client.get("/v1/models").json()["data"]
    assert models[0]["id"] == "mlx-community/Qwen3-32B-4bit"
    assert models[0]["owned_by"] == "cloxy-upstream"


def test_chat_streams_from_upstream(client, cluster_llm):
    with client.stream("POST", "/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}], "stream": True}) as r:
        assert r.status_code == 200
        events = [json.loads(l[6:]) for l in r.iter_lines()
                  if l.startswith("data: ") and l != "data: [DONE]"]
    text = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
    assert text == "Hello"
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    assert events[-1]["usage"]["completion_tokens"] == 2
    sent = json.loads([q for q in SEEN if q.method == "POST"][-1].content)
    assert sent["stream_options"] == {"include_usage": True}


def test_unreachable_upstream_is_503_not_hang(client, monkeypatch, tmp_path):
    def down(request):
        raise httpx.ConnectError("refused", request=request)
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "none.json")
    monkeypatch.setattr(config, "LLM_URL", "http://rank0.lan:9056")
    monkeypatch.setattr(openai_backend, "_transport", httpx.MockTransport(down))
    monkeypatch.setattr(openai_backend, "_model_id", None)
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 503 and "unreachable" in r.json()["detail"]


def test_upstream_dies_after_load_is_502(client, cluster_llm, monkeypatch):
    client.get("/v1/models")
    monkeypatch.setattr(openai_backend, "_model_id", "m")

    def down(request):
        raise httpx.ConnectError("refused", request=request)
    monkeypatch.setattr(openai_backend, "_transport", httpx.MockTransport(down))
    for stream in (False, True):
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "x"}], "stream": stream})
        assert r.status_code == 502


# ---------------------------------------------------------------------------
# argv we hand to Apple's tooling
# ---------------------------------------------------------------------------

def test_serve_argv_runs_mlx_lm_server_under_mlx_launch():
    argv = cluster.serve_argv("mlx-community/Qwen3-32B-4bit", "/h.json", 9056,
                              backend="jaccl", python="/opt/py", pipeline=True,
                              extra=["--", "--max-tokens", "2048"])
    launch = argv.index("--")
    assert argv[0].endswith("mlx.launch") or argv[:3][-1] == "mlx._distributed_utils.launch"
    assert argv[launch - 6:launch] == ["--hostfile", "/h.json", "--python", "/opt/py",
                                       "--backend", "jaccl"]
    # mlx.launch prefixes --python itself; repeating it runs the interpreter as a script.
    assert argv[launch + 1:] == ["-m", "mlx_lm", "server",
                                 "--model", "mlx-community/Qwen3-32B-4bit",
                                 "--host", "0.0.0.0", "--port", "9056", "--pipeline",
                                 "--max-tokens", "2048"]


def test_setup_argv():
    argv = cluster.setup_argv("m1,m2,m3", "/h.json", backend="jaccl-ring", auto_setup=True)
    tail = argv[argv.index("--hosts"):]
    assert tail == ["--hosts", "m1,m2,m3", "--over", "thunderbolt",
                    "--output-hostfile", "/h.json", "--backend", "jaccl-ring", "--auto-setup"]
    eth = cluster.setup_argv("m1,m2", "/h.json", over="ethernet", backend="jaccl")
    assert "--backend" not in eth          # backend only applies to thunderbolt discovery


def test_hostfile_describe_and_rank0_url(tmp_path):
    hf = tmp_path / "h.json"
    hf.write_text(json.dumps({"backend": "jaccl", "hosts": [
        {"ssh": "roy@m4-a.local", "ips": ["169.254.1.1"], "rdma": [None, "rdma_en3"]},
        {"ssh": "m4-b.local", "ips": ["169.254.1.2"], "rdma": ["rdma_en3", None]},
    ]}))
    data = cluster.read_hostfile(str(hf))
    lines = cluster.describe(data)
    assert "ranks: 2" in lines[0] and "rank 0 — serves HTTP" in lines[1]
    assert cluster.rank0_url(data, 9056) == "http://m4-a.local:9056"
    assert cluster.rank0_url({"hosts": []}, 9056) is None
    hf.write_text(json.dumps([{"ssh": "x", "ips": [], "rdma": []}]))   # bare-list form
    assert cluster.read_hostfile(str(hf))["hosts"][0]["ssh"] == "x"


def test_cluster_cli_wiring(monkeypatch, tmp_path, capsys):
    p = cli.build_parser()
    a = p.parse_args(["cluster", "serve", "--model", "m", "--", "--max-tokens", "9"])
    assert a.func is cli.cmd_cluster_serve and a.extra == ["--", "--max-tokens", "9"]

    hf = tmp_path / "h.json"
    hf.write_text(json.dumps({"backend": "jaccl-ring", "hosts": [{"ssh": "a", "ips": ["1"]}]}))
    ran = []
    monkeypatch.setattr(cli, "_run", lambda argv: ran.append(argv) or 0)
    assert cli.main(["cluster", "serve", "--model", "m", "--hostfile", str(hf)]) == 0
    assert ran[0][ran[0].index("--backend") + 1] == "jaccl-ring"     # hostfile's backend
    assert "CLOXY_LLM_URL=http://a:" in capsys.readouterr().out

    assert cli.main(["cluster", "serve", "--model", "m",
                     "--hostfile", str(tmp_path / "missing.json")]) == 1
