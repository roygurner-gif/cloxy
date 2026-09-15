"""client.env loading: fills in CLOXY_* defaults without overriding real env."""
from cloxy import config


def test_env_file_sets_defaults_only(tmp_path, monkeypatch):
    envf = tmp_path / "client.env"
    envf.write_text(
        "# comment\n"
        "CLOXY_URL=http://10.0.0.5:9055\n"
        'CLOXY_API_KEY="secret"\n'
        "CLOXY_PORT=7\n"
        "NOT_OURS=1\n"
        "garbage line\n"
    )
    monkeypatch.delenv("CLOXY_URL", raising=False)
    monkeypatch.delenv("CLOXY_API_KEY", raising=False)
    monkeypatch.setenv("CLOXY_PORT", "9999")     # already set → must win
    monkeypatch.delenv("NOT_OURS", raising=False)

    config._load_env_file(str(envf))

    import os
    assert os.environ["CLOXY_URL"] == "http://10.0.0.5:9055"
    assert os.environ["CLOXY_API_KEY"] == "secret"
    assert os.environ["CLOXY_PORT"] == "9999"
    assert "NOT_OURS" not in os.environ


def test_env_file_missing_is_fine(tmp_path):
    config._load_env_file(str(tmp_path / "nope.env"))
