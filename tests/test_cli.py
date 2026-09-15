"""CLI: parser wiring and the launchd plist we generate."""
import sys
from pathlib import Path

from cloxy import cli, config


def test_parser_has_all_commands():
    p = cli.build_parser()
    cmds = p._subparsers._group_actions[0].choices
    assert {"init", "show", "list", "start", "mcp", "status", "recall", "ingest",
            "install-service", "uninstall-service"} <= set(cmds)


def test_service_plist_uses_console_script_and_data_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CLOXY_API_KEY", "k")
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", "/models")
    monkeypatch.setenv("UNRELATED", "x")
    plist = cli.service_plist(tmp_path / "logs")

    # Not `python -m cloxy` from $HOME — a checkout at ~/cloxy would shadow the package.
    assert plist["WorkingDirectory"] == str(tmp_path)
    script = Path(sys.executable).parent / "cloxy"
    if script.is_file():
        assert plist["ProgramArguments"] == [str(script), "start"]
    else:
        assert plist["ProgramArguments"] == [sys.executable, "-m", "cloxy", "start"]

    env = plist["EnvironmentVariables"]
    assert env["CLOXY_API_KEY"] == "k" and env["FASTEMBED_CACHE_PATH"] == "/models"
    assert "UNRELATED" not in env and "PATH" in env
    assert plist["KeepAlive"] is True and plist["RunAtLoad"] is True
