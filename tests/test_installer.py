"""One-click installer support: non-interactive setup, frozen-app streams, first launch, selftest."""

import io
import json
import sys
from pathlib import Path

import pytest

from away import config as C
from away import paths
from away.app import frozen, models, selftest, setup_wizard
from away.cli import build_parser
from away.cli import main as cli_main


def _fake_dl(mf, d, out):
    d.mkdir(parents=True, exist_ok=True)
    (d / mf.name).write_bytes(b"x")
    return d / mf.name


# ---------------------------------------------------------------- never block without a user
def test_ask_without_stdin_returns_default(monkeypatch):
    def boom(_p):
        raise AssertionError("input() must not be called without stdin")

    monkeypatch.setattr("builtins.input", boom)
    monkeypatch.setattr(sys, "stdin", None)
    assert setup_wizard.ask("q", True, None) is True
    assert setup_wizard.ask("q", False, None) is False
    closed = io.StringIO("")
    closed.close()
    monkeypatch.setattr(sys, "stdin", closed)
    assert setup_wizard.ask("q", True, None) is True


def test_ask_lost_stdin_runtimeerror(monkeypatch):
    def lost(_p):
        raise RuntimeError("input(): lost sys.stdin")

    monkeypatch.setattr("builtins.input", lost)
    assert setup_wizard.ask("q", True, None) is True


def test_setup_yes_never_reads_stdin(monkeypatch):
    """`setup --yes` in a windowed frozen exe (sys.stdin is None) must complete without input()."""
    monkeypatch.setattr(sys, "stdin", None)
    monkeypatch.setattr("builtins.input", lambda p: (_ for _ in ()).throw(AssertionError("blocked on input")))
    got = []
    monkeypatch.setattr(models, "download_asr", lambda m, d, out: got.append(m))
    rep = setup_wizard.run(yes=True, asr_model="small", skip_probe=True, out=lambda s: None, downloader=_fake_dl)
    assert got == ["small"] and rep.models["asr"] == "ok"
    assert Path(rep.config_path).exists()


def test_setup_without_yes_and_without_stdin_uses_defaults(monkeypatch):
    monkeypatch.setattr(sys, "stdin", None)
    got = []
    monkeypatch.setattr(models, "download_asr", lambda m, d, out: got.append(m))
    rep = setup_wizard.run(skip_probe=True, out=lambda s: None, downloader=_fake_dl)
    assert rep.models["asr"] == "ok" and got == [C.AppConfig().voice.asr_model]


def test_uninstall_without_stdin_does_not_delete(monkeypatch):
    from away.app import uninstall

    paths.data_dir().mkdir(parents=True)
    monkeypatch.setattr(sys, "stdin", None)
    assert uninstall.run() == []
    assert paths.data_dir().exists()


# ---------------------------------------------------------------- windowed-exe streams
def test_ensure_streams_tty_untouched():
    assert frozen.ensure_streams(["doctor"]) == "tty"


def test_ensure_streams_log_file(monkeypatch, tmp_path):
    log = tmp_path / "logs" / "a.log"
    monkeypatch.setattr(sys, "stdin", None)
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    where = frozen.ensure_streams(["tray"], platform="darwin", env={"AWAY_LOG_FILE": str(log)})
    assert where == str(log)
    print("hello")
    sys.stderr.write("progress\n")
    with pytest.raises(EOFError):
        input()
    sys.stdout.flush()
    assert "hello" in log.read_text() and "progress" in log.read_text()


def test_ensure_streams_windows_console(monkeypatch, tmp_path):
    con = io.StringIO()
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    assert frozen.ensure_streams(["setup", "--yes"], platform="win32", env={}, console=lambda: con) == "console"
    print("downloading")
    assert "downloading" in con.getvalue()


def test_ensure_streams_windows_falls_back_to_log(monkeypatch, tmp_path):
    log = tmp_path / "w.log"
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    env = {"AWAY_LOG_FILE": str(log)}
    # tray does not want a console; setup with no console available or AWAY_NO_CONSOLE=1 logs to file
    assert frozen.ensure_streams(["tray"], platform="win32", env=env, console=lambda: io.StringIO()) == str(log)
    monkeypatch.setattr(sys, "stdout", None)
    assert frozen.ensure_streams(["setup"], platform="win32", env=env, console=lambda: None) == str(log)
    monkeypatch.setattr(sys, "stdout", None)
    env2 = {**env, "AWAY_NO_CONSOLE": "1"}
    assert frozen.ensure_streams(["setup"], platform="win32", env=env2, console=lambda: io.StringIO()) == str(log)


def test_stdin_usable():
    assert frozen.stdin_usable(None) in (True, False)
    s = io.StringIO("")
    assert frozen.stdin_usable(s) is False
    s.close()
    assert frozen.stdin_usable(s) is False


def test_wants_console():
    assert frozen.wants_console(["setup", "--yes"]) and frozen.wants_console(["--version"])
    assert not frozen.wants_console(["tray"]) and not frozen.wants_console([])


# ---------------------------------------------------------------- first launch
def test_missing_models_and_first_launch(monkeypatch):
    cfg = C.AppConfig()
    mdir = paths.models_dir()
    missing = frozen.missing_models(cfg, mdir)
    assert "hand_landmarker.task" in missing and f"whisper-{cfg.voice.asr_model}" in missing
    assert frozen.needs_setup(cfg)

    calls, notes = [], []

    def fake_cli(argv):
        calls.append(argv)
        for mf in models.MEDIAPIPE_MODELS:
            _fake_dl(mf, mdir, None)
        w = mdir / f"whisper-{cfg.voice.asr_model}"
        w.mkdir(parents=True, exist_ok=True)
        (w / "model.bin").write_bytes(b"x")
        C.save(cfg)
        return 0

    rc = frozen.first_launch(fake_cli, cfg, notifier=lambda t, m: notes.append(m))
    assert rc == 0
    assert calls == [["setup", "--yes", "--skip-probe", "--asr-model", cfg.voice.asr_model]]
    assert "GB" in notes[0] and notes[-1] == "Models ready."
    assert frozen.missing_models(cfg, mdir) == [] and not frozen.needs_setup(cfg)
    # second launch: nothing to do
    assert frozen.first_launch(fake_cli, cfg, notifier=lambda t, m: None) is None
    assert len(calls) == 1


def test_missing_models_ignores_asr_when_voice_off_or_parakeet():
    cfg = C.AppConfig()
    cfg.voice.enabled = False
    assert not any(m.startswith("whisper-") for m in frozen.missing_models(cfg))
    cfg.voice.enabled = True
    cfg.voice.asr_backend = "parakeet"
    assert not any(m.startswith("whisper-") for m in frozen.missing_models(cfg))


def test_first_launch_failure_note():
    notes = []
    cfg = C.AppConfig()
    cfg.voice.asr_model = "small"
    rc = frozen.first_launch(lambda argv: 1, cfg, notifier=lambda t, m: notes.append(m))
    assert rc == 1 and "retry" in notes[-1]
    assert "0.5 GB" in notes[0]  # MediaPipe (~12 MB) + Whisper small (~485 MB)


def test_frozen_entry_runs_setup_then_tray(monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "away_entry", Path(__file__).parents[1] / "packaging" / "pyinstaller" / "away_entry.py"
    )
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    calls = []
    monkeypatch.setattr("away.cli.main", lambda argv: calls.append(argv) or 0)
    monkeypatch.setattr(frozen, "notify", lambda t, m, platform=None: True)
    monkeypatch.setattr(frozen, "first_launch", lambda cli: calls.append("first_launch"))
    monkeypatch.setattr(sys, "argv", ["ProjectAway"])
    assert entry.main() == 0
    assert calls == ["first_launch", ["tray"]]
    calls.clear()
    monkeypatch.setattr(sys, "argv", ["ProjectAway", "doctor"])
    entry.main()
    assert calls == [["doctor"]]


# ---------------------------------------------------------------- whisper path + parakeet
def test_resolve_whisper_model_uses_setup_download(tmp_path):
    from away.voice.asr import resolve_whisper_model

    assert resolve_whisper_model("small", None) == "small"
    assert resolve_whisper_model("small", str(tmp_path)) == "small"
    (tmp_path / "model.bin").write_bytes(b"x")
    assert resolve_whisper_model("small", str(tmp_path)) == str(tmp_path)


def test_parakeet_missing_message(monkeypatch):
    from away.voice.asr import PARAKEET_MISSING, make_asr

    monkeypatch.setitem(sys.modules, "nemo", None)
    monkeypatch.setitem(sys.modules, "nemo.collections", None)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", None)
    with pytest.raises(ImportError, match="pipx install"):
        make_asr("parakeet", "large-v3-turbo")
    assert "not bundle" in PARAKEET_MISSING


# ---------------------------------------------------------------- selftest
def test_selftest_pass_fail_and_skip(tmp_path, capsys):
    ok = selftest.Check("a", "feature a", lambda: "fine")
    gui = selftest.Check("g", "gui", lambda: "x", needs_display=True)

    def broken():
        raise OSError("libfoo.so missing")

    bad = selftest.Check("b", "feature b", broken)
    res = selftest.run([ok, gui], display=False)
    assert res["a"]["ok"] is True and res["g"]["ok"] is None
    assert selftest.main(str(tmp_path / "r.json"), [ok, gui], display=False) == 0
    assert json.loads((tmp_path / "r.json").read_text())["ok"] is True
    assert selftest.main(None, [ok, bad], display=True) == 1
    assert selftest.run([bad])["b"]["detail"] == "OSError: libfoo.so missing"


def test_selftest_core_checks_run_in_dev_env():
    res = selftest.run([c for c in selftest.CHECKS if c.name in ("core", "away-data")])
    assert all(r["ok"] for r in res.values()), res


def test_selftest_has_display():
    assert selftest.has_display("win32") and selftest.has_display("darwin")


def test_selftest_cli_hidden_and_runs(monkeypatch, capsys, tmp_path):
    assert "selftest" not in build_parser().format_help()
    monkeypatch.setattr(selftest, "CHECKS", (selftest.Check("a", "a", lambda: "ok"),))
    assert cli_main(["selftest", "--json", str(tmp_path / "s.json")]) == 0
    assert json.loads((tmp_path / "s.json").read_text())["checks"]["a"]["ok"] is True


def test_doctor_lists_settings_and_parakeet(capsys):
    cli_main(["doctor"])
    info = json.loads(capsys.readouterr().out)
    assert "tkinter" in info["extras"]["settings window"]
    assert any(k.startswith("parakeet") for k in info["extras"])


# ---------------------------------------------------------------- packaging files stay consistent
ROOT = Path(__file__).parents[1]


def test_inno_script_one_click():
    iss = (ROOT / "packaging" / "windows" / "away.iss").read_text()
    assert "setup --yes --skip-probe" in iss and "--asr-model small" in iss
    for task in ("autostart", "desktopicon"):
        line = next(ln for ln in iss.splitlines() if ln.startswith(f'Name: "{task}"'))
        assert "unchecked" not in line, line
    assert 'Parameters: "tray"' in iss and "postinstall" in iss


def test_release_workflow_smoke_and_strict():
    yml = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    assert yml.count("selftest --json") + yml.count('"selftest", "--json"') + yml.count("'selftest','--json'") >= 5
    assert 'AWAY_SPEC_STRICT: "1"' in yml and yml.count("scripts/check_selftest.py") >= 8
    assert "download.pytorch.org/whl/cpu" in yml and ".[all,train]" in yml
    assert "draft: true" in yml and "PYPI_API_TOKEN != ''" in yml


def test_spec_bundles_everything_but_nemo():
    spec = (ROOT / "packaging" / "pyinstaller" / "away.spec").read_text()
    for pkg in ("torch", "onnx", "tkinter", "pystray", "pynput", "sounddevice", "mediapipe", "faster_whisper"):
        assert f'optional("{pkg}"' in spec, pkg
    assert '"nemo"' in spec and '"matplotlib"' not in spec  # mediapipe.tasks.python.vision imports matplotlib


def test_autostart_uses_appimage_path(monkeypatch):
    from away.app import autostart

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("APPIMAGE", "/home/u/Apps/ProjectAway.AppImage")
    assert autostart.launch_command() == ["/home/u/Apps/ProjectAway.AppImage", "tray"]


def _load_check_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location("check_selftest", ROOT / "scripts" / "check_selftest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_check_selftest_script(tmp_path, capsys):
    chk = _load_check_script()
    good = {"frozen": True, "checks": {"torch": {"ok": True, "feature": "train", "detail": "2.x"}}}
    skip = {"frozen": True, "checks": {"pystray": {"ok": None, "feature": "tray", "detail": "skipped"}}}
    bad = {"frozen": False, "checks": {"tkinter": {"ok": False, "feature": "settings", "detail": "ImportError"}}}
    for name, rep in (("g", good), ("s", skip), ("b", bad)):
        (tmp_path / f"{name}.json").write_text(json.dumps(rep))
    assert chk.main([str(tmp_path / "g.json")]) == 0
    assert chk.main([str(tmp_path / "s.json")]) == 1
    assert chk.main([str(tmp_path / "s.json"), "--allow-skip"]) == 0
    assert chk.main([str(tmp_path / "b.json")]) == 1
    mdir = tmp_path / "models"
    assert chk.main(["--models", str(mdir), "--asr", "tiny"]) == 1
    for m in chk.MEDIAPIPE:
        mdir.mkdir(exist_ok=True)
        (mdir / m).write_bytes(b"x")
    (mdir / "whisper-tiny").mkdir()
    (mdir / "whisper-tiny" / "model.bin").write_bytes(b"x")
    assert chk.main(["--models", str(mdir), "--asr", "tiny"]) == 0
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * 3 * 1024 * 1024)
    assert chk.main(["--max-size-mb", "2", str(big)]) == 1
    assert chk.main(["--max-size-mb", "5", str(big)]) == 0
