"""ui_helper: missing/corrupt session spec must fail clean, never traceback.

Regression test for the FileNotFoundError crash in ``ui_helper.main`` when the
agent-link UI was opened with a session id whose ``<id>.json`` was never
written (e.g. two app copies fighting over AppData). ``main`` must print a
human-readable message to stderr and return exit code 2.
"""

from __future__ import annotations

import json

from away.agentlink import ui_helper


def test_main_missing_spec_file_returns_2_with_clean_message(tmp_path, capsys):
    missing = tmp_path / "no-such-session.json"
    assert not missing.exists()

    rc = ui_helper.main("prompt", missing)

    assert rc == 2
    err = capsys.readouterr().err
    assert "no session file" in err
    assert "pairing link" in err
    assert "Traceback" not in err


def test_main_corrupt_spec_file_returns_2_with_clean_message(tmp_path, capsys):
    bad = tmp_path / "corrupt.json"
    bad.write_text("{not valid json", encoding="utf-8")

    rc = ui_helper.main("toast", bad)

    assert rc == 2
    err = capsys.readouterr().err
    assert "unreadable session file" in err
    assert "Traceback" not in err


def test_main_valid_spec_passes_through_to_the_window(tmp_path, monkeypatch):
    # A well-formed spec must get past the loading guard and reach the window code. The window itself is
    # replaced so no Tk main loop runs (a real one would block forever on a desktop CI runner).
    spec = tmp_path / "ok.json"
    spec.write_text(json.dumps({"agent": "cookie", "text": "hi"}), encoding="utf-8")
    seen = {}

    def fake_message(s, ack):
        seen["spec"], seen["ack"] = s, ack
        return 0

    monkeypatch.setattr(ui_helper, "_message", fake_message)
    assert ui_helper.main("toast", spec) == 0
    assert seen == {"spec": {"agent": "cookie", "text": "hi"}, "ack": None}


def test_main_unknown_kind_fails_clean_without_opening_tk(tmp_path, capsys):
    spec = tmp_path / "ok.json"
    spec.write_text("{}", encoding="utf-8")
    assert ui_helper.main("bogus-kind", spec) == 2
    assert "unknown window kind" in capsys.readouterr().err


def test_main_non_object_spec_returns_2(tmp_path, capsys):
    spec = tmp_path / "list.json"
    spec.write_text("[1, 2]", encoding="utf-8")
    assert ui_helper.main("prompt", spec) == 2
    assert "expected a JSON object" in capsys.readouterr().err


def test_main_spec_path_is_a_directory_returns_2(tmp_path, capsys):
    assert ui_helper.main("prompt", tmp_path) == 2
    assert "unreadable session file" in capsys.readouterr().err


def test_prompt_texts_defaults():
    t = ui_helper.prompt_texts({})
    assert t["title"] == "Your AI agent asks"
    assert [b for b, _ in t["buttons"]] == ["Yes", "No"]


def test_toast_texts_truncates():
    t = ui_helper.toast_texts({"agent": "cookie", "text": "x" * 500})
    assert t["title"] == "Message from cookie"
    assert len(t["text"]) == 300


def test_toast_geometry_stays_inside_work_area():
    x, y = ui_helper.toast_geometry((0, 0, 1920, 1040), 380, 120)
    assert x == 1920 - 380 - ui_helper.TOAST_MARGIN
    assert y == 1040 - 120 - ui_helper.TOAST_MARGIN
