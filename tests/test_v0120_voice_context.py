"""v0.1.20: context-aware dictation cleanup (rules + optional local LLM) and the personal dictionary."""

from __future__ import annotations

import http.server
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from away import config as C
from away.app import fastdl
from away.app import optional_models as om
from away.app.voice_panel import VoicePanel, human_size
from away.voice import smart_cleanup as sc
from away.voice.appcategory import categorize, resolve_tone
from away.voice.cleanup import RuleCleaner
from away.voice.dictation import DictationSession
from away.voice.dictionary import CorrectionWatcher, Dictionary, find_span, learn_pairs, parse_command
from away.voice.llm_backends import (
    EngineError,
    LlamaCppBackend,
    OllamaBackend,
    OpenAIBackend,
    make_backend,
    pick_model_key,
)
from away.voice.output import ListSink
from away.voice.voice_setup import setup_voice_context


# ------------------------------------------------------------------------------------------ fakes
class FakeLLM:
    """Answers with a fixed text (or a function of the prompt) after an optional delay."""

    def __init__(self, answer: Any = None, delay: float = 0.0, error: Exception | None = None) -> None:
        self.answer = answer
        self.delay = delay
        self.error = error
        self.calls: list[tuple[str, str, int]] = []

    def complete(self, system: str, user: str, max_tokens: int, timeout: float) -> str:
        self.calls.append((system, user, max_tokens))
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        if callable(self.answer):
            return str(self.answer(user))
        if self.answer is None:  # echo the dictated text
            return user.rsplit("Dictated text to fix:\n", 1)[-1]
        return str(self.answer)


def cleaner(
    llm: Any = None,
    mode: str = "llm",
    process: str | None = "notepad.exe",
    dictionary: Dictionary | None = None,
    before: str | None = None,
    **kw: Any,
) -> sc.SmartCleaner:
    return sc.SmartCleaner(
        RuleCleaner(),
        mode,
        dictionary,
        llm,
        kw.pop("timeout_ms", 1500),
        kw.pop("tones", None),
        process=lambda: process,
        before_caret=(lambda: before) if before is not None else None,
        **kw,
    )


# ------------------------------------------------------------------------------------------ app category + tone
@pytest.mark.parametrize(
    ("process", "title", "cat"),
    [
        ("slack.exe", None, "chat"),
        ("C:\\Program Files\\Discord\\Discord.exe", None, "chat"),
        ("OUTLOOK.EXE", None, "email"),
        ("WINWORD.EXE", "Report.docx - Word", "docs"),
        ("Code.exe", None, "code"),
        ("WindowsTerminal.exe", None, "terminal"),
        ("pwsh", None, "terminal"),
        ("chrome.exe", "Inbox (3) - me@gmail.com - Gmail", "email"),
        ("chrome.exe", "Project plan - Google Docs", "docs"),
        ("msedge.exe", "(2) WhatsApp", "chat"),
        ("chrome.exe", "Change your password - Example", "other"),  # "password" must not read as Word
        (None, None, "other"),
    ],
)
def test_categorize(process: str | None, title: str | None, cat: str) -> None:
    assert categorize(process, title) == cat


def test_resolve_tone_defaults_and_overrides() -> None:
    assert resolve_tone("chat") == "casual" and resolve_tone("email") == "formal"
    assert resolve_tone("terminal") == "verbatim" and resolve_tone("code") == "code"
    assert resolve_tone("other") == "neutral"
    ov = {"Slack": "formal", "work-browser": "casual", "notepad.exe": "auto"}
    assert resolve_tone("chat", ov, "slack.exe") == "formal"  # process, case- and .exe-insensitive
    assert resolve_tone("other", ov, "chrome.exe", profile="work-browser") == "casual"  # bindings profile name
    assert resolve_tone("docs", ov, "notepad.exe") == "formal"  # auto = the category default


# ------------------------------------------------------------------------------------------ rule corrections
@pytest.mark.parametrize(
    ("spoken", "typed"),
    [
        ("lets meet on monday no wait make it tuesday", "Lets meet on Tuesday."),
        ("um so i think we should uh meet on monday no wait make it tuesday at three", None),
        ("meet monday at three no wait tuesday. see you then", "Meet Tuesday at three. See you then."),
        ("call me at 3 pm no wait 4 pm", "Call me at 4 pm."),
        ("send it to john no wait make it mary", "Send it to mary."),
        ("the total is 50 no wait 60 dollars", "The total is 60 dollars."),
        ("see you on monday, no wait, friday, okay", "See you on Friday, okay."),
        ("i moved in june no wait july", "I moved in July."),
        ("hello there scratch that hi ana how are you", "Hi ana how are you."),
    ],
)
def test_spoken_edits(spoken: str, typed: str | None) -> None:
    out = RuleCleaner().clean(spoken)
    if typed is None:
        assert out == "So I think we should meet on Tuesday at three."
    else:
        assert out == typed


def test_may_and_march_are_not_capitalised_as_months() -> None:
    assert RuleCleaner().clean("you may march on friday") == "You may march on Friday."


# ------------------------------------------------------------------------------------------ guard
def test_guard_accepts_a_cleanup_and_rejects_new_content() -> None:
    src = "hey can you send me the report by friday i need it for the meeting with jonie"
    ok = sc.guard(src, "Hey, can you send me the report by Friday? I need it for the meeting with Jonie.")
    assert ok.ok and ok.similarity > 0.9
    answer = sc.guard("what time is the standup tomorrow", "The standup is at 9:30 AM tomorrow in room B.")
    assert not answer.ok and "new word" in answer.reason
    assert not sc.guard("send it", "").ok
    rewrite = sc.guard("thanks for the update", "Thank you so much for the wonderful and detailed update, truly.")
    assert not rewrite.ok
    dropped = sc.guard("please book the room for ten people on friday afternoon", "Book it.")
    assert not dropped.ok and "length" in dropped.reason
    leaked = sc.guard("meet on tuesday", "Meet on Monday, no wait, Tuesday.")
    assert not leaked.ok
    assert sc.guard("talk to jonie", "Talk to Jonie.", ["Jonie"]).ok
    assert sc.guard("i got twenty five", "I got 25.").ok  # numbers either way round
    assert sc.guard("the report is done", "The report is done, it's done.").ok is True  # small additions are fine


def test_strip_wrapping() -> None:
    assert sc.strip_wrapping('"Hello there."') == "Hello there."
    assert sc.strip_wrapping("Here is the fixed text:\nHello there.") == "Hello there."
    assert sc.strip_wrapping("Here's the corrected text: Hello there.") == "Hello there."
    assert sc.strip_wrapping("Hello there.") == "Hello there."


# ------------------------------------------------------------------------------------------ SmartCleaner
def test_llm_polish_is_used_when_the_guard_accepts_it() -> None:
    llm = FakeLLM("Hey, can you send me the report by Friday? I need it for the meeting.")
    c = cleaner(llm)
    out = c.clean("hey can you send me the report by friday i need it for the meeting")
    assert out == "Hey, can you send me the report by Friday? I need it for the meeting."
    assert c.stats.method == "llm" and c.stats.tone == "formal" and c.stats.category == "docs"
    system, user, max_tokens = llm.calls[0]
    assert "email or document" in system and user.endswith("I need it for the meeting.")
    assert 16 < max_tokens < 100


def test_llm_that_invents_content_falls_back_to_rules() -> None:
    c = cleaner(FakeLLM("The standup is at 9:30 tomorrow, see you there."))
    assert c.clean("what time is the standup tomorrow") == "What time is the standup tomorrow?"
    assert c.stats.method == "fallback" and "new word" in (c.stats.rejected or "")


def test_timeout_types_the_rule_output_within_the_budget_and_skips_while_busy() -> None:
    slow = FakeLLM("Never used.", delay=0.6)
    c = cleaner(slow, timeout_ms=150)
    t = time.monotonic()
    out = c.clean("this is a slow one")
    took = time.monotonic() - t
    assert out == "This is a slow one." and c.stats.method == "fallback" and "150 ms" in (c.stats.error or "")
    assert took < 0.45  # the budget plus scheduling slack, never the model's 0.6 s
    # the slow call is still running: the next dictation does not queue behind it
    t = time.monotonic()
    assert c.clean("next one please") == "Next one please." and time.monotonic() - t < 0.1
    assert "still running" in (c.stats.error or "") and len(slow.calls) == 1
    time.sleep(0.6)
    c.clean("and again")
    assert len(slow.calls) == 2


def test_engine_error_is_friendly_and_carries_a_fix() -> None:
    c = cleaner(FakeLLM(error=EngineError("Smart cleanup needs its model.", "download")))
    assert c.clean("hello world again") == "Hello world again."
    assert c.stats.method == "fallback" and c.stats.fix == "download" and "needs its model" in (c.stats.error or "")
    c2 = cleaner(FakeLLM(error=ConnectionError("refused")))
    assert c2.clean("hello world again") == "Hello world again." and "ConnectionError" in (c2.stats.error or "")


@pytest.mark.parametrize("process", ["Code.exe", "WindowsTerminal.exe"])
def test_code_editors_and_terminals_never_call_the_model(process: str) -> None:
    llm = FakeLLM("SHOULD NOT APPEAR")
    c = cleaner(llm, process=process)
    assert c.clean("git status") == "git status"
    assert llm.calls == [] and c.stats.tone in ("code", "verbatim")


def test_chat_is_casual_and_a_continued_sentence_stays_lowercase() -> None:
    c = cleaner(None, mode="rules", process="slack.exe")
    assert c.clean("sounds good see you then") == "Sounds good see you then"  # one short chat line: no period
    assert c.clean("ok. see you at noon") == "Ok. See you at noon."  # two sentences keep their punctuation
    d = cleaner(None, mode="rules", process="notepad.exe", before="Thanks for the")
    assert d.clean("quick reply") == "quick reply"
    e = cleaner(None, mode="rules", process="notepad.exe", before="Thanks for the update.")
    assert e.clean("see you soon") == "See you soon."


def test_mode_off_types_what_whisper_heard_plus_dictionary_fixes(tmp_path: Path) -> None:
    d = Dictionary(tmp_path / "d.json")
    d.add_fix("jonnie", "Jonie")
    c = cleaner(None, mode="off", dictionary=d)
    assert c.clean("um  ask jonnie   about it") == "um ask Jonie about it"
    assert c.stats.method == "off"


def test_dictionary_fixes_apply_before_and_after_the_model(tmp_path: Path) -> None:
    d = Dictionary(tmp_path / "d.json")
    d.add_fix("git hub", "GitHub")
    llm = FakeLLM(lambda user: "Push it to github now.")
    c = cleaner(llm, dictionary=d)
    assert c.clean("push it to git hub now") == "Push it to GitHub now."
    assert "GitHub" in llm.calls[0][1]  # the model already saw the right spelling


def test_dictionary_voice_commands_type_nothing(tmp_path: Path) -> None:
    d = Dictionary(tmp_path / "d.json")
    events: list[tuple[str, dict[str, Any]]] = []
    c = cleaner(None, mode="rules", dictionary=d, on_event=lambda t, data: events.append((t, data)))
    assert c.clean("Add Jonie to my dictionary.") == ""
    assert d.words() == ["Jonie"] and events[-1][0] == "dictionary.command"
    # "correct X to Y" needs X in what was just dictated, otherwise it is prose and gets typed
    assert c.clean("correct the plan to match the budget") == "Correct the plan to match the budget."
    assert c.clean("ping ana about the sink") == "Ping ana about the sink."
    assert c.clean("Correct Ana to Anna.") == ""
    assert c.clean("ping ana now please") == "Ping Anna now please."
    assert c.clean("remove Jonie from the dictionary") == "" and "Jonie" not in d.words()


def test_dictation_session_with_smart_cleaner_types_and_runs_commands(tmp_path: Path) -> None:
    class ASR:
        def __init__(self) -> None:
            self.next = ""

        def transcribe(self, audio: Any, prompt: str | None = None) -> str:
            return self.next

    class VAD:
        def trim(self, audio: Any, pad_ms: int = 150) -> Any:
            return audio

    asr, sink = ASR(), ListSink()
    d = Dictionary(tmp_path / "d.json")
    sess = DictationSession(asr, cleaner(None, mode="rules", dictionary=d), sink, VAD(), context=lambda: "Notes")
    for text in ("add Jonie to my dictionary", "say hi to jonie"):
        asr.next = text
        sess.start()
        sess.feed(np.ones(16000, dtype=np.float32) * 0.1)
        res = sess.stop()
    assert sink.items == ["Say hi to Jonie. "] and res.status == "ok"


# ------------------------------------------------------------------------------------------ dictionary
def test_dictionary_words_fixes_casing_and_persistence(tmp_path: Path) -> None:
    p = tmp_path / "dictionary.json"
    d = Dictionary(p)
    assert d.add("  Jonie. ") is None and d.add("") == "type a word or name"
    assert "too many words" in (d.add("one two three four five") or "")
    assert d.add("a -> b") is not None
    d.add("GitHub")
    d.add("Will")
    assert d.apply("ask will about the github repo") == "ask will about the GitHub repo"  # "Will" stays a word
    assert d.add_fix("jonnie", "Jonie") is None
    assert d.apply("Jonnie and JONNIE met jonnie's dog") == "Jonie and Jonie met Jonie's dog"
    assert d.add_fix("same", "same") == "the two spellings are the same"
    d2 = Dictionary.load(p)
    assert set(d2.words()) == {"Jonie", "GitHub", "Will"}
    assert d2.apply("jonnie") == "Jonie"
    assert d2.hotwords(["Ruff"]) == "Ruff Jonie GitHub Will"
    assert d2.remove("github") and not d2.remove("github")


def test_learned_fix_needs_to_be_seen_twice(tmp_path: Path) -> None:
    d = Dictionary(tmp_path / "d.json", learn_after=2)
    assert not d.observe_fix("Jonnie", "Jonie") and d.apply("Jonnie") == "Jonnie"
    assert Dictionary.load(tmp_path / "d.json", 2).to_json()["pending"]  # pending counts survive a restart
    assert d.observe_fix("jonnie", "Jonie")
    assert d.apply("hi jonnie") == "hi Jonie" and d.entries()[0].source == "learned"
    assert not d.observe_fix("jonnie", "Jonie")  # already known


def test_settings_and_app_share_the_file(tmp_path: Path) -> None:
    p = tmp_path / "d.json"
    app, settings = Dictionary.load(p), Dictionary.load(p)
    seen: list[int] = []
    app.on_change.append(lambda: seen.append(1))
    settings.add_fix("kubernetes cluster", "Kubernetes cluster")
    settings.add_fix("jonnie", "Jonie")
    time.sleep(0.01)
    assert app.apply("ask jonnie") == "ask Jonie" and seen  # the app picks the Settings edit up by itself
    app.add("Ruff")  # and an app-side change keeps the Settings words
    assert set(Dictionary.load(p).words()) == {"Kubernetes cluster", "Jonie", "Ruff"}


def test_corrupt_file_is_kept_aside(tmp_path: Path) -> None:
    p = tmp_path / "d.json"
    p.write_text("{not json", encoding="utf-8")
    d = Dictionary.load(p)
    assert d.words() == [] and (tmp_path / "d.json.bad").exists()


def test_import_export_text_and_json(tmp_path: Path) -> None:
    d = Dictionary(tmp_path / "d.json")
    d.add("Jonie")
    d.add_fix("pie torch", "PyTorch")
    txt, js = tmp_path / "w.txt", tmp_path / "w.json"
    d.export_file(txt)
    d.export_file(js)
    assert "pie torch -> PyTorch" in txt.read_text(encoding="utf-8")
    e = Dictionary(tmp_path / "e.json")
    assert e.import_file(txt) == 3
    assert set(e.words()) == {"Jonie", "PyTorch"} and e.apply("pie torch") == "PyTorch"
    f = Dictionary(tmp_path / "f.json")
    assert f.import_file(js) == 2 and f.apply("pie torch") == "PyTorch"
    g = Dictionary(tmp_path / "g.json")
    assert g.import_text("# comment\nRTX 2060\nwisper -> Whisper\n\n") == 2
    assert g.apply("rtx 2060 and wisper") == "RTX 2060 and Whisper"


@pytest.mark.parametrize(
    ("text", "action", "term", "wrong"),
    [
        ("Add Jonie to my dictionary.", "add", "Jonie", ""),
        ("remember the name Ana Reyes to the dictionary", "add", "Ana Reyes", ""),
        ("Remove GitHub from my dictionary", "remove", "GitHub", ""),
        ("Correct Jonnie to Jonie.", "correct", "Jonie", "Jonnie"),
        ("fix git hub as GitHub", "correct", "GitHub", "git hub"),
        ("Spell Jonie as J O N I E.", "correct", "JONIE", "Jonie"),
        ("Spell Jonie as J-O-N-I-E", "correct", "JONIE", "Jonie"),
    ],
)
def test_parse_command(text: str, action: str, term: str, wrong: str) -> None:
    cmd = parse_command(text)
    assert cmd is not None and (cmd.action, cmd.term, cmd.wrong) == (action, term, wrong)


def test_parse_command_ignores_prose() -> None:
    assert parse_command("I want to add more tests to the suite") is None
    assert parse_command("hello there") is None
    assert parse_command("correct " + "word " * 20 + "to x") is None


def test_find_span_and_learn_pairs() -> None:
    field = "Dear team,\nPlease ask Jonie to review the PyTorch port today. Thanks"
    assert find_span("please ask Jonnie to review the pie torch port today.", field) == (
        "Please ask Jonie to review the PyTorch port today."
    )
    assert find_span("something completely different", field) is None
    assert learn_pairs("ask Jonnie to review it", "ask Jonie to review it") == [("Jonnie", "Jonie")]
    assert learn_pairs("push to github today", "push to GitHub today") == [("github", "GitHub")]
    assert learn_pairs("port it to pie torch", "port it to PyTorch") == [("pie torch", "PyTorch")]
    assert learn_pairs("send a file", "send the file") == []  # grammar, not vocabulary
    assert learn_pairs("ask him to call me", "ask him to call me tomorrow") == []  # an addition
    assert learn_pairs("the cat sat on the mat today", "a completely different sentence now") == []
    assert learn_pairs("one two three four five six", "uno two tres four cinco six") == []  # too many edits


def test_correction_watcher_learns_from_the_final_edit() -> None:
    d = Dictionary(None, learn_after=1)
    now = [0.0]
    field = {"text": "", "app": "Notes"}
    learned: list[tuple[str, str]] = []
    w = CorrectionWatcher(
        d,
        lambda: field["text"],
        app=lambda: field["app"],
        watch_seconds=10,
        clock=lambda: now[0],
        on_learned=lambda a, b: learned.append((a, b)),
    )
    w.on_inserted("Ask Jonnie about the PR.", "Notes")
    field["text"] = "Hi! Ask Jonnie about the PR."
    w.poll()
    now[0] = 3
    field["text"] = "Hi! Ask Jon about the PR."  # mid-edit
    w.poll()
    now[0] = 6
    field["text"] = "Hi! Ask Jonie about the PR."
    w.poll()
    assert learned == []  # still inside the watch window
    now[0] = 11
    w.poll()
    assert learned == [("Jonnie", "Jonie")] and d.apply("jonnie") == "Jonie"


def test_correction_watcher_ignores_other_apps_and_unreadable_fields() -> None:
    d = Dictionary(None, learn_after=1)
    now = [0.0]
    state = {"app": "Notes", "text": None}
    w = CorrectionWatcher(d, lambda: state["text"], app=lambda: state["app"], watch_seconds=5, clock=lambda: now[0])
    w.on_inserted("Ask Jonnie.", "Notes")
    w.poll()  # unreadable field: nothing to compare
    state["app"] = "Browser"
    state["text"] = "Ask Jonie."  # another app's text must never be compared
    w.poll()
    now[0] = 9
    w.poll()
    w.flush()
    assert w.observed == [] and d.words() == []


# ------------------------------------------------------------------------------------------ engines
class _Handler(http.server.BaseHTTPRequestHandler):
    seen: list[dict[str, Any]] = []  # noqa: RUF012

    def log_message(self, *a: Any) -> None:
        pass

    def _send(self, obj: Any) -> None:
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._send({"models": [{"name": "qwen2.5:1.5b"}]})

    def do_POST(self) -> None:
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.seen.append({"path": self.path, "body": data, "auth": self.headers.get("Authorization")})
        if self.path == "/api/chat":
            self._send({"message": {"content": " Hello there. "}})
        else:
            self._send({"choices": [{"message": {"content": "Hello there."}}]})


@pytest.fixture
def server() -> Any:
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    _Handler.seen.clear()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_ollama_backend_over_http(server: str) -> None:
    b = OllamaBackend(server, "qwen2.5:1.5b")
    b.check()
    assert b.complete("sys", "hello there", 20, 5) == "Hello there."
    body = _Handler.seen[-1]["body"]
    assert body["stream"] is False and body["options"] == {"temperature": 0, "num_predict": 20}
    with pytest.raises(EngineError) as e:
        OllamaBackend(server, "llama3").check()
    assert e.value.fix == "pick_model"
    with pytest.raises(EngineError) as e2:
        OllamaBackend("http://127.0.0.1:9", "x").check(timeout=0.5)
    assert e2.value.fix == "start_ollama"


def test_openai_backend_over_http(server: str) -> None:
    b = OpenAIBackend(server, "m", "secret")
    assert b.complete("sys", "hello there", 30, 5) == "Hello there."
    assert _Handler.seen[-1]["path"] == "/chat/completions" and _Handler.seen[-1]["auth"] == "Bearer secret"
    with pytest.raises(EngineError):
        OpenAIBackend().complete("s", "u", 5, 1)  # AWAY_LLM_BASE_URL is unset in tests


def test_make_backend_and_model_choice(tmp_path: Path) -> None:
    cfg = C.AppConfig()
    c = cfg.voice.cleanup
    assert pick_model_key(tmp_path, gpu=True) == "qwen2.5-1.5b"
    assert pick_model_key(tmp_path, gpu=False) == "qwen2.5-0.5b"
    light = om.get("qwen2.5-0.5b")
    (tmp_path / light.filename).write_bytes(b"\0" * 10)  # wrong size: not installed
    assert pick_model_key(tmp_path, gpu=True) == "qwen2.5-1.5b"
    b = make_backend(c, tmp_path, gpu=False)
    assert isinstance(b, LlamaCppBackend) and b.model_path == tmp_path / light.filename
    with pytest.raises(EngineError) as e:
        LlamaCppBackend(tmp_path / "missing.gguf").check()
    assert e.value.fix == "download"
    c.model = str(tmp_path / "mine.gguf")
    assert make_backend(c, tmp_path, gpu=False).model_path == tmp_path / "mine.gguf"  # type: ignore[attr-defined]
    c.backend, c.model = "ollama", "auto"
    ob = make_backend(c, tmp_path, gpu=True)
    assert isinstance(ob, OllamaBackend) and ob.model == "qwen2.5:1.5b"
    c.backend = "openai"
    assert isinstance(make_backend(c, tmp_path, gpu=True), OpenAIBackend)


# ------------------------------------------------------------------------------------------ config
def test_config_keys_round_trip_and_validate() -> None:
    cfg, unknown = C.from_dict(
        {
            "voice": {
                "cleanup": {"mode": "llm", "model": "qwen2.5-0.5b", "timeout_ms": 900, "tones": {"slack": "formal"}},
                "dictionary": {"learn_after": 3, "watch_seconds": 30},
            }
        }
    )
    C.validate(cfg)
    assert unknown == [] and cfg.voice.cleanup.timeout_ms == 900 and cfg.voice.dictionary.learn_after == 3
    d = C.to_dict(cfg)["voice"]
    assert d["cleanup"]["tones"] == {"slack": "formal"} and d["dictionary"]["watch_seconds"] == 30
    assert C.AppConfig().voice.cleanup.mode == "rules"  # nothing changes for existing users
    off, _ = C.from_dict({"voice": {"cleanup": {"mode": False}}})  # YAML `off` (from_dict validates)
    assert off.voice.cleanup.mode == "off"
    for bad, msg in (
        ({"mode": "smart"}, "voice.cleanup.mode"),
        ({"backend": "gpt"}, "voice.cleanup.backend"),
        ({"timeout_ms": 50}, "voice.cleanup.timeout_ms"),
        ({"tones": {"slack": "loud"}}, "voice.cleanup.tones.slack"),
        ({"context": "yes"}, "voice.cleanup.context"),
    ):
        with pytest.raises(ValueError, match=msg.replace(".", r"\.")):
            C.from_dict({"voice": {"cleanup": bad}})
    with pytest.raises(ValueError, match="learn_after"):
        C.from_dict({"voice": {"dictionary": {"learn_after": 0}}})


def test_build_keeps_the_old_llm_cleanup_switch_working(tmp_path: Path) -> None:
    cfg = C.AppConfig()
    cfg.voice.llm_cleanup = True
    c = sc.build(cfg, tmp_path, RuleCleaner())
    assert c.mode == "llm" and isinstance(c.backend, OpenAIBackend)
    cfg2 = C.AppConfig()
    assert sc.build(cfg2, tmp_path, RuleCleaner()).backend is None  # rules: no engine at all


# ------------------------------------------------------------------------------------------ registry
def test_cleanup_models_are_apache_pinned_and_not_restricted() -> None:
    keys = [m.key for m in om.CLEANUP_MODELS]
    assert keys == ["qwen2.5-1.5b", "qwen2.5-0.5b"]
    for m in om.CLEANUP_MODELS:
        assert m.license == "Apache-2.0" and not m.restricted and m.kind == "cleanup"
        assert m.url.startswith("https://huggingface.co/Qwen/") and "/resolve/" in m.url and "/main/" not in m.url
        assert m.url.endswith("-q4_k_m.gguf") and len(m.sha256) == 64 and m.filename.endswith(".gguf")
        assert om.get(m.key) is m
    assert om.get("qwen2.5-1.5b").size_bytes == 1_117_320_736
    assert [m.key for m in om.MODELS] == ["l2cs", "mobilegaze"]  # the gaze list is unchanged
    assert {s.model.key for s in om.status(Path("/nonexistent"), kind="cleanup")} == set(keys)


# ------------------------------------------------------------------------------------------ Settings panel
def test_voice_panel_toggle_download_and_friendly_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = C.AppConfig()
    engine = {"ok": False}
    p = VoicePanel(
        cfg,
        tmp_path / "models",
        tmp_path / "d.json",
        gpu=False,
        engine_available=lambda: engine["ok"],
        engine_offer=lambda: None,  # no ready-made engine for this computer: the "Learn how" path
    )
    assert not p.smart_on() and p.problem() is None
    prob = p.set_smart(True)
    assert cfg.voice.cleanup.mode == "llm" and prob is not None and prob.fix == "download"
    assert prob.button == "Download (491 MB)" and "Qwen" not in prob.message + prob.button  # no model names
    m = om.get("qwen2.5-0.5b")
    seen: list[tuple[int, int]] = []

    def fake_download(url: str, dest: Path, rf: Any, meter: Any) -> Path:
        assert rf.sha256 == m.sha256 and url == m.url
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            fh.truncate(m.size_bytes)
        meter.add(m.size_bytes // 2)
        meter.add(m.size_bytes - m.size_bytes // 2)
        return dest

    monkeypatch.setattr(fastdl, "download_file", fake_download)
    assert p.download(lambda d, t: seen.append((d, t))) is None
    assert seen[-1] == (m.size_bytes, m.size_bytes) and p.model_ready()
    prob = p.problem()
    assert prob is not None and prob.fix == "install_engine" and prob.button == "Learn how"
    assert "Basic cleanup keeps working" in prob.message
    engine["ok"] = True
    assert p.problem() is None
    assert p.set_smart(False) is None and cfg.voice.cleanup.mode == "rules"
    monkeypatch.setattr(fastdl, "download_file", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("x")))
    om.remove("qwen2.5-0.5b", tmp_path / "models")
    err = p.download()
    assert err is not None and "resume" in err and "ConnectionError" in err
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
    assert p.fix_action("start_ollama") == "https://ollama.com/download" and opened
    assert p.fix_action("download") is None


def test_voice_panel_download_also_fetches_the_ready_made_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v0.1.20 installers: Download fetches the model and the pinned llama-cpp-python wheel in one go."""
    from types import SimpleNamespace

    cfg = C.AppConfig()
    state = {"engine": False}
    wheel = SimpleNamespace(size=8_000_000)
    calls: list[str] = []

    def install_engine(progress: Any = None) -> Path:
        calls.append("engine")
        if progress is not None:
            progress(wheel.size, wheel.size)
        state["engine"] = True
        return tmp_path

    p = VoicePanel(
        cfg,
        tmp_path / "models",
        tmp_path / "d.json",
        gpu=False,
        engine_available=lambda: state["engine"],
        engine_offer=lambda: wheel,
        engine_install=install_engine,
    )
    prob = p.set_smart(True)
    m = om.get("qwen2.5-0.5b")
    assert prob is not None and prob.fix == "download" and prob.button == "Download (499 MB)"

    def fake_download(url: str, dest: Path, rf: Any, meter: Any) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            fh.truncate(m.size_bytes)
        meter.add(m.size_bytes)
        return dest

    monkeypatch.setattr(fastdl, "download_file", fake_download)
    seen: list[tuple[int, int]] = []
    assert p.download(lambda d, t: seen.append((d, t))) is None
    assert calls == ["engine"] and seen[-1] == (m.size_bytes + wheel.size, m.size_bytes + wheel.size)
    assert p.problem() is None
    # the model already there, the engine missing: Download fetches only the engine
    state["engine"] = False
    prob = p.problem()
    assert prob is not None and prob.fix == "download" and prob.button == "Download (8 MB)"
    from away.voice.engine import EngineInstallError

    def broken(progress: Any = None) -> Path:
        raise EngineInstallError("Smart cleanup's engine needs the Microsoft Visual C++ Redistributable (x64).")

    p.engine_install = broken
    assert "Visual C++" in (p.download() or "")


def test_voice_panel_advanced_and_words(tmp_path: Path) -> None:
    cfg = C.AppConfig()
    p = VoicePanel(cfg, tmp_path, tmp_path / "d.json", gpu=True, engine_available=lambda: True)
    assert p.download_label() == "Download (1.1 GB)"
    assert len(p.backend_choices()) == 3 and p.backend_label().startswith("Built-in")
    assert p.set_backend_label("Ollama (a separate local app)") is None and cfg.voice.cleanup.backend == "ollama"
    assert p.set_backend_label("nope") is not None
    assert p.set_timeout_ms("800") is None and cfg.voice.cleanup.timeout_ms == 800
    assert p.set_timeout_ms("abc") and p.set_timeout_ms(50)
    assert p.set_tones_text("slack.exe = casual\n# note\nOUTLOOK = Formal\n") is None
    assert cfg.voice.cleanup.tones == {"slack.exe": "casual", "OUTLOOK": "formal"}
    assert "slack.exe = casual" in p.tones_text()
    assert "line 1" in (p.set_tones_text("slack loud") or "") and cfg.voice.cleanup.tones  # unchanged on error
    C.validate(cfg)
    assert p.add_word("Jonie") is None and p.add_word("ana") is None
    assert "Couldn't add" in (p.add_word("") or "")
    assert p.words() == ["ana", "Jonie"] and p.remove_word("ana") and p.words() == ["Jonie"]
    out = tmp_path / "words.txt"
    assert p.export_words(out) is None and "Jonie" in out.read_text(encoding="utf-8")
    (tmp_path / "in.txt").write_text("Kubernetes\n", encoding="utf-8")
    assert p.import_words(tmp_path / "in.txt") == (None, 1) and "Kubernetes" in p.words()
    err, n = p.import_words(tmp_path / "missing.txt")
    assert err and n == 0
    assert human_size(491_400_032) == "491 MB"


# ------------------------------------------------------------------------------------------ app wiring
def test_setup_voice_context_shares_live_terms_with_whisper(tmp_path: Path) -> None:
    cfg = C.AppConfig()
    cfg.voice.dictionary.file = str(tmp_path / "d.json")

    class Bus:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, Any]]] = []

        def publish(self, topic: str, data: dict[str, Any]) -> None:
            self.events.append((topic, data))

    class App:
        def __init__(self) -> None:
            self.cfg, self.models_dir, self.bus = cfg, tmp_path, Bus()
            self.stop_event = threading.Event()

    from away.voice.cleanup import Vocabulary

    app = App()
    vocab = Vocabulary.parse("Ruff\n")
    ctx = setup_voice_context(app, RuleCleaner(vocab), vocab)
    assert isinstance(ctx.cleaner, sc.SmartCleaner) and ctx.terms == ["Ruff"]

    class Sess:
        vocabulary: list[str] = []  # noqa: RUF012

    class ASR:
        hotwords: str | None = "Ruff"

    sess, asr = Sess(), ASR()
    ctx.attach(asr, sess, None)
    assert ctx.dictionary is not None
    ctx.dictionary.add("Jonie")
    assert sess.vocabulary is ctx.terms and ctx.terms == ["Ruff", "Jonie"] and asr.hotwords == "Ruff Jonie"
    assert ctx.hotwords() == "Ruff Jonie"
    app.stop_event.set()


def test_setup_voice_context_falls_back_to_rules_on_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = C.AppConfig()
    cfg.voice.dictionary.file = str(tmp_path / "d.json")

    class App:
        bus = None
        models_dir = tmp_path

    App.cfg = cfg  # type: ignore[attr-defined]
    monkeypatch.setattr(sc, "build", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    from away.voice.cleanup import Vocabulary

    rules = RuleCleaner()
    assert setup_voice_context(App(), rules, Vocabulary()).cleaner is rules


# ------------------------------------------------------------------------------------------ real model (optional)
@pytest.mark.slow
def test_real_model_smoke() -> None:
    """Runs only with llama-cpp-python installed and AWAY_TEST_GGUF pointing at a cleanup model file."""
    pytest.importorskip("llama_cpp")
    path = os.environ.get("AWAY_TEST_GGUF")
    if not path or not Path(path).is_file():
        pytest.skip("set AWAY_TEST_GGUF to a downloaded cleanup model (.gguf) to run the real-model smoke test")
    c = cleaner(LlamaCppBackend(Path(path), n_ctx=1024, gpu_layers=0), timeout_ms=30000)
    out = c.clean("hey can you send me the report by friday i need it for the meeting")
    assert out and c.stats.method in ("llm", "fallback")
    assert "report" in out.lower() and "friday" in out.lower()
