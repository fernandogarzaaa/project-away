import io
import json

import pytest

from away.voice import cleanup as C


@pytest.fixture
def rc():
    return C.RuleCleaner()


def test_fillers_and_stutter():
    assert C.remove_fillers("um so uh we, erm, go") == "so we, go"
    assert C.remove_fillers("the the cat") == "the cat"
    assert C.remove_fillers("Umbrella hummus") == "Umbrella hummus"  # word boundaries
    assert C.remove_fillers("I I I think") == "I think"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("um so i think we should uh meet at 3 no wait 4 pm", "So I think we should meet at 4 pm."),
        ("send it to john no wait mary", "Send it to mary."),
        ("meet at the cafe no, wait at the library tomorrow", "Meet at the library tomorrow."),
        ("hello there period this is wrong scratch that this is right", "Hello there. This is right."),
        ("first one. second one scratch that", "First one."),
        (
            "deploy the release tonight period no wait scratch that deploy the release tomorrow morning period",
            "Deploy the release tomorrow morning.",
        ),
        ("scratch that hi", "Hi"),
        ("is it ready question mark new line yes it is", "Is it ready?\nYes it is."),
        ("the the meeting is at at noon", "The meeting is at noon."),
        ("call me exclamation point", "Call me!"),
        ("note colon buy milk semicolon eggs full stop", "Note: buy milk; eggs."),
        ("line one new paragraph line two", "Line one\n\nLine two."),
        ("i'm sure i can", "I'm sure I can."),
        ("ok", "Ok"),
        ("", ""),
    ],
)
def test_rule_cleaner(rc, raw, expected):
    assert rc.clean(raw) == expected


def test_correction_counts_words_when_no_anchor():
    assert C.apply_corrections("buy red apples no wait green pears") == "buy green pears"
    assert C.apply_corrections("buy apples no wait one two three four") == "one two three four"
    assert C.apply_corrections("no wait hello") == "hello"
    assert C.apply_corrections("hello no wait") == "hello"


def test_number_correction():
    assert C.apply_corrections("set volume to 30 no wait 40") == "set volume to 40"
    assert C.apply_corrections("at 3:30 actually no 4:15 today") == "at 4:15 today"


def test_scratch_that_multiple_sentences():
    assert C.apply_scratch_that("Keep this. Drop this scratch that. And this.") == "Keep this. And this."
    assert C.apply_scratch_that("a! b scratch that c") == "a! c"


def test_scratch_that_after_correction_marker():
    # "no wait" / "actually no" are correction commands, not sentence text:
    # the sentence before the marker is what the user wants gone
    assert C.apply_scratch_that("Deploy tonight. no wait scratch that deploy tomorrow.") == "deploy tomorrow."
    assert C.apply_scratch_that("Deploy tonight no wait scratch that deploy tomorrow") == "deploy tomorrow"
    assert C.apply_scratch_that("Keep this. Drop this actually no scratch that. And this.") == "Keep this. And this."
    assert C.apply_scratch_that("no wait scratch that hi") == "hi"
    # plain scratch-that behaviour is unchanged
    assert C.apply_scratch_that("Deploy tonight scratch that deploy tomorrow.") == "deploy tomorrow."


def test_vocabulary(tmp_path):
    v = C.Vocabulary.parse("# comment\nKubernetes\ngit hub -> GitHub\n\n  pie torch->PyTorch  # trailing\n -> x\n")
    assert v.apply("we use kubernetes and git hub with pie torch") == "we use Kubernetes and GitHub with PyTorch"
    assert v.terms() == ["Kubernetes", "GitHub", "PyTorch"]
    assert v.apply("kubernetesx") == "kubernetesx"  # whole words only
    p = tmp_path / "vocab.txt"
    p.write_text("Inan\n", encoding="utf-8")
    assert C.Vocabulary.load(p).apply("hi inan") == "hi Inan"
    assert C.Vocabulary.load(None).apply("x") == "x"
    assert C.Vocabulary.load(tmp_path / "missing.txt").replacements == []


def test_tidy_and_capitalize():
    assert C.tidy("a ,b  c ..") == "a, b c."
    assert C.tidy(" , hi") == "hi"
    assert C.tidy("a\n \n\n b") == "a\n\nb"
    assert C.tidy("x\n  y") == "x\ny"
    assert C.capitalize("hi. how are you? fine! i think") == "Hi. How are you? Fine! I think"
    assert C.capitalize("this is it") == "This is it"
    assert C.capitalize("iphone and i'll") == "Iphone and I'll"


def test_ensure_terminal():
    assert C.ensure_terminal("one two three") == "one two three."
    assert C.ensure_terminal("one two") == "one two"
    assert C.ensure_terminal("one two three?") == "one two three?"
    assert C.ensure_terminal("") == ""
    assert C.ensure_terminal("a b", min_words=2) == "a b."


def test_code_context_skips_period(rc):
    assert rc.clean("um git status short flag", context="Windows PowerShell") == "git status short flag"
    assert rc.clean("git status short flag", context="Notepad") == "Git status short flag."
    assert C.RuleCleaner(terminal_period=False).clean("one two three") == "One two three"
    assert not C._is_code_context(None)
    assert C._is_code_context("iTerm2")


class FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_llm_cleaner_not_configured_uses_rules(rc):
    llm = C.LLMCleaner(rc, base_url="")
    assert not llm.configured
    assert llm.clean("um hello there friend") == "Hello there friend."


def test_llm_cleaner_success(monkeypatch, rc):
    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["auth"] = req.headers.get("Authorization")
        seen["body"] = json.loads(req.data)
        seen["timeout"] = timeout
        return FakeResp(json.dumps({"choices": [{"message": {"content": " Hello there, friend. "}}]}).encode())

    monkeypatch.setattr(C.urllib.request, "urlopen", fake_urlopen)
    llm = C.LLMCleaner(rc, base_url="http://localhost:11434/v1/", model="m", api_key="k", timeout=3)
    assert llm.clean("um hello there friend", context="Slack") == "Hello there, friend."
    assert seen["url"] == "http://localhost:11434/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    assert seen["timeout"] == 3
    assert seen["body"]["model"] == "m"
    assert seen["body"]["temperature"] == 0
    assert seen["body"]["messages"][0]["content"] == C.SYSTEM_PROMPT
    assert seen["body"]["messages"][1]["content"] == "[active app: Slack]\nHello there friend."
    assert llm.last_error is None


def test_llm_cleaner_no_key_no_auth_header(monkeypatch, rc):
    seen = {}

    def fake_urlopen(req, timeout):
        seen["auth"] = req.headers.get("Authorization")
        seen["user"] = json.loads(req.data)["messages"][1]["content"]
        return FakeResp(json.dumps({"choices": [{"message": {"content": "Hello there friend."}}]}).encode())

    monkeypatch.setattr(C.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.delenv("AWAY_LLM_API_KEY", raising=False)
    llm = C.LLMCleaner(rc, base_url="http://x")
    llm.clean("hello there friend")
    assert seen["auth"] is None
    assert seen["user"] == "Hello there friend."


def test_llm_cleaner_env_config(monkeypatch, rc):
    monkeypatch.setenv("AWAY_LLM_BASE_URL", "http://h/v1")
    monkeypatch.setenv("AWAY_LLM_MODEL", "qwen")
    monkeypatch.setenv("AWAY_LLM_API_KEY", "sk")
    llm = C.LLMCleaner(rc)
    assert (llm.base_url, llm.model, llm.api_key) == ("http://h/v1", "qwen", "sk")


def test_llm_cleaner_falls_back_on_error(monkeypatch, rc):
    def boom(req, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(C.urllib.request, "urlopen", boom)
    llm = C.LLMCleaner(rc, base_url="http://x")
    assert llm.clean("um hello there friend") == "Hello there friend."
    assert "OSError" in llm.last_error


@pytest.mark.parametrize("answer", ["", "x", "Hello there friend. " * 5])
def test_llm_cleaner_rejects_suspicious_output(monkeypatch, rc, answer):
    monkeypatch.setattr(
        C.urllib.request,
        "urlopen",
        lambda req, timeout: FakeResp(json.dumps({"choices": [{"message": {"content": answer}}]}).encode()),
    )
    llm = C.LLMCleaner(rc, base_url="http://x")
    assert llm.clean("hello there friend") == "Hello there friend."
    assert llm.last_error.startswith("rejected")


def test_llm_cleaner_skips_empty(monkeypatch, rc):
    called = []
    monkeypatch.setattr(C.urllib.request, "urlopen", lambda *a, **k: called.append(1))
    assert C.LLMCleaner(rc, base_url="http://x").clean("um") == ""
    assert called == []
