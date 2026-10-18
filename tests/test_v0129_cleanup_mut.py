"""Mutation-strengthening tests for voice/cleanup.py (v0.1.29)."""

from __future__ import annotations

from pathlib import Path

from away.voice.cleanup import (
    Vocabulary,
    apply_spoken_punctuation,
    capitalize,
    ensure_terminal,
    tidy,
)


def test_spoken_punctuation() -> None:
    assert "?" in apply_spoken_punctuation("what question mark")
    assert "," in apply_spoken_punctuation("hello comma world")


def test_spoken_punctuation_case_insensitive() -> None:
    assert "?" in apply_spoken_punctuation("what QUESTION MARK")


def test_vocabulary_parse_arrow() -> None:
    v = Vocabulary.parse("teh -> the")
    assert v.apply("teh cat") == "the cat"


def test_vocabulary_parse_bare_term() -> None:
    v = Vocabulary.parse("Inan")
    assert v.apply("hello inan") == "hello Inan"


def test_vocabulary_parse_skips_comments_and_blanks() -> None:
    v = Vocabulary.parse("# comment\n\nteh -> the")
    assert len(v.replacements) == 1


def test_vocabulary_parse_empty_wrong_skipped() -> None:
    v = Vocabulary.parse("-> the\n")
    assert len(v.replacements) == 0


def test_vocabulary_load_missing() -> None:
    v = Vocabulary.load(None)
    assert v.replacements == []
    v = Vocabulary.load("/nonexistent/path.txt")
    assert v.replacements == []


def test_vocabulary_load_file(tmp_path: Path) -> None:
    p = tmp_path / "vocab.txt"
    p.write_text("teh -> the\n", encoding="utf-8")
    v = Vocabulary.load(p)
    assert v.apply("teh") == "the"


def test_vocabulary_terms() -> None:
    v = Vocabulary.parse("teh -> the\nInan")
    assert "the" in v.terms()
    assert "Inan" in v.terms()


def test_vocabulary_hotwords() -> None:
    v = Vocabulary.parse("teh -> the\nInan")
    hw = v.hotwords()
    assert hw is not None
    assert "the" in hw and "Inan" in hw


def test_vocabulary_hotwords_empty() -> None:
    v = Vocabulary()
    assert v.hotwords() is None


def test_vocabulary_hotwords_dedupes() -> None:
    v = Vocabulary.parse("a -> the\nb -> the")
    hw = v.hotwords()
    assert hw is not None
    assert hw.count("the") == 1


def test_vocabulary_hotwords_limit() -> None:
    v = Vocabulary.parse("\n".join(f"w{i} -> term{i}" for i in range(100)))
    hw = v.hotwords(limit=10)
    assert hw is not None
    assert len(hw.split()) == 10


def test_tidy_collapses_spaces() -> None:
    assert tidy("hello   world") == "hello world"


def test_tidy_fixes_punctuation_spacing() -> None:
    assert tidy("hello , world") == "hello, world"


def test_tidy_removes_duplicate_punct() -> None:
    assert tidy("hello!!") == "hello!"


def test_tidy_strips_leading_punct() -> None:
    assert tidy(", hello") == "hello"


def test_capitalize_i() -> None:
    assert capitalize("i am here") == "I am here"


def test_capitalize_sentence() -> None:
    assert capitalize("hello. world") == "Hello. World"


def test_capitalize_weekday() -> None:
    assert "Tuesday" in capitalize("see you tuesday")


def test_ensure_terminal_adds_period() -> None:
    assert ensure_terminal("hello world foo") == "hello world foo."


def test_ensure_terminal_short_no_period() -> None:
    assert ensure_terminal("hi") == "hi"


def test_ensure_terminal_existing_punct() -> None:
    assert ensure_terminal("hello world foo?") == "hello world foo?"
    assert ensure_terminal("hello world foo!") == "hello world foo!"
