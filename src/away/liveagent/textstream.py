"""Turning a model's streamed reply into things worth saying out loud.

* :class:`ThinkFilter` drops reasoning traces as they stream (``<think>...</think>``, ``<reasoning>``,
  ``<thinking>``, ``◁think▷...◁/think▷``). Reasoning models (Nemotron, DeepSeek R1, Qwen3...) put them first.
* :class:`SentenceSplitter` hands each finished sentence to the voice as soon as it is complete, so the first
  sentence is spoken while the model is still writing the rest.
* :func:`for_speech` removes markdown, code and links a voice should not read.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

OPEN_TAGS = ("<think>", "<thinking>", "<reasoning>", "\u25c1think\u25b7")
CLOSE_FOR = {
    "<think>": "</think>",
    "<thinking>": "</thinking>",
    "<reasoning>": "</reasoning>",
    "\u25c1think\u25b7": "\u25c1/think\u25b7",
}
_WATCH = (*OPEN_TAGS, "</think>")
_MAX_TAG = max(len(t) for t in (*OPEN_TAGS, *CLOSE_FOR.values()))


class ThinkFilter:
    """Feed chunks, get back only the visible text. A reply that starts with a bare ``</think>`` (some servers
    strip the opening tag) has everything before it dropped too."""

    def __init__(self) -> None:
        self.buf = ""
        self.inside: str | None = None  # the closing tag we wait for
        self.started = False  # any visible text emitted yet

    def feed(self, chunk: str) -> str:
        self.buf += chunk
        out: list[str] = []
        while self.buf:
            low = self.buf.lower()
            if self.inside is not None:
                i = low.find(self.inside)
                if i < 0:
                    keep = len(self.inside) - 1
                    self.buf = self.buf[-keep:] if len(self.buf) > keep else self.buf
                    break
                self.buf = self.buf[i + len(self.inside) :]
                self.inside = None
                continue
            if not self.started and not "".join(out).strip():
                j = low.find("</think>")
                opens = [low.find(t) for t in OPEN_TAGS if low.find(t) >= 0]
                if j >= 0 and (not opens or j < min(opens)):
                    self.buf = self.buf[j + len("</think>") :]
                    out.clear()
                    continue
            hits = [(low.find(t), t) for t in OPEN_TAGS if low.find(t) >= 0]
            if hits:
                i, tag = min(hits)
                out.append(self.buf[:i])
                self.buf = self.buf[i + len(tag) :]
                self.inside = CLOSE_FOR[tag]
                continue
            cut = len(self.buf)  # keep a possible partial tag at the end for the next chunk
            for n in range(1, min(_MAX_TAG, len(self.buf) + 1)):
                if any(t.startswith(low[-n:]) for t in _WATCH):
                    cut = len(self.buf) - n
            out.append(self.buf[:cut])
            self.buf = self.buf[cut:]
            break
        text = "".join(out)
        if text.strip():
            self.started = True
        return text

    def flush(self) -> str:
        if self.inside is not None:
            self.buf = ""
            return ""
        rest, self.buf = self.buf, ""
        return rest


def strip_reasoning(text: str) -> str:
    f = ThinkFilter()
    return (f.feed(text) + f.flush()).strip()


_END = re.compile(r"([.!?\u3002\uff01\uff1f]+[\"')\]]*)(\s+)|(\n\s*\n)")
_MIN_LEN = 12


class SentenceSplitter:
    """Feed visible text; :meth:`feed` returns sentences that are complete. Very short pieces ("Hi.") are joined
    with the next one so the voice does not stutter. The first sentence may go out early at a comma when it is
    long (lower time to the first spoken word)."""

    def __init__(self, first_min: int = 80) -> None:
        self.buf = ""
        self.first_min = first_min
        self.sent = 0

    def _next(self) -> tuple[str, str] | None:
        pos = 0
        while True:
            m = _END.search(self.buf, pos)
            if m is None:
                return None
            end = m.end(1) if m.group(1) else m.start()
            piece = self.buf[:end].strip()
            if len(piece) >= _MIN_LEN:
                return piece, self.buf[m.end() :]
            pos = m.end()

    def feed(self, text: str) -> list[str]:
        self.buf += text
        out: list[str] = []
        while True:
            nxt = self._next()
            if nxt is None:
                break
            piece, self.buf = nxt
            out.append(piece)
        if not out and self.sent == 0 and len(self.buf) > self.first_min:
            i = max(self.buf.rfind(", "), self.buf.rfind("; "))
            if i > 20:
                out.append(self.buf[: i + 1].strip())
                self.buf = self.buf[i + 2 :]
        spoken = [s for s in (for_speech(p) for p in out) if s]
        self.sent += len(spoken)
        return spoken

    def flush(self) -> list[str]:
        rest, self.buf = self.buf.strip(), ""
        s = for_speech(rest)
        return [s] if s else []


_CODE = re.compile(r"```[\s\S]*?(```|$)")
_INLINE = re.compile(r"`([^`]*)`")
_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")
_URL = re.compile(r"https?://\S+")
_MARKS = re.compile(r"(\*\*|__|(?<!\w)\*|(?<!\w)_(?=\w)|#{1,6}\s|^>\s)", re.M)


def for_speech(text: str) -> str:
    t = _CODE.sub(" ", text)
    t = _LINK.sub(r"\1", t)
    t = _URL.sub("a link", t)
    t = _INLINE.sub(r"\1", t)
    t = _MARKS.sub("", t)
    t = re.sub(r"^\s*[-*\u2022]\s+", "", t, flags=re.M)
    return re.sub(r"\s+", " ", t).strip()


def sentences(chunks: Iterable[str]) -> Iterator[str]:
    """Reasoning-free sentences from a stream of raw model chunks."""
    f, s = ThinkFilter(), SentenceSplitter()
    for c in chunks:
        yield from s.feed(f.feed(c))
    yield from s.feed(f.flush())
    yield from s.flush()
