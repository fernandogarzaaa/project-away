"""Edit the selected text by voice (``voice.edit_hotkey``).

Press the hotkey: the selection is copied, the mic records your instruction.
Press it again: the instruction is transcribed and applied, the result is pasted
over the selection, and the clipboard is restored.

:func:`apply_edit` tries deterministic rule transforms first ("uppercase", "replace
X with Y", "clean up", "numbers as digits", "make it a bullet list"...), then the
optional OpenAI-compatible LLM (the same ``AWAY_LLM_*`` settings as dictation
cleanup) for free-form instructions. If neither applies the selection is left
untouched. Idea from OpenWhispr's edit-in-place hotkey (MIT); original code.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from away.voice.cleanup import RuleCleaner, tidy
from away.voice.hallucination import clean_transcript
from away.voice.numbers import words_to_digits

EDIT_SYSTEM_PROMPT = (
    "You edit text. Apply the user's instruction to the TEXT and output only the edited text, "
    "with no explanations, quotes or preamble. Keep the language of the text unless told otherwise."
)


def _bullets(s: str) -> str:
    items = [x.strip(" -*\u2022") for x in re.split(r"\n+|(?<=[.;])\s+", s.strip()) if x.strip(" -*\u2022")]
    return "\n".join(f"- {x.rstrip('.;')}" for x in items)


def _sentence_case(s: str) -> str:
    low = s.lower()
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), low)


RULES: list[tuple[re.Pattern[str], Callable[[re.Match[str], str], str]]] = [
    (re.compile(r"^(?:make it |convert (?:it |this )?to )?(?:upper ?case|all caps|capitals)$"), lambda m, s: s.upper()),
    (re.compile(r"^(?:make it |convert (?:it |this )?to )?lower ?case$"), lambda m, s: s.lower()),
    (re.compile(r"^(?:make it |convert (?:it |this )?to )?title case$"), lambda m, s: s.title()),
    (re.compile(r"^(?:make it |convert (?:it |this )?to )?sentence case$"), lambda m, s: _sentence_case(s)),
    (
        re.compile(
            r"^(?:clean (?:it |this )?up|tidy (?:it |this )?up|fix (?:the )?(?:punctuation|capitali[sz]ation))$"
        ),
        lambda m, s: RuleCleaner().clean(s),
    ),
    (re.compile(r"^(?:delete|remove|erase) (?:it|this|that|everything)$"), lambda m, s: ""),
    (re.compile(r"^(?:wrap (?:it |this )?in|add) (?:double )?quotes$"), lambda m, s: f'"{s}"'),
    (re.compile(r"^(?:make (?:it|this) )?(?:a )?bullet(?:ed)? (?:list|points)$"), lambda m, s: _bullets(s)),
    (re.compile(r"^(?:numbers (?:as|to) digits|use digits)$"), lambda m, s: words_to_digits(s, 0)),
    (re.compile(r"^(?:remove|trim) (?:extra )?(?:spaces|whitespace)$"), lambda m, s: tidy(s)),
    (
        re.compile(r"^(?:replace|change) (.+?) (?:with|to) (.+)$"),
        lambda m, s: re.sub(re.escape(m.group(1)), m.group(2).replace("\\", r"\\"), s, flags=re.IGNORECASE),
    ),
]


def normalize_instruction(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"^(?:please |can you |could you )+", "", t)
    return re.sub(r"[.!?,]+$", "", t).strip()


class EditLLM(Protocol):
    def edit(self, instruction: str, text: str) -> str: ...


class LLMEditor:
    """Free-form edits via an OpenAI-compatible ``/chat/completions`` endpoint (``AWAY_LLM_BASE_URL``...)."""

    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None) -> None:
        self.base_url = (base_url or os.environ.get("AWAY_LLM_BASE_URL") or "").rstrip("/")
        self.model = model or os.environ.get("AWAY_LLM_MODEL") or "llama3.2"
        self.api_key = api_key or os.environ.get("AWAY_LLM_API_KEY")

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    def edit(self, instruction: str, text: str) -> str:  # pragma: no cover - network
        body = json.dumps(
            {
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": EDIT_SYSTEM_PROMPT},
                    {"role": "user", "content": f"INSTRUCTION: {instruction}\nTEXT:\n{text}"},
                ],
            }
        ).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base_url + "/chat/completions", data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read())
        return str(data["choices"][0]["message"]["content"]).strip()


def apply_edit(instruction: str, selection: str, llm: EditLLM | None = None) -> tuple[str, str]:
    """Returns ``(new_text, method)`` with method ``rule`` | ``llm`` | ``unchanged``."""
    ins = normalize_instruction(instruction)
    if not ins:
        return selection, "unchanged"
    for pat, fn in RULES:
        m = pat.match(ins)
        if m:
            return fn(m, selection), "rule"
    if llm is not None and getattr(llm, "configured", True):
        try:
            out = llm.edit(instruction.strip(), selection)
        except Exception:
            return selection, "unchanged"
        if out:
            return out, "llm"
    return selection, "unchanged"


class Clipboard(Protocol):
    def get(self) -> str: ...
    def set(self, text: str) -> None: ...


class SelectionEditor:
    """Hotkey-driven record -> transcribe -> transform -> paste. Every device is injected."""

    def __init__(
        self,
        asr: Any,
        clipboard: Clipboard,
        copy_keys: Callable[[], None],
        paste_keys: Callable[[], None],
        llm: EditLLM | None = None,
        bus: Any = None,
        sample_rate: int = 16000,
        on_record: Callable[[bool], None] | None = None,  # e.g. mute ambient listening while recording
        sleep: Callable[[float], None] = time.sleep,
        settle_s: float = 0.15,
    ) -> None:
        self.asr = asr
        self.clip = clipboard
        self.copy_keys = copy_keys
        self.paste_keys = paste_keys
        self.llm = llm
        self.bus = bus
        self.sample_rate = sample_rate
        self.on_record = on_record
        self.sleep = sleep
        self.settle_s = settle_s
        self.recording = False
        self._chunks: list[np.ndarray] = []
        self._selection = ""
        self._saved_clip: str | None = None

    def _publish(self, topic: str, data: dict[str, Any]) -> None:
        if self.bus is not None:
            self.bus.publish(topic, data)

    def feed(self, chunk: Any) -> None:
        if self.recording:
            self._chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))

    def start(self) -> bool:
        try:
            self._saved_clip = self.clip.get()
        except Exception:
            self._saved_clip = None
        marker = f"\u2063away-{time.monotonic_ns()}"
        self.clip.set(marker)  # tells "nothing selected" apart from "selection equals the old clipboard"
        self.copy_keys()
        self.sleep(self.settle_s)
        sel = self.clip.get()
        if not sel or sel == marker:
            self._restore()
            self._publish("edit.result", {"status": "no_selection"})
            return False
        self._selection = sel
        self._chunks = []
        self.recording = True
        if self.on_record:
            self.on_record(True)
        self._publish("edit.start", {"chars": len(sel)})
        return True

    def stop(self) -> tuple[str, str]:
        if not self.recording:
            return "", "idle"
        self.recording = False
        if self.on_record:
            self.on_record(False)
        audio = np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)
        self._chunks = []
        instruction = self.asr.transcribe(audio, None) if len(audio) else ""
        # "Thank you." on a silent recording must not become an edit instruction
        instruction = clean_transcript(instruction, None, (), len(audio) / self.sample_rate)[0]
        new, method = apply_edit(instruction, self._selection, self.llm)
        if method != "unchanged":
            self.clip.set(new)
            self.paste_keys()
            self.sleep(self.settle_s)
        self._restore()
        self._publish("edit.result", {"instruction": instruction, "method": method, "chars": len(new)})
        return new, method

    def toggle(self) -> tuple[str, str] | None:
        if self.recording:
            return self.stop()
        self.start()
        return None

    def _restore(self) -> None:
        if self._saved_clip is not None:
            self.clip.set(self._saved_clip)


class SystemClipboard:  # pragma: no cover - needs a desktop session
    def __init__(self) -> None:
        import pyperclip

        self._p = pyperclip

    def get(self) -> str:
        return str(self._p.paste() or "")

    def set(self, text: str) -> None:
        self._p.copy(text)
