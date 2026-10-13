"""Personal dictionary: names and terms that dictation should get right.

* **Words** (``Jonie``, ``GitHub``, ``RTX 2060``) go to Whisper as ``initial_prompt`` vocabulary and as
  ``hotwords``, so it is more likely to hear them.
* **Fixes** (``Jonnie -> Jonie``) are applied after Whisper, case-insensitively on whole words.

Words come from the Settings word list, the voice commands ``add X to my dictionary`` / ``correct X to Y``
(:func:`parse_command`), or are learned from your own corrections: :class:`CorrectionWatcher` re-reads the text
field shortly after a dictation was typed and :func:`learn_pairs` finds the few words you changed. A fix learned
from an edit only counts once it was seen ``learn_after`` times, so one rewrite does not teach a wrong rule.

Stored locally as JSON (``dictionary.json`` in the config folder, nothing leaves the PC). Import/export: the same
JSON, or plain text with one ``word`` or ``wrong -> right`` per line (the ``voice.vocabulary_file`` format).
"""

from __future__ import annotations

import contextlib
import difflib
import json
import re
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_TERM_LEN = 48
MAX_WORDS_PER_TERM = 4
_STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "am",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "my",
        "your",
        "our",
        "their",
        "this",
        "that",
        "these",
        "those",
        "as",
        "so",
        "not",
        "no",
        "yes",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "will",
        "would",
        "can",
        "could",
        "should",
        "just",
        "very",
    ]
)
# names that are also everyday words: their capitalised spelling is not forced on the ordinary word
_NAME_WORDS = frozenset(
    [
        "will",
        "mark",
        "bill",
        "may",
        "june",
        "april",
        "august",
        "rose",
        "grace",
        "hope",
        "joy",
        "faith",
        "frank",
        "art",
        "bob",
        "pat",
        "sue",
        "jack",
        "ray",
        "dawn",
        "summer",
        "iris",
        "ruby",
        "amber",
        "jade",
        "rich",
        "guy",
        "gene",
        "drew",
        "chase",
        "hunter",
        "max",
        "sky",
        "river",
        "lily",
        "daisy",
        "penny",
        "sandy",
        "rob",
        "dean",
        "carol",
        "crystal",
        "sunny",
        "wade",
        "reed",
        "cliff",
        "brook",
        "jay",
        "robin",
        "sage",
        "glen",
        "heath",
        "holly",
        "ivy",
        "miles",
        "page",
        "victor",
        "don",
        "destiny",
        "harmony",
        "melody",
        "patience",
        "sterling",
        "stone",
        "storm",
        "violet",
        "wren",
        "sol",
        "earl",
        "duke",
        "king",
        "prince",
        "major",
        "hazel",
        "olive",
        "pearl",
        "apple",
        "amazon",
        "windows",
        "word",
        "excel",
        "teams",
        "office",
        "slack",
        "signal",
        "edge",
        "chrome",
    ]
)
_WORD = re.compile(r"[\w'][\w'.+#-]*")


def clean_term(term: str) -> str:
    """Trim spaces and surrounding punctuation; ``J O N I E`` (spelled out) becomes ``JONIE``."""
    t = " ".join(str(term).split()).strip(" \t.,;:!?\"'()[]")
    parts = t.split(" ")
    if len(parts) >= 3 and all(len(p) == 1 and p.isalnum() for p in parts):
        t = "".join(parts)
    return t


def valid_term(term: str) -> str | None:
    """Error message for a word the list cannot hold, else None."""
    if not term:
        return "type a word or name"
    if len(term) > MAX_TERM_LEN:
        return f"too long (max {MAX_TERM_LEN} characters)"
    if len(term.split()) > MAX_WORDS_PER_TERM:
        return f"too many words (max {MAX_WORDS_PER_TERM})"
    if "->" in term:
        return "use one word or name (fixes are learned automatically)"
    return None


@dataclass
class Entry:
    term: str
    variants: list[str] = field(default_factory=list)  # what Whisper heard instead (replaced by ``term``)
    source: str = "manual"  # manual | learned | command | import
    added: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"term": self.term, "variants": list(self.variants), "source": self.source, "added": self.added}


def _pattern(text: str) -> re.Pattern[str]:
    # whole words; a possessive "'s" still counts ("jonnie's" -> "Jonie's")
    return re.compile(r"(?<![\w'])" + re.escape(text) + r"(?!\w)(?!'(?!s\b)\w)", re.IGNORECASE)


def _fix_casing(term: str) -> bool:
    """Re-case every occurrence of the term itself ("jonie" -> "Jonie", "github" -> "GitHub"), except for an
    all-lowercase term and names that are also everyday words ("Will", "Mark", "May")."""
    if term == term.lower():
        return False
    low = term.lower()
    return low not in _NAME_WORDS and low not in _STOP


class Dictionary:
    """The word list plus learned fixes. Thread-safe; ``on_change`` callbacks fire after every change."""

    def __init__(self, path: Path | None = None, learn_after: int = 2, clock: Callable[[], float] = time.time) -> None:
        self.path = path
        self.learn_after = max(1, int(learn_after))
        self.clock = clock
        self._entries: dict[str, Entry] = {}  # key: term.lower()
        self._pending: dict[str, int] = {}  # "wrong\tright" -> times seen in an edit
        self._lock = threading.RLock()
        self.on_change: list[Callable[[], None]] = []
        self._compiled: list[tuple[re.Pattern[str], str]] | None = None
        self._mtime: tuple[int, int] | None = (
            None  # (mtime_ns, size) of the file as last read/written (Settings and the app share it)
        )

    # ------------------------------------------------------------------ storage
    @classmethod
    def load(cls, path: Path | None, learn_after: int = 2) -> Dictionary:
        d = cls(path, learn_after)
        d._read()
        return d

    def _stat(self) -> tuple[int, int] | None:
        try:
            st = self.path.stat() if self.path is not None else None
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size) if st is not None else None

    def _read(self) -> None:
        path = self.path
        if path is None or not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._entries.clear()
            self._pending.clear()
            self._from_json(data)
            self._mtime = self._stat()
        except (OSError, ValueError, TypeError, AttributeError):
            bad = path.with_suffix(path.suffix + ".bad")
            with contextlib.suppress(OSError):
                path.replace(bad)  # keep the unreadable file for the user instead of overwriting it

    def reload_if_changed(self) -> bool:
        """Re-read the file when someone else (Settings, another process) saved it since we last did."""
        m = self._stat()
        if m is None or m == self._mtime:
            return False
        with self._lock:
            self._read()
            self._compiled = None
        return True

    def _from_json(self, data: dict[str, Any]) -> int:
        n = 0
        for item in data.get("words") or []:
            if isinstance(item, str):
                item = {"term": item}
            term = clean_term(str(item.get("term", "")))
            if valid_term(term):
                continue
            e = self._entries.setdefault(term.lower(), Entry(term, [], str(item.get("source", "import")), 0.0))
            e.added = float(item.get("added") or e.added or self.clock())
            for v in item.get("variants") or []:
                v = clean_term(str(v))
                if v and v.lower() != term.lower() and v.lower() not in (x.lower() for x in e.variants):
                    e.variants.append(v)
            n += 1
        for k, c in (data.get("pending") or {}).items():
            if isinstance(k, str) and "\t" in k and isinstance(c, int):
                self._pending[k] = c
        self._compiled = None
        return n

    def to_json(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": 1,
                "words": [e.to_dict() for e in self._entries.values()],
                "pending": dict(self._pending),
            }

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_json(), indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)
        self._mtime = self._stat()

    def _changed(self) -> None:
        self._compiled = None
        with contextlib.suppress(OSError):
            self.save()
        for cb in list(self.on_change):
            with contextlib.suppress(Exception):
                cb()

    # ------------------------------------------------------------------ editing
    def words(self) -> list[str]:
        with self._lock:
            return [e.term for e in self._entries.values()]

    def entries(self) -> list[Entry]:
        with self._lock:
            return list(self._entries.values())

    def add(self, term: str, source: str = "manual") -> str | None:
        """Add a word or name. Returns an error message or None."""
        t = clean_term(term)
        err = valid_term(t)
        if err:
            return err
        self.reload_if_changed()
        with self._lock:
            old = self._entries.get(t.lower())
            if old is not None:
                if old.term == t:
                    return None
                old.term = t  # same word, new casing: the user's latest spelling wins
            else:
                self._entries[t.lower()] = Entry(t, [], source, self.clock())
        self._changed()
        return None

    def remove(self, term: str) -> bool:
        self.reload_if_changed()
        with self._lock:
            gone = self._entries.pop(clean_term(term).lower(), None) is not None
        if gone:
            self._changed()
        return gone

    def add_fix(self, wrong: str, right: str, source: str = "command") -> str | None:
        """``wrong`` is replaced by ``right`` from now on (``right`` joins the word list)."""
        w, r = clean_term(wrong), clean_term(right)
        self.reload_if_changed()
        err = valid_term(r) or (valid_term(w) and "the misheard word is invalid")
        if err:
            return err
        if w.lower() == r.lower() and w == r:
            return "the two spellings are the same"
        with self._lock:
            e = self._entries.get(r.lower())
            if e is None:
                e = self._entries[r.lower()] = Entry(r, [], source, self.clock())
            else:
                e.term = r
            if w.lower() != r.lower() and w.lower() not in (v.lower() for v in e.variants):
                e.variants.append(w)
            for other in self._entries.values():  # a word cannot be both a term and another term's variant
                if other is not e:
                    other.variants = [v for v in other.variants if v.lower() != w.lower()]
            self._pending.pop(f"{w.lower()}\t{r}", None)
        self._changed()
        return None

    def observe_fix(self, wrong: str, right: str) -> bool:
        """A fix seen in one of your edits; it is learned once seen ``learn_after`` times. True when learned now."""
        w, r = clean_term(wrong), clean_term(right)
        if not w or not r or valid_term(r) or valid_term(w):
            return False
        key = f"{w.lower()}\t{r}"
        self.reload_if_changed()
        with self._lock:
            known = self._entries.get(r.lower())
            if known is not None and (w.lower() in (v.lower() for v in known.variants) or w == r):
                return False
            self._pending[key] = self._pending.get(key, 0) + 1
            ready = self._pending[key] >= self.learn_after
        if ready:
            self.add_fix(w, r, source="learned")
            return True
        with contextlib.suppress(OSError):
            self.save()
        return False

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._pending.clear()
        self._changed()

    # ------------------------------------------------------------------ use
    def _rules(self) -> list[tuple[re.Pattern[str], str]]:
        with self._lock:
            if self._compiled is None:
                rules: list[tuple[re.Pattern[str], str]] = []
                for e in self._entries.values():
                    for v in sorted(e.variants, key=len, reverse=True):
                        rules.append((_pattern(v), e.term))
                    if _fix_casing(e.term):
                        rules.append((_pattern(e.term), e.term))
                # longer phrases first, so "git hub" is fixed before "hub"
                rules.sort(key=lambda r: len(r[0].pattern), reverse=True)
                self._compiled = rules
            return self._compiled

    def apply(self, text: str) -> str:
        if self.reload_if_changed():
            for cb in list(self.on_change):  # the Settings word list changed: refresh the Whisper prompt too
                with contextlib.suppress(Exception):
                    cb()
        for pat, rep in self._rules():
            text = pat.sub(rep.replace("\\", "\\\\"), text)
        return text

    def hotwords(self, extra: Iterable[str] = (), limit: int = 60) -> str | None:
        seen = list(dict.fromkeys([*extra, *self.words()]))
        return " ".join(t for t in seen[:limit] if t) or None

    # ------------------------------------------------------------------ import / export
    def export_text(self) -> str:
        lines = ["# Project Away personal dictionary: one word per line, or 'misheard -> right'"]
        for e in self.entries():
            lines.append(e.term)
            lines += [f"{v} -> {e.term}" for v in e.variants]
        return "\n".join(lines) + "\n"

    def import_text(self, text: str) -> int:
        n = 0
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if "->" in line:
                w, r = (s.strip() for s in line.split("->", 1))
                ok = self.add_fix(w, r, source="import") is None if w.lower() != r.lower() else self.add(r) is None
            else:
                ok = self.add(line, source="import") is None
            n += int(ok)
        return n

    def export_file(self, path: Path) -> None:
        if path.suffix.lower() == ".json":
            path.write_text(json.dumps(self.to_json(), indent=1, ensure_ascii=False), encoding="utf-8")
        else:
            path.write_text(self.export_text(), encoding="utf-8")

    def import_file(self, path: Path) -> int:
        """Adds what the file holds (never deletes). Returns how many words/fixes were read."""
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            with self._lock:
                n = self._from_json(json.loads(text))
            self._changed()
            return n
        return self.import_text(text)


def default_path() -> Path:
    from away import paths

    return paths.config_dir() / "dictionary.json"


# ---------------------------------------------------------------------- voice commands
_CMD_CORRECT = re.compile(r"^(?:correct|fix|change)\s+(.{1,48}?)\s+(?:to|into|as)\s+(.{1,48}?)$", re.IGNORECASE)
_CMD_SPELL = re.compile(r"^spell\s+(.{1,48}?)\s+(?:as|like)\s+(.{1,60}?)$", re.IGNORECASE)
_CMD_ADD = re.compile(
    r"^(?:add|remember)\s+(?:the\s+)?(?:word\s+|name\s+|term\s+)?(.{1,60}?)\s+to\s+(?:the\s+|my\s+)?dictionary$",
    re.IGNORECASE,
)
_CMD_REMOVE = re.compile(
    r"^(?:remove|forget|delete)\s+(?:the\s+)?(?:word\s+|name\s+|term\s+)?(.{1,60}?)\s+from\s+(?:the\s+|my\s+)?"
    r"dictionary$",
    re.IGNORECASE,
)


@dataclass
class Command:
    action: str  # add | remove | correct
    term: str
    wrong: str = ""


def parse_command(text: str) -> Command | None:
    """``add Jonie to my dictionary``, ``remove X from the dictionary``, ``correct Jonnie to Jonie``,
    ``spell Jonie as J O N I E``. Punctuation and casing from Whisper are ignored."""
    t = " ".join(text.split()).strip(" .!?,")
    if len(t.split()) > 14:
        return None
    m = _CMD_ADD.match(t)
    if m:
        return Command("add", clean_term(m.group(1)))
    m = _CMD_REMOVE.match(t)
    if m:
        return Command("remove", clean_term(m.group(1)))
    m = _CMD_SPELL.match(t)
    if m:
        return Command("correct", clean_term(re.sub(r"[,.-]", " ", m.group(2))), clean_term(m.group(1)))
    m = _CMD_CORRECT.match(t)
    if m:
        return Command("correct", clean_term(m.group(2)), clean_term(m.group(1)))
    return None


# ---------------------------------------------------------------------- learning from edits
def _tokens(text: str) -> list[str]:
    return _WORD.findall(text)


def _key(tok: str) -> str:
    return tok.lower().strip(".'")


def find_span(inserted: str, field_text: str) -> str | None:
    """The part of ``field_text`` that is the (possibly edited) inserted dictation, or None when it is gone.

    Aligns the inserted words with the field's words; at least half of the inserted words must still be there,
    in order. The span reaches as far as the inserted text's unmatched first/last words would."""
    ins = _tokens(inserted)
    fld = _tokens(field_text)
    if not ins or not fld:
        return None
    sm = difflib.SequenceMatcher(None, [_key(t) for t in ins], [_key(t) for t in fld], autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size]
    if not blocks or sum(b.size for b in blocks) * 2 < len(ins):
        return None
    first, last = blocks[0], blocks[-1]
    start = max(0, first.b - first.a)
    end = min(len(fld), last.b + last.size + (len(ins) - (last.a + last.size)))
    return " ".join(fld[start:end])


def learn_pairs(inserted: str, edited: str, max_edits: int = 2) -> list[tuple[str, str]]:
    """``(wrong, right)`` word fixes between what was typed and what the user changed it to.

    Only small, term-like substitutions count (1-3 words for 1-3 words, at most ``max_edits`` of them, the rest of
    the text unchanged); a rewrite, an insertion, a deletion or a change between common words teaches nothing."""
    a = _tokens(inserted)
    b = _tokens(edited)
    if not a or not b:
        return []
    ka, kb = [_key(t) for t in a], [_key(t) for t in b]
    sm = difflib.SequenceMatcher(None, ka, kb, autojunk=False)
    if sm.ratio() < 0.5:
        return []  # a rewrite, not a correction
    out: list[tuple[str, str]] = []
    case_fixes: list[tuple[str, str]] = []
    edits = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for x, y in zip(a[i1:i2], b[j1:j2]):  # same word, new casing: "github" -> "GitHub"
                if x != y and x.lower() == y.lower() and y.lower() not in _STOP and y != y.lower():
                    case_fixes.append((x, y))
            continue
        edits += 1
        if tag != "replace" or not (1 <= i2 - i1 <= 3 and 1 <= j2 - j1 <= 3):
            continue
        wrong, right = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        if all(_key(t) in _STOP for t in b[j1:j2]) or all(_key(t) in _STOP for t in a[i1:i2]):
            continue  # "a" -> "the" is grammar, not vocabulary
        similar = difflib.SequenceMatcher(None, wrong.lower(), right.lower()).ratio() >= 0.45
        termish = any(c.isupper() or c.isdigit() for c in right)
        if similar or (termish and abs(len(wrong) - len(right)) <= max(4, len(right))):
            out.append((wrong, right))
    if edits > max_edits:
        return []
    return out + case_fixes


@dataclass
class _Pending:
    text: str
    app: str | None
    t: float
    last_span: str | None = None


class CorrectionWatcher:
    """Notices when you fix just-dictated text and teaches the dictionary.

    ``on_inserted`` is called with every typed dictation; ``poll`` (every ~2 s, from a background thread) re-reads
    the focused field with ``reader`` (UI Automation; None when unreadable) while the same app has the focus. When
    the watch window ends, the last version of the span is compared with what was typed (:func:`learn_pairs`)."""

    def __init__(
        self,
        dictionary: Dictionary,
        reader: Callable[[], str | None],
        app: Callable[[], str | None] = lambda: None,
        watch_seconds: float = 20.0,
        clock: Callable[[], float] = time.monotonic,
        on_learned: Callable[[str, str], None] | None = None,
    ) -> None:
        self.dictionary = dictionary
        self.reader = reader
        self.app = app
        self.watch_seconds = watch_seconds
        self.clock = clock
        self.on_learned = on_learned
        self._pending: list[_Pending] = []
        self._lock = threading.Lock()
        self.observed: list[tuple[str, str]] = []  # every fix seen (for tests and diagnostics)

    def on_inserted(self, text: str, app: str | None = None) -> None:
        if not text.strip():
            return
        with self._lock:
            self._pending.append(_Pending(text.strip(), app, self.clock()))
            self._pending = self._pending[-3:]

    def poll(self) -> None:
        with self._lock:
            pending = list(self._pending)
        if not pending:
            return
        now = self.clock()
        focused = self.app()
        field_text: str | None = None
        read = False
        done: list[_Pending] = []
        for p in pending:
            same_app = p.app is None or focused is None or focused == p.app
            if same_app and now - p.t <= self.watch_seconds:
                if not read:
                    try:
                        field_text = self.reader()
                    except Exception:
                        field_text = None
                    read = True
                if field_text:
                    span = find_span(p.text, field_text)
                    if span is not None:
                        p.last_span = span
                continue
            done.append(p)
        for p in done:
            self._finish(p)
        with self._lock:
            self._pending = [p for p in self._pending if p not in done]

    def flush(self) -> None:
        """Finish every pending watch now (app shutting down)."""
        with self._lock:
            pending, self._pending = self._pending, []
        for p in pending:
            self._finish(p)

    def _finish(self, p: _Pending) -> None:
        if not p.last_span:
            return
        typed = p.text
        for wrong, right in learn_pairs(typed, p.last_span):
            self.observed.append((wrong, right))
            if self.dictionary.observe_fix(wrong, right) and self.on_learned is not None:
                self.on_learned(wrong, right)
