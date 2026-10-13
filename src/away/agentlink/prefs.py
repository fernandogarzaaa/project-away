"""The preference memory (``away://preferences``): a short plain-language file the user and their agents share.

One preference per line ("I hate accidental right-clicks", "prefer a slower cursor in Photoshop"). Agents read it
to decide and justify changes; they may add lines and remove a line by number, but the file is capped in size and
every agent edit is tagged with who made it. It lives in the data folder as ``agentlink/preferences.md`` so the
user can also open and edit it by hand.
"""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path

MAX_LINES = 200
MAX_LINE = 300
HEADER = "# My preferences for Project Away\n# One per line. My AI agents read these before changing anything.\n"
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


class Preferences:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def _read_lines(self) -> list[str]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [ln[2:] if ln.startswith("- ") else ln for ln in lines if ln.strip() and not ln.startswith("#")]

    def items(self) -> list[str]:
        with self._lock:
            return self._read_lines()

    def text(self) -> str:
        items = self.items()
        if not items:
            return HEADER + "\n(nothing yet)\n"
        return HEADER + "\n" + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(items)) + "\n"

    def _write(self, items: list[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(HEADER + "".join(f"- {t}\n" for t in items), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, text: str, who: str = "you") -> list[str]:
        line = _CTRL.sub(" ", str(text)).strip()[:MAX_LINE]
        if not line:
            raise ValueError("a preference cannot be empty")
        tag = "" if who in ("", "you") else f"  (added by {who}, {time.strftime('%Y-%m-%d')})"
        with self._lock:
            items = self._read_lines()
            if len(items) >= MAX_LINES:
                raise ValueError(f"the preference list is full ({MAX_LINES} lines); remove one first")
            items.append(line + tag)
            self._write(items)
            return items

    def remove(self, number: int) -> list[str]:
        with self._lock:
            items = self._read_lines()
            if not 1 <= int(number) <= len(items):
                raise ValueError(f"there is no preference number {number}")
            del items[int(number) - 1]
            self._write(items)
            return items
