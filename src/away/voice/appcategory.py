"""What kind of app has the focus, and which tone dictation cleanup should use there.

:func:`categorize` maps the focused process name (``slack.exe``, ``Code``) and window title to one of
``chat | email | docs | code | terminal | other``. Browsers are told apart by their window title (a Gmail tab is
``email``, a Google Docs tab is ``docs``). :func:`resolve_tone` turns the category into a tone, after the
user's per-app overrides (``voice.cleanup.tones``, keyed by process name or bindings profile name).
"""

from __future__ import annotations

from collections.abc import Mapping

CATEGORIES = ("chat", "email", "docs", "code", "terminal", "other")
DEFAULT_TONES = {
    "chat": "casual",
    "email": "formal",
    "docs": "formal",
    "code": "code",
    "terminal": "verbatim",
    "other": "neutral",
}

# process names, lower case, without ".exe"
_PROCESSES: dict[str, tuple[str, ...]] = {
    "terminal": (
        "windowsterminal",
        "wt",
        "cmd",
        "powershell",
        "pwsh",
        "conhost",
        "openconsole",
        "wsl",
        "bash",
        "mintty",
        "putty",
        "terminal",
        "iterm2",
        "alacritty",
        "wezterm",
        "wezterm-gui",
        "kitty",
        "konsole",
        "gnome-terminal-server",
        "xterm",
        "tilix",
        "warp",
        "tabby",
    ),
    "code": (
        "code",
        "code - insiders",
        "cursor",
        "windsurf",
        "devenv",
        "pycharm64",
        "pycharm",
        "idea64",
        "idea",
        "webstorm64",
        "clion64",
        "rider64",
        "goland64",
        "studio64",
        "sublime_text",
        "notepad++",
        "vim",
        "nvim",
        "gvim",
        "emacs",
        "zed",
        "xcode",
        "atom",
        "spyder",
        "jupyter-lab",
    ),
    "chat": (
        "slack",
        "discord",
        "teams",
        "ms-teams",
        "whatsapp",
        "telegram",
        "signal",
        "messenger",
        "line",
        "viber",
        "skype",
        "element",
        "zoom",
        "messages",
        "wechat",
    ),
    "email": ("outlook", "olk", "thunderbird", "mail", "hxoutlook", "spark", "mailspring", "betterbird"),
    "docs": (
        "winword",
        "word",
        "microsoft word",
        "soffice",
        "swriter",
        "libreoffice",
        "pages",
        "notion",
        "obsidian",
        "onenote",
        "wordpad",
        "textedit",
        "notepad",
        "evernote",
        "typora",
        "logseq",
        "powerpnt",
    ),
}

# window-title hints (browser tabs, web apps), lower case; first match wins in this order
_TITLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("terminal", ("powershell", "command prompt", "cmd.exe", "windows terminal", "bash", "zsh", "ssh ")),
    ("code", ("visual studio code", "github.dev", "codespaces", "replit", "jupyter", "colab", "stackblitz")),
    ("email", ("gmail", "outlook", "inbox", "proton mail", "yahoo mail", "compose mail", "- mail")),
    (
        "chat",
        ("whatsapp", "messenger", "slack", "discord", "telegram", "microsoft teams", "chatgpt", "claude", "x.com"),
    ),
    (
        "docs",
        (
            "google docs",
            "- word",
            "microsoft word",
            "notion",
            "confluence",
            "dropbox paper",
            "quip",
            "- docs",
            "overleaf",
        ),
    ),
)


def _norm_process(process: str | None) -> str:
    p = (process or "").strip().replace("\\", "/").rsplit("/", 1)[-1].lower()
    return p[:-4] if p.endswith(".exe") else p


def categorize(process: str | None, title: str | None = None) -> str:
    """``chat | email | docs | code | terminal | other`` for the focused app."""
    p = _norm_process(process)
    if p:
        for cat, names in _PROCESSES.items():
            if p in names:
                return cat
    t = (title or "").lower()
    if t:
        for cat, hints in _TITLES:
            if any(h in t for h in hints):
                return cat
    return "other"


def resolve_tone(
    category: str,
    overrides: Mapping[str, str] | None = None,
    process: str | None = None,
    profile: str | None = None,
) -> str:
    """The tone for this app: an override for the process (with or without ``.exe``) or for the active
    bindings profile wins; ``auto`` (or no override) uses the category's default."""
    tone = None
    if overrides:
        low = {str(k).lower(): v for k, v in overrides.items()}
        p = _norm_process(process)
        for key in (p, p + ".exe", (profile or "").lower()):
            if key and key in low:
                tone = low[key]
                break
    if tone in (None, "auto"):
        tone = DEFAULT_TONES.get(category, "neutral")
    return str(tone)
