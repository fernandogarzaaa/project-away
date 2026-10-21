"""Diagnostics bundle: one zip with everything needed to understand a problem, nothing private on purpose.

Contents: the tails of away.log / crash.log / update-install.log / agent-access.log with tokens, keys, passwords,
e-mail addresses and the home folder hidden (:func:`redact_text`), the config with secrets and the home folder
redacted, the latest camera diagnosis, fps / latency / CPU / GPU numbers and the live context, and
the last 30 s of landmarks + events (a ``.awayrec``; no video, no audio, no screenshots). Saved as
``<data>/diagnostics/bundle-<time>.zip`` (the newest 20 are kept).
"""

from __future__ import annotations

import io
import json
import platform
import re
import time
import zipfile
from pathlib import Path
from typing import Any

import yaml

# Fail closed on key names (audit Phase 4): a structured key whose name suggests a secret loses its
# value even when the name was never listed up front, so a secret stored under an unrecognized name
# (private_key, passphrase, a seed phrase) cannot leave the PC in a bundle. Two layers:
# - SECRET_KEY: substrings that are unambiguous anywhere in a name.
# - _SECRET_TOKENS: whole words after splitting on separators and camelCase humps, for short words
#   that must not fire inside innocent words (key in keyboard/monkey, pass in passage). A *_key name
#   still counts (hold_key's value is hidden too; that is the intended fail-closed trade-off).
SECRET_KEY = re.compile(
    r"(token|password|passwd|passcode|passphrase|secret|credentials?|mnemonic|api_?key)", re.IGNORECASE
)
_SECRET_TOKENS = frozenset(
    {
        "key",
        "keys",
        "pass",
        "phrase",
        "auth",
        "oauth",
        # Whole words, not the auth substring: "authorization" and "authentication"
        # are credentials, but "author" is innocent and must not match.
        "authorization",
        "authentication",
        "bearer",
        "cookie",
        "cookies",
        "session",
        "seed",
        "seeds",
        "cert",
        "certs",
        "cred",
        "creds",
        "pwd",
        "private",
    }
)
_KEY_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _key_tokens(name: str) -> list[str]:
    out: list[str] = []
    for chunk in re.split(r"[^A-Za-z0-9]+", name):
        for part in _KEY_CAMEL.split(chunk):
            tok = part.lower().rstrip("0123456789")
            if tok:
                out.append(tok)
    return out


def _is_secret_key(name: Any) -> bool:
    text = str(name)
    if SECRET_KEY.search(text):
        return True
    return any(tok in _SECRET_TOKENS for tok in _key_tokens(text))


LOG_FILES = ("away.log", "crash.log", "update-install.log", "agent-access.log")
TAIL_BYTES = 2_000_000
KEEP = 20


HIDDEN = "***"
# v0.1.24 hardening: what is hidden in raw log text before it leaves the PC (bundles, get_logs, read_file)
_TEXT_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=\-]+"), "Bearer " + HIDDEN),
    (re.compile(r"\b(?:paway|pa)_[A-Za-z0-9_\-]{16,}"), HIDDEN),  # Agent Link tokens (persistent and session)
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]{4,}"), HIDDEN),  # JWTs
    (re.compile(r"\b(?:sk|rk|pk)-[A-Za-z0-9_\-]{16,}"), HIDDEN),  # OpenAI / Anthropic / Stripe style keys
    (re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"), HIDDEN),  # GitHub
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"), HIDDEN),  # Slack
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), HIDDEN),  # AWS access key id
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"), HIDDEN),  # Google API key
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}"), HIDDEN),  # Hugging Face
    # key=value, key: value, "key": "value" for anything named like a secret
    (
        re.compile(
            r"(?i)([\"']?\b[\w.-]*(?:token|password|passwd|pwd|secret|api[_-]?key|auth\w*|credential|cookie"
            r"|passphrase|passcode|mnemonic|private[_-]?key)[\w.-]*"
            r"[\"']?\s*[:=]\s*)([\"']?)[^\s\"',;&]+"
        ),
        r"\1\2" + HIDDEN,
    ),
    (re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@"), r"\1" + HIDDEN + "@"),  # user:pass@ in URLs
    (re.compile(r"/c/[a-z0-9]{6,32}\b"), "/c/" + HIDDEN),  # pairing links
    (re.compile(r"(?i)\b[A-Z]:\\Users\\[^\\/\s]+"), "~"),  # other people's home folders too
    (re.compile(r"/(?:home|Users)/[^/\s]+"), "~"),
)
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def redact_text(text: str, home: str | None = None, emails: bool = True) -> str:
    """Raw log text with tokens (``paway_``/``pa_``, ``Bearer``, JWTs), API keys, passwords and other secret
    ``key=value`` pairs, credentials in URLs, pairing links, e-mail addresses (``emails``) and the home folder
    hidden."""
    home = home if home is not None else str(Path.home())
    out = str(text)
    if home and len(home) > 3:
        for h in {home, home.replace("\\", "/"), home.replace("/", "\\")}:
            out = re.sub(re.escape(h), "~", out, flags=re.IGNORECASE)
    for rx, repl in _TEXT_RULES:
        out = rx.sub(repl, out)
    if emails:
        out = EMAIL.sub(HIDDEN + "@" + HIDDEN, out)
    return out


def _redact_bytes(raw: bytes) -> bytes:
    return redact_text(raw.decode("utf-8", "replace")).encode("utf-8")


def redact(obj: Any, home: str | None = None) -> Any:
    home = home if home is not None else str(Path.home())
    if isinstance(obj, dict):
        return {
            k: ("***" if _is_secret_key(k) and v not in (None, "", False) else redact(v, home)) for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [redact(v, home) for v in obj]
    if isinstance(obj, str):
        return redact_text(obj, home)
    return obj


def _tail(path: Path, n: int = TAIL_BYTES) -> bytes:
    with path.open("rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - n))
        return fh.read()


def build(
    data_dir: Path,
    cfg: Any = None,
    stats: dict[str, Any] | None = None,
    recording: tuple[dict[str, Any], list[dict[str, Any]]] | None = None,
    note: str = "",
    out_dir: Path | None = None,
    clock: Any = time.time,
) -> Path:
    from away import __version__
    from away import config as cfgmod
    from away.agentlink import recording as rec

    data_dir = Path(data_dir)
    out = Path(out_dir) if out_dir is not None else data_dir / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(clock()))
    path = out / f"bundle-{stamp}.zip"
    n = 1
    while path.exists():
        path = out / f"bundle-{stamp}-{n}.zip"
        n += 1
    included: list[str] = []
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in LOG_FILES:
            p = data_dir / "logs" / name
            if p.is_file():
                z.writestr(f"logs/{name}", _redact_bytes(_tail(p)))
                included.append(f"logs/{name}")
        conf = cfgmod.to_dict(cfg) if cfg is not None else None
        if conf is None:
            try:
                conf = cfgmod.to_dict(cfgmod.load()[0])
            except Exception as exc:
                conf = {"error": f"could not load the config: {exc}"}
        z.writestr("config-redacted.yaml", yaml.safe_dump(redact(conf), sort_keys=False))
        included.append("config-redacted.yaml")
        diag = sorted((data_dir / "diagnostics").glob("cameras-*.txt"), key=lambda p: p.stat().st_mtime)
        if diag:
            z.writestr("cameras-diagnose.txt", _redact_bytes(_tail(diag[-1], 500_000)))
            included.append("cameras-diagnose.txt")
        info = {
            "app": __version__,
            "python": platform.python_version(),
            "os": platform.platform(),
            "machine": platform.machine(),
            "created": clock(),
            "note": note,
            "stats": redact(stats or {}),
        }
        z.writestr("stats.json", json.dumps(info, indent=1, default=str))
        included.append("stats.json")
        if recording is not None and recording[1]:
            buf = io.BytesIO()
            tmp = out / f".rec-{stamp}{rec.SUFFIX}"
            rec.write(tmp, recording[0], recording[1])
            buf.write(tmp.read_bytes())
            tmp.unlink(missing_ok=True)
            z.writestr(f"last-seconds{rec.SUFFIX}", buf.getvalue())
            included.append(f"last-seconds{rec.SUFFIX}")
        z.writestr(
            "README.txt",
            "Project Away diagnostics bundle\n\nIncluded: " + ", ".join(included) + "\n\n"
            "Not included, by design: video, audio, screenshots, passwords/tokens (redacted), your files.\n"
            "Replay the recording with: away replay last-seconds.awayrec\n",
        )
    for old in sorted(out.glob("bundle-*.zip"), key=lambda p: p.stat().st_mtime)[:-KEEP]:
        old.unlink(missing_ok=True)
    return path
