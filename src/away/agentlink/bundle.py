"""Diagnostics bundle: one zip with everything needed to understand a problem, nothing private on purpose.

Contents: the tails of away.log / crash.log / update-install.log / agent-access.log, the config with secrets and
the home folder redacted, the latest camera diagnosis, fps / latency / CPU / GPU numbers and the live context, and
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

SECRET_KEY = re.compile(r"(token|password|passwd|secret|api_?key|auth|credential|cookie)", re.IGNORECASE)
LOG_FILES = ("away.log", "crash.log", "update-install.log", "agent-access.log")
TAIL_BYTES = 2_000_000
KEEP = 20


def redact(obj: Any, home: str | None = None) -> Any:
    home = home if home is not None else str(Path.home())
    if isinstance(obj, dict):
        return {
            k: ("***" if SECRET_KEY.search(str(k)) and v not in (None, "", False) else redact(v, home))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [redact(v, home) for v in obj]
    if isinstance(obj, str) and home and len(home) > 3 and home in obj:
        return obj.replace(home, "~")
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
                z.writestr(f"logs/{name}", _tail(p))
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
            z.writestr("cameras-diagnose.txt", _tail(diag[-1], 500_000))
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
