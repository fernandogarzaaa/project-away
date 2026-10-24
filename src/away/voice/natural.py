"""Natural voices (v0.1.23): what agents say is read aloud by a voice that sounds like a person.

Engines, best first (``speech.engine: auto`` tries them in this order; a failure falls back silently to the next):

* **Natural (Kokoro)** - Kokoro-82M (Apache-2.0 model) through kokoro-onnx (MIT), int8 model on the CPU (the CUDA
  provider when onnxruntime has it). Voices: af_heart, af_bella, am_michael, bf_emma, bm_george.
* **Light (Piper)** - a Piper voice (en_US lessac / amy, medium) run directly with onnxruntime, so the GPL
  piper-tts package is not needed.
* **Windows natural** - the Windows 10/11 OneCore voices through WinRT ``Windows.Media.SpeechSynthesis`` (called
  from PowerShell, so no extra packages), which sound better than the old desktop voices.
* the classic voices of :mod:`away.voice.speech` (SAPI, PowerShell System.Speech, say, spd-say, espeak).

Nothing heavy is in the installer. **Download** in Settings > Voice & input fetches, into
``<data>/engines/voices``, the pinned model files and the few small Python wheels kokoro-onnx needs (kokoro-onnx,
phonemizer, espeakng-loader with the espeak-ng phoneme library, joblib, cloudpickle, attrs, dlinfo); onnxruntime
and numpy are already in the app. Every file's size and SHA-256 are pinned below and checked before it is used.
espeak-ng (downloaded on request, never bundled) is GPL-3.0.

Playback is chunked: the text is split into short sentences, the first chunk is spoken while the next ones are
synthesized, so speech starts quickly (about 0.3 s for Kokoro, 0.06 s for Piper on a laptop CPU).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import platform as _pf
import queue
import re
import shutil
import subprocess
import sys
import threading
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from away.voice import latency

ENGINES = ("auto", "kokoro", "piper", "windows", "classic")
ENGINE_TITLES = {
    "auto": "Best available",
    "kokoro": "Natural (Kokoro)",
    "piper": "Light (Piper)",
    "windows": "Windows natural",
    "classic": "Classic system voice",
}
KOKORO_VOICES = ("af_heart", "af_bella", "am_michael", "bf_emma", "bm_george")
PIPER_VOICES = ("lessac", "amy")
SPEEDS = {"slow": 0.85, "normal": 1.0, "fast": 1.2}
VOLUMES = {"quiet": 0.55, "normal": 0.9, "loud": 1.0}
FIRST_CHUNK_WORDS = 8  # the first chunk is short so the voice starts quickly
CHUNK_WORDS = 28


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    size: int
    sha256: str


KOKORO_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/"
PYPI = "https://files.pythonhosted.org/packages/"

KOKORO_MODEL = [
    Asset(
        "kokoro-v1.0.int8.onnx",
        KOKORO_BASE + "kokoro-v1.0.int8.onnx",
        92361271,
        "6e742170d309016e5891a994e1ce1559c702a2ccd0075e67ef7157974f6406cb",
    ),
    Asset(
        "voices-v1.0.bin",
        KOKORO_BASE + "voices-v1.0.bin",
        28214398,
        "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
    ),
]
PIPER_MODELS = {
    "lessac": [
        Asset(
            "en_US-lessac-medium.onnx",
            PIPER_BASE + "lessac/medium/en_US-lessac-medium.onnx",
            63201294,
            "5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f",
        ),
        Asset(
            "en_US-lessac-medium.onnx.json",
            PIPER_BASE + "lessac/medium/en_US-lessac-medium.onnx.json",
            4885,
            "efe19c417bed055f2d69908248c6ba650fa135bc868b0e6abb3da181dab690a0",
        ),
    ],
    "amy": [
        Asset(
            "en_US-amy-medium.onnx",
            PIPER_BASE + "amy/medium/en_US-amy-medium.onnx",
            63201294,
            "b3a6e47b57b8c7fbe6a0ce2518161a50f59a9cdd8a50835c02cb02bdd6206c18",
        ),
        Asset(
            "en_US-amy-medium.onnx.json",
            PIPER_BASE + "amy/medium/en_US-amy-medium.onnx.json",
            4882,
            "95a23eb4d42909d38df73bb9ac7f45f597dbfcde2d1bf9526fdeaf5466977d77",
        ),
    ],
}
# pure-python wheels (any platform) + espeakng-loader per platform; sizes and sha256 checked by downloading each
WHEELS_ANY = [
    Asset(
        "kokoro_onnx-0.6.1-py3-none-any.whl",
        PYPI + "60/e1/a27e5a70a525a5ee1fd5357596f07b724d02ff317f134e86cb6e3d9db968/kokoro_onnx-0.6.1-py3-none-any.whl",
        26084,
        "50c8de4950d601df41428ee5462a48c8a78bef441bf671f2492e070ef44d8a32",
    ),
    Asset(
        "phonemizer-3.4.0-py3-none-any.whl",
        PYPI + "37/41/09bafd0ccf4f4ddd1530c9f9d10e75a24056e10e120b9daa97f691fcabc3/phonemizer-3.4.0-py3-none-any.whl",
        108529,
        "5ff1215d0efa3606dd1b92449f6d71b5c1741efcc84c6d40c17bfaf64f6ded5f",
    ),
    Asset(
        "joblib-1.6.0-py3-none-any.whl",
        PYPI + "18/53/84099323c2ec4be98d935f63c033ac4151ee83836ca1050ede3b3aadf155/joblib-1.6.0-py3-none-any.whl",
        306115,
        "3dbbf9f6e4b592a2357b854608e980fe6390d131d7a82f011a377ef2ebef7aba",
    ),
    Asset(
        "cloudpickle-3.1.2-py3-none-any.whl",
        PYPI + "88/39/799be3f2f0f38cc727ee3b4f1445fe6d5e4133064ec2e4115069418a5bb6/cloudpickle-3.1.2-py3-none-any.whl",
        22228,
        "9acb47f6afd73f60dc1df93bb801b472f05ff42fa6c84167d25cb206be1fbf4a",
    ),
    Asset(
        "attrs-26.1.0-py3-none-any.whl",
        PYPI + "64/b4/17d4b0b2a2dc85a6df63d1157e028ed19f90d4cd97c36717afef2bc2f395/attrs-26.1.0-py3-none-any.whl",
        67548,
        "c647aa4a12dfbad9333ca4e71fe62ddc36f4e63b2d260a37a8b83d2f043ac309",
    ),
    Asset(
        "dlinfo-2.0.0-py3-none-any.whl",
        PYPI + "da/90/022c79d6e5e6f843268c10b84d4a021ee3afba0621d3c176d3ff2024bfc8/dlinfo-2.0.0-py3-none-any.whl",
        3654,
        "b32cc18e3ea67c0ca9ca409e5b41eed863bd1363dbc9dd3de90fedf11b61e7bc",
    ),
]
ESPEAK_WHEELS = {
    "win32": Asset(
        "espeakng_loader-0.2.4-py3-none-win_amd64.whl",
        PYPI + "9d/ed/a3d872fbad4f3a3f3db0e8c31768ab14e77cd77306de16b8b20b1e1df7ea/"
        "espeakng_loader-0.2.4-py3-none-win_amd64.whl",
        9437292,
        "41f1e08ac9deda2efd1ea9de0b81dab9f5ae3c4b24284f76533d0a7b1dd7abd7",
    ),
    "linux": Asset(
        "espeakng_loader-0.2.4-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        PYPI + "de/1e/25ec5ab07528c0fbb215a61800a38eca05c8a99445515a02d7fa5debcb32/"
        "espeakng_loader-0.2.4-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        10078484,
        "08721baf27d13d461f6be6eed9a65277e70d68234ff484fd8b9897b222cdcb6d",
    ),
    "linux-aarch64": Asset(
        "espeakng_loader-0.2.4-py3-none-manylinux_2_28_aarch64.whl",
        PYPI + "d9/ad/1b768d8daffc2996e07bbcb6f534d8de3202cd75fce1f1c45eced1ce6465/"
        "espeakng_loader-0.2.4-py3-none-manylinux_2_28_aarch64.whl",
        10037736,
        "d1e798141b46a050cdb75fcf3c17db969bb2c40394f3f4a48910655d547508b9",
    ),
    "darwin": Asset(
        "espeakng_loader-0.2.4-py3-none-macosx_11_0_arm64.whl",
        PYPI + "a8/26/258c0cd43b9bc1043301c5f61767d6a6c3b679df82790c9cb43a3277b865/"
        "espeakng_loader-0.2.4-py3-none-macosx_11_0_arm64.whl",
        9892565,
        "d27cdca31112226e7299d8562e889d3e38a1e48055c9ee381b45d669072ee59f",
    ),
}


# v0.1.23 licensing. Nothing below is bundled in the installers; each piece is downloaded only when the person
# presses Download for a natural voice, and these notes are shown next to that button first.
LICENSES: dict[str, list[tuple[str, str]]] = {
    "kokoro": [
        ("Kokoro-82M voice model", "Apache-2.0"),
        ("kokoro-onnx", "MIT"),
        ("phonemizer", "GPL-3.0-or-later"),
        ("espeak-ng (espeakng-loader)", "GPL-3.0-or-later"),
    ],
    "piper": [
        ("Piper voice (en_US lessac / amy)", "voice data: Blizzard 2013 Lessac licence / mimic3-voices (model card)"),
        ("phonemizer", "GPL-3.0-or-later"),
        ("espeak-ng (espeakng-loader)", "GPL-3.0-or-later"),
    ],
}
LICENSE_NOTE = (
    "Natural voices are downloaded separately, only when you press Download; they are not part of Project Away's "
    "MIT-licensed installer. They include espeak-ng and phonemizer, which are GPL-3.0 (free software; source at "
    "github.com/espeak-ng/espeak-ng and github.com/bootphon/phonemizer). Piper's voices come with their own data "
    "licence (see the Piper model card)."
)


def license_lines(engine: str) -> list[str]:
    """Plain lines for the voice picker: "phonemizer: GPL-3.0-or-later"."""
    return [f"{what}: {lic}" for what, lic in LICENSES.get(engine, [])]


class VoiceDownloadError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------ files
def voices_dir(data_dir: Path | None = None) -> Path:
    if data_dir is None:
        from away import paths

        data_dir = Path(paths.data_dir())
    return Path(data_dir) / "engines" / "voices"


def lib_dir(data_dir: Path | None = None) -> Path:
    return voices_dir(data_dir) / "lib"


# the processors the pinned espeak-ng wheels are built for (plat_key values)
ESPEAK_PLATFORMS = ("win32", "linux", "linux-aarch64", "darwin")


def plat_key(plat: str | None = None, machine: str | None = None) -> str:
    p = plat or sys.platform
    if p.startswith("win"):
        return "win32"
    if p == "darwin":  # the pinned espeak-ng wheel is Apple silicon only
        m = (machine if machine is not None else (_pf.machine() if plat is None else "arm64")).lower()
        return "darwin" if m in ("arm64", "aarch64") else f"darwin-{m}"
    m = (machine if machine is not None else (_pf.machine() if plat is None else "x86_64")).lower()
    if m in ("aarch64", "arm64"):
        return "linux-aarch64"
    return "linux" if m in ("x86_64", "amd64") else f"linux-{m}"


def assets_for(engine: str, voice: str = "", plat: str | None = None) -> list[Asset]:
    """What :func:`download` fetches for ``engine`` (kokoro / piper)."""
    key = plat_key(plat)
    if key not in ESPEAK_PLATFORMS:  # no espeak-ng build for this processor: the engine cannot run here
        return []
    esp = ESPEAK_WHEELS.get(key)
    wheels = WHEELS_ANY + ([esp] if esp else [])
    if engine == "kokoro":
        return KOKORO_MODEL + wheels
    if engine == "piper":
        v = voice if voice in PIPER_MODELS else PIPER_VOICES[0]
        return PIPER_MODELS[v] + [w for w in wheels if not w.name.startswith("kokoro_onnx")]
    return []


def _path(d: Path, a: Asset) -> Path:
    return d / ("wheels" if a.name.endswith(".whl") else "") / a.name


def _have(d: Path, a: Asset) -> bool:
    try:
        return _path(d, a).stat().st_size == a.size
    except OSError:
        return False


def download_size(engine: str, voice: str = "", plat: str | None = None, data_dir: Path | None = None) -> int:
    d = voices_dir(data_dir)
    return sum(a.size for a in assets_for(engine, voice, plat) if not _have(d, a))


def installed(engine: str, voice: str = "", data_dir: Path | None = None, plat: str | None = None) -> bool:
    d = voices_dir(data_dir)
    need = assets_for(engine, voice, plat)
    return bool(need) and all(_have(d, a) for a in need) and (lib_dir(data_dir) / ".unpacked").exists()


def _fetch(a: Asset, dest: Path, opener: Callable[..., Any], progress: Callable[[int], None]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    req = urllib.request.Request(a.url, headers={"User-Agent": "project-away"})
    with opener(req, timeout=60) as resp, part.open("wb") as fh:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            fh.write(chunk)
            h.update(chunk)
            progress(len(chunk))
    if part.stat().st_size != a.size or h.hexdigest() != a.sha256:
        part.unlink(missing_ok=True)
        raise VoiceDownloadError(f"{a.name} did not match its pinned checksum; nothing was installed")
    part.replace(dest)


def _safe(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    out = []
    for m in zf.infolist():
        p = PurePosixPath(m.filename)
        if p.is_absolute() or ".." in p.parts:
            raise VoiceDownloadError(f"unsafe path in a wheel: {m.filename}")
        out.append(m)
    return out


def unpack_wheels(data_dir: Path | None = None, only: list[Asset] | None = None) -> None:
    """Extract the wheels into ``lib`` (on ``sys.path`` once activated). ``only``: just these pinned wheels, each
    re-checked against its sha256 first (a stale or planted file in ``wheels/`` is never unpacked)."""
    lib = lib_dir(data_dir)
    lib.mkdir(parents=True, exist_ok=True)
    marker = lib / ".unpacked"
    marker.unlink(missing_ok=True)  # a half-finished unpack must not count as done
    d = voices_dir(data_dir)
    if only is None:
        files = sorted((d / "wheels").glob("*.whl"))
    else:
        files = []
        for a in only:
            if not a.name.endswith(".whl"):
                continue
            f = _path(d, a)
            if hashlib.sha256(f.read_bytes()).hexdigest() != a.sha256:
                f.unlink(missing_ok=True)
                raise VoiceDownloadError(f"{a.name} is damaged; download it again")
            files.append(f)
    for whl in files:
        with zipfile.ZipFile(whl) as zf:
            zf.extractall(lib, _safe(zf))
    marker.write_text("ok", encoding="utf-8")


def download(
    engine: str,
    voice: str = "",
    data_dir: Path | None = None,
    progress: Callable[[int, int], None] | None = None,
    opener: Callable[..., Any] = urllib.request.urlopen,
    plat: str | None = None,
) -> Path:
    """Fetch everything ``engine`` needs (pinned + verified), unpack the wheels. ``progress(done, total)``."""
    need = assets_for(engine, voice, plat)
    if not need:
        raise VoiceDownloadError(f"{ENGINE_TITLES.get(engine, engine)} needs no download")
    d = voices_dir(data_dir)
    todo = [a for a in need if not _have(d, a)]
    total = sum(a.size for a in todo)
    done = [0]

    def step(n: int) -> None:
        done[0] += n
        if progress is not None:
            with contextlib.suppress(Exception):
                progress(done[0], total)

    for a in todo:
        _fetch(a, _path(d, a), opener, step)
    unpack_wheels(data_dir, only=need)
    return d


def remove(data_dir: Path | None = None) -> bool:
    d = voices_dir(data_dir)
    if not d.exists():
        return False
    shutil.rmtree(d, ignore_errors=True)
    return True


def activate(data_dir: Path | None = None) -> bool:
    """Put the downloaded wheels on ``sys.path`` (after the app's own packages, so bundled ones win)."""
    lib = lib_dir(data_dir)
    if not (lib / ".unpacked").exists():
        return False
    if str(lib) not in sys.path:
        sys.path.append(str(lib))
    return True


# ------------------------------------------------------------------------------------------------ text
def chunks(text: str, first: int = FIRST_CHUNK_WORDS, size: int = CHUNK_WORDS) -> list[str]:
    """Short pieces to synthesize one after another: sentences, a short first piece, long sentences split at a
    comma near the limit or every ``size`` words."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?;:])\s+", " ".join(str(text).split())) if s.strip()]
    out: list[str] = []
    for s in sents:
        words = s.split()
        limit = first if not out else size
        while len(words) > limit:
            cut = limit
            for i in range(min(limit, len(words) - 1), max(2, limit // 2) - 1, -1):
                if words[i - 1].endswith(","):
                    cut = i
                    break
            out.append(" ".join(words[:cut]))
            words = words[cut:]
            limit = size
        if words:
            out.append(" ".join(words))
    return out


# ------------------------------------------------------------------------------------------------ audio
def play_chunks(synth: Callable[[str], tuple[Any, int]], pieces: list[str], play: Callable[[Any, int], None]) -> int:
    """Synthesize piece n+1 while piece n plays. Returns how many pieces were played."""
    q: queue.Queue[Any] = queue.Queue(maxsize=2)
    err: list[BaseException] = []
    stop = threading.Event()  # set when playback ends early (play raised): the producer must not block on put

    def put(item: Any) -> bool:
        while not stop.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def producer() -> None:
        try:
            for p in pieces:
                if stop.is_set() or not put(synth(p)):
                    return
        except BaseException as exc:  # re-raised on the speaking thread
            err.append(exc)
        finally:
            put(None)

    threading.Thread(target=producer, daemon=True, name="voice-synth").start()
    n = 0
    try:
        while True:
            item = q.get()
            if item is None:
                break
            samples, sr = item
            if n == 0:
                latency.mark("first_audio")  # v0.1.26 latency log: the first piece is synthesized and starts playing
            play(samples, sr)
            n += 1
    finally:
        stop.set()
    if err and n == 0:
        raise err[0]
    return n


def sd_play(samples: Any, sr: int, volume: float = 1.0) -> None:  # pragma: no cover - audio device
    import numpy as np

    from away.voice import audio_out

    audio_out.play_blocking(np.asarray(samples, dtype=np.float32) * float(volume), int(sr))


def providers() -> list[str]:
    """onnxruntime providers: CUDA first when this onnxruntime has it, then the CPU."""
    try:
        import onnxruntime as ort

        have = set(ort.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"]
    return [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in have] or ["CPUExecutionProvider"]


# ------------------------------------------------------------------------------------------------ engines
class KokoroEngine:
    name = "Natural (Kokoro)"

    def __init__(
        self,
        data_dir: Path | None = None,
        voice: Callable[[], str] = lambda: "af_heart",
        play: Callable[[Any, int, float], None] = sd_play,
        model: Any = None,
    ) -> None:
        self.voice = voice
        self.play = play
        self.used_voice = ""
        if model is None:
            if not installed("kokoro", data_dir=data_dir):
                raise RuntimeError("the natural voice is not downloaded yet (Settings > Voice & input > Download)")
            activate(data_dir)
            import onnxruntime as ort
            from kokoro_onnx import Kokoro

            d = voices_dir(data_dir)
            sess = ort.InferenceSession(str(d / KOKORO_MODEL[0].name), providers=providers())
            model = Kokoro.from_session(sess, str(d / KOKORO_MODEL[1].name))
        self.model = model

    def speak(self, text: str, v: Any) -> None:
        name = self.voice() or KOKORO_VOICES[0]
        if name not in KOKORO_VOICES:
            name = KOKORO_VOICES[0]
        speed = SPEEDS.get(str(getattr(v, "rate", "normal")), 1.0)
        vol = VOLUMES.get(str(getattr(v, "volume", "normal")), 0.9)
        self.used_voice = name

        def synth(piece: str) -> tuple[Any, int]:
            samples, sr = self.model.create(piece, voice=name, speed=speed, lang="en-us")
            return samples, int(sr)

        play_chunks(synth, chunks(text), lambda s, sr: self.play(s, sr, vol))

    def close(self) -> None:
        pass


def piper_ids(phonemes: str, id_map: dict[str, list[int]]) -> list[int]:
    """Piper's phoneme ids: ``^``, each known phoneme followed by the pad ``_``, then ``$``."""
    pad = id_map.get("_", [0])
    ids = list(id_map.get("^", [1])) + list(pad)
    for ch in phonemes:
        if ch in id_map:
            ids += list(id_map[ch]) + list(pad)
    return ids + list(id_map.get("$", [2]))


def espeak_phonemizer(language: str = "en-us") -> Callable[[str], str]:  # pragma: no cover - native library
    """IPA phonemes from espeak-ng (the downloaded espeakng-loader library) through phonemizer."""
    import espeakng_loader
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    if hasattr(EspeakWrapper, "set_data_path"):
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    from phonemizer.backend import EspeakBackend

    backend = EspeakBackend(language, preserve_punctuation=True, with_stress=True)

    def run(text: str) -> str:
        return str(backend.phonemize([text], strip=True)[0])

    return run


class PiperEngine:
    name = "Light (Piper)"

    def __init__(
        self,
        data_dir: Path | None = None,
        voice: Callable[[], str] = lambda: "lessac",
        play: Callable[[Any, int, float], None] = sd_play,
        session: Any = None,
        config: dict[str, Any] | None = None,
        phonemize: Callable[[str], str] | None = None,
    ) -> None:
        self.play = play
        name = voice() or PIPER_VOICES[0]
        self.used_voice = name if name in PIPER_MODELS else PIPER_VOICES[0]
        if session is None:
            if not installed("piper", self.used_voice, data_dir=data_dir):
                raise RuntimeError("the light voice is not downloaded yet (Settings > Voice & input > Download)")
            activate(data_dir)
            import onnxruntime as ort

            d = voices_dir(data_dir)
            files = PIPER_MODELS[self.used_voice]
            session = ort.InferenceSession(str(d / files[0].name), providers=providers())
            config = json.loads((d / files[1].name).read_text(encoding="utf-8"))
        self.session = session
        self.config = config or {}
        self.phonemize = phonemize or espeak_phonemizer(str((self.config.get("espeak") or {}).get("voice", "en-us")))

    def synth(self, text: str, speed: float = 1.0) -> tuple[Any, int]:
        import numpy as np

        inf = self.config.get("inference") or {}
        ids = piper_ids(self.phonemize(text), self.config.get("phoneme_id_map") or {})
        scales = np.array(
            [
                float(inf.get("noise_scale", 0.667)),
                float(inf.get("length_scale", 1.0)) / max(0.5, speed),
                float(inf.get("noise_w", 0.8)),
            ],
            dtype=np.float32,
        )
        audio = self.session.run(
            None,
            {
                "input": np.array([ids], dtype=np.int64),
                "input_lengths": np.array([len(ids)], dtype=np.int64),
                "scales": scales,
            },
        )[0]
        sr = int((self.config.get("audio") or {}).get("sample_rate", 22050))
        return np.asarray(audio, dtype=np.float32).reshape(-1), sr

    def speak(self, text: str, v: Any) -> None:
        speed = SPEEDS.get(str(getattr(v, "rate", "normal")), 1.0)
        vol = VOLUMES.get(str(getattr(v, "volume", "normal")), 0.9)
        play_chunks(lambda p: self.synth(p, speed), chunks(text), lambda s, sr: self.play(s, sr, vol))

    def close(self) -> None:
        pass


class WindowsNaturalEngine:
    """The OneCore voices (Windows 10/11) via WinRT SpeechSynthesizer, from Windows PowerShell 5.1. The text goes
    through stdin; the audio goes to a temporary .wav played with System.Media.SoundPlayer, then deleted."""

    name = "Windows natural"
    SCRIPT = (
        "$ErrorActionPreference='Stop'; Add-Type -AssemblyName System.Runtime.WindowsRuntime; "
        "$null=[Windows.Media.SpeechSynthesis.SpeechSynthesizer,Windows.Media.SpeechSynthesis,"
        "ContentType=WindowsRuntime]; "
        "$null=[Windows.Storage.Streams.DataReader,Windows.Storage.Streams,ContentType=WindowsRuntime]; "
        "function Await($t,$r){$m=([System.WindowsRuntimeSystemExtensions].GetMethods()|?{$_.Name -eq 'AsTask' "
        "-and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'})"
        "[0].MakeGenericMethod($r); $k=$m.Invoke($null,@($t)); $k.Wait(-1)|Out-Null; $k.Result}; "
        "$s=New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer; {select}"
        "$s.Options.SpeakingRate={rate}; $s.Options.AudioVolume={volume}; "
        "$st=Await ($s.SynthesizeTextToStreamAsync([Console]::In.ReadToEnd())) "
        "([Windows.Media.SpeechSynthesis.SpeechSynthesisStream]); "
        "$rd=New-Object Windows.Storage.Streams.DataReader($st.GetInputStreamAt(0)); "
        "$n=Await ($rd.LoadAsync([uint32]$st.Size)) ([uint32]); $b=New-Object byte[] $n; $rd.ReadBytes($b); "
        "$f=[IO.Path]::Combine([IO.Path]::GetTempPath(),'away-voice-'+[guid]::NewGuid().ToString()+'.wav'); "
        "[IO.File]::WriteAllBytes($f,$b); $p=New-Object System.Media.SoundPlayer $f; $p.PlaySync(); "
        "Remove-Item $f -ErrorAction SilentlyContinue"
    )

    def __init__(self, exe: str, run: Callable[..., Any] = subprocess.run) -> None:
        self.exe = exe
        self.run = run

    def command(self, v: Any) -> list[str]:
        want = str(getattr(v, "voice", "") or "")
        select = ""
        if want and all(c.isalnum() or c in " -_" for c in want):
            select = (
                "$m=[Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices|"
                f"?{{$_.DisplayName -like '*{want}*'}}|Select-Object -First 1; if($m){{$s.Voice=$m}}; "
            )
        rate = SPEEDS.get(str(getattr(v, "rate", "normal")), 1.0)
        vol = VOLUMES.get(str(getattr(v, "volume", "normal")), 0.9)
        script = self.SCRIPT.replace("{select}", select).replace("{rate}", f"{rate:g}").replace("{volume}", f"{vol:g}")
        return [self.exe, "-NoProfile", "-NonInteractive", "-Command", script]

    def speak(self, text: str, v: Any) -> None:
        kw: dict[str, Any] = {"input": text, "text": True, "timeout": 120, "capture_output": True}
        if sys.platform == "win32":  # pragma: no cover
            kw["creationflags"] = 0x08000000
        res = self.run(self.command(v), **kw)
        if getattr(res, "returncode", 0):
            raise RuntimeError(f"Windows natural voice failed ({str(getattr(res, 'stderr', ''))[:120]})")

    def close(self) -> None:
        pass


def natural_factories(
    settings: Callable[[], Any],
    data_dir: Path | None = None,
    platform: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
    have_kokoro: Callable[[], bool] | None = None,
    have_piper: Callable[[], bool] | None = None,
) -> list[tuple[str, Callable[[], Any]]]:
    """The natural engines in the order ``speech.engine`` asks for (``auto``: Kokoro, Piper, Windows natural; a
    chosen one first, the others after it as fallbacks); engines not downloaded are left out. The classic voices of
    :mod:`away.voice.speech` come after these."""
    plat = platform or sys.platform

    def cfg(name: str, default: str) -> str:
        s = settings()
        return str(getattr(s, name, default) or default) if s is not None else default

    want = cfg("engine", "auto")
    if want == "classic":
        return []
    hk = have_kokoro or (lambda: installed("kokoro", data_dir=data_dir, plat=plat))
    hp = have_piper or (lambda: installed("piper", cfg("piper_voice", "lessac"), data_dir=data_dir, plat=plat))
    table: dict[str, tuple[str, Callable[[], Any]] | None] = {
        "kokoro": (KokoroEngine.name, lambda: KokoroEngine(data_dir, lambda: cfg("kokoro_voice", "af_heart")))
        if hk()
        else None,
        "piper": (PiperEngine.name, lambda: PiperEngine(data_dir, lambda: cfg("piper_voice", "lessac")))
        if hp()
        else None,
        "windows": None,
    }
    if plat.startswith("win"):
        ps = which("powershell") or which("powershell.exe")
        if ps:
            table["windows"] = (WindowsNaturalEngine.name, lambda: WindowsNaturalEngine(ps))
    order = ["kokoro", "piper", "windows"]
    if want in order:
        order = [want] + [o for o in order if o != want]
    return [e for e in (table[o] for o in order) if e is not None]


def preview_text(engine: str) -> str:
    title = ENGINE_TITLES.get(engine, "chosen").split(" (")[0].lower()
    return f"Hi, this is the {title} voice. Your agent will sound like this."
