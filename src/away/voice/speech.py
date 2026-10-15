"""Speaking to the person (v0.1.21): text-to-speech for Agent Link's notify_user / ask_user.

v0.1.20 only spoke through pyttsx3, which is optional and not in the installers, so ``notify(speak=true)`` always
answered ``spoken: false`` on the user's PC. Now, in order, the first engine this computer has:

* Windows: the built-in Windows voice (SAPI ``SAPI.SpVoice``) through comtypes, which the installer already ships
  for the smart pointer (late-bound COM: no generated wrapper modules); without comtypes, the same voice through
  PowerShell's System.Speech;
* pyttsx3 when it is installed (any OS);
* macOS ``say``; Linux ``spd-say`` (speech-dispatcher), ``espeak-ng`` or ``espeak``.

Speaking never blocks the caller: :class:`Speaker` queues the text for its own worker thread (COM objects are
used only on the thread that created them) and returns at once. The settings (``speech:`` in the config) are read
at every message: "Let agents talk to me" (``enabled``), how fast (``rate``: slow | normal | fast), how loud
(``volume``: quiet | normal | loud) and which voice (``voice``: part of a voice's name, "" = the system default).
"""

from __future__ import annotations

import contextlib
import functools
import logging
import queue
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from away.voice import latency

log = logging.getLogger("away.speech")

ECHO_TAIL_S = 0.7  # v0.1.22: the microphone keeps ignoring sound this long after the voice stops (room echo)


class EchoGuard:
    """v0.1.22: while the computer's voice speaks (and ``tail_s`` after), the microphone must not hear it: the
    agent's spoken "Can you hear me now?" was transcribed and typed. :class:`Speaker` marks each spoken message;
    the voice input asks :meth:`active` (drop the audio) and :meth:`overlaps` (drop an utterance that ran into it).
    Times are ``time.monotonic()`` seconds."""

    def __init__(self, tail_s: float = ECHO_TAIL_S, clock: Callable[[], float] | None = None) -> None:
        import time as _time

        self.tail_s = float(tail_s)
        self.clock = clock or _time.monotonic
        self._lock = threading.Lock()
        self._speaking = 0  # nested begin() calls
        self._since = 0.0
        self.spans: list[tuple[float, float]] = []  # finished (start, end + tail), the last few

    def begin(self) -> None:
        with self._lock:
            if self._speaking == 0:
                self._since = self.clock()
            self._speaking += 1

    def end(self) -> None:
        with self._lock:
            if self._speaking == 0:
                return
            self._speaking -= 1
            if self._speaking == 0:
                self.spans = [*self.spans[-19:], (self._since, self.clock() + self.tail_s)]

    @contextlib.contextmanager
    def speaking(self) -> Any:
        self.begin()
        try:
            yield
        finally:
            self.end()

    def cut(self, keep_s: float = 1.2) -> None:
        """v0.1.26 barge-in: you started talking over the voice. The microphone listens again at once (no echo
        tail), and the last ``keep_s`` seconds no longer count as the computer's voice: they hold your first
        words (the voice input replays them, :attr:`barged`)."""
        with self._lock:
            now = self.clock()
            since = self._since if self._speaking else now
            self._speaking = 0
            self.spans = [(a, min(b, now - keep_s)) for a, b in self.spans[-19:]]
            self.spans.append((since, now - float(keep_s)))
            self.barged = True

    barged = False

    def take_barge(self) -> bool:
        with self._lock:
            b, self.barged = self.barged, False
            return b

    def active(self, now: float | None = None) -> bool:
        """True while the voice speaks or within ``tail_s`` after it stopped."""
        t = self.clock() if now is None else now
        with self._lock:
            if self._speaking:
                return True
            return any(a <= t <= b for a, b in self.spans)

    def overlaps(self, t0: float, t1: float) -> bool:
        """True when ``[t0, t1]`` touches a time the voice was speaking (or its tail)."""
        with self._lock:
            if self._speaking and t1 >= self._since:
                return True
            return any(t0 <= b and t1 >= a for a, b in self.spans)


ECHO = EchoGuard()  # one per process: the app's microphone and Agent Link's voice share it

RATES = ("slow", "normal", "fast")
VOLUMES = ("quiet", "normal", "loud")
SAPI_RATE = {"slow": -3, "normal": 0, "fast": 3}  # SpVoice.Rate: -10..10
SAPI_VOLUME = {"quiet": 45, "normal": 80, "loud": 100}  # SpVoice.Volume: 0..100
WPM = {"slow": 145, "normal": 180, "fast": 225}  # say / pyttsx3 / espeak words per minute
SPD_RATE = {"slow": "-35", "normal": "0", "fast": "35"}  # spd-say -r: -100..100
SPD_VOLUME = {"quiet": "-50", "normal": "0", "loud": "60"}  # spd-say -i: -100..100
ESPEAK_AMP = {"quiet": "60", "normal": "110", "loud": "170"}  # espeak -a: 0..200
MAX_TEXT = 600
MAX_QUEUE = 4
IDLE_S = 30.0
WARM = "\x00warm"  # v0.1.26: queue marker: load the voice engine now, say nothing
CREATE_NO_WINDOW = 0x08000000


@dataclass
class Voice:
    rate: str = "normal"
    volume: str = "normal"
    voice: str = ""  # part of a voice name ("Zira", "David"); "" = the system default


class Engine(Protocol):
    name: str

    def speak(self, text: str, v: Voice) -> None: ...  # blocking; called on the speaker's worker thread
    def close(self) -> None: ...


def _clean(text: str) -> str:
    t = " ".join(str(text).split())
    return t[:MAX_TEXT]


def pick(table: dict[str, Any], key: str) -> Any:
    return table.get(str(key), table["normal"])


# ------------------------------------------------------------------------------------------------ engines
class SapiEngine:
    """The Windows voice through comtypes' late-bound COM (``SAPI.SpVoice``). Create it on the thread that uses
    it (the speaker's worker)."""

    name = "Windows voice (SAPI)"

    def __init__(self, create: Callable[[str], Any] | None = None, co_init: Callable[[], Any] | None = None) -> None:
        if create is None:  # pragma: no cover - Windows only
            import comtypes
            import comtypes.client

            if co_init is None:
                co_init = comtypes.CoInitialize

            def create(progid: str) -> Any:
                return comtypes.client.CreateObject(progid, dynamic=True)

        if co_init is not None:
            with contextlib.suppress(Exception):  # already initialised on this thread is fine
                co_init()
        self.voice = create("SAPI.SpVoice")
        self._voice_name = ""

    def voices(self) -> list[str]:
        out = []
        tokens = self.voice.GetVoices()
        for i in range(int(tokens.Count)):
            with contextlib.suppress(Exception):
                out.append(str(tokens.Item(i).GetDescription()))
        return out

    def _choose(self, wanted: str) -> None:
        if wanted == self._voice_name:
            return
        self._voice_name = wanted
        if not wanted:
            return
        tokens = self.voice.GetVoices()
        for i in range(int(tokens.Count)):
            tok = tokens.Item(i)
            with contextlib.suppress(Exception):
                if wanted.lower() in str(tok.GetDescription()).lower():
                    self.voice.Voice = tok
                    return
        log.info("speech: no Windows voice named like %r; using the default voice", wanted)

    def speak(self, text: str, v: Voice) -> None:
        with contextlib.suppress(Exception):
            self._choose(v.voice)
        self.voice.Rate = int(pick(SAPI_RATE, v.rate))
        self.voice.Volume = int(pick(SAPI_VOLUME, v.volume))
        self.voice.Speak(text, 0)  # synchronous here: this is the speaker's own thread

    def close(self) -> None:
        self.voice = None


class PowerShellEngine:
    """The same Windows voice through PowerShell's System.Speech (when comtypes is not installed). The text goes
    through stdin, never the command line."""

    name = "Windows voice (PowerShell)"
    SCRIPT = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.Rate = {rate}; $s.Volume = {volume}; "
        "{select}"
        "$s.Speak([Console]::In.ReadToEnd())"
    )

    def __init__(self, exe: str, run: Callable[..., Any] = subprocess.run) -> None:
        self.exe = exe
        self.run = run

    def command(self, v: Voice) -> list[str]:
        select = ""
        if v.voice and all(c.isalnum() or c in " -_" for c in v.voice):
            select = (
                f"$m = $s.GetInstalledVoices() | Where-Object {{ $_.VoiceInfo.Name -like '*{v.voice}*' }} | "
                "Select-Object -First 1; if ($m) { $s.SelectVoice($m.VoiceInfo.Name) }; "
            )
        script = self.SCRIPT.format(
            rate=int(pick(SAPI_RATE, v.rate)), volume=int(pick(SAPI_VOLUME, v.volume)), select=select
        )
        return [self.exe, "-NoProfile", "-NonInteractive", "-Command", script]

    def speak(self, text: str, v: Voice) -> None:
        kw: dict[str, Any] = {"input": text, "text": True, "timeout": 120, "capture_output": True}
        if sys.platform == "win32":  # pragma: no cover
            kw["creationflags"] = CREATE_NO_WINDOW
        self.run(self.command(v), **kw)

    def close(self) -> None:
        pass


class Pyttsx3Engine:
    name = "pyttsx3"

    def __init__(self) -> None:  # pragma: no cover - optional package with audio
        import pyttsx3

        self.eng = pyttsx3.init()

    def speak(self, text: str, v: Voice) -> None:  # pragma: no cover - audio
        self.eng.setProperty("rate", pick(WPM, v.rate))
        self.eng.setProperty("volume", pick(SAPI_VOLUME, v.volume) / 100.0)
        self.eng.say(text)
        self.eng.runAndWait()

    def close(self) -> None:  # pragma: no cover
        with contextlib.suppress(Exception):
            self.eng.stop()


class CommandEngine:
    """``say`` (macOS), ``spd-say``, ``espeak-ng`` / ``espeak`` (Linux): the text is one argument (no shell)."""

    def __init__(self, exe: str, kind: str, run: Callable[..., Any] = subprocess.run) -> None:
        self.exe, self.kind, self.run = exe, kind, run
        self.name = kind

    def command(self, text: str, v: Voice) -> list[str]:
        if self.kind == "say":
            cmd = [self.exe, "-r", str(pick(WPM, v.rate))]
            if v.voice and all(c.isalnum() or c in " -_" for c in v.voice):
                cmd += ["-v", v.voice]
            return [*cmd, "--", text]
        if self.kind == "spd-say":
            return [self.exe, "-w", "-r", pick(SPD_RATE, v.rate), "-i", pick(SPD_VOLUME, v.volume), "--", text]
        return [self.exe, "-s", str(pick(WPM, v.rate)), "-a", pick(ESPEAK_AMP, v.volume), "--", text]

    def speak(self, text: str, v: Voice) -> None:
        self.run(self.command(text, v), timeout=120, capture_output=True, check=False)

    def close(self) -> None:
        pass


def engine_factories(
    platform: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
    have: Callable[[str], bool] | None = None,
) -> list[tuple[str, Callable[[], Engine]]]:
    """The engines this computer could use, best first: ``[(name, factory)]``. Cheap (nothing is created)."""
    import importlib.util

    plat = platform or sys.platform
    has = have or (lambda mod: importlib.util.find_spec(mod) is not None)
    out: list[tuple[str, Callable[[], Engine]]] = []
    if plat == "win32":
        if has("comtypes"):
            out.append((SapiEngine.name, SapiEngine))
        ps = which("powershell") or which("powershell.exe")
        if ps:
            out.append((PowerShellEngine.name, lambda: PowerShellEngine(ps)))
    if has("pyttsx3"):
        out.append((Pyttsx3Engine.name, Pyttsx3Engine))
    if plat == "darwin":
        say = which("say")
        if say:
            out.append(("say", lambda: CommandEngine(say, "say")))
    elif plat.startswith("linux"):
        for exe, kind in (("spd-say", "spd-say"), ("espeak-ng", "espeak"), ("espeak", "espeak")):
            path = which(exe)
            if path:
                out.append((exe, functools.partial(CommandEngine, path, kind)))
                break
    return out


# ------------------------------------------------------------------------------------------------ speaker
class Speaker:
    """Speaks on its own thread, one message after another (at most ``MAX_QUEUE`` waiting; older ones are dropped
    so the voice never lags far behind). ``settings()`` returns the config's ``speech`` section (read per
    message)."""

    def __init__(
        self,
        settings: Callable[[], Any] | None = None,
        factories: list[tuple[str, Callable[[], Engine]]] | None = None,
        on_error: Callable[[str], None] | None = None,
        echo: EchoGuard | None = None,
    ) -> None:
        self.echo = echo or ECHO  # v0.1.22: the microphone ignores the voice while it speaks
        self.settings = settings or (lambda: None)
        # v0.1.23: no explicit list = the natural voices (away.voice.natural) first, re-checked whenever an engine
        # starts (a voice downloaded in Settings is used from the next message on), then the classic voices
        self._auto = factories is None
        self._failed: set[str] = set()
        self.factories = self._auto_factories() if factories is None else list(factories)
        self.on_error = on_error
        self.engine_name = ""
        self.last_error = ""
        self.spoken: list[str] = []  # what was said (the last few; tests, diagnostics)
        self._q: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        # Distinguish actual engine playback from queued/loading speech. Barge-in must not calibrate echo
        # against the user's voice while a model is still loading.
        self._speaking = threading.Event()
        # v0.1.26: how long the voice engine stays loaded after the last message (the live agent raises it so a
        # follow-up never waits for the natural voice to load again)
        self.idle_s = IDLE_S

    # ---------------------------------------------------------------- settings
    def _cfg(self) -> Any:
        try:
            return self.settings()
        except Exception:
            return None

    def allowed(self) -> bool:
        """ "Let agents talk to me" (``speech.enabled``, default on)."""
        s = self._cfg()
        return bool(getattr(s, "enabled", True)) if s is not None else True

    def read_aloud_by_default(self) -> bool:
        s = self._cfg()
        return bool(getattr(s, "read_messages_aloud", False)) if s is not None else False

    def voice(self) -> Voice:
        s = self._cfg()
        if s is None:
            return Voice()
        return Voice(
            str(getattr(s, "rate", "normal") or "normal"),
            str(getattr(s, "volume", "normal") or "normal"),
            str(getattr(s, "voice", "") or ""),
        )

    def available(self) -> bool:
        return bool(self.factories)

    def status(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed(),
            "engine": self.engine_name or (self.factories[0][0] if self.factories else None),
            "available": self.available(),
            "last_error": self.last_error or None,
        }

    # ------------------------------------------------------------------- speak
    def say(self, text: str, *, force: bool = False) -> dict[str, Any]:
        """Queue ``text``; returns at once: ``{"spoken": True, "engine": ...}`` when it will be said, else
        ``{"spoken": False, "reason": ...}`` (the setting is off, no engine, nothing to say)."""
        t = _clean(text)
        if not t:
            return {"spoken": False, "reason": "nothing to say"}
        if not force and not self.allowed():
            return {"spoken": False, "reason": "speaking is turned off (Settings > Agent Link: Let agents talk to me)"}
        if not self.factories:
            return {"spoken": False, "reason": "no voice is available on this computer"}
        with self._lock:
            while self._q.qsize() >= MAX_QUEUE:
                with contextlib.suppress(queue.Empty):
                    self._q.get_nowait()  # drop the oldest waiting message
            self._idle.clear()
            self._q.put(t)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, daemon=True, name="away-speech")
                self._thread.start()
        return {
            "spoken": True,
            "engine": self.engine_name or self.factories[0][0],
            "voice": self.voice_label(),
            "queued": True,
        }

    __call__ = say

    def warm(self) -> bool:
        """v0.1.26: load the voice engine now (on its thread) so the first sentence of an answer plays at once.
        Called when you start talking to an agent. Says nothing; False when no voice is available."""
        if not self.factories:
            return False
        with self._lock:
            if self._thread is not None and self._thread.is_alive() and self._warm:
                return True
            self._q.put(WARM)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, daemon=True, name="away-speech")
                self._thread.start()
        return True

    @property
    def _warm(self) -> bool:
        return bool(getattr(self, "_eng_ready", False))

    def wait(self, timeout: float = 30.0) -> bool:
        """Until everything queued was spoken (tests, the self-test)."""
        return self._idle.wait(timeout)

    def stop(self) -> None:
        self._q.put(None)

    def _auto_factories(self) -> list[tuple[str, Callable[[], Engine]]]:
        out: list[tuple[str, Callable[[], Engine]]] = []
        with contextlib.suppress(Exception):  # v0.1.24: Kokoro fp16 on the GPU / pocket-tts, only when chosen
            from away.voice.tts_next import next_factories

            out += next_factories(self._cfg_or_none)
        with contextlib.suppress(Exception):
            from away.voice.natural import natural_factories

            out += natural_factories(self._cfg_or_none)
        out += engine_factories()
        return [f for f in out if f[0] not in self._failed]

    def _cfg_or_none(self) -> Any:
        return self._cfg()

    def voice_label(self) -> str:
        """v0.1.23: which voice speaks ("Natural (Kokoro): af_heart"), reported back to agents."""
        name = self.engine_name or (self.factories[0][0] if self.factories else "")
        s = self._cfg()
        if name.startswith("Natural (Kokoro)"):
            return f"{name}: {getattr(s, 'kokoro_voice', 'af_heart')}"
        if name.startswith("Light (Piper)"):
            return f"{name}: {getattr(s, 'piper_voice', 'lessac')}"
        return name

    def _make(self) -> Engine | None:
        if self._auto:
            self.factories = self._auto_factories()
        while self.factories:
            name, factory = self.factories[0]
            try:
                eng = factory()
                self.engine_name = name
                return eng
            except Exception as exc:
                self._fail(f"{name} could not start: {exc}")
                self._failed.add(name)
                self.factories.pop(0)  # try the next engine
        return None

    def _fail(self, msg: str) -> None:
        self.last_error = msg
        log.warning("speech: %s", msg)
        if self.on_error is not None:
            with contextlib.suppress(Exception):
                self.on_error(msg)

    def _run(self) -> None:
        eng: Engine | None = None
        try:
            while True:
                try:
                    text = self._q.get(timeout=max(1.0, float(self.idle_s)))
                except queue.Empty:
                    break  # idle: let the thread (and its COM object) go; the next message starts a new one
                if text is None:
                    break
                # Switching/deleting a custom profile must take effect on the next queued message rather than
                # leaving a stale cloned voice loaded for the entire idle timeout.
                profile = getattr(eng, "profile", None) if eng is not None else None
                if profile is not None:
                    selected = str(getattr(self._cfg(), "custom_voice_profile", "") or "")
                    if selected != str(getattr(profile, "id", "")):
                        with contextlib.suppress(Exception):
                            eng.close()
                        eng = None
                        self.engine_name = ""
                        self._eng_ready = False
                if eng is None:
                    eng = self._make()
                    self._eng_ready = eng is not None
                if text == WARM:
                    if self._q.empty():
                        self._idle.set()
                    continue
                if eng is None:
                    self._fail("no voice engine could start")
                else:
                    for _attempt in range(4):  # v0.1.23: a failing engine falls back silently to the next one
                        try:
                            latency.mark("tts_start")
                            self._speaking.set()
                            try:
                                with self.echo.speaking():
                                    eng.speak(text, self.voice())
                            finally:
                                self._speaking.clear()
                            latency.spoken()  # engines that cannot tell when their sound started
                            self.spoken = [*self.spoken[-19:], text]
                            break
                        except Exception as exc:
                            self._fail(f"{self.engine_name} failed: {exc}")
                            with contextlib.suppress(Exception):
                                eng.close()
                            eng = None
                            if len(self.factories) <= 1:
                                break
                            self._failed.add(self.engine_name)
                            self.factories.pop(0)
                            eng = self._make()
                            self._eng_ready = eng is not None
                            if eng is None:
                                break
                if self._q.empty():
                    self._idle.set()
        finally:
            self._eng_ready = False
            if eng is not None:
                with contextlib.suppress(Exception):
                    eng.close()
            self._idle.set()


def selftest_check(platform: str | None = None) -> str:
    """The installer self-test's ``speech`` check: on Windows the built-in voice must load through comtypes
    (SAPI.SpVoice, its voice list, rate and volume); elsewhere it reports which engine would be used."""
    plat = platform or sys.platform
    if plat == "win32":  # pragma: no cover - Windows only (runs on the Windows release runner)
        import comtypes
        import comtypes.client

        comtypes.CoInitialize()
        try:
            voice = comtypes.client.CreateObject("SAPI.SpVoice", dynamic=True)
            voice.Rate = SAPI_RATE["normal"]
            voice.Volume = SAPI_VOLUME["normal"]
            assert int(voice.Rate) == SAPI_RATE["normal"], "SpVoice.Rate did not stick"
            n = int(voice.GetVoices().Count)
            names = []
            for i in range(min(n, 3)):
                with contextlib.suppress(Exception):
                    names.append(str(voice.GetVoices().Item(i).GetDescription()))
            synth = ""
            if n:
                try:
                    stream = comtypes.client.CreateObject("SAPI.SpMemoryStream", dynamic=True)
                    voice.AudioOutputStream = stream
                    voice.Speak("Project Away", 0)
                    size = len(bytes(stream.GetData()))
                    synth = f"; synthesized {size} bytes"
                except Exception as exc:
                    synth = f"; synthesis not checked ({type(exc).__name__})"
            return f"SAPI.SpVoice ok, {n} voice(s){': ' + ', '.join(names) if names else ''}{synth}"
        finally:
            with contextlib.suppress(Exception):
                comtypes.CoUninitialize()
    names = [n for n, _f in engine_factories(plat)]
    return f"engines here: {', '.join(names)}" if names else "no voice engine here (optional: say / spd-say / espeak)"
