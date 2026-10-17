"""``away`` command line."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from away import __version__, paths
from away import config as cfgmod

EXTRAS = {
    "hands/face": ["mediapipe", "cv2"],
    "runtime models": ["onnxruntime"],
    "voice": ["faster_whisper", "sounddevice"],
    "control": ["pynput", "pyperclip"],
    "tray": ["pystray", "PIL"],
    "train": ["torch", "onnx"],
    "settings window": ["tkinter"],
    "parakeet (optional, not in installers)": ["nemo"],
}


def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def _models_dir(cfg: cfgmod.AppConfig) -> Path:
    return Path(cfg.models_dir) if cfg.models_dir else paths.models_dir()


def cmd_doctor(args: argparse.Namespace) -> int:
    cfg, unknown = cfgmod.load()
    info: dict[str, object] = {
        "version": __version__,
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "config": str(paths.config_file()),
        "data_dir": str(paths.data_dir()),
        "extras": {k: {m: _has(m) for m in mods} for k, mods in EXTRAS.items()},
        "models": {
            n: (_models_dir(cfg) / n).exists()
            for n in ("hand_landmarker.task", "face_landmarker.task", "gesture_tcn.onnx", "static_pose.npz")
        },
        "unknown_config_keys": unknown,
    }
    from away.hands.runtime import bundled_model

    info["bundled_models"] = {n: bundled_model(n) is not None for n in ("static_pose.npz", "gesture_tcn.onnx")}
    print(json.dumps(info, indent=2, default=str))
    return 0


def cmd_hardware(args: argparse.Namespace) -> int:
    from away.app import gpupack, hardware, performance

    cfg, _ = cfgmod.load()
    hw = hardware.detect()
    prof = performance.current_profile(cfg, hw)
    if args.json:
        from away.control import monitors

        print(
            json.dumps(
                {
                    "hardware": hw.to_dict(),
                    "profile": prof.to_dict(),
                    "gpu_pack": gpupack.status(),
                    "monitors": monitors.enumerate_monitors().to_dict(),
                },
                indent=2,
            )
        )
        return 0
    print("\n".join(performance.describe(hw, prof)))
    from away.control import monitors

    print("Monitors: " + "\n".join(monitors.enumerate_monitors().lines()))
    if hw.notes:
        print("Notes:")
        for n in hw.notes:
            print(f"  - {n}")
    if prof.gpu_pack == "recommended":
        from away.app.frozen import is_frozen

        print("\nEnable the GPU: " + ("away gpu install" if is_frozen() else gpupack.PIP_HINT))
    return 0


def cmd_monitors(args: argparse.Namespace) -> int:
    from away.control import monitors

    layout = monitors.enumerate_monitors()
    cfg, _ = cfgmod.load()
    sel = monitors.select_monitors(layout, cfg.hands.cursor.monitors)
    if args.json:
        d = layout.to_dict()
        d["dpi_awareness"] = monitors.enable_dpi_awareness()
        d["cursor_monitors"] = [layout.monitors.index(m) + 1 for m in sel]
        print(json.dumps(d, indent=2))
        return 0
    print("\n".join(layout.lines()))
    used = ", ".join(str(layout.monitors.index(m) + 1) for m in sel)
    print(f"  cursor uses: {used} (hands.cursor.monitors: {cfg.hands.cursor.monitors})")
    print('  jump with the voice commands "next monitor", "previous monitor", "monitor one/two/three"')
    return 0


def cmd_gpu(args: argparse.Namespace) -> int:
    from away.app import gpupack, hardware
    from away.app.frozen import is_frozen

    if args.action == "status":
        st = gpupack.status()
        hw = hardware.detect()
        g = hw.nvidia
        st["nvidia_gpu"] = (
            f"{g.name} ({g.vram_mb or '?'} MB, compute {g.cc_str}, driver {g.driver_version})" if g else None
        )
        st["cublas_loadable"] = gpupack.cublas_available()
        print(json.dumps(st, indent=2))
        return 0
    if args.action == "remove":
        removed = gpupack.remove()
        print("GPU pack removed." if removed else "GPU pack is not installed.")
        return 0
    # install
    if gpupack.platform_tag() is None:
        return _fail("the GPU pack needs Windows or Linux on x86-64 with an NVIDIA GPU (macOS has no CUDA)")
    if not is_frozen() and not args.force:
        print(gpupack.pip_instructions())
        print("(To download it into the Project Away data folder anyway: away gpu install --force)")
        return 0
    if hardware.detect().nvidia is None and not args.force:
        return _fail("no NVIDIA GPU detected (use --force to download the GPU pack anyway)")
    try:
        gpupack.install(connections=max(1, args.connections))
    except Exception as exc:
        return _fail(f"GPU pack download failed: {exc} (run it again to resume)")
    print("Done. Speech recognition uses the GPU from the next start (performance.mode: auto).")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """v0.1.24: the one-click PC self-test (``--json``/``--bundle``: the installer bundle check, unchanged)."""
    from away.selftest import cli as stcli

    return stcli.main(args)


def cmd_check(args: argparse.Namespace) -> int:
    """v0.1.24: Check my PC, the guided real-hardware report."""
    from away.selftest import checkpc

    return checkpc.main(args)


def cmd_setup(args: argparse.Namespace) -> int:
    from away.app import setup_wizard

    rep = setup_wizard.run(
        yes=args.yes,
        asr_model=args.asr_model,
        skip_asr=args.skip_asr,
        skip_probe=args.skip_probe,
        connections=args.connections,
        gpu=args.gpu,
    )
    return 1 if any(v.startswith("failed") for v in rep.models.values() if v) else 0


def cmd_run(args: argparse.Namespace) -> int:  # pragma: no cover - needs devices
    from away.app.runner import AwayApp

    cfg, unknown = cfgmod.load(Path(args.config) if args.config else None)
    for k in unknown:
        print(f"warning: unknown config key {k}", file=sys.stderr)
    if args.no_hands:
        cfg.hands.enabled = False
    if args.no_voice:
        cfg.voice.enabled = False
    if args.no_face:
        cfg.face.enabled = False
    if args.wifi:
        cfg.wifi.enabled = True
    if not args.dry_run:
        from away.app import instance

        if not instance.acquire():  # v0.1.23: one app at a time (the big cursor got stuck with two)
            print("Project Away is already running (tray icon). Quit it first to run it here.", file=sys.stderr)
            return 1
    app = AwayApp(cfg, dry_run=args.dry_run)
    if args.cursor_overlay or cfg.cursor_overlay:
        import threading

        from away.app.cursor_overlay import run_cursor_overlay

        app.start()
        threading.Thread(target=lambda: app.stop_event.wait(), daemon=True).start()
        run_cursor_overlay(app.pipeline.cursor_state, app.bus if (args.overlay or cfg.overlay) else None)
        app.stop()
        return 0
    if args.overlay or cfg.overlay:
        import threading

        from away.app.overlay import run_overlay

        app.start()
        threading.Thread(target=lambda: app.stop_event.wait(), daemon=True).start()
        run_overlay(app.bus)
        app.stop()
        return 0
    print("Project Away running. Ctrl+C to quit. Open palm (hold) = pause/resume.")
    app.run_forever()
    return 0


def cmd_tray(args: argparse.Namespace) -> int:  # pragma: no cover
    from away.app.tray import main as tray_main

    return tray_main([])


def cmd_demo(args: argparse.Namespace) -> int:
    """Drive the full pipeline with synthetic hands, faces and CSI (no devices needed)."""
    import numpy as np

    from away.app.runner import Pipeline
    from away.control.mouse import RecordingBackend
    from away.hands.synthetic import gesture_sequence
    from away.wifi.sources import synthetic_csi

    cfg, _ = cfgmod.load(Path(args.config) if args.config else None)
    cfg.wifi.enabled = True
    backend = RecordingBackend()
    pipe = Pipeline(cfg, backend)
    rng = np.random.default_rng(args.seed)
    t = 0.0
    fps = 30.0
    script = ["move", "pinch_click", "idle", "pinch_drag", "scroll", "swipe_right", "palm_clutch"]
    for label in script:
        seen: list[str] = []
        for lm in gesture_sequence(label, 24, rng):
            t += 1 / fps
            for k in pipe.on_hand(t, lm):
                if not seen or seen[-1] != k:
                    seen.append(k)
        pipe.on_hand(t + 1 / fps, None)
        t += 0.5
        print(f"{label:<12} -> {' > '.join(seen) or '(nothing)'}")
    for kind in ("empty", "push", "swipe"):
        _t, H = synthetic_csi(kind, seed=args.seed)
        label, conf = pipe.on_csi_window(t, H, None, 100.0)
        print(f"wifi window '{kind}' -> {label} ({conf:.2f})")
    counts: dict[str, int] = {}
    for name, _a in backend.calls:
        counts[name] = counts.get(name, 0) + 1
    print("mouse backend calls:", counts)
    return 0


def register_custom_gesture(label: str, cfg_path: Path | None = None) -> bool:
    """Add ``label`` to bindings.custom_gestures (unbound until you bind it). Returns True if it was added."""
    from away.hands.recorder import check_label
    from away.hands.synthetic import GESTURES

    check_label(label, allow_custom=True)
    if label in GESTURES:
        return False
    cfg, _ = cfgmod.load(cfg_path)
    if label in cfg.bindings.custom_gestures:
        return False
    cfg.bindings.custom_gestures.append(label)
    cfgmod.save(cfg, cfg_path)
    return True


def cmd_record(args: argparse.Namespace) -> int:  # pragma: no cover - needs a camera
    from away.hands.recorder import check_label, record

    try:
        check_label(args.label, args.custom)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    cfg, _ = cfgmod.load()
    out = Path(args.out) if args.out else paths.recordings_dir()
    record(
        args.label,
        args.count,
        args.seconds,
        out,
        _models_dir(cfg) / "hand_landmarker.task",
        cfg.hands.camera_index,
        cfg.hands.mirror,
        allow_custom=args.custom,
    )
    if args.custom and register_custom_gesture(args.label):
        print(
            f"registered custom gesture {args.label!r}. Next: `away calibrate --model`, then "
            f"`away bindings set gesture:{args.label} key:ctrl+t` (or use `away settings`)."
        )
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """``away calibrate``: the live tracking calibration window (v0.1.25); ``--reset`` forgets this camera's
    calibration; ``--model`` (or ``--recordings``) fine-tunes the temporal gesture model as before."""
    if args.model or args.recordings:
        return cmd_calibrate_model(args)
    if args.reset:
        from away.hands import livecal

        cfg, _ = cfgmod.load(_cfg_path(args))
        key = livecal.camera_key(cfg)
        livecal.save(_cfg_path(args), key, None)
        print(f"tracking calibration for camera {key!r} removed; the default settings are used")
        return 0
    from away.app.calibrate_gui import main as calibrate_main  # pragma: no cover - needs a display

    return calibrate_main(_cfg_path(args))  # pragma: no cover


def cmd_calibrate_model(args: argparse.Namespace) -> int:  # pragma: no cover - needs torch + recordings
    from away.hands.train import calibrate

    cfg, _ = cfgmod.load()
    rec = Path(args.recordings) if args.recordings else paths.recordings_dir()
    m = calibrate(_models_dir(cfg), rec, epochs=args.epochs)
    print(json.dumps(m, indent=2))
    if "recorded_holdout_accuracy" in m:
        print(
            "Recorded holdout metric available. Review the metrics and test on real gestures"
            " before enabling the model; it remains disabled by default."
        )
    elif "recorded_holdout_warning" in m:
        print(m["recorded_holdout_warning"])
    return 0


def cmd_train(args: argparse.Namespace) -> int:  # pragma: no cover - needs torch
    from away.hands import train

    out = Path(args.out)
    if args.what == "static":
        hdir = Path(args.hagrid_dir)
        if args.download:
            from away.hands.dataset import fetch_hagrid_subset

            fetch_hagrid_subset(hdir)
        m = train.train_static(hdir, out, max_per_class=args.max_per_class, epochs=args.epochs)
    else:
        m = train.train_temporal(
            out,
            recordings=Path(args.recordings) if args.recordings else None,
            n_synthetic=args.n_synthetic,
            epochs=args.epochs,
        )
    print(json.dumps({k: v for k, v in m.items() if k != "confusion"}, indent=2))
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    if args.action == "path":
        print(cfgmod.config_path())
    elif args.action == "init":
        p = cfgmod.save(cfgmod.AppConfig())
        print(f"wrote defaults to {p}")
    else:
        cfg, unknown = cfgmod.load()
        import yaml

        print(yaml.safe_dump(cfgmod.to_dict(cfg), sort_keys=False))
        for k in unknown:
            print(f"# unknown key ignored: {k}")
    return 0


# --------------------------------------------------------------------------- controls
def _cfg_path(args: argparse.Namespace) -> Path:
    return Path(args.config) if getattr(args, "config", None) else cfgmod.config_path()


def _load_for_edit(args: argparse.Namespace) -> cfgmod.AppConfig:
    cfg, unknown = cfgmod.load(_cfg_path(args))
    for k in unknown:
        print(f"warning: unknown config key {k}", file=sys.stderr)
    return cfg


def _fail(msg: object) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 2


def cmd_bindings(args: argparse.Namespace) -> int:
    import yaml

    from away.bindings import BindingsError, list_rows, load_bindings_file, merge_bindings

    path = _cfg_path(args)
    try:
        cfg = _load_for_edit(args)
    except ValueError as exc:
        return _fail(exc)
    b = cfg.bindings
    prof = getattr(args, "profile", None) or "default"
    if prof not in b.profiles:
        return _fail(f"no profile named {prof!r} (have: {', '.join(b.profiles)})")
    if args.action == "list":
        rows = list_rows(b, prof)
        if args.json:
            print(json.dumps([{"trigger": t, "action": a, "from": src} for t, a, src in rows], indent=2))
        else:
            w = max(len(t) for t, _a, _s in rows)
            print(f"# profile: {prof}   (active_profile: {b.active_profile}; allow_shell: {b.allow_shell})")
            for t, a, src in rows:
                mark = "" if src == prof else "   (from default)"
                print(f"{t:<{w}}  ->  {a}{mark}")
        return 0
    if args.action == "set":
        if not args.trigger or args.value is None:
            return _fail(
                "usage: away bindings set TRIGGER ACTION [--profile NAME], e.g. gesture:swipe_left key:alt+left"
            )
        try:
            act = b.set(args.trigger, args.value, prof)
        except ValueError as exc:
            return _fail(exc)
        cfgmod.save(cfg, path)
        print(f"[{prof}] {args.trigger} -> {act}")
        return 0
    if args.action == "reset":
        try:
            if args.trigger:
                b.unset(args.trigger, prof)
                msg = f"[{prof}] {args.trigger} reset"
            else:
                b.reset(None if args.all else prof)
                msg = "all bindings reset to defaults" if args.all else f"profile {prof!r} reset"
        except ValueError as exc:
            return _fail(exc)
        cfgmod.save(cfg, path)
        print(msg)
        return 0
    if args.action == "export":
        text = yaml.safe_dump({"bindings": b.to_dict()}, sort_keys=False, allow_unicode=True)
        if not args.trigger or args.trigger == "-":
            print(text, end="")
        else:
            Path(args.trigger).write_text(text, encoding="utf-8")
            print(f"exported bindings to {args.trigger}")
        return 0
    # import
    if not args.trigger:
        return _fail("usage: away bindings import FILE [--replace]")
    try:
        incoming, raw = load_bindings_file(args.trigger)
        cfg.bindings = incoming if args.replace else merge_bindings(b, incoming, raw)
    except BindingsError as exc:
        return _fail(exc)
    except (OSError, yaml.YAMLError) as exc:
        return _fail(f"cannot read {args.trigger}: {exc}")
    cfgmod.save(cfg, path)
    print(f"imported bindings from {args.trigger} ({'replaced' if args.replace else 'merged'})")
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    path = _cfg_path(args)
    try:
        cfg = _load_for_edit(args)
    except ValueError as exc:
        return _fail(exc)
    b = cfg.bindings
    if args.action == "list":
        for name, p in b.profiles.items():
            active = " *" if b.active_profile == name else ""
            apps = (
                "all apps" if name == "default" else (", ".join(p.apps) or "(no apps; select with `away profile use`)")
            )
            print(f"{name}{active}: {apps}")
        print(f"active_profile: {b.active_profile}")
        return 0
    if not args.name:
        return _fail(f"usage: away profile {args.action} NAME")
    try:
        if args.action == "use":
            if args.name != "auto" and args.name not in b.profiles:
                raise ValueError(f"no profile named {args.name!r} (have: {', '.join(b.profiles)}, or 'auto')")
            b.active_profile = args.name
            msg = f"active profile: {args.name}"
        elif args.action == "new":
            b.new_profile(args.name, args.apps or [], args.copy_from)
            msg = f"created profile {args.name!r} for {', '.join(args.apps) if args.apps else 'no apps yet'}"
        else:  # remove
            if args.name == "default" or args.name not in b.profiles:
                raise ValueError(f"cannot remove {args.name!r}")
            del b.profiles[args.name]
            if b.active_profile == args.name:
                b.active_profile = "auto"
            msg = f"removed profile {args.name!r}"
        cfgmod.save(cfg, path)
    except ValueError as exc:
        return _fail(exc)
    print(msg)
    return 0


VOICE_SHOW_KEYS = (
    "enabled",
    "mode",
    "hotkey",
    "hold_key",
    "mute_hotkey",
    "wake_word_enabled",
    "wake_word",
    "wake_window",
    "intent",
    "ambient_fallback",
    "vad_sensitivity",
    "end_silence_ms",
    "commands_enabled",
    "listen_seconds",
    "conversational",
    "agent_end_silence_ms",
    "listen_sound",
)


def cmd_voice(args: argparse.Namespace) -> int:
    path = _cfg_path(args)
    try:
        cfg = _load_for_edit(args)
    except ValueError as exc:
        return _fail(exc)
    v = cfg.voice
    if args.action == "show":
        for k in VOICE_SHOW_KEYS:
            print(f"voice.{k}: {getattr(v, k)}")
        print(f"app.always_on: {cfg.app.always_on}")
        return 0
    val = args.value
    attr = {
        "mode": "mode",
        "intent": "intent",
        "sensitivity": "vad_sensitivity",
        "listen-seconds": "listen_seconds",
        "conversational": "conversational",
        "agent-pause-ms": "agent_end_silence_ms",
        "listen-sound": "listen_sound",
    }.get(args.action)
    if val is None:
        if attr:
            print(getattr(v, attr))
        else:
            print(v.wake_word if v.wake_word_enabled else "off")
        return 0
    try:
        if args.action == "wake-word":
            if val.lower() in ("off", "none", "false"):
                v.wake_word_enabled = False
            else:
                v.wake_word_enabled = True
                if val.lower() not in ("on", "true"):
                    v.wake_word = val
        elif attr == "vad_sensitivity":
            v.vad_sensitivity = float(val)
        elif attr == "listen_seconds":
            v.listen_seconds = float(val)
        elif attr == "agent_end_silence_ms":
            v.agent_end_silence_ms = int(float(val))
        elif attr in ("conversational", "listen_sound"):
            if val.lower() not in ("on", "off", "true", "false", "1", "0", "yes", "no"):
                raise ValueError(f"voice {args.action} must be on or off")
            setattr(v, attr, val.lower() in ("on", "true", "1", "yes"))
        elif attr:
            setattr(v, attr, val)
        cfgmod.save(cfg, path)  # validates
    except ValueError as exc:
        return _fail(exc)
    shown = "off" if args.action == "wake-word" and not v.wake_word_enabled else val
    print(f"voice {args.action}: {shown}")
    return 0


def cmd_cameras(args: argparse.Namespace) -> int:
    from away.multicam import capture

    devs = capture.list_cameras(probe=args.probe, log_list=True)
    cfg, _ = cfgmod.load(Path(args.config) if args.config else None)
    if args.json:
        print(json.dumps([d.__dict__ for d in devs], indent=2))
        return 0
    if not devs:
        print("no cameras found (none connected, or no permission to /dev/video*)")
    elif capture.last_source:
        print(
            f"source: {capture.last_source}"
            + ("  (index = DirectShow index)" if capture.last_source == "dshow" else "")
        )
    for d in devs:
        state = "" if d.ok is None else ("  ok" if d.ok else "  (could not open)")
        if d.in_use:
            state = "  (in use by Project Away)"
        virt = " (virtual)" if d.virtual else ""
        print(f"{d.index if d.index is not None else '-':>3}  {d.path:<14} {d.name}{virt}{state}")
    if cfg.hands.cameras:
        print("\nconfigured (hands.cameras):")
        for c in cfg.hands.cameras:
            print(f"  {c.role:<9} device={c.device!r} {c.width}x{c.height}@{c.fps} hfov={c.hfov_deg}")
    else:
        print(f"\nsingle-camera mode: hands.camera_index = {cfg.hands.camera_index!r}")
        print("pick it by name with `away cameras use NAME`; add a second one with `away cameras add NAME`")
    return 0


def cmd_cameras_test(args: argparse.Namespace, lease_factory: Any = None, tester: Any = None, save: Any = None) -> int:
    """``away cameras test [--seconds 10] [--save]``: can the configured cameras stream at the same time?"""
    import functools

    from away.app.camera_share import borrowed_cameras
    from away.multicam import camtest, capture

    path = _cfg_path(args)
    cfg, _ = cfgmod.load(path)
    capture.configure(cfg.hands)
    cams = list(cfg.hands.cameras)
    if not cams:
        return _fail("no cameras in hands.cameras; add them with `away cameras add NAME` (two for multi-camera)")
    borrow = lease_factory or (lambda: borrowed_cameras(timeout=CALIBRATE_LEASE_S, who="camera-test"))
    from away.app.camera_share import confirm_handover

    with borrow() as lease:
        confirm_handover(lease, say=lambda m: print(m, flush=True))  # never open what the app still releases
        if lease.app_running and not lease.handed_over:
            return _fail("Project Away is using the cameras and did not hand them over in time - pause or quit it")
        if lease.handed_over:
            print("Project Away released the cameras for the test; they are handed back afterwards.")
        if tester is None:
            capture.use_mode_store(capture.default_mode_file())
            run: Any = functools.partial(
                camtest.run_camera_test,
                min_fps=float(cfg.hands.min_camera_fps),
                virtual_fallback=bool(cfg.hands.virtual_fallback),
            )
        else:
            run = tester
        seconds = 10.0 if getattr(args, "seconds", None) is None else float(args.seconds)
        report = run(cams, seconds, progress=lambda m: print(m, flush=True))
    for line in report.lines():
        print(line)
    offer = report.save_offer(cams)
    if offer is not None:
        if args.save:
            changed = camtest.save_combination(cfg, offer, cams)
            try:
                (save or cfgmod.save)(cfg, path)
            except ValueError as exc:
                return _fail(exc)
            print("saved: " + "; ".join(changed))
        else:
            print("run `away cameras test --save` to save the combination that works")
    return 0 if report.works else 1


def cmd_cameras_diagnose(
    args: argparse.Namespace, lease_factory: Any = None, runner: Any = None, lister: Any = None
) -> int:
    """``away cameras diagnose [DEVICE] [--seconds 1]`` (v0.1.19): every camera on every backend (DirectShow,
    Media Foundation, WinRT): its native formats where the backend lists them, and whether each mode opens and
    how fast it streams. The cameras are borrowed from the running app first."""
    from away.app.camera_share import borrowed_cameras, confirm_handover
    from away.multicam import capture, diagnose

    cfg, _ = cfgmod.load(_cfg_path(args))
    capture.configure(cfg.hands)
    spec = cfg.hands.cameras[0] if cfg.hands.cameras else None
    perf_w = int(getattr(getattr(cfg, "performance", None), "camera_width", 0) or 0)
    perf_h = int(getattr(getattr(cfg, "performance", None), "camera_height", 0) or 0)
    width, height = (spec.width, spec.height) if spec is not None else (perf_w or 640, perf_h or 480)
    fps = spec.fps if spec is not None else 30
    device = args.device
    if isinstance(device, str) and device.isdigit():
        device = int(device)
    borrow = lease_factory or (lambda: borrowed_cameras(timeout=CALIBRATE_LEASE_S, who="camera-diagnose"))
    with borrow() as lease:
        confirm_handover(lease, say=lambda m: print(m, flush=True))
        if lease.app_running and not lease.handed_over:
            return _fail("Project Away is using the cameras and did not hand them over in time - pause or quit it")
        if lease.handed_over:
            print("Project Away released the cameras for the diagnosis; they are handed back afterwards.")
        devs = (lister or diagnose.diagnose_devices)(device)
        if not devs:
            return _fail(f"camera {device!r} not found (see `away cameras list`)" if device else "no cameras found")
        seconds = diagnose.MEASURE_S if args.seconds is None else float(args.seconds)
        quick = bool(getattr(args, "quick", False))
        if quick:
            # v0.1.28 (CodeRabbit): the mode store is what saved_modes reads; configure() does not activate it
            capture.use_mode_store(capture.default_mode_file())
            print("quick check: opening each camera on its remembered backend only...", flush=True)
        else:
            print("diagnosing (each backend and mode is opened and measured; this takes a while)...", flush=True)
        budget = getattr(args, "budget", None)
        seen_fixes: list[str] = []

        def show(line: str) -> None:
            action, text = diagnose.split_fix_marker(str(line))  # v0.1.28: markers become plain words here
            if action and action not in seen_fixes:
                seen_fixes.append(action)
            print(text, flush=True)

        report = (runner or diagnose.run_diagnose)(
            devs,
            width=width,
            height=height,
            fps=fps,
            measure_s=seconds,
            out=show,
            budget_s=diagnose.BUDGET_S if budget is None else float(budget),
            include_virtual=bool(getattr(args, "include_virtual", False) or device not in (None, "")),
            consent=diagnose.read_consent,
            quick=quick,
        )
        if "open:camera_privacy" in seen_fixes:
            print("fix: open Settings > Privacy > Camera and allow camera access, then diagnose again.", flush=True)
        # (proposal #10): one machine-readable line carrying the structured per-camera results, so an
        # Agent Link check can lift it out of this subprocess's output into the job's ``result`` dict.
        structured = getattr(report, "result", None)
        if isinstance(structured, dict) and isinstance(structured.get("cameras"), list):
            print(diagnose.RESULT_MARKER, json.dumps(structured, separators=(",", ":")), flush=True)
    for line in pc_context_lines():  # v0.1.23: who else uses the camera / the CPU
        print(line)
    return 0


def pc_context_lines(users: Any = None, top: Any = None, source: Any = None) -> list[str]:
    """v0.1.23: apps using a camera now (Windows privacy records), NVIDIA Broadcast's source camera and the 5
    programs using the most CPU, for ``away cameras diagnose`` and diagnostics bundles."""
    from away.app.sysload import top_processes
    from away.multicam import camera_users as cu

    out = []
    us = cu.current_users() if users is None else users
    out.append("apps using a camera now: " + (", ".join(u.name for u in us) or "none"))
    src = cu.broadcast_source() if source is None else source
    if src:
        out.append(f"NVIDIA Broadcast takes its picture from: {src}")
    rows = top_processes(5, interval=0.5) if top is None else top
    if rows:
        out.append("busiest programs: " + ", ".join(f"{r['name']} {r['cpu_pct']:.0f}%" for r in rows))
    return out


def pick_camera(arg: str, devs: list[Any]) -> tuple[int | str, int | None]:
    """``away cameras add/use`` argument -> ``(what to save, index hint)``: a listed camera is saved by name
    (``"HD User Facing"``; the DevicePath or ``name #2`` when two share a name), with its index as a hint.
    A path, or anything not in the list, is saved as given."""
    from away.multicam.capture import config_device, find_device

    raw: int | str = int(arg) if str(arg).strip().isdigit() else str(arg)
    if isinstance(raw, str) and raw.startswith(("/dev/", "http://", "https://", "rtsp://")):
        return raw, None
    c = find_device(raw, devs) if devs else None
    if c is None:
        return raw, None
    value = config_device(c, devs)
    return value, (c.index if isinstance(value, str) else None)


def cmd_cameras_edit(args: argparse.Namespace) -> int:
    from away.config import CameraSpec
    from away.multicam.capture import list_cameras

    path = _cfg_path(args)
    try:
        cfg = _load_for_edit(args)
    except ValueError as exc:
        return _fail(exc)
    cams = cfg.hands.cameras
    h = cfg.hands
    if args.action in ("add", "use"):
        if args.device is None:
            return _fail(
                f"usage: away cameras {args.action} DEVICE" + (" [--role secondary]" if args.action == "add" else "")
            )
        try:
            devs = list_cameras()
        except Exception:
            devs = []
        dev, hint = pick_camera(args.device, devs)
        if args.action == "use":
            if cams:
                return _fail(
                    "several cameras are configured; use `away cameras remove` / `add` (or Settings > Cameras)"
                )
            h.camera_index, h.camera_hint = dev, hint
        else:
            if not cams:  # going multi-camera: the current single camera becomes the primary
                cams.append(
                    CameraSpec(
                        h.camera_index, "primary", index_hint=h.camera_hint, mirror=h.mirror, fourcc=h.camera_fourcc
                    )
                )
            cams.append(CameraSpec(dev, args.role or "secondary", index_hint=hint, mirror=h.mirror))
            if args.role == "primary":
                for c in cams[:-1]:
                    c.role = "secondary"
        print(f"camera: {dev!r}")
    elif args.action == "remove":
        keep = [c for c in cams if str(c.device) != str(args.device)]
        if len(keep) == len(cams):
            return _fail(f"no configured camera {args.device!r}")
        cfg.hands.cameras = keep if len(keep) > 1 else []  # one camera left: back to single-camera mode
        if len(keep) == 1:
            h.camera_index, h.camera_hint = keep[0].device, keep[0].index_hint
    try:
        cfgmod.save(cfg, path)
    except ValueError as exc:
        return _fail(exc)
    n = len(cfg.hands.cameras)
    print(f"{n} cameras configured" if n else "single-camera mode")
    return 0


def cmd_calibrate_cameras(args: argparse.Namespace) -> int:
    from away.multicam import geometry as G

    if args.print_board:
        import cv2

        cols, rows = (int(v) for v in args.checkerboard.lower().split("x"))
        cv2.imwrite(args.print_board, G.make_checkerboard_image(cols, rows))
        print(
            f"wrote a {cols}x{rows} inner-corner checkerboard to {args.print_board}; print it flat, measure one square"
        )
        return 0
    cfg, _ = cfgmod.load(Path(args.config) if args.config else None)
    from away.multicam import capture as _capture

    _capture.configure(cfg.hands)
    status = getattr(args, "status_file", None)
    if len(cfg.hands.cameras) < 2:
        msg = "calibration needs two cameras in hands.cameras (see `away cameras list` / `away cameras add`)"
        write_calibration_status(status, state="failed", message=msg)
        return _fail(msg)
    return calibrate_with_lease(cfg, args)


CALIBRATE_LEASE_S = 10.0  # the app closes two cameras one after another (each join up to 2 s)


def calibrate_with_lease(
    cfg: cfgmod.AppConfig, args: argparse.Namespace, lease_factory: Any = None, run: Any = None
) -> int:
    """Borrow the cameras from the running app (a CameraLease, ``who="calibrate"``), calibrate, give them
    back. When the app holds them and does not let go in time, say so instead of failing on a busy device."""
    from away.app.camera_share import borrowed_cameras

    status = getattr(args, "status_file", None)
    borrow = lease_factory or (lambda: borrowed_cameras(timeout=CALIBRATE_LEASE_S, who="calibrate"))
    calibrate = run or _run_camera_calibration
    write_calibration_status(status, state="running", progress=0, message="asking Project Away for the cameras...")
    with borrow() as lease:
        if lease.app_running and not lease.handed_over:
            msg = (
                "Project Away is using the cameras and did not hand them over in time - pause or quit it "
                "(tray menu) and calibrate again"
            )
            write_calibration_status(status, state="failed", message=msg)
            return _fail(msg)
        if lease.handed_over:
            print("Project Away released the cameras for calibration; they are handed back afterwards.")
        try:
            return int(calibrate(cfg, args))
        except Exception as exc:
            write_calibration_status(status, state="failed", message=f"calibration failed: {exc}")
            raise


def open_calibration_cameras(
    cams: list[Any],
    pairer: Any,
    stop: Any,
    *,
    virtual_fallback: bool = True,
    start: Any = None,
    retry_opener: Any = None,
    wait_s: float = 30.0,
    clock: Any = None,
    sleep: Any = None,
    say: Any = None,
    min_fps: float | None = None,
) -> list[Any]:
    """Start the calibration cameras with the shared sequential start-up (primary first, one after another,
    the same opener and fallbacks as the app). A secondary that still fails is retried once, after the
    primary is streaming, at 320x240 MJPG @ 15 (``low_res`` modes)."""
    import copy
    import time

    from away.multicam import capture

    if start is None:
        import functools

        start = functools.partial(capture.start_camera_threads, min_fps=min_fps)
    clock = clock or time.monotonic
    sleep = sleep or time.sleep
    say = say or (lambda m: print(m, flush=True))
    threads = start(list(cams), pairer, stop, virtual_fallback=virtual_fallback)
    first = capture.primary_index(list(cams))
    deadline = clock() + wait_s
    while clock() < deadline and not stop.is_set():
        if all(th.frames > 0 or not th.is_alive() for th in threads):
            break
        sleep(0.05)
    prim = threads[first]
    if prim.frames == 0:
        return threads  # the primary itself does not work: nothing to retry against
    for i, th in enumerate(threads):
        if i == first or th.frames > 0 or (th.is_alive() and not th.error):
            continue
        say(f"camera {cams[i].device!r} did not start ({th.error or 'no frames'}); retrying it at 320x240 MJPG...")
        spec = copy.copy(cams[i])
        spec.width, spec.height, spec.fps, spec.fourcc = 320, 240, 15, "MJPG"
        opener = retry_opener or capture.camera_opener(virtual_fallback, low_res=True, min_fps=min_fps)
        again = capture.CameraThread(i, spec, pairer, stop, opener=opener)
        again.start()
        threads[i] = again
    return threads


CALIBRATION_SPEED_S = 3.0  # streaming this long before calibration measures each camera's frame rate


def calibration_speed_check(
    cams: list[Any],
    threads: list[Any],
    min_fps: float,
    *,
    settle_s: float = CALIBRATION_SPEED_S,
    clock: Any = None,
    sleep: Any = None,
    stop: Any = None,
) -> str | None:
    """Calibration needs every camera at ``min_fps`` (v0.1.16): stream ``settle_s``, then return why it cannot
    run (a camera too slow, or not streaming), or None when every camera is fast enough. On the user's laptop a
    camera at 1 fps made hand-motion calibration fail with only "no secondary camera could be calibrated"."""
    import time

    clock = clock or time.monotonic
    sleep = sleep or time.sleep
    end = clock() + settle_s
    while clock() < end and not (stop is not None and stop.is_set()):
        sleep(0.05)
    problems = []
    for spec, th in zip(cams, threads):
        if getattr(th, "error", None):
            problems.append(f"camera {spec.device!r} does not work: {th.error}")
            continue
        if getattr(th, "frames", 0) == 0:
            problems.append(f"camera {spec.device!r} sent no frames")
            continue
        check = getattr(th, "too_slow", None)
        if min_fps > 0 and callable(check) and check(min_fps, window_s=min(2.0, settle_s * 0.6)):
            fps = th.fps() if callable(getattr(th, "fps", None)) else 0.0
            problems.append(
                f"camera {spec.device!r} is too slow for calibration: {fps:.1f} fps (minimum {min_fps:g}, "
                "hands.min_camera_fps)"
            )
    if not problems:
        return None
    return (
        "calibration cannot run: "
        + "; ".join(problems)
        + ". Run Settings > Cameras > Test both cameras (or `away cameras test`) to find a mode that works."
    )


def write_calibration_status(path: str | None, **data: Any) -> None:
    """Progress/result for the Settings window (``--status-file``); never raises."""
    if not path:
        return
    import contextlib
    import os
    import time

    with contextlib.suppress(OSError):
        tmp = f"{path}.tmp"
        Path(tmp).write_text(json.dumps({**data, "t": time.time()}), encoding="utf-8")
        os.replace(tmp, path)


def _run_camera_calibration(cfg: cfgmod.AppConfig, args: argparse.Namespace) -> int:  # pragma: no cover - cameras
    import threading
    import time

    import numpy as np

    from away.hands.tracker import HandTracker
    from away.multicam import capture
    from away.multicam import geometry as G
    from away.multicam.calibration import CheckerboardCalibrator, HandMotionCalibrator
    from away.multicam.capture import FramePairer
    from away.multicam.fusion import ViewObs

    status = getattr(args, "status_file", None)
    write_calibration_status(status, state="running", progress=0, message="opening the cameras...")
    cams = cfg.hands.cameras
    primary = next(i for i, c in enumerate(cams) if c.role == "primary")
    pairer = FramePairer(len(cams), primary, cfg.hands.multicam.pair_tolerance_ms / 1000, 0.5)
    stop = threading.Event()
    capture.use_mode_store(capture.default_mode_file())
    threads = open_calibration_cameras(
        cams,
        pairer,
        stop,
        virtual_fallback=bool(cfg.hands.virtual_fallback),
        min_fps=float(cfg.hands.min_camera_fps),
    )
    write_calibration_status(status, state="running", progress=0, message="cameras open; checking their speed...")
    slow = calibration_speed_check(cams, threads, float(cfg.hands.min_camera_fps), stop=stop)
    if slow is not None:
        capture.stop_camera_threads(stop, threads)
        write_calibration_status(status, state="failed", message=slow)
        return _fail(slow)
    write_calibration_status(status, state="running", progress=0, message="cameras open; starting...")
    board = args.checkerboard if args.method == "checkerboard" else None
    try:
        if board:
            cols, rows = (int(v) for v in board.lower().split("x"))
            cal_b = CheckerboardCalibrator(cams, cols, rows, args.square_mm / 1000.0)
            print(f"show the {cols}x{rows} board to both cameras, tilting it; collecting {args.views} views...")
            last = 0.0
            deadline = time.monotonic() + max(args.seconds, 60)
            while time.monotonic() < deadline and min(len(v) for v in cal_b.views.values()) < args.views:
                s = pairer.pop_set()
                if s is None or s[0] - last < 0.7:
                    time.sleep(0.01)
                    continue
                if cal_b.add([None if f is None else f[1] for f in s[1]]):
                    last = s[0]
                    n = min(len(v) for v in cal_b.views.values())
                    print(f"  views: {n}/{args.views}", flush=True)
                    write_calibration_status(status, state="running", message=f"checkerboard views {n}/{args.views}")
            ex, rep = cal_b.solve(min_views=min(args.views, 8))
        else:
            model = _models_dir(cfg) / "hand_landmarker.task"
            trackers = [HandTracker(model, 1) for _ in cams]
            cal_h = HandMotionCalibrator(cams)
            print(f"move your OPEN hand slowly around the space both cameras see, for {args.seconds:.0f} s...")
            end = time.monotonic() + args.seconds
            shown = -1
            while time.monotonic() < end:
                s = pairer.pop_set()
                if s is None:
                    time.sleep(0.003)
                    continue
                views: list[ViewObs | None] = []
                for i, item in enumerate(s[1]):
                    if item is None:
                        views.append(None)
                        continue
                    fr = trackers[i].process(item[1], item[0])
                    h, w = np.asarray(item[1]).shape[:2]
                    views.append(
                        None
                        if fr.landmarks is None
                        else ViewObs(i, item[0], fr.landmarks, fr.handedness, fr.score, (w, h))
                    )
                cal_h.add(views)
                pct = int(cal_h.progress() * 100)
                if pct // 10 != shown:
                    shown = pct // 10
                    left = int(end - time.monotonic())
                    print(f"  {pct}% ({left} s left)", flush=True)
                    write_calibration_status(
                        status, state="running", progress=pct, message=f"calibrating: {pct}% ({left} s left)"
                    )
            ex, rep = cal_h.solve(cfg.hands.multicam.palm_length_m)
    finally:
        capture.stop_camera_threads(stop, threads)  # devices released before the lease hands them back
    errors = [f"camera {th.spec.device!r}: {th.error}" for th in threads if th.error]
    for e in errors:
        print(e, file=sys.stderr)
    print(json.dumps(rep.per_camera, indent=2))
    if ex is None:
        msg = rep.message + (f" ({'; '.join(errors)})" if errors else "")
        write_calibration_status(status, state="failed", message=msg)
        return _fail(msg)
    p = G.save_extrinsics(ex)
    done = f"calibrated, reprojection error {ex.reprojection_error_px:.2f} px"
    write_calibration_status(status, state="done", message=done, error_px=ex.reprojection_error_px)
    print(f"{rep.message}\n{done}\nsaved extrinsics to {p}")
    return 0


def cmd_iphone(args: argparse.Namespace, webcam: Any = None) -> int:
    """Experimental: iPhone face tracking through Epic's Live Link Face app (see docs/iphone.md)."""
    from away.face import iphone_cli
    from away.face.livelink import LiveLinkReceiver

    cfg, _ = cfgmod.load(Path(args.config) if args.config else None)
    ip = cfg.face.iphone
    port = args.port or ip.port
    rx = LiveLinkReceiver(args.host or ip.host, port, ip.stale_after)
    if args.action == "listen":
        return iphone_cli.run_listen(rx, args.seconds)
    # compare
    seconds = args.seconds or 30.0
    out_path = Path(args.out) if args.out else iphone_cli.default_compare_path(paths.data_dir())
    if webcam is None:  # pragma: no cover - needs a camera + the [face] extra
        model = _models_dir(cfg) / "face_landmarker.task"
        if not model.exists():
            return _fail(f"webcam face model missing ({model}); run `away setup` first")
        webcam = iphone_cli.webcam_blendshapes(model, cfg.hands.camera_index, cfg.hands.mirror)
    if not rx.start():
        return _fail(str(rx.error))
    keys = iphone_cli.SHARED_KEYS if args.all else iphone_cli.COMPARE_KEYS
    print(f"Comparing webcam vs iPhone for {seconds:g} s: talk, blink, wink, raise your brows, open your mouth...")
    try:
        summary = iphone_cli.run_compare(rx, webcam, seconds, out_path, keys, ip.swap_sides)
    finally:
        rx.stop()
    return 0 if summary["iphone_face_rows"] else 1


def _gaze_monitors(cfg: cfgmod.AppConfig) -> tuple[Any, ...]:
    from away.control import monitors

    return monitors.select_monitors(monitors.enumerate_monitors(), cfg.hands.cursor.monitors)


def gaze_status_lines(cfg: cfgmod.AppConfig, models_dir: Path, mons: Any = None) -> list[str]:
    """``away gaze status``: calibrations per camera, the gaze backend in use and the optional models."""
    import time as _time

    from away.app import optional_models as om
    from away.face import gaze as gz
    from away.face.headgaze import calibration_camera

    cam = calibration_camera(cfg)
    lines = [f"camera: {cam}"]
    cals = gz.list_calibrations()
    if not cals:
        lines.append("screen map: not calibrated (run `away gaze calibrate`); gaze features stay off")
    for name, c in cals.items():
        when = _time.strftime("%Y-%m-%d %H:%M", _time.localtime(c.created))
        acc = c.accuracy_px()
        mark = " (current camera)" if name == cam else ""
        layout = ""
        if mons is not None and name == cam and not c.matches_layout(mons):
            layout = " - monitor layout changed since: recalibrate"
        lines.append(
            f"screen map for {name!r}{mark}: {when}, {c.n_dots} dots on {len(c.monitors)} monitor(s), "
            f"~{acc:.0f} px typical error (leave-one-dot-out)"
            + (f", with {c.net_model}" if c.net_model else "")
            + layout
            if acc is not None
            else f"screen map for {name!r}{mark}: {when}"
        )
    lines.append(f"gaze.model: {cfg.gaze.model} -> {om.resolve_gaze_model(cfg.gaze.model, models_dir)}")
    lines += [s.line() for s in om.status(models_dir, kind="gaze")]
    return lines


def cmd_gaze(args: argparse.Namespace) -> int:
    """``away gaze calibrate | test | clear | status | model install|status|remove``."""
    from away.app import optional_models as om
    from away.face import gaze as gz
    from away.face.headgaze import calibration_camera

    cfg, _ = cfgmod.load(_cfg_path(args))
    mdir = _models_dir(cfg)
    action = args.action
    if action == "model":
        sub = args.name or "status"
        if sub == "status":
            for line in gaze_status_lines(cfg, mdir)[-len(om.MODELS) :]:
                print(line)
            return 0
        if sub not in ("install", "remove"):
            return _fail("usage: away gaze model install NAME [--accept-license] | remove NAME | status")
        key = args.model
        if not key:
            return _fail(f"which model? {', '.join(m.key for m in om.MODELS)}")
        try:
            m = om.get(key)
        except KeyError as exc:
            return _fail(exc)
        if sub == "remove":
            print(f"removed {m.name}" if om.remove(key, mdir) else f"{m.name} was not installed")
            return 0
        accepted = bool(args.accept_license)
        if m.restricted and not accepted:
            from away.app.frozen import stdin_usable

            print(m.notice)
            if not stdin_usable():
                return _fail("pass --accept-license to download a research-only model")
            accepted = input("Accept these terms and download? [y/N] ").strip().lower() in ("y", "yes")
            if not accepted:
                return _fail("not downloaded (license not accepted)")
        print(f"downloading {m.name} ({m.size_mb:.1f} MB) ...")
        try:
            path = om.install(key, mdir, accept_license=accepted, write=_progress_writer())
        except Exception as exc:
            return _fail(f"download failed: {exc} (run it again to resume)")
        print(f"\ninstalled {path}; gaze.model auto uses it ({om.resolve_gaze_model(cfg.gaze.model, mdir)} now)")
        print("recalibrate (`away gaze calibrate`) so the screen map uses its features")
        return 0
    cam = str(args.camera) if args.camera else calibration_camera(cfg)
    if action == "status":
        mons = None
        with contextlib.suppress(Exception):  # monitor detection failed: still show what is saved
            mons = _gaze_monitors(cfg)
        for line in gaze_status_lines(cfg, mdir, mons):
            print(line)
        return 0
    if action == "clear":
        gone = gz.clear_calibration(None if args.all else cam)
        print(("cleared" if gone else "nothing to clear") + (" (all cameras)" if args.all else f" for {cam!r}"))
        return 0
    return _gaze_camera_command(args, cfg, mdir, cam)  # pragma: no cover - needs a camera and a display


def _progress_writer() -> Any:
    def write(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    return write


def _gaze_camera_command(
    args: argparse.Namespace, cfg: cfgmod.AppConfig, mdir: Path, cam: str
) -> int:  # pragma: no cover - needs a camera, mediapipe and a display
    from away.app import gaze_ui
    from away.app.camera_share import borrowed_cameras, confirm_handover
    from away.face import gaze as gz

    if not (mdir / "face_landmarker.task").exists():
        return _fail("the face model is missing: run `away setup` first")
    mons = _gaze_monitors(cfg)
    with borrowed_cameras(timeout=CALIBRATE_LEASE_S, who="gaze") as lease:
        confirm_handover(lease, say=lambda m: print(m, flush=True))
        if lease.app_running and not lease.handed_over:
            return _fail("Project Away is using the camera and did not hand it over in time - pause or quit it")
        if args.action == "calibrate":
            print(f"calibrating {len(mons)} monitor(s), {cfg.gaze.points_per_monitor} dots each, camera {cam!r}")
            try:
                cal = gaze_ui.calibrate(cfg, mdir, mons, cam)
            except ValueError as exc:
                return _fail(exc)
            if cal is None:
                return 1
            path = gz.save_calibration(cal)
            acc = cal.accuracy_px()
            print(f"saved the screen map to {path}")
            if acc is not None:
                print(
                    f"typical error ~{acc:.0f} px (leave-one-dot-out). Webcam gaze is good for 'which monitor and "
                    "roughly where' (about 2-4 degrees), not for small buttons."
                )
            _status_file_write(args, {"done": True, "rc": 0, "accuracy_px": acc})
            return 0
        cal = gz.load_calibration(cam)
        if cal is None:
            return _fail(f"no screen map for camera {cam!r}: run `away gaze calibrate` first")
        print("a red ring follows your gaze; Esc or the time limit ends the test")
        return gaze_ui.live_test(cfg, mdir, cal, float(args.seconds))


def _status_file_write(args: argparse.Namespace, data: dict[str, Any]) -> None:
    path = getattr(args, "status_file", None)
    if path:
        with contextlib.suppress(OSError):
            Path(path).write_text(json.dumps(data), encoding="utf-8")


def cmd_tour(args: argparse.Namespace) -> int:
    """``away tour``: the guided first run window (v0.1.20); ``--text`` without a window."""
    if args.text:
        from away.app.first_run import text_tour

        return text_tour(cfgmod.load(_cfg_path(args))[0])
    from away.app.first_run_gui import main as tour_main  # pragma: no cover - needs a display

    return tour_main(_cfg_path(args))  # pragma: no cover


def cmd_cleanup_engine(args: argparse.Namespace) -> int:
    """v0.1.20: download / check the Smart cleanup engine the installers do not bundle (away.voice.engine)."""
    from away.voice import engine

    if args.action == "status":
        print(engine.status_line())
        return 0
    if args.action == "check":  # run in its own process by install(): a broken native library cannot crash the app
        try:
            print(engine.self_check())
        except Exception as exc:
            print(f"engine check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.action == "remove":
        print("removed" if engine.remove() else "nothing to remove")
        return 0
    try:
        path = engine.install(progress=None)
    except engine.EngineInstallError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Smart cleanup engine ready: {path}")
    return 0


def cmd_settings(args: argparse.Namespace) -> int:
    from away.app import settings_gui

    return settings_gui.main(
        config_path=Path(args.config) if args.config else None,
        tab=getattr(args, "tab", None),
        teach_gesture=bool(getattr(args, "teach_gesture", False)),
    )


def cmd_autostart(args: argparse.Namespace) -> int:
    from away.app import autostart

    if args.state == "on":
        print(f"start at login enabled: {autostart.enable()}")
    else:
        print("start at login disabled" if autostart.disable() else "start at login was not enabled")
    return 0


def cmd_uninstall(args: argparse.Namespace) -> int:
    from away.app import uninstall

    uninstall.run(yes=args.yes, keep_config=args.keep_config, dry_run=args.dry_run)
    return 0


EXIT_UPDATE_AVAILABLE = 10


def _confirm(prompt: str) -> bool:
    """Ask yes/no on a real terminal; without one (windowed exe, piped stdin) the answer is no."""
    from away.app.frozen import stdin_usable

    if not stdin_usable():
        return False
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except (EOFError, RuntimeError, OSError):
        return False


def cmd_update(args: argparse.Namespace, fetch: Any = None, opener: Any = None, installer: Any = None) -> int:
    """`away update [--check] [--yes]`: exit 0 up to date / installed, 10 update available (not installed), 1 error."""
    from away.app import models, updater

    try:
        cfg, _ = cfgmod.load()
        ucfg = cfg.updates
    except Exception:  # a broken config must not block updating to a fixed version
        ucfg = cfgmod.UpdatesConfig()
    res = updater.check(channel=ucfg.channel, fetch=fetch)
    print(f"current: {res.current}")
    print(f"latest:  {res.latest or 'unknown'}")
    if res.error:
        print(f"error: {res.error}", file=sys.stderr)
        return 1
    if res.target.kind == "store":  # v0.1.25: a Microsoft Store copy is updated by the Store only
        print(res.summary())
        return 0
    if not res.update_available:
        print("Project Away is up to date.")
        return 0
    if args.check:
        print(res.summary())
        return EXIT_UPDATE_AVAILABLE
    if res.target.kind == "pip":
        print(f"Installed with pip/pipx; not modifying it. Upgrade with:\n  {updater.PIP_HINT}")
        return EXIT_UPDATE_AVAILABLE
    if not res.can_install or res.asset is None:
        print(res.summary())
        return EXIT_UPDATE_AVAILABLE
    question = f"Download and install {res.asset.name} ({res.asset.size / 1_000_000:,.0f} MB)? [y/N] "
    if res.target.kind == "deb":
        question = f"Download {res.asset.name} and install it with `pkexec apt install` (admin password)? [y/N] "
    if not args.yes and not _confirm(question):
        print("Not installed. Run `away update --yes` to install without asking.")
        return EXIT_UPDATE_AVAILABLE

    def bar(done: int, total: int) -> None:
        sys.stdout.write("\r" + models.progress_bar(done, total))
        sys.stdout.flush()

    try:
        path = updater.download(res.asset, progress=bar, opener=opener)
        print()
        source = updater.verify_download(path, res.asset, res.release, fetch, ucfg.allow_unverified)
        print(f"verified SHA-256 ({source})" if source != "none" else "WARNING: installing an unverified download")
        do_install = installer or updater.install
        # the CLI only relaunches on Windows (the installer closes a running tray app and restarts it)
        outcome = do_install(path, res.target, relaunch=res.target.kind == "windows")
    except updater.UpdateError as exc:
        print()
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(outcome.message)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="away",
        description="Project Away: the bridge between you and your AI agents, controlled by your hands, face and "
        "voice. Hands-free PC control is one part of it (local-first).",
    )
    p.add_argument("--version", action="version", version=f"project-away {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup", help="first-run wizard: download models, check camera/mic, write config")
    s.add_argument("--yes", "-y", action="store_true", help="accept defaults (non-interactive)")
    s.add_argument(
        "--asr-model",
        help="Whisper model (default: picked from your hardware, see `away hardware`; "
        "'small' ~485 MB for slow PCs or connections)",
    )
    s.add_argument("--skip-asr", action="store_true")
    s.add_argument("--skip-probe", action="store_true", help="do not open camera/mic")
    s.add_argument(
        "--connections",
        type=int,
        default=8,
        metavar="N",
        help="parallel connections for the Whisper download (default 8; 1 = single stream)",
    )
    s.add_argument(
        "--gpu",
        dest="gpu",
        action="store_true",
        default=None,
        help="with an NVIDIA GPU: download the GPU pack without asking",
    )
    s.add_argument("--no-gpu", dest="gpu", action="store_false", help="do not offer the GPU pack")
    s.set_defaults(fn=cmd_setup)

    r = sub.add_parser("run", help="run gestures + dictation in the foreground")
    r.add_argument("--config")
    r.add_argument("--dry-run", action="store_true", help="do not move the mouse or type (log only)")
    r.add_argument("--no-hands", action="store_true")
    r.add_argument("--no-voice", action="store_true")
    r.add_argument("--no-face", action="store_true")
    r.add_argument("--wifi", action="store_true", help="enable Wi-Fi sensing (see docs/wifi-setup.md)")
    r.add_argument("--overlay", action="store_true", help="show the on-screen status overlay")
    r.add_argument("--cursor-overlay", action="store_true", help="ring at the cursor: pinch progress, drag, dwell")
    r.set_defaults(fn=cmd_run)

    sub.add_parser("tray", help="run in the system tray").set_defaults(fn=cmd_tray)

    d = sub.add_parser("demo", help="synthetic end-to-end run (no camera/mic needed)")
    d.add_argument("--config")
    d.add_argument("--seed", type=int, default=1)
    d.set_defaults(fn=cmd_demo)

    rc = sub.add_parser("record", help="record your own labelled gesture clips")
    rc.add_argument("--label", required=True)
    rc.add_argument("--count", type=int, default=10)
    rc.add_argument("--seconds", type=float, default=1.2)
    rc.add_argument("--out")
    rc.add_argument("--custom", action="store_true", help="record a new custom gesture and register it for bindings")
    rc.set_defaults(fn=cmd_record)

    c = sub.add_parser(
        "calibrate",
        help="calibrate tracking for your camera and hand (a short guided window); --model: tune the gesture model",
    )
    c.add_argument("--reset", action="store_true", help="forget this camera's tracking calibration (defaults again)")
    c.add_argument("--model", action="store_true", help="fine-tune the temporal gesture model on your recordings")
    c.add_argument("--recordings", help="with --model: the recordings folder")
    c.add_argument("--config")
    c.add_argument("--epochs", type=int, default=15)
    c.set_defaults(fn=cmd_calibrate)

    t = sub.add_parser("train", help="train models (needs the [train] extra)")
    t.add_argument("what", choices=["static", "temporal"])
    t.add_argument("--out", default=str(paths.models_dir()))
    t.add_argument("--hagrid-dir", default=str(paths.data_dir() / "hagrid"))
    t.add_argument("--download", action="store_true", help="fetch the HaGRID landmark subset first (~20 MB)")
    t.add_argument("--max-per-class", type=int, default=3000)
    t.add_argument("--recordings")
    t.add_argument("--n-synthetic", type=int, default=200)
    t.add_argument("--epochs", type=int, default=30)
    t.set_defaults(fn=cmd_train)

    cf = sub.add_parser("config", help="show | path | init")
    cf.add_argument("action", choices=["show", "path", "init"], nargs="?", default="show")
    cf.set_defaults(fn=cmd_config)

    bd = sub.add_parser("bindings", help="list | set | reset | export | import trigger -> action bindings")
    bd.add_argument("action", choices=["list", "set", "reset", "export", "import"])
    bd.add_argument("trigger", nargs="?", help="trigger (gesture:pinch, 'voice:scroll down', ...) or FILE")
    bd.add_argument("value", nargs="?", help="action for `set` (key:ctrl+tab, mouse:right_click, none, ...)")
    bd.add_argument("--profile", help="profile to show/edit (default: default)")
    bd.add_argument("--json", action="store_true", help="list as JSON")
    bd.add_argument("--all", action="store_true", help="reset: every profile")
    bd.add_argument("--replace", action="store_true", help="import: replace instead of merging")
    bd.add_argument("--config", help="config file (default: per-OS path or $AWAY_CONFIG)")
    bd.set_defaults(fn=cmd_bindings)

    pf = sub.add_parser("profile", help="list | use NAME|auto | new NAME --apps ... | remove NAME")
    pf.add_argument("action", choices=["list", "use", "new", "remove"])
    pf.add_argument("name", nargs="?")
    pf.add_argument("--apps", nargs="*", help="process names this profile applies to (chrome firefox.exe Code)")
    pf.add_argument("--copy-from", help="start from a copy of this profile")
    pf.add_argument("--config")
    pf.set_defaults(fn=cmd_profile)

    vc = sub.add_parser(
        "voice",
        help="show | mode [always_on|push_to_talk|toggle] | wake-word [WORD|off] | listen-seconds [3-30] | "
        "conversational [on|off] | ...",
    )
    vc.add_argument(
        "action",
        choices=[
            "show",
            "mode",
            "wake-word",
            "intent",
            "sensitivity",
            "listen-seconds",
            "conversational",
            "agent-pause-ms",
            "listen-sound",
        ],
    )
    vc.add_argument("value", nargs="?")
    vc.add_argument("--config")
    vc.set_defaults(fn=cmd_voice)

    cm = sub.add_parser(
        "cameras",
        help="list | use NAME (single camera) | add NAME | remove DEVICE | test [--save] | diagnose [NAME]: "
        "choose, test and diagnose cameras",
    )
    cm.add_argument("action", choices=["list", "use", "add", "remove", "test", "diagnose"], nargs="?", default="list")
    cm.add_argument("device", nargs="?", help="camera name (or part of it), index from `away cameras list`, or path")
    cm.add_argument("--role", choices=["primary", "secondary"])
    cm.add_argument("--probe", action="store_true", help="open each camera to check it delivers frames")
    cm.add_argument("--json", action="store_true")
    cm.add_argument(
        "--seconds", type=float, default=None, help="test: how long to stream (default 10); diagnose: per mode (1)"
    )
    cm.add_argument("--save", action="store_true", help="test: save the camera settings that worked together")
    cm.add_argument(
        "--budget", type=float, default=None, help="diagnose: stop after this many seconds (default 60, 0 = no limit)"
    )
    cm.add_argument(
        "--quick",
        action="store_true",
        help="diagnose: quick triage - open each camera on its remembered backend only, skip the mode sweep",
    )
    cm.add_argument(
        "--include-virtual", action="store_true", help="diagnose: also check virtual cameras (NVIDIA Broadcast, OBS)"
    )
    cm.add_argument("--config")
    cm.set_defaults(
        fn=lambda a: (
            cmd_cameras(a)
            if a.action == "list"
            else (
                cmd_cameras_test(a)
                if a.action == "test"
                else (cmd_cameras_diagnose(a) if a.action == "diagnose" else cmd_cameras_edit(a))
            )
        )
    )

    cc = sub.add_parser("calibrate-cameras", help="calibrate two cameras (hand motion ~20 s, or a checkerboard)")
    cc.add_argument("--method", choices=["hand", "checkerboard"], default="hand")
    cc.add_argument("--seconds", type=float, default=20.0)
    cc.add_argument("--checkerboard", default="9x6", help="inner corners COLSxROWS (default 9x6)")
    cc.add_argument("--square-mm", type=float, default=25.0, help="checkerboard square size in mm")
    cc.add_argument("--views", type=int, default=15, help="checkerboard views to collect")
    cc.add_argument("--print-board", metavar="PNG", help="write a printable checkerboard image and exit")
    cc.add_argument("--config")
    cc.add_argument("--status-file", help=argparse.SUPPRESS)  # progress JSON for the Settings window
    cc.set_defaults(fn=cmd_calibrate_cameras)

    ip = sub.add_parser("iphone", help="listen | compare: experimental iPhone face tracking (Live Link Face app)")
    ip.add_argument("action", choices=["listen", "compare"], nargs="?", default="listen")
    ip.add_argument("--port", type=int, help="UDP port (default face.iphone.port, 11111)")
    ip.add_argument("--host", help="interface to listen on (default face.iphone.host, 0.0.0.0)")
    ip.add_argument("--seconds", type=float, help="listen: stop after N s (default: until Ctrl+C); compare: default 30")
    ip.add_argument("--out", help="compare: CSV path (default: <data dir>/iphone/compare-<time>.csv)")
    ip.add_argument("--all", action="store_true", help="compare: log all 51 shared blendshapes, not just the key ones")
    ip.add_argument("--config")
    ip.set_defaults(fn=cmd_iphone)

    gz = sub.add_parser(
        "gaze",
        help="calibrate | test | clear | status | model install|status|remove: eye tracking + the screen map",
    )
    gz.add_argument("action", choices=["calibrate", "test", "clear", "status", "model"])
    gz.add_argument("name", nargs="?", help="model: install | status | remove")
    gz.add_argument("model", nargs="?", help="model name for install/remove (l2cs, mobilegaze)")
    gz.add_argument("--camera", help="calibration for this camera name (default: the configured camera)")
    gz.add_argument("--all", action="store_true", help="clear: every camera's calibration")
    gz.add_argument("--seconds", type=float, default=30.0, help="test: how long to show the gaze dot")
    gz.add_argument("--accept-license", action="store_true", help="model install: accept research-only terms")
    gz.add_argument("--config", help=argparse.SUPPRESS)
    gz.add_argument("--status-file", help=argparse.SUPPRESS)
    gz.set_defaults(fn=cmd_gaze)

    tr = sub.add_parser("tour", help="guided first run: camera, sitting position, a 2-minute gesture tutorial")
    tr.add_argument("--text", action="store_true", help="text version (no window): camera check and tips")
    tr.set_defaults(fn=cmd_tour)

    ce = sub.add_parser("cleanup-engine", help="Smart cleanup's engine (a pinned llama-cpp-python wheel)")
    ce.add_argument("action", choices=["status", "install", "check", "remove"])
    ce.set_defaults(fn=cmd_cleanup_engine)

    st = sub.add_parser("settings", help="open the settings editor (needs a display + tkinter)")
    st.add_argument("--config")
    st.add_argument("--tab", help='open on this tab (part of its title, e.g. "studio", "shortcuts")')
    st.add_argument("--teach-gesture", action="store_true", help="open Gesture Studio's 'teach a gesture' guide")
    st.set_defaults(fn=cmd_settings)

    from away.control.macro_cli import add_parser as add_shortcut_parser

    add_shortcut_parser(sub)  # v0.1.20: away shortcut ... (alias: away macro)

    a = sub.add_parser("autostart", help="start the tray app at login")
    a.add_argument("state", choices=["on", "off"])
    a.set_defaults(fn=cmd_autostart)

    u = sub.add_parser("uninstall", help="delete models, recordings, config and autostart")
    u.add_argument("--yes", "-y", action="store_true")
    u.add_argument("--keep-config", action="store_true")
    u.add_argument("--dry-run", action="store_true")
    u.set_defaults(fn=cmd_uninstall)

    up = sub.add_parser("update", help="check GitHub for a new version and install it (exit 10 = update available)")
    up.add_argument("--check", action="store_true", help="only print current/latest (exit 0 up to date, 10 available)")
    up.add_argument("--yes", "-y", action="store_true", help="install without asking")
    up.set_defaults(fn=cmd_update)

    hwp = sub.add_parser("hardware", help="show detected CPU/RAM/GPU and the performance profile picked from them")
    hwp.add_argument("--json", action="store_true")
    hwp.set_defaults(fn=cmd_hardware)

    mp = sub.add_parser(
        "monitors", help="list the monitors the gesture cursor can reach (numbered 1..N, left to right)"
    )
    mp.add_argument("--json", action="store_true")
    mp.set_defaults(fn=cmd_monitors)

    gp = sub.add_parser("gpu", help="install | status | remove the NVIDIA GPU pack (cuBLAS for Whisper on CUDA)")
    gp.add_argument("action", choices=["install", "status", "remove"])
    gp.add_argument("--force", action="store_true", help="download even from a pip install / without a detected GPU")
    gp.add_argument("--connections", type=int, default=8, metavar="N")
    gp.set_defaults(fn=cmd_gpu)

    sub.add_parser("doctor", help="show installed extras, models and paths").set_defaults(fn=cmd_doctor)
    cu = sub.add_parser("cursor", help="reset: put the mouse pointer back to its normal size")
    cu.add_argument("action", choices=["reset"])
    cu.set_defaults(fn=cmd_cursor)

    # hidden: used by the release workflow to verify the frozen installer build
    stt = sub.add_parser("selftest", help="check cameras, tracking, pointer, microphone, voices, link and models")
    from away.selftest.cli import add_arguments as selftest_arguments

    selftest_arguments(stt)
    stt.set_defaults(fn=cmd_selftest)
    chk = sub.add_parser("check", help="Check my PC: a guided 2-minute test of cameras, voice, pointer and link")
    from away.selftest.checkpc import add_arguments as check_arguments

    check_arguments(chk)
    chk.set_defaults(fn=cmd_check)
    from away.agentlink.cli import add_parsers  # Agent Link (v0.1.20): agent, mcp, replay, diagnose

    add_parsers(sub)
    hidden = ("agent-ui", "link", "remote")
    sub.metavar = "{" + ",".join(c for c in sub.choices if c not in hidden) + "}"
    return p


def cmd_cursor(args: argparse.Namespace) -> int:
    """``away cursor reset`` (v0.1.23): the pointer back to your own size (32 px when unknown), marker cleared."""
    from away.app.bigcursor import BigCursor, default_backend

    size = BigCursor(default_backend(), 0).reset()
    if sys.platform.startswith("win"):
        print(f"Mouse pointer size reset to {size} px.")
    else:
        print("Nothing to reset: Project Away only changes the pointer size on Windows.")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows: per-monitor DPI awareness before pynput or tkinter exist, so monitor rectangles and cursor
    # moves share physical pixels (no-op elsewhere; AWAY_DPI_AWARE=0 turns it off)
    from away.control.monitors import enable_dpi_awareness

    enable_dpi_awareness()
    args = build_parser().parse_args(argv)
    # crash.log for every entry point (v0.1.16): faulthandler dumps all threads on a fatal error, uncaught
    # exceptions go to away.log and crash.log; idempotent (the frozen exe and the tray app call it too)
    from away.app import crashlog

    cmd = str(getattr(args, "command", "") or getattr(getattr(args, "fn", None), "__name__", "cli"))
    crashlog.install({"cmd_settings": "settings", "cmd_tray": "tray"}.get(cmd, f"cli {cmd}".strip()))
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
