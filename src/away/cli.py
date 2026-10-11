"""``away`` command line."""

from __future__ import annotations

import argparse
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
    from away.app import selftest

    return selftest.main(args.json)


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
            f"registered custom gesture {args.label!r}. Next: `away calibrate`, then "
            f"`away bindings set gesture:{args.label} key:ctrl+t` (or use `away settings`)."
        )
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:  # pragma: no cover - needs torch + recordings
    from away.hands.train import calibrate

    cfg, _ = cfgmod.load()
    rec = Path(args.recordings) if args.recordings else paths.recordings_dir()
    m = calibrate(_models_dir(cfg), rec, epochs=args.epochs)
    print(json.dumps(m, indent=2))
    if "recorded_holdout_accuracy" in m:
        cfg.hands.gestures.use_model = True
        cfgmod.save(cfg)
        print("enabled hands.gestures.use_model in config")
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
    attr = {"mode": "mode", "intent": "intent", "sensitivity": "vad_sensitivity"}.get(args.action)
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
                cams.append(CameraSpec(h.camera_index, "primary", index_hint=h.camera_hint, mirror=h.mirror))
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
    status = getattr(args, "status_file", None)
    if len(cfg.hands.cameras) < 2:
        msg = "calibration needs two cameras in hands.cameras (see `away cameras list` / `away cameras add`)"
        write_calibration_status(status, state="failed", message=msg)
        return _fail(msg)
    from away.app.camera_share import borrowed_cameras

    write_calibration_status(status, state="running", progress=0, message="asking Project Away for the cameras...")
    with borrowed_cameras(who="calibrate") as lease:  # pragma: no cover - needs cameras
        if lease.handed_over:
            print("Project Away released the cameras for calibration; they are handed back afterwards.")
        try:
            return _run_camera_calibration(cfg, args)
        except Exception as exc:
            write_calibration_status(status, state="failed", message=f"calibration failed: {exc}")
            raise


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
    from away.multicam import geometry as G
    from away.multicam.calibration import CheckerboardCalibrator, HandMotionCalibrator
    from away.multicam.capture import CameraThread, FramePairer
    from away.multicam.fusion import ViewObs

    status = getattr(args, "status_file", None)
    write_calibration_status(status, state="running", progress=0, message="opening the cameras...")
    cams = cfg.hands.cameras
    primary = next(i for i, c in enumerate(cams) if c.role == "primary")
    pairer = FramePairer(len(cams), primary, cfg.hands.multicam.pair_tolerance_ms / 1000, 0.5)
    stop = threading.Event()
    threads = [CameraThread(i, c, pairer, stop) for i, c in enumerate(cams)]
    for th in threads:
        th.start()
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
        stop.set()
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


def cmd_settings(args: argparse.Namespace) -> int:
    from away.app import settings_gui

    return settings_gui.main(config_path=Path(args.config) if args.config else None)


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
    p = argparse.ArgumentParser(prog="away", description="Project Away: hands-free PC control (local-first).")
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

    c = sub.add_parser("calibrate", help="fine-tune the temporal model on your recordings")
    c.add_argument("--recordings")
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

    vc = sub.add_parser("voice", help="show | mode [always_on|push_to_talk|toggle] | wake-word [WORD|off] | ...")
    vc.add_argument("action", choices=["show", "mode", "wake-word", "intent", "sensitivity"])
    vc.add_argument("value", nargs="?")
    vc.add_argument("--config")
    vc.set_defaults(fn=cmd_voice)

    cm = sub.add_parser(
        "cameras", help="list | use NAME (single camera) | add NAME | remove DEVICE: choose cameras by name"
    )
    cm.add_argument("action", choices=["list", "use", "add", "remove"], nargs="?", default="list")
    cm.add_argument("device", nargs="?", help="camera name (or part of it), index from `away cameras list`, or path")
    cm.add_argument("--role", choices=["primary", "secondary"])
    cm.add_argument("--probe", action="store_true", help="open each camera to check it delivers frames")
    cm.add_argument("--json", action="store_true")
    cm.add_argument("--config")
    cm.set_defaults(fn=lambda a: cmd_cameras(a) if a.action == "list" else cmd_cameras_edit(a))

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

    st = sub.add_parser("settings", help="open the settings editor (needs a display + tkinter)")
    st.add_argument("--config")
    st.set_defaults(fn=cmd_settings)

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

    # hidden: used by the release workflow to verify the frozen installer build
    stt = sub.add_parser("selftest")  # no help= => not listed in --help
    stt.add_argument("--json", help="also write the report to this file")
    stt.set_defaults(fn=cmd_selftest)
    sub.metavar = "{" + ",".join(c for c in sub.choices if c != "selftest") + "}"
    return p


def main(argv: list[str] | None = None) -> int:
    # Windows: per-monitor DPI awareness before pynput or tkinter exist, so monitor rectangles and cursor
    # moves share physical pixels (no-op elsewhere; AWAY_DPI_AWARE=0 turns it off)
    from away.control.monitors import enable_dpi_awareness

    enable_dpi_awareness()
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
