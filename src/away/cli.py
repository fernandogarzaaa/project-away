"""``away`` command line."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from away import __version__, paths
from away import config as cfgmod

EXTRAS = {
    "hands/face": ["mediapipe", "cv2"],
    "runtime models": ["onnxruntime"],
    "voice": ["faster_whisper", "sounddevice", "silero_vad"],
    "control": ["pynput", "pyperclip"],
    "tray": ["pystray", "PIL"],
    "train": ["torch", "onnx"],
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


def cmd_setup(args: argparse.Namespace) -> int:
    from away.app import setup_wizard

    rep = setup_wizard.run(yes=args.yes, asr_model=args.asr_model, skip_asr=args.skip_asr, skip_probe=args.skip_probe)
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


def cmd_record(args: argparse.Namespace) -> int:  # pragma: no cover - needs a camera
    from away.hands.recorder import record

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
        print(paths.config_file())
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="away", description="Project Away: hands-free PC control (local-first).")
    p.add_argument("--version", action="version", version=f"project-away {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup", help="first-run wizard: download models, check camera/mic, write config")
    s.add_argument("--yes", "-y", action="store_true", help="accept defaults (non-interactive)")
    s.add_argument("--asr-model", help="Whisper model (large-v3-turbo | small | ...)")
    s.add_argument("--skip-asr", action="store_true")
    s.add_argument("--skip-probe", action="store_true", help="do not open camera/mic")
    s.set_defaults(fn=cmd_setup)

    r = sub.add_parser("run", help="run gestures + dictation in the foreground")
    r.add_argument("--config")
    r.add_argument("--dry-run", action="store_true", help="do not move the mouse or type (log only)")
    r.add_argument("--no-hands", action="store_true")
    r.add_argument("--no-voice", action="store_true")
    r.add_argument("--no-face", action="store_true")
    r.add_argument("--wifi", action="store_true", help="enable Wi-Fi sensing (see docs/wifi-setup.md)")
    r.add_argument("--overlay", action="store_true", help="show the on-screen status overlay")
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

    a = sub.add_parser("autostart", help="start the tray app at login")
    a.add_argument("state", choices=["on", "off"])
    a.set_defaults(fn=cmd_autostart)

    u = sub.add_parser("uninstall", help="delete models, recordings, config and autostart")
    u.add_argument("--yes", "-y", action="store_true")
    u.add_argument("--keep-config", action="store_true")
    u.add_argument("--dry-run", action="store_true")
    u.set_defaults(fn=cmd_uninstall)

    sub.add_parser("doctor", help="show installed extras, models and paths").set_defaults(fn=cmd_doctor)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
