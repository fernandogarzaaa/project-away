"""v0.1.19 optional learned gaze model: decoding, crop/preprocess, the rate-limited runner with a fake ONNX
session, the optional-model registry (license opt-in, sha256 download, status, auto resolution), the
Settings Models panel logic and the `away gaze` CLI."""

from __future__ import annotations

import argparse
import hashlib
import io
import math
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest

from away import cli
from away import config as C
from away.app import fastdl
from away.app import optional_models as om
from away.app.models_panel import ModelsPanel
from away.face import gaze_net as gn
from away.face.headgaze import NetRunner


# ------------------------------------------------------------------------------------------ decoding
def onehot(deg: float) -> np.ndarray:
    logits = np.full((1, gn.BINS), -50.0)
    logits[0, round((deg + gn.ANGLE_OFFSET) / gn.BIN_WIDTH)] = 50.0
    return logits


def test_decode_is_the_softmax_expectation_over_4_degree_bins():
    assert gn.decode(onehot(0.0)) == pytest.approx(0.0, abs=1e-9)
    assert gn.decode(onehot(20.0)) == pytest.approx(math.radians(20.0))
    assert gn.decode(onehot(-40.0)) == pytest.approx(math.radians(-40.0))
    two = np.full((1, gn.BINS), -50.0)
    two[0, 45] = two[0, 46] = 10.0  # 0 and +4 degrees equally likely -> +2
    assert gn.decode(two) == pytest.approx(math.radians(2.0))
    assert gn.softmax(np.array([[1.0, 1.0]])) == pytest.approx(np.array([[0.5, 0.5]]))


def test_crop_box_pads_squares_and_clamps():
    assert gn.crop_box((0.5, 0.5, 0.2), 640, 480) == (240, 160, 400, 320)
    assert gn.crop_box((0.02, 0.5, 0.2), 640, 480)[0] == 0
    assert gn.crop_box((0.5, 0.5, 0.0), 640, 480) is None
    assert gn.crop_box((float("nan"), 0.5, 0.2), 640, 480) is None
    assert gn.crop_box((1.2, 0.5, 0.01), 640, 480) is None  # off the image


def test_preprocess_shapes_and_normalises():
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    x = gn.preprocess(rgb, (0.5, 0.5, 0.3), 64)
    assert x is not None and x.shape == (1, 3, 64, 64) and x.dtype == np.float32
    assert x[0, 0].mean() == pytest.approx((1.0 - 0.485) / 0.229, abs=1e-4)  # red channel at full scale
    assert x[0, 2].mean() == pytest.approx((0.0 - 0.406) / 0.225, abs=1e-4)
    assert gn.preprocess(rgb, (0.5, 0.5, 0.0)) is None


def test_providers_prefer_the_gpu():
    assert gn.providers(["CPUExecutionProvider"]) == ["CPUExecutionProvider"]
    assert gn.providers(["CUDAExecutionProvider", "CPUExecutionProvider"])[0] == "CUDAExecutionProvider"
    assert gn.gpu_available(["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert not gn.gpu_available(["CPUExecutionProvider"])


class FakeSession:
    """Mimics onnxruntime.InferenceSession for a MobileGaze export (outputs named yaw, pitch)."""

    def __init__(self, yaw: float = 12.0, pitch: float = -8.0, names: tuple[str, str] = ("yaw", "pitch")):
        self.yaw, self.pitch, self.names = yaw, pitch, names
        self.calls: list[tuple[int, ...]] = []

    def get_inputs(self) -> list[Any]:
        return [SimpleNamespace(name="input", shape=[1, 3, 448, 448])]

    def get_outputs(self) -> list[Any]:
        return [SimpleNamespace(name=n) for n in self.names]

    def run(self, _out: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls.append(tuple(feeds["input"].shape))
        vals = {"yaw": onehot(self.yaw), "pitch": onehot(self.pitch)}
        return [vals[n] if n in vals else onehot(0.0) for n in self.names]


def test_gaze_net_is_rate_limited_and_reads_outputs_by_name(tmp_path):
    sess = FakeSession(names=("pitch", "yaw"))  # reversed order: picked by name
    net = gn.GazeNet(tmp_path / "m.onnx", "mobilegaze", max_hz=10.0, session_factory=lambda _p: sess)
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    out = net.update(0.0, rgb, (0.5, 0.5, 0.3))
    assert out == pytest.approx((math.radians(12.0), math.radians(-8.0)))
    assert sess.calls == [(1, 3, 448, 448)]
    for k in range(1, 3):  # 30 fps frames within 0.1 s: reuse the last answer
        assert net.update(k / 30, rgb, (0.5, 0.5, 0.3)) == out
    assert len(sess.calls) == 1
    net.update(0.11, rgb, (0.5, 0.5, 0.3))
    assert len(sess.calls) == 2 and net.runs == 2
    assert net.update(0.5, rgb, None) == out  # no face: last answer, no run
    assert net.fresh(0.3) == out and net.fresh(2.0) is None


# ------------------------------------------------------------------------------------------ registry
def test_registry_entries_are_complete_and_restricted():
    keys = [m.key for m in om.MODELS]
    assert keys == ["l2cs", "mobilegaze"]
    for m in om.MODELS:
        assert m.url.startswith("https://github.com/yakhyo/gaze-estimation/releases/download/")
        assert len(m.sha256) == 64 and m.size_bytes > 1_000_000 and m.restricted
        assert "Gaze360" in m.notice and "research" in m.license
    with pytest.raises(KeyError):
        om.get("nope")
    assert set(C.GAZE_MODELS) == {"auto", "iris", *keys}


def fake_downloader(payload: bytes):
    def dl(url: str, dest: Path, rf: Any, meter: Any) -> Path:
        assert url.startswith("https://")
        dest.parent.mkdir(parents=True, exist_ok=True)
        for i in range(0, len(payload), 1000):
            meter.add(min(1000, len(payload) - i))
        dest.write_bytes(payload)
        return dest

    return dl


def test_install_requires_opt_in_reports_progress_and_removes(tmp_path, monkeypatch):
    m = om.get("mobilegaze")
    payload = b"x" * m.size_bytes
    with pytest.raises(om.LicenseNotAccepted, match="Gaze360"):
        om.install("mobilegaze", tmp_path, downloader=fake_downloader(payload))
    assert not om.installed("mobilegaze", tmp_path)
    seen: list[tuple[int, int]] = []
    p = om.install(
        "mobilegaze",
        tmp_path,
        accept_license=True,
        progress=lambda d, t: seen.append((d, t)),
        downloader=fake_downloader(payload),
    )
    assert p == tmp_path / m.filename and om.installed("mobilegaze", tmp_path)
    assert seen[-1] == (m.size_bytes, m.size_bytes) and len(seen) > 3
    assert (tmp_path / (m.filename + ".license.json")).exists()
    # second install: nothing downloaded
    om.install("mobilegaze", tmp_path, accept_license=True, downloader=lambda *a: pytest.fail("downloaded again"))
    st = {s.model.key: s for s in om.status(tmp_path)}
    assert st["mobilegaze"].installed and not st["l2cs"].installed
    assert "installed" in st["mobilegaze"].line()
    assert om.remove("mobilegaze", tmp_path) and not om.remove("mobilegaze", tmp_path)
    assert not (tmp_path / (m.filename + ".license.json")).exists()


def test_wrong_size_is_not_installed_and_partial_download_is_reported(tmp_path):
    m = om.get("l2cs")
    (tmp_path / m.filename).write_bytes(b"short")
    assert not om.installed("l2cs", tmp_path)
    (tmp_path / (m.filename + ".part")).write_bytes(b"y" * 2_000_000)
    s = next(s for s in om.status(tmp_path) if s.model.key == "l2cs")
    assert s.partial_bytes == 2_000_000 and "resumes" in s.line()


def test_real_fastdl_path_verifies_sha256(tmp_path, monkeypatch):
    """The registry downloads through fastdl.download_file: a body with the wrong checksum is rejected."""
    body = b"\x00" * 4096
    good = hashlib.sha256(body).hexdigest()
    m = om.OptionalModel(
        "tiny",
        "Tiny",
        "tiny.onnx",
        "https://example.invalid/tiny.onnx",
        good,
        len(body),
        "MIT",
        "",
        False,
        "gaze",
        "test",
        "-",
    )
    monkeypatch.setattr(om, "MODELS", (m,))

    class Resp(io.BytesIO):
        status = 200
        headers: ClassVar[dict[str, str]] = {}

        def getcode(self) -> int:
            return 200

        def geturl(self) -> str:
            return m.url

        def __enter__(self) -> Resp:
            return self

        def __exit__(self, *a: object) -> None:
            self.close()

    def opener(req: urllib.request.Request, timeout: float) -> Any:
        return Resp(served)

    served = body
    real = fastdl.download_file
    monkeypatch.setattr(fastdl, "download_file", lambda url, dest, rf, meter: real(url, dest, rf, meter, opener=opener))
    assert om.install("tiny", tmp_path).read_bytes() == body
    om.remove("tiny", tmp_path)
    served = b"\x01" * 4096
    with pytest.raises(fastdl.ChecksumError):
        om.install("tiny", tmp_path)
    assert not (tmp_path / "tiny.onnx").exists()


def test_resolve_gaze_model(tmp_path):
    def put(key: str) -> None:
        (tmp_path / om.get(key).filename).write_bytes(b"z" * om.get(key).size_bytes)

    assert om.resolve_gaze_model("auto", tmp_path, gpu=True) == "iris"
    assert om.resolve_gaze_model("l2cs", tmp_path) == "iris"  # explicit but missing: keep working on iris
    put("l2cs")
    assert om.resolve_gaze_model("auto", tmp_path, gpu=True) == "l2cs"
    assert om.resolve_gaze_model("auto", tmp_path, gpu=False) == "iris"  # l2cs on the CPU is too heavy for auto
    assert om.resolve_gaze_model("auto", tmp_path, gpu_check=lambda: True) == "l2cs"
    put("mobilegaze")
    assert om.resolve_gaze_model("auto", tmp_path, gpu=False) == "mobilegaze"
    assert om.resolve_gaze_model("l2cs", tmp_path, gpu=False) == "l2cs"  # explicit choice wins
    assert om.resolve_gaze_model("iris", tmp_path, gpu=True) == "iris"


# ------------------------------------------------------------------------------------------ NetRunner
def test_net_runner_runs_only_for_a_calibration_that_uses_the_resolved_model(tmp_path):
    cfg = C.AppConfig()
    made: list[str] = []
    state = {"resolved": "iris"}

    def factory(_d: Path, name: str, hz: float) -> Any:
        made.append(name)
        return gn.GazeNet(Path("x"), name, hz, session_factory=lambda _p: FakeSession())

    nr = NetRunner(cfg, tmp_path, factory=factory, resolve=lambda _c, _d: state["resolved"], check_s=5.0)
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    box = (0.5, 0.5, 0.3)
    cal = SimpleNamespace(net_model="mobilegaze")
    assert nr.update(0.0, rgb, box, None) is None  # no calibration
    assert nr.update(0.0, rgb, box, cal) is None and made == []  # model not downloaded yet
    state["resolved"] = "mobilegaze"  # downloaded from Settings while the app runs
    assert nr.update(1.0, rgb, box, cal) is None  # re-resolved only every check_s
    out = nr.update(5.5, rgb, box, cal)
    assert out is not None and made == ["mobilegaze"]
    nr.update(6.0, rgb, box, cal)
    assert made == ["mobilegaze"]  # the session is kept
    assert nr.update(7.0, rgb, box, SimpleNamespace(net_model=None)) is None
    cfg.gaze.enabled = False
    assert nr.update(8.0, rgb, box, cal) is None


def test_net_runner_survives_a_broken_model(tmp_path):
    cfg = C.AppConfig()

    def boom(*_a: Any) -> Any:
        raise RuntimeError("bad onnx")

    nr = NetRunner(cfg, tmp_path, factory=boom, resolve=lambda _c, _d: "l2cs")
    assert nr.update(0.0, None, (0.5, 0.5, 0.3), SimpleNamespace(net_model="l2cs")) is None
    assert nr.error is not None and "bad onnx" in nr.error


# ------------------------------------------------------------------------------------------ Settings panel
def test_models_panel_rows_consent_install_and_gaze_status(tmp_path, monkeypatch):
    cfg = C.AppConfig()
    panel = ModelsPanel(cfg, tmp_path / "config.yaml", tmp_path / "models")
    rows = panel.rows()
    assert [r["key"] for r in rows] == ["l2cs", "mobilegaze"]
    assert rows[1]["size"] == "5.0 MB" and rows[1]["status"] == "not installed" and rows[1]["restricted"]
    assert "Gaze360" in (panel.needs_consent("l2cs") or "")
    assert "license was not accepted" in (panel.install("mobilegaze", accepted=False) or "")
    monkeypatch.setattr(fastdl, "download_file", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("offline")))
    err = panel.install("mobilegaze", accepted=True)
    assert err is not None and "offline" in err and "resume" in err
    argv = panel.gaze_command("calibrate")
    assert argv[-4:] == ["gaze", "calibrate", "--config", str(tmp_path / "config.yaml")]
    assert "not calibrated" in panel.gaze_status() and "Backend: iris" in panel.gaze_status()
    assert panel.remove("l2cs") == "was not installed"
    assert panel.clear_gaze() == "nothing to clear"


# ------------------------------------------------------------------------------------------ CLI
def ns(**kw: Any) -> argparse.Namespace:
    base: dict[str, Any] = dict(
        action="status",
        name=None,
        model=None,
        camera=None,
        all=False,
        seconds=1.0,
        accept_license=False,
        config=None,
        status_file=None,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_cli_gaze_parser_and_status(capsys, monkeypatch):
    args = cli.build_parser().parse_args(["gaze", "model", "install", "mobilegaze", "--accept-license"])
    assert (args.action, args.name, args.model, args.accept_license) == ("model", "install", "mobilegaze", True)
    monkeypatch.setattr(cli, "_gaze_monitors", lambda cfg: ())
    assert cli.cmd_gaze(ns()) == 0
    out = capsys.readouterr().out
    assert "not calibrated" in out and "gaze.model: auto -> iris" in out and "[mobilegaze]: not installed" in out


def test_cli_gaze_clear_and_model_commands(capsys, monkeypatch, tmp_path):
    assert cli.cmd_gaze(ns(action="clear")) == 0 and "nothing to clear" in capsys.readouterr().out
    assert cli.cmd_gaze(ns(action="model", name="install")) == 2  # which model?
    assert cli.cmd_gaze(ns(action="model", name="install", model="bogus")) == 2
    monkeypatch.setattr("away.app.frozen.stdin_usable", lambda stream=None: False)
    assert cli.cmd_gaze(ns(action="model", name="install", model="l2cs")) == 2  # needs --accept-license
    assert "Gaze360" in capsys.readouterr().out
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        om, "install", lambda key, d, accept_license=False, **k: calls.append((key, accept_license)) or d / "x.onnx"
    )
    assert cli.cmd_gaze(ns(action="model", name="install", model="l2cs", accept_license=True)) == 0
    assert calls == [("l2cs", True)] and "recalibrate" in capsys.readouterr().out
    assert cli.cmd_gaze(ns(action="model", name="remove", model="l2cs")) == 0
    assert "was not installed" in capsys.readouterr().out
    assert cli.cmd_gaze(ns(action="model")) == 0 and "[l2cs]" in capsys.readouterr().out
    assert cli.cmd_gaze(ns(action="model", name="frobnicate")) == 2


def test_setup_mentions_the_optional_gaze_models_without_downloading(monkeypatch):
    from away.app import setup_wizard

    lines: list[str] = []
    got: list[str] = []
    setup_wizard.run(
        yes=True,
        skip_asr=True,
        skip_probe=True,
        out=lines.append,
        downloader=lambda mf, d, out: got.append(mf.name) or d / mf.name,
        gpu=False,
    )
    text = "\n".join(lines)
    assert "research-only" in text and "--accept-license" in text
    assert not any("gaze" in g for g in got)
