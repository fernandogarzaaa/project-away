"""iPhone face tracking via Live Link Face: parser, receiver, source selection, config, CLI (no real phone)."""

import argparse
import contextlib
import csv
import struct
import threading
import time

import pytest

from away import cli
from away import config as C
from away.app.runner import Pipeline
from away.face import iphone_cli, livelink
from away.face.livelink import (
    ARKIT_BLENDSHAPES,
    ARKIT_TO_MEDIAPIPE,
    POSE_VALUES,
    VALUE_COUNT,
    LiveLinkError,
    LiveLinkReceiver,
    encode_packet,
    parse_packet,
    to_mediapipe,
)
from away.face.source import IPHONE_BOX, FaceSourceSelector, IPhoneFaceFeed

# MediaPipe Face Landmarker's 52 categories (face_blendshapes output), for the name-mapping check
MEDIAPIPE_NAMES = {
    "_neutral",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
}


def handmade_packet(values, device=b"ABC-123", subject=b"iPhone"):
    """Byte-for-byte the layout described in Unreal's AppleARKitLiveLinkSource (independent of encode_packet)."""
    out = bytes([6]) + struct.pack(">I", len(device)) + device + struct.pack(">I", len(subject)) + subject
    out += struct.pack(">i", 4321) + struct.pack(">f", 0.47) + struct.pack(">I", 60) + struct.pack(">I", 1)
    if values is not None:
        out += bytes([len(values)]) + b"".join(struct.pack(">f", v) for v in values)
    return out


# -- parser ---------------------------------------------------------------------------------
def test_tables():
    assert len(ARKIT_BLENDSHAPES) == 52 and len(set(ARKIT_BLENDSHAPES)) == 52
    assert len(POSE_VALUES) == 9 and VALUE_COUNT == 61


def test_parse_handmade_packet():
    vals = [0.0] * 61
    vals[ARKIT_BLENDSHAPES.index("jawOpen")] = 0.75
    vals[ARKIT_BLENDSHAPES.index("eyeBlinkLeft")] = 0.9
    vals[ARKIT_BLENDSHAPES.index("tongueOut")] = 0.25
    vals[52] = -0.3  # headYaw
    vals[60] = 0.125  # rightEyeRoll
    fr = parse_packet(handmade_packet(vals))
    assert fr.version == 6 and fr.device_id == "ABC-123" and fr.subject == "iPhone"
    assert fr.frame == 4321 and fr.fps == 60 and fr.denominator == 1
    assert fr.sub_frame == pytest.approx(0.47)
    assert fr.has_face
    assert fr.blendshapes["jawOpen"] == pytest.approx(0.75)
    assert fr.blendshapes["eyeBlinkLeft"] == pytest.approx(0.9)
    assert fr.blendshapes["tongueOut"] == pytest.approx(0.25)
    assert fr.pose["headYaw"] == pytest.approx(-0.3)
    assert fr.pose["rightEyeRoll"] == pytest.approx(0.125)
    assert len(fr.blendshapes) == 52 and len(fr.pose) == 9


def test_encode_matches_handmade_layout():
    vals = [(i % 10) / 10 for i in range(61)]
    bs = dict(zip(ARKIT_BLENDSHAPES, vals[:52]))
    pose = dict(zip(POSE_VALUES, vals[52:]))
    enc = encode_packet(bs, pose, device_id="ABC-123", subject="iPhone", frame=4321, sub_frame=0.47)
    assert enc == handmade_packet(vals)
    fr = parse_packet(enc)
    for k, v in bs.items():
        assert fr.blendshapes[k] == pytest.approx(v)
    assert encode_packet(None, device_id="ABC-123", frame=4321, sub_frame=0.47) == handmade_packet(None)


def test_header_only_packet_is_no_face():
    fr = parse_packet(handmade_packet(None))
    assert not fr.has_face and fr.blendshapes is None and fr.pose == {}


def test_values_are_sanitized():
    vals = [1.7, -0.2, float("nan"), float("inf")] + [0.0] * 48 + [float("nan")] + [0.0] * 8
    fr = parse_packet(handmade_packet(vals))
    bs = fr.blendshapes
    assert bs["eyeBlinkLeft"] == 1.0 and bs["eyeLookDownLeft"] == 0.0
    assert bs["eyeLookInLeft"] == 0.0 and bs["eyeLookOutLeft"] == 0.0
    assert fr.pose["headYaw"] == 0.0


def test_fewer_values_accepted_extra_bytes_ignored():
    fr = parse_packet(handmade_packet([0.5] * 52))  # blendshapes only, no pose
    assert fr.blendshapes["jawOpen"] == 0.5 and fr.pose == {}
    fr2 = parse_packet(handmade_packet([0.1] * 61) + b"\x00\x01trailing")
    assert fr2.blendshapes["jawOpen"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    "data, msg",
    [
        (b"", "empty"),
        (bytes([5]) + b"\x00" * 40, "version 5"),
        (bytes([6]) + b"\x00\x00", "device id length"),
        (bytes([6]) + struct.pack(">I", 50) + b"short", "truncated device id"),
        (bytes([6]) + struct.pack(">I", 100000) + b"x" * 10, "implausible"),
        (bytes([6]) + struct.pack(">I", 1) + b"a" + struct.pack(">I", 1) + b"b" + b"\x00" * 5, "frame time"),
    ],
)
def test_malformed_packets_raise(data, msg):
    with pytest.raises(LiveLinkError, match=msg):
        parse_packet(data)


def test_truncated_blendshapes_and_bad_count():
    good = handmade_packet([0.2] * 61)
    for cut in (1, 4, 100, 243):
        with pytest.raises(LiveLinkError):
            parse_packet(good[:-cut])
    with pytest.raises(LiveLinkError, match="value count 10"):
        parse_packet(handmade_packet([0.0] * 10))
    with pytest.raises(LiveLinkError, match="value count 70"):
        parse_packet(handmade_packet([0.0] * 70))


def test_fuzz_never_raises_anything_but_livelinkerror():
    import random

    rnd = random.Random(1)
    good = handmade_packet([0.3] * 61)
    for _ in range(2000):
        b = bytearray(good[: rnd.randint(0, len(good))])
        for _ in range(rnd.randint(0, 4)):
            if b:
                b[rnd.randrange(len(b))] = rnd.randrange(256)
        with contextlib.suppress(LiveLinkError):
            parse_packet(bytes(b))


def test_invalid_utf8_names_do_not_crash():
    fr = parse_packet(handmade_packet([0.0] * 61, device=b"\xff\xfe", subject=b"ok"))
    assert fr.subject == "ok" and "\ufffd" in fr.device_id


# -- name mapping ---------------------------------------------------------------------------
def test_arkit_to_mediapipe_mapping_is_explicit():
    mapped = {v for v in ARKIT_TO_MEDIAPIPE.values() if v is not None}
    assert mapped == MEDIAPIPE_NAMES - {"_neutral"}
    assert ARKIT_TO_MEDIAPIPE["tongueOut"] is None
    assert set(ARKIT_BLENDSHAPES) - mapped == {"tongueOut"}
    assert MEDIAPIPE_NAMES - mapped == set(livelink.MEDIAPIPE_ONLY)


def test_to_mediapipe_drops_tongue_and_swaps_sides():
    bs = {"tongueOut": 0.9, "eyeBlinkLeft": 0.8, "eyeBlinkRight": 0.1, "jawOpen": 0.4, "browInnerUp": 0.2}
    mp = to_mediapipe(bs)
    assert "tongueOut" not in mp and mp["eyeBlinkLeft"] == 0.8 and mp["jawOpen"] == 0.4
    sw = to_mediapipe(bs, swap_sides=True)
    assert sw["eyeBlinkRight"] == 0.8 and sw["eyeBlinkLeft"] == 0.1
    assert sw["jawOpen"] == 0.4 and sw["browInnerUp"] == 0.2


# -- receiver -------------------------------------------------------------------------------
class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


def face_packet(**bs):
    return encode_packet(bs)


def test_receiver_latest_rate_and_staleness():
    clk = Clock()
    rx = LiveLinkReceiver(stale_after=0.5, clock=clk)
    assert rx.latest() is None and not rx.is_fresh() and rx.age() is None and rx.rate() == 0.0
    for i in range(30):
        rx.handle(face_packet(jawOpen=0.6), ("192.168.1.50", 5000), t=100.0 + i / 60)
    clk.t = 100.5
    assert rx.packets == 30 and rx.seq == 30 and rx.last_sender == "192.168.1.50"
    assert rx.rate() == pytest.approx(30.0)
    t, bs = rx.latest_face()
    assert bs["jawOpen"] == pytest.approx(0.6) and t == pytest.approx(100.0 + 29 / 60)
    assert rx.is_fresh()
    clk.t = 101.2  # 0.7 s after the last packet
    assert not rx.is_fresh() and rx.latest_face() is None
    assert rx.age() == pytest.approx(101.2 - (100 + 29 / 60))
    clk.t = 101.6
    assert rx.rate() == 0.0


def test_receiver_no_face_packet_is_not_fresh():
    rx = LiveLinkReceiver(clock=Clock())
    rx.handle(face_packet(jawOpen=0.3), t=100.0)
    assert rx.is_fresh(100.0)
    rx.handle(encode_packet(None), t=100.01)
    assert not rx.is_fresh(100.02) and rx.no_face_packets == 1 and rx.packets == 2


def test_receiver_counts_bad_packets_and_keeps_last_good():
    rx = LiveLinkReceiver(clock=Clock())
    rx.handle(face_packet(jawOpen=0.3), t=100.0)
    for junk in (b"", b"\x07hello", face_packet(jawOpen=0.9)[:-3], b"GET / HTTP/1.1\r\n"):
        assert rx.handle(junk, ("1.2.3.4", 1), t=100.1) is None
    assert rx.bad_packets == 4 and rx.packets == 1 and rx.last_error
    assert rx.latest_face(100.2)[1]["jawOpen"] == pytest.approx(0.3)


class FakeSocket:
    """Stands in for a UDP socket: hands out queued datagrams, then times out (or errors)."""

    def __init__(self, packets=(), forever=None, errors=()):
        self.packets = list(packets)
        self.forever = forever
        self.errors = list(errors)
        self.bound = None
        self.closed = False
        self.timeout = None

    def setsockopt(self, *a):
        pass

    def bind(self, addr):
        self.bound = addr

    def settimeout(self, t):
        self.timeout = t

    def recvfrom(self, n):
        if self.closed:
            raise OSError("closed")
        if self.errors:
            raise self.errors.pop(0)
        if self.packets:
            return self.packets.pop(0), ("192.168.1.77", 50000)
        if self.forever is not None:
            time.sleep(0.002)
            return self.forever, ("192.168.1.77", 50000)
        time.sleep(0.01)
        raise TimeoutError

    def close(self):
        self.closed = True


def test_receiver_thread_with_fake_socket(monkeypatch):
    errors = [ConnectionResetError(10054, "reset"), OSError("transient"), TimeoutError()]
    sock = FakeSocket([b"garbage", face_packet(jawOpen=0.5), face_packet(eyeBlinkLeft=1.0)], errors=errors)
    monkeypatch.setattr(livelink, "_new_socket", lambda: sock)
    rx = LiveLinkReceiver("127.0.0.1", 12345, stale_after=5.0)
    with rx:
        deadline = time.monotonic() + 3
        while rx.packets < 2 and time.monotonic() < deadline:
            rx.wait(rx.seq, 0.05)
    assert sock.bound == ("127.0.0.1", 12345) and sock.closed
    assert rx.packets == 2 and rx.bad_packets == 1 and rx.error is None
    assert rx.latest()[1].blendshapes["eyeBlinkLeft"] == 1.0


def test_receiver_bind_failure_is_reported_not_raised(monkeypatch):
    class Busy(FakeSocket):
        def bind(self, addr):
            raise OSError(98, "Address already in use")

    monkeypatch.setattr(livelink, "_new_socket", lambda: Busy())
    rx = LiveLinkReceiver(port=11111)
    assert rx.start() is False
    assert "11111" in rx.error and "in use" in rx.error
    rx.stop()


def test_receiver_real_udp_loopback():
    import socket

    rx = LiveLinkReceiver("127.0.0.1", 0)
    rx.open()  # port 0: the OS picks a free one
    port = rx._sock.getsockname()[1]
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.sendto(face_packet(jawOpen=0.42), ("127.0.0.1", port))
        for _ in range(20):
            rx.poll_once()
            if rx.packets:
                break
    finally:
        s.close()
        rx.stop()
    assert rx.packets == 1 and rx.latest()[1].blendshapes["jawOpen"] == pytest.approx(0.42)


def test_lan_ips_filters_loopback():
    ips = livelink.lan_ips()
    assert all(not ip.startswith(("127.", "169.254.")) for ip in ips)
    assert len(ips) == len(set(ips))


# -- source selection + pipeline --------------------------------------------------------------
def test_selector_modes():
    rx = LiveLinkReceiver(stale_after=0.5, clock=Clock())
    assert FaceSourceSelector("webcam", None).active(0) == "webcam"
    with pytest.raises(ValueError):
        FaceSourceSelector("iphone", None)
    with pytest.raises(ValueError):
        FaceSourceSelector("kinect", rx)
    ip = FaceSourceSelector("iphone", rx)
    assert ip.active(100.0) == "iphone" and not ip.needs_webcam_model and ip.needs_receiver
    auto = FaceSourceSelector("auto", rx)
    assert auto.needs_webcam_model and auto.needs_receiver
    assert auto.update(100.0) == ("webcam", False)
    rx.handle(face_packet(jawOpen=0.1), t=100.0)
    assert auto.update(100.1) == ("iphone", True)
    assert not auto.use_webcam(100.2)
    assert auto.update(100.4) == ("iphone", False)
    assert auto.update(101.0) == ("webcam", True)  # stale -> webcam fallback
    assert auto.use_webcam(101.0)


class RecordingPipeline:
    def __init__(self):
        self.calls = []
        self.enrolls = 0

    def on_faces(self, t, boxes, shapes):
        self.calls.append((t, boxes, shapes))
        return None

    def re_enroll(self):
        self.enrolls += 1


def test_feed_rate_limit_no_face_and_switch():
    rx = LiveLinkReceiver(stale_after=0.5, clock=Clock())
    pipe = RecordingPipeline()
    switched = []
    feed = IPhoneFaceFeed(
        rx, pipe, FaceSourceSelector("auto", rx), max_rate=30, lock=threading.Lock(), on_switch=switched.append
    )
    assert feed.step(100.0) is False  # nothing yet: webcam is active, nothing fed
    rx.handle(face_packet(jawOpen=0.7, tongueOut=1.0), t=100.0)
    assert feed.step(100.001) is True
    _t, boxes, shapes = pipe.calls[-1]
    assert boxes == [IPHONE_BOX] and shapes[0]["jawOpen"] == pytest.approx(0.7) and "tongueOut" not in shapes[0]
    assert pipe.enrolls == 1 and switched == ["iphone"]
    assert feed.step(100.01) is False  # < 1/30 s later: rate-limited
    rx.handle(encode_packet(None), t=100.05)  # phone lost the face
    assert feed.step(100.06) is False  # no fresh face -> auto falls back to the webcam
    assert switched == ["iphone", "webcam"] and pipe.enrolls == 2
    iphone_only = IPhoneFaceFeed(rx, pipe, FaceSourceSelector("iphone", rx))
    assert iphone_only.step(100.1) is True and pipe.calls[-1][1:] == ([], [])  # "no face" is fed
    assert iphone_only.fed_empty == 1


def _arkit_stream(fn, t0, n, fps=60):
    for i in range(n):
        yield t0 + i / fps, encode_packet(fn(i))


def test_iphone_drives_wink_mouth_brow_triggers():
    cfg = C.AppConfig()
    cfg.face.source = "iphone"
    cfg.face.wink_trigger = cfg.face.mouth_open_trigger = cfg.face.brow_raise_toggle = True
    pipe = Pipeline(cfg)
    rx = LiveLinkReceiver(stale_after=0.5, clock=Clock())
    feed = IPhoneFaceFeed(rx, pipe, FaceSourceSelector("iphone", rx), max_rate=30)

    def run(fn, t0, seconds):
        for t, pkt in _arkit_stream(fn, t0, int(seconds * 60)):
            rx.handle(pkt, t=t)
            feed.step(t)

    run(lambda i: {}, 10.0, 0.5)  # neutral face: enroll
    run(lambda i: {"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.05}, 10.5, 0.6)  # left wink (person's left)
    run(lambda i: {}, 11.1, 0.3)
    run(lambda i: {"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.9}, 11.4, 0.6)  # a blink: both eyes, no wink
    run(lambda i: {}, 12.0, 0.3)
    run(lambda i: {"jawOpen": 0.8}, 12.3, 1.2)
    run(lambda i: {}, 13.5, 0.3)
    run(lambda i: {"browInnerUp": 0.8}, 13.8, 1.0)
    topics = [e.topic for e in pipe.bus.history() if e.topic.startswith("face.")]
    assert "face.enrolled" in topics
    assert topics.count("face.wink_left") == 1 and "face.wink_right" not in topics
    assert topics.count("face.mouth_open") == 1 and topics.count("face.brow_toggle") == 1


def test_iphone_lip_motion_gates_dictation():
    import math

    cfg = C.AppConfig()
    pipe = Pipeline(cfg)
    rx = LiveLinkReceiver(stale_after=0.5, clock=Clock())
    feed = IPhoneFaceFeed(rx, pipe, FaceSourceSelector("iphone", rx), max_rate=30)
    t = 20.0
    for i in range(120):  # 2 s of talking: jaw oscillates, mic is loud
        t = 20.0 + i / 60
        rx.handle(encode_packet({"jawOpen": 0.35 + 0.3 * math.sin(i / 3)}), t=t)
        pipe.mic_energy = 0.5
        feed.step(t)
    assert pipe.gate.primary is not None
    assert pipe.gate.allow(20.5, t) and pipe.gate.allow_ambient(20.5, t)
    t2 = t
    for i in range(120):  # phone loses the face (stream stale): nobody at the screen
        t2 = t + 0.6 + i / 60
        feed.step(t2)
    assert not pipe.gate.allow_ambient(t + 0.7, t2)


# -- config ---------------------------------------------------------------------------------
def test_config_defaults_and_roundtrip(tmp_path):
    cfg = C.AppConfig()
    assert cfg.face.source == "webcam" and cfg.face.iphone.port == 11111
    cfg2, unknown = C.from_dict({"face": {"source": "auto", "iphone": {"port": 12000, "swap_sides": True}}})
    assert unknown == [] and cfg2.face.source == "auto" and cfg2.face.iphone.port == 12000
    assert cfg2.face.iphone.swap_sides is True and cfg2.face.iphone.stale_after == 0.5
    p = C.save(cfg2, tmp_path / "c.yaml")
    cfg3, unknown3 = C.load(p)
    assert unknown3 == [] and cfg3.face.source == "auto" and cfg3.face.iphone.port == 12000
    _, unk = C.from_dict({"face": {"iphone": {"prot": 1}}})
    assert unk == ["face.iphone.prot"]


@pytest.mark.parametrize(
    "face",
    [
        {"source": "kinect"},
        {"iphone": {"port": 0}},
        {"iphone": {"port": 70000}},
        {"iphone": {"port": "11111"}},
        {"iphone": {"port": True}},
        {"iphone": {"host": ""}},
        {"iphone": {"stale_after": 0}},
        {"iphone": {"max_rate": 1000}},
        {"iphone": {"swap_sides": "yes"}},
    ],
)
def test_config_rejects_bad_iphone_values(face):
    with pytest.raises(ValueError, match="face"):
        C.from_dict({"face": face})


def test_app_builds_receiver_only_when_asked():
    from away.app.runner import AwayApp

    app = AwayApp(C.AppConfig(), dry_run=True)
    assert app.iphone is None and app.face_source.mode == "webcam"
    cfg = C.AppConfig()
    cfg.face.source = "auto"
    app2 = AwayApp(cfg, dry_run=True)
    assert app2.iphone is not None and app2.iphone.port == 11111 and app2.face_source.mode == "auto"
    cfg.face.enabled = False
    assert AwayApp(cfg, dry_run=True).iphone is None


# -- CLI ------------------------------------------------------------------------------------
def test_cli_listen_with_mocked_socket(monkeypatch, capsys):
    sock = FakeSocket(forever=face_packet(jawOpen=0.55, eyeBlinkLeft=0.1, mouthSmileLeft=0.4, mouthSmileRight=0.2))
    monkeypatch.setattr(livelink, "_new_socket", lambda: sock)
    monkeypatch.setattr(iphone_cli, "lan_ips", lambda: ["192.168.1.10"])
    rc = cli.main(["iphone", "listen", "--seconds", "0.6"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "192.168.1.10" in out and "11111" in out
    assert "Live Link (ARKit)" in out and "same Wi-Fi" in out and "Firewall" in out
    assert "pkt/s" in out and "jawOpen 0.55" in out and "mouthSmile 0.30" in out
    assert sock.bound == ("0.0.0.0", 11111) and sock.closed


def test_cli_listen_nothing_received_and_port_override(monkeypatch, capsys):
    sock = FakeSocket([b"\x05old-protocol"])
    monkeypatch.setattr(livelink, "_new_socket", lambda: sock)
    monkeypatch.setattr(iphone_cli, "lan_ips", lambda: [])
    rc = cli.main(["iphone", "listen", "--seconds", "0.6", "--port", "12001"])
    out = capsys.readouterr().out
    assert rc == 1 and sock.bound == ("0.0.0.0", 12001)
    assert "version 5" in out and "Nothing received" in out


def test_status_line_states():
    clk = Clock()
    rx = LiveLinkReceiver(stale_after=0.5, clock=clk)
    assert "waiting" in iphone_cli.status_line(rx, 100.0)
    rx.handle(encode_packet(None), ("10.0.0.2", 1), t=100.0)
    assert "no face" in iphone_cli.status_line(rx, 100.1)
    assert "STALE" in iphone_cli.status_line(rx, 102.0)


def test_run_compare_csv_and_summary(tmp_path):
    rx = LiveLinkReceiver(stale_after=0.5, clock=Clock())

    def webcam():
        for i in range(90):
            t = 50.0 + i / 30
            j = 0.5 + 0.4 * ((i % 10) / 10)
            rx.handle(encode_packet({"jawOpen": j, "eyeBlinkLeft": 0.2}), t=t - 0.01)
            yield t, ({"jawOpen": j * 0.9, "eyeBlinkLeft": 0.25} if i % 3 else None)

    out_path = tmp_path / "cmp" / "x.csv"
    lines = []
    s = iphone_cli.run_compare(rx, webcam(), 2.0, out_path, ("jawOpen", "eyeBlinkLeft"), out=lines.append)
    rows = list(csv.DictReader(out_path.open()))
    assert s["rows"] == len(rows) == 61  # t in [0, 2.0]
    assert s["iphone_face_rows"] == 61 and s["webcam_face_rows"] == 40
    assert rows[0]["webcam_face"] == "0" and rows[0]["webcam_jawOpen"] == "" and rows[0]["iphone_jawOpen"] != ""
    assert rows[1]["iphone_age_ms"] == "10"
    k = s["keys"]["jawOpen"]
    assert k["pairs"] == 40 and k["corr"] == pytest.approx(1.0) and k["mean_abs_diff"] > 0
    assert s["keys"]["eyeBlinkLeft"]["corr"] is None  # constant: no correlation
    assert any("jawOpen" in ln for ln in lines)


def test_cli_compare_with_mocked_socket_and_webcam(monkeypatch, capsys, tmp_path):
    sock = FakeSocket(forever=face_packet(jawOpen=0.3, browInnerUp=0.6))
    monkeypatch.setattr(livelink, "_new_socket", lambda: sock)

    def webcam():
        while True:
            time.sleep(0.02)
            yield time.monotonic(), {"jawOpen": 0.25, "browInnerUp": 0.5}

    args = argparse.Namespace(action="compare", port=None, host=None, seconds=0.5, out=None, all=True, config=None)
    rc = cli.cmd_iphone(args, webcam=webcam())
    out = capsys.readouterr().out
    assert rc == 0, out
    files = list((tmp_path / "data" / "iphone").glob("compare-*.csv"))
    assert len(files) == 1
    header = next(csv.reader(files[0].open()))
    assert len(header) == 4 + 2 * 51 and "iphone_tongueOut" not in header
    assert "browInnerUp" in out and sock.closed


def test_setup_steps_mention_everything():
    text = "\n".join(iphone_cli.setup_steps(["10.0.0.5"], 11111))
    for needle in ("Live Link Face", "Live Link (ARKit)", "10.0.0.5", "11111", "same Wi-Fi", "Firewall", "UDP"):
        assert needle in text
