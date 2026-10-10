import json

import numpy as np
import pytest

from away.wifi import features as F
from away.wifi.classifier import CoarseCSIClassifier, CSIThresholds, _margin
from away.wifi.sources import (
    IAXSource,
    frames_from_array,
    parse_iax_csv_row,
    parse_iax_line,
    parse_netsh,
    parse_proc_net_wireless,
    parse_wdutil,
    picoscenes_frame,
    synthetic_csi,
)

PROC = """Inter-| sta-|   Quality        |   Discarded packets               | Missed | WE
 face | tus | link level noise |  nwid  crypt   frag  retry   misc | beacon | 22
 wlp2s0: 0000   54.  -56.  -256        0      0      0      0      0        0
 wlan1: 0000   30.  -80.  -256        0      0      0      0      0        0
"""


def test_parse_proc_net_wireless():
    assert parse_proc_net_wireless(PROC) == -56.0
    assert parse_proc_net_wireless(PROC, "wlan1") == -80.0
    assert parse_proc_net_wireless(PROC, "eth0") is None
    assert parse_proc_net_wireless("h\nh\n") is None


def test_parse_netsh_and_wdutil():
    assert parse_netsh("    Name : Wi-Fi\n    Signal                 : 80%\n") == -60.0
    assert parse_netsh("nothing") is None
    assert parse_wdutil("    RSSI                 : -61 dBm") == -61.0
    assert parse_wdutil("     agrCtlRSSI: -45") == -45.0
    assert parse_wdutil("") is None


def test_iax_jsonl(tmp_path):
    line = json.dumps({"t": 1.5, "rssi": -40, "re": [[1, 0], [0, 2]], "im": [[0, 1], [3, 0]]})
    fr = parse_iax_line(line)
    assert fr.t == 1.5 and fr.rssi == -40.0
    assert fr.csi.shape == (2, 2)
    assert fr.csi[1, 0] == 0 + 3j
    assert parse_iax_line("# comment") is None
    assert parse_iax_line("   ") is None
    no_rssi = parse_iax_line(json.dumps({"t": 0, "re": [1, 2], "im": [0, 0]}))
    assert no_rssi.rssi is None and no_rssi.csi.shape == (1, 2)
    with pytest.raises(ValueError):
        parse_iax_line(json.dumps({"t": 0, "re": [1, 2], "im": [0]}))
    p = tmp_path / "cap.jsonl"
    p.write_text("# header\n" + line + "\n" + line + "\n")
    assert len(list(IAXSource(p))) == 2


def test_iax_csv(tmp_path):
    fr = parse_iax_csv_row(["0.5", "-50", "1", "2", "3", "4"])
    assert fr.t == 0.5 and fr.rssi == -50.0
    assert np.allclose(fr.csi, [[1 + 2j, 3 + 4j]])
    assert parse_iax_csv_row(["0", "", "1", "1"]).rssi is None
    for bad in (["0", "1", "2"], ["0", "1", "2", "3", "4"]):
        with pytest.raises(ValueError):
            parse_iax_csv_row(bad)
    p = tmp_path / "cap.csv"
    p.write_text("t,rssi,re0,im0\n# c\n0.1,-40,1,0\n\n0.2,-41,0,1\n")
    frames = list(IAXSource(p))
    assert [f.t for f in frames] == [0.1, 0.2]


def test_picoscenes_frame_conversion():
    raw = {"RxSBasic": {"systemns": 2_000_000_000, "rssi": -42}, "CSI": {"CSI": np.arange(8) + 1j, "numTones": 4}}
    fr = picoscenes_frame(raw)
    assert fr.t == 2.0 and fr.rssi == -42.0
    assert fr.csi.shape == (2, 4)
    empty = picoscenes_frame({})
    assert empty.csi is None and empty.rssi is None and empty.t == 0.0
    odd = picoscenes_frame({"CSI": {"CSI": np.ones(5), "numTones": 4}})
    assert odd.csi.shape == (1, 5)


def test_synthetic_shapes_and_validation():
    t, H = synthetic_csi("push", seconds=1.0, fs=50, n_sub=30, n_rx=3)
    assert t.shape == (50,) and H.shape == (50, 3, 30)
    with pytest.raises(ValueError):
        synthetic_csi("dance")
    frames = frames_from_array(t, H)
    assert len(frames) == 50 and frames[0].csi.shape == (3, 30)


def test_hampel_removes_spikes_keeps_signal():
    x = np.sin(np.linspace(0, 6, 100))
    y = x.copy()
    y[[20, 60]] += 5.0
    out = F.hampel(y, k=3)
    assert np.allclose(out, x, atol=0.1)
    assert np.allclose(F.hampel(x, k=3), x)
    m = np.stack([y, y], axis=1)
    assert np.allclose(F.hampel(m, k=3)[:, 1], out)
    flat = np.ones(10)
    flat[5] = 2
    assert F.hampel(flat, k=2)[5] == 2.0  # MAD 0 -> no replacement (no scale information)
    with pytest.raises(ValueError):
        F.hampel(x, k=0)


def test_hampel_threshold():
    x = np.array([1, -1, 1, -1, 1, 4, -1, 1, -1, 1, -1], dtype=float)
    # window median 1, MAD 2 -> scaled 2.9652; |4 - 1| = 3
    assert F.hampel(x, k=5, n_sigma=1.0)[5] == 1.0
    assert F.hampel(x, k=5, n_sigma=1.1)[5] == 4.0


def test_sanitize_phase_removes_linear_trend():
    k = np.arange(30)
    true = 0.3 * np.sin(k / 3.0)
    garbage = 1.3 + 0.2 * k
    csi = np.exp(1j * (true + garbage))
    out = F.sanitize_phase(csi[None, :])[0]
    detr = true - np.polyval(np.polyfit(k, true, 1), k)
    assert np.allclose(out, detr, atol=1e-9)
    assert abs(out.mean()) < 1e-9
    out2 = F.sanitize_phase(csi, subcarriers=k * 2.0)
    assert np.allclose(out2, detr, atol=1e-9)


def test_csi_ratio_cancels_common_phase():
    h = np.ones((2, 2, 4), dtype=complex)
    h[:, 0] *= 2
    h[1] *= np.exp(1j * 1.234)
    r = F.csi_ratio(h)
    assert np.allclose(r, 2.0)


def test_normalize_amplitude_and_energy():
    amp = np.array([[1.0, 0.0], [3.0, 0.0]])
    n = F.normalize_amplitude(amp)
    assert np.allclose(n[:, 0], [0.5, 1.5]) and np.allclose(n[:, 1], 0)
    assert F.motion_energy(np.ones((10, 4))) == 0.0
    assert F.motion_energy(amp) == pytest.approx(0.25)
    assert F.rssi_motion_energy(np.array([1.0])) == 0.0
    assert F.rssi_motion_energy(np.array([-50.0, -48.0, -51.0])) == pytest.approx(2.5)


def test_first_pc_and_doppler():
    t = np.arange(200) / 100
    s = np.sin(2 * np.pi * 10 * t)
    x = np.stack([s, 2 * s, -s], axis=1) + 5
    pc = F.first_pc(x)
    assert abs(np.corrcoef(pc, s)[0, 1]) > 0.999
    f, tt, S = F.doppler_spectrogram(s, 100.0, nperseg=64)
    peak = f[np.argmax(S.mean(axis=1))]
    assert abs(peak - 10) < 2
    f2, _, _ = F.doppler_spectrogram(s[:20], 100.0, nperseg=64)
    assert len(f2) == 11  # nperseg clipped to the series length


def test_ratio_rotation_direction():
    _, push = synthetic_csi("push", seed=1)
    _, pull = synthetic_csi("pull", seed=1)
    _, swipe = synthetic_csi("swipe", seed=1)
    n_push, tot_push = F.ratio_rotation(push)
    n_pull, _ = F.ratio_rotation(pull)
    n_sw, tot_sw = F.ratio_rotation(swipe)
    assert n_push > 3 and n_pull < -3
    assert tot_push >= abs(n_push)
    assert abs(n_sw) < 0.5 and tot_sw > 5


def test_window_features_csi_and_rssi():
    _, H = synthetic_csi("push", seed=2)
    f = F.window_features(H, np.array([-50.0, -52.0]), 100.0)
    assert set(f) == {"energy", "doppler_peak_hz", "burst_fraction", "rotation_net", "rotation_total", "rssi_energy"}
    assert f["rssi_energy"] == 2.0
    single = F.window_features(H[:, 0, :], None, 100.0)  # (T, n_sub) single antenna
    assert "rotation_net" not in single and single["energy"] > 0
    assert F.window_features(None, np.array([-50.0, -50.0]), 10.0) == {"rssi_energy": 0.0}
    assert F.window_features(None, None, 10.0) == {}


@pytest.mark.parametrize("kind", ["empty", "push", "pull", "swipe"])
def test_classifier_on_synthetic(kind):
    c = CoarseCSIClassifier()
    hits = 0
    for seed in range(5):
        _, H = synthetic_csi(kind, seed=seed)
        label, conf = c.classify(F.window_features(H, None, 100.0))
        hits += label == kind
        assert 0.0 <= conf <= 1.0
    assert hits == 5


def test_classifier_rules_directly():
    th = CSIThresholds(presence=0.1, motion=0.2)
    c = CoarseCSIClassifier(th)
    assert c.classify({"energy": 0.05})[0] == "empty"
    assert c.classify({"energy": 0.1})[0] == "presence"
    assert c.classify({"energy": 0.15})[0] == "presence"
    assert c.classify({"energy": 0.2})[0] == "motion"
    assert c.classify({"energy": 0.3, "rotation_net": 2.0, "rotation_total": 2.5})[0] == "push"
    assert c.classify({"energy": 0.3, "rotation_net": -2.0, "rotation_total": 2.5})[0] == "pull"
    assert c.classify({"energy": 0.3, "rotation_net": 0.5, "rotation_total": 4.0})[0] == "swipe"
    assert c.classify({"energy": 0.3, "rotation_net": 0.5, "rotation_total": 1.0})[0] == "motion"
    assert c.classify({"energy": 0.3, "rotation_net": 0.0, "rotation_total": 0.0})[0] == "motion"
    assert c.classify({"energy": 0.3, "rotation_net": 0.9, "rotation_total": 1.0})[0] == "motion"  # < min net
    th.direction_sign = -1.0
    assert c.classify({"energy": 0.3, "rotation_net": 2.0, "rotation_total": 2.5})[0] == "pull"
    assert c.classify({"rssi_energy": 0.1})[0] == "empty"
    assert c.classify({"rssi_energy": 0.3})[0] == "presence"
    assert c.classify({"rssi_energy": 1.5})[0] == "motion"
    assert c.classify({}) == ("empty", 0.0)


def test_classifier_boundaries_ratio():
    c = CoarseCSIClassifier(CSIThresholds(presence=0.1, motion=0.2))
    assert c.classify({"energy": 0.3, "rotation_net": 1.2, "rotation_total": 2.0})[0] == "push"  # 0.6 exactly
    assert c.classify({"energy": 0.3, "rotation_net": 0.6, "rotation_total": 2.0})[0] == "swipe"  # 0.3 exactly
    assert c.classify({"energy": 0.3, "rotation_net": 1.0, "rotation_total": 1.0})[0] == "push"
    assert c.classify({"energy": 0.3, "rotation_net": 0.0, "rotation_total": 2.0})[0] == "swipe"
    assert c.classify({"energy": 0.3, "rotation_net": 0.0, "rotation_total": 1.99})[0] == "motion"


def test_classifier_confidences():
    c = CoarseCSIClassifier(CSIThresholds(presence=0.1, motion=0.2))
    assert c.classify({"energy": 0.0})[1] == 1.0
    assert c.classify({"energy": 0.15})[1] == pytest.approx(0.75)
    assert c.classify({"energy": 0.3, "rotation_net": 2.0, "rotation_total": 2.5})[1] == pytest.approx(0.8)
    assert c.classify({"energy": 0.3, "rotation_net": 0.4, "rotation_total": 4.0})[1] == pytest.approx(0.9)
    assert _margin(1.0, 0.0) == 0.5
    assert _margin(0.0, 1.0) == 0.5
    assert _margin(5.0, 1.0) == 1.0


def test_calibration_learns_thresholds_and_sign():
    c = CoarseCSIClassifier()
    samples = [
        ({"energy": 0.01}, "empty"),
        ({"energy": 0.04}, "presence"),
        ({"energy": 0.16, "rotation_net": -3.0}, "push"),
        ({"energy": 0.25}, "swipe"),
    ]
    th = c.calibrate(samples)
    assert th.presence == pytest.approx(0.02)
    assert th.motion == pytest.approx(0.08)
    assert th.direction_sign == -1.0
    # positive push net -> sign +1
    assert CoarseCSIClassifier().calibrate([({"rotation_net": 2.0}, "push")]).direction_sign == 1.0


def test_calibration_ignores_inconsistent_groups():
    c = CoarseCSIClassifier(CSIThresholds(presence=0.015, motion=0.08))
    th = c.calibrate([({"energy": 0.5}, "empty"), ({"energy": 0.1}, "presence"), ({"energy": 0.05}, "motion")])
    assert th.presence == 0.015 and th.motion == 0.08
    th2 = c.calibrate([({"energy": 0.0}, "empty"), ({"energy": 0.04}, "presence")])
    assert th2.presence == pytest.approx(0.02)
    assert c.calibrate([]).presence == pytest.approx(0.02)
