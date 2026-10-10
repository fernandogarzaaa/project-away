"""CSI feature extraction (amplitude/phase sanitisation, outlier removal, motion, Doppler)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import signal

F = NDArray[np.float64]
Cx = NDArray[np.complex128]


def hampel(x: F, k: int = 5, n_sigma: float = 3.0) -> F:
    """Hampel filter along axis 0: replace samples > n_sigma*MAD from the rolling median."""
    x = np.asarray(x, dtype=np.float64)
    if k < 1:
        raise ValueError("k must be >= 1")
    y = x.copy()
    n = len(x)
    scale = 1.4826  # MAD -> std for Gaussian data
    for i in range(n):
        lo, hi = max(0, i - k), min(n, i + k + 1)
        win = x[lo:hi]
        med = np.median(win, axis=0)
        mad = scale * np.median(np.abs(win - med), axis=0)
        out = np.abs(x[i] - med) > n_sigma * mad
        out &= mad > 0
        if np.ndim(out) == 0:
            if out:
                y[i] = med
        else:
            y[i] = np.where(out, med, x[i])
    return y


def amplitude(csi: Cx) -> F:
    return np.abs(np.asarray(csi))


def sanitize_phase(csi: Cx, subcarriers: F | None = None) -> F:
    """Unwrap across subcarriers and remove the per-packet linear trend (STO slope + CFO offset).

    Input (..., n_sub); output same shape, phase in radians.
    """
    ph = np.unwrap(np.angle(np.asarray(csi)), axis=-1)
    n_sub = ph.shape[-1]
    k = np.arange(n_sub, dtype=np.float64) if subcarriers is None else np.asarray(subcarriers, dtype=np.float64)
    k_c = k - k.mean()
    slope = (ph * k_c).sum(axis=-1, keepdims=True) / (k_c * k_c).sum()
    offset = ph.mean(axis=-1, keepdims=True)
    out: F = ph - slope * k_c - offset
    return out


def csi_ratio(csi: Cx, a: int = 0, b: int = 1, eps: float = 1e-9) -> Cx:
    """Antenna ratio H_a / H_b over (T, n_rx, n_sub): cancels per-packet CFO/STO common to both chains."""
    h = np.asarray(csi)
    out: Cx = h[:, a, :] / (h[:, b, :] + eps)
    return out


def normalize_amplitude(amp: F) -> F:
    """(T, n_sub): divide each subcarrier by its mean (removes static gain)."""
    m = amp.mean(axis=0, keepdims=True)
    out: F = amp / np.where(m > 0, m, 1.0)
    return out


def motion_energy(amp: F) -> float:
    """Mean over subcarriers of the temporal std of normalized amplitude."""
    a = normalize_amplitude(np.asarray(amp, dtype=np.float64).reshape(len(amp), -1))
    return float(a.std(axis=0).mean())


def rssi_motion_energy(rssi: F) -> float:
    r = np.asarray(rssi, dtype=np.float64)
    if len(r) < 2:
        return 0.0
    return float(np.abs(np.diff(r)).mean())


def first_pc(x: F) -> F:
    """First principal component time series of (T, D)."""
    x = np.asarray(x, dtype=np.float64)
    xc = x - x.mean(axis=0)
    _u, _s, vt = np.linalg.svd(xc, full_matrices=False)
    out: F = xc @ vt[0]
    return out


def doppler_spectrogram(series: F, fs: float, nperseg: int = 64, noverlap: int | None = None) -> tuple[F, F, F]:
    """|STFT| of a (T,) series (e.g. first PC of amplitude). Returns freqs, times, magnitude."""
    nper = min(nperseg, len(series))
    f, tt, z = signal.stft(
        np.asarray(series, dtype=np.float64),
        fs=fs,
        nperseg=nper,
        noverlap=noverlap if noverlap is not None else nper * 3 // 4,
    )
    return f, tt, np.abs(z)


def ratio_phase_track(csi: Cx) -> F:
    """Unwrapped phase of the antenna ratio over time, averaged over subcarriers (T,)."""
    r = csi_ratio(csi)
    ph = np.unwrap(np.angle(r), axis=0)
    out: F = ph.mean(axis=1) - ph[0].mean()
    return out


def ratio_rotation(csi: Cx) -> tuple[float, float]:
    """(net, total) rotations of the antenna ratio around its own centre, median over subcarriers.

    A reflector moving along the link bisector rotates the dynamic CSI vector,
    which maps to a circle in the ratio plane (a Mobius transform of the unit
    circle). Net rotations ~ path-length change / wavelength (sign = direction,
    geometry-dependent, so calibrate the sign); swipes go and come back
    (net ~ 0, total large).
    """
    r = csi_ratio(csi)
    ang = np.unwrap(np.angle(r - r.mean(axis=0)), axis=0)
    net = (ang[-1] - ang[0]) / (2 * np.pi)
    total = np.abs(np.diff(ang, axis=0)).sum(axis=0) / (2 * np.pi)
    return float(np.median(net)), float(np.median(total))


def window_features(csi: Cx | None, rssi: F | None, fs: float) -> dict[str, float]:
    """Features for one window. Works with RSSI only (csi=None)."""
    feats: dict[str, float] = {}
    if csi is not None:
        h = np.asarray(csi)
        if h.ndim == 2:
            h = h[:, None, :]
        amp = hampel(amplitude(h).reshape(len(h), -1), k=3)
        feats["energy"] = motion_energy(amp)
        pc = first_pc(normalize_amplitude(amp))
        f, _t, S = doppler_spectrogram(pc, fs)
        prof = S.mean(axis=1)
        prof[0] = 0.0
        feats["doppler_peak_hz"] = float(f[int(np.argmax(prof))]) if prof.any() else 0.0
        # burst duration: fraction of time the short-time energy exceeds half its max
        st = np.convolve(np.abs(np.diff(pc)), np.ones(5) / 5, mode="same")
        feats["burst_fraction"] = float((st > 0.5 * st.max()).mean()) if st.max() > 0 else 0.0
        if h.shape[1] >= 2:
            net, total = ratio_rotation(h)
            feats["rotation_net"] = net
            feats["rotation_total"] = total
    if rssi is not None and len(rssi):
        feats["rssi_energy"] = rssi_motion_energy(rssi)
    return feats
