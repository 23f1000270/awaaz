"""Which side is the siren on? Is it coming closer?

Side: two microphones a known distance apart hear the siren at slightly
different times. GCC-PHAT (generalised cross-correlation with phase
transform) finds that delay; its sign says left or right. Only the siren
band is used, so engine rumble and wind count for little.

Closer or further: a siren's level rises steadily as it approaches. We fit a
line to the siren-band level over the last few seconds.

The C version in csrc/ follows this code step by step.
"""

from __future__ import annotations

import numpy as np

from .features import SR

SPEED_OF_SOUND = 343.0
BAND = (400.0, 4000.0)


def next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def gcc_phat(left: np.ndarray, right: np.ndarray, max_tau: float, fs: int = SR) -> tuple[float, float]:
    """Delay of the right channel relative to the left, in seconds (positive: the
    sound reached the left microphone first). Also returns the correlation peak (0 to 1)."""
    n = next_pow2(len(left) + len(right))
    L = np.fft.rfft(left.astype(np.float64), n)
    R = np.fft.rfft(right.astype(np.float64), n)
    G = R * np.conj(L)
    G /= np.abs(G) + 1e-12
    f = np.arange(len(G)) * fs / n
    G[(f < BAND[0]) | (f > BAND[1])] = 0
    cc = np.fft.irfft(G, n)
    m = int(np.ceil(max_tau * fs))
    win = np.concatenate([cc[-m:], cc[:m + 1]])        # lags -m .. +m
    k = int(np.argmax(win))
    lag = float(k - m)
    if 0 < k < len(win) - 1:                            # parabolic sub-sample refinement
        a, b, c = win[k - 1], win[k], win[k + 1]
        den = a - 2 * b + c
        if den != 0:
            lag += 0.5 * (a - c) / den
    in_band = np.count_nonzero((f >= BAND[0]) & (f <= BAND[1]))
    peak = float(win[k] * n / (2 * in_band)) if in_band else 0.0
    return lag / fs, peak


def side(left, right, spacing_m: float, fs: int = SR, min_angle_deg: float = 10.0):
    """'left', 'right' or 'ahead/behind' (too close to the axis to tell), plus the angle in degrees."""
    max_tau = spacing_m / SPEED_OF_SOUND
    tau, peak = gcc_phat(left, right, max_tau, fs)
    s = np.clip(tau / max_tau, -1.0, 1.0)
    angle = float(np.degrees(np.arcsin(s)))   # positive: towards the left microphone
    if abs(angle) < min_angle_deg:
        return "ahead/behind", angle, peak
    return ("left" if angle > 0 else "right"), angle, peak


def band_level_db(x: np.ndarray, fs: int = SR) -> float:
    n = next_pow2(len(x))
    X = np.fft.rfft(x.astype(np.float64) * np.hanning(len(x)), n)
    f = np.arange(len(X)) * fs / n
    p = np.sum(np.abs(X[(f >= BAND[0]) & (f <= BAND[1])]) ** 2) / len(x)
    return float(10 * np.log10(p + 1e-12))


def trend(levels_db: list[float], step_s: float) -> float:
    """Least-squares slope of the level in dB per second."""
    if len(levels_db) < 3:
        return 0.0
    t = np.arange(len(levels_db)) * step_s
    return float(np.polyfit(t, np.asarray(levels_db), 1)[0])


def approach(levels_db: list[float], step_s: float, threshold_db_s: float = 1.0) -> str:
    s = trend(levels_db, step_s)
    if s > threshold_db_s:
        return "approaching"
    if s < -threshold_db_s:
        return "moving away"
    return "steady"
