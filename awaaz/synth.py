"""Synthetic sounds for training and for tests the dataset cannot cover.

* Sirens in the three patterns used on Indian roads: wail (slow sweep),
  yelp (fast sweep) and hi-lo (two alternating tones). Used only for testing,
  never for training, to check that the detector learned what a siren is
  rather than the particular recordings it saw.
* Horns: two detuned square-ish tones, honked in short bursts the way Indian
  traffic does. Hard negatives: a horn is loud and tonal like a siren.
* A car-cabin filter: closed windows cut high frequencies and overall level.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from .features import SR


def _tone_from_freq(freq: np.ndarray, harmonics=(1.0, 0.45, 0.25, 0.12), rng=None) -> np.ndarray:
    phase = 2 * np.pi * np.cumsum(freq) / SR
    if rng is not None:
        phase += rng.uniform(0, 2 * np.pi)
    out = sum(a * np.sin(k * phase) for k, a in enumerate(harmonics, start=1))
    return (out / np.max(np.abs(out) + 1e-9)).astype(np.float32)


def siren(kind: str, seconds: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    if kind == "wail":
        lo, hi, period = rng.uniform(550, 750), rng.uniform(1300, 1650), rng.uniform(3.0, 6.0)
        f = lo + (hi - lo) * 0.5 * (1 - np.cos(2 * np.pi * t / period + rng.uniform(0, 6.28)))
    elif kind == "yelp":
        lo, hi, period = rng.uniform(600, 800), rng.uniform(1300, 1600), rng.uniform(0.25, 0.4)
        f = lo + (hi - lo) * ((t / period + rng.uniform(0, 1)) % 1.0)
    elif kind == "hilo":
        lo, hi, period = rng.uniform(580, 700), rng.uniform(750, 1000), rng.uniform(0.9, 1.3)
        f = np.where(((t / period + rng.uniform(0, 1)) % 1.0) < 0.5, lo, hi)
    else:
        raise ValueError(kind)
    return _tone_from_freq(f.astype(np.float64), rng=rng)


def horn(seconds: float, rng: np.random.Generator) -> np.ndarray:
    """Bursts of a two-tone electric horn (typical car horns sit around 350 to 550 Hz)."""
    n = int(seconds * SR)
    out = np.zeros(n, np.float32)
    f1 = rng.uniform(330, 470)
    f2 = f1 * rng.uniform(1.18, 1.3)
    t = 0.0
    while t < seconds:
        dur = rng.uniform(0.12, 1.2)
        a, b = int(t * SR), min(n, int((t + dur) * SR))
        if b > a:
            tt = np.arange(b - a) / SR
            tone = np.sign(np.sin(2 * np.pi * f1 * tt)) * 0.6 + np.sign(np.sin(2 * np.pi * f2 * tt)) * 0.4
            env = np.minimum(1.0, np.minimum(tt / 0.01, (tt[-1] - tt + 1e-3) / 0.02))
            out[a:b] += (tone * env).astype(np.float32)
        t += dur + rng.uniform(0.05, 1.5)
    return out


def cabin(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Closed windows: low-pass around 0.8 to 2 kHz and 10 to 25 dB quieter."""
    sos = butter(2, rng.uniform(800, 2000), btype="low", fs=SR, output="sos")
    return (sosfilt(sos, x) * 10 ** (-rng.uniform(10, 25) / 20)).astype(np.float32)


def mix(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add noise so that signal power / noise power equals snr_db."""
    ps = float(np.mean(signal ** 2)) + 1e-12
    pn = float(np.mean(noise ** 2)) + 1e-12
    return (signal + noise * np.sqrt(ps / (pn * 10 ** (snr_db / 10)))).astype(np.float32)
