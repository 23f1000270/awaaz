"""Log-mel features. This file is the reference: the C code in csrc/ and the
JavaScript in site/ must produce the same numbers. Tests check that they do.

16 kHz mono, 512-sample Hann window (32 ms), 256-sample hop (16 ms),
40 triangular mel bands from 100 Hz to 6 kHz, natural log. One window for the
classifier is 64 frames, a little over one second. The mean of the window is
subtracted, so a quieter or louder microphone gives the same input.
"""

from __future__ import annotations

import numpy as np

SR = 16000
N_FFT = 512
HOP = 256
N_MELS = 40
F_MIN = 100.0
F_MAX = 6000.0
FRAMES = 64
WINDOW_SAMPLES = N_FFT + HOP * (FRAMES - 1)   # 16640 samples, 1.04 s
EPS = 1e-6


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank() -> np.ndarray:
    """(N_MELS, N_FFT // 2 + 1) triangular filters on the HTK mel scale."""
    edges = mel_to_hz(np.linspace(hz_to_mel(F_MIN), hz_to_mel(F_MAX), N_MELS + 2))
    bins = np.arange(N_FFT // 2 + 1) * SR / N_FFT
    fb = np.zeros((N_MELS, N_FFT // 2 + 1), np.float32)
    for m in range(N_MELS):
        lo, mid, hi = edges[m], edges[m + 1], edges[m + 2]
        up = (bins - lo) / (mid - lo)
        down = (hi - bins) / (hi - mid)
        fb[m] = np.maximum(0.0, np.minimum(up, down)).astype(np.float32)
    return fb


HANN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)  # periodic Hann
FB = mel_filterbank()


def logmel(x: np.ndarray) -> np.ndarray:
    """(n_frames, N_MELS) log-mel of a mono float signal at 16 kHz."""
    x = np.asarray(x, np.float32)
    n = 1 + max(0, (len(x) - N_FFT) // HOP)
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    frames = x[idx] * HANN
    spec = np.fft.rfft(frames, n=N_FFT, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    return np.log(power @ FB.T + EPS).astype(np.float32)


def normalise(window: np.ndarray) -> np.ndarray:
    """Subtract the window's mean, making the input independent of microphone gain."""
    return (window - window.mean()).astype(np.float32)


def windows(x: np.ndarray, hop_frames: int = 32) -> np.ndarray:
    """Classifier inputs every hop_frames (0.5 s): (n, FRAMES, N_MELS)."""
    lm = logmel(x)
    if len(lm) < FRAMES:
        lm = np.concatenate([lm, np.full((FRAMES - len(lm), N_MELS), np.log(EPS), np.float32)])
    starts = range(0, len(lm) - FRAMES + 1, hop_frames)
    return np.stack([normalise(lm[s:s + FRAMES]) for s in starts])
