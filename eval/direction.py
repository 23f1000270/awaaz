"""Can two microphones tell which side the siren is on and whether it is coming closer?

Simulation, not a road test: test-split siren recordings are played from a
point moving past the listener in a straight line. Each microphone receives
the sound with its own propagation delay (so Doppler and timing come out
naturally) and 1/r spreading, plus independent road noise from the test
split. We then run the same side and approach estimates the firmware uses.

Writes eval/results/direction.md.

    python eval/direction.py --data data/clips.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from awaaz import direction  # noqa: E402
from awaaz.features import SR  # noqa: E402
from evaluate import load_clips  # noqa: E402

C = direction.SPEED_OF_SOUND
DIR_N = 1024        # samples per channel for GCC-PHAT, as on the device (64 ms)
LEVEL_N = 2048      # samples for the siren-band level (128 ms)
STEP_S = 0.512     # 32 hops of 256 samples, as on the device


def propagate(src: np.ndarray, path, mic, t):
    """Sound at microphone position `mic` from a source moving along path(t)."""
    pos = path(t)
    r = np.linalg.norm(pos - mic, axis=1)
    te = t - r / C
    s = np.interp(te * SR, np.arange(len(src)), src, left=0.0, right=0.0)
    return s / np.maximum(r, 1.0)


def scenario(src, noise_l, noise_r, side_sign, offset_m, speed, spacing, snr_db, rng):
    dur = len(src) / SR
    t = np.arange(len(src)) / SR
    y0 = -speed * dur * 0.7                            # starts behind, passes about 70% of the way through
    path = lambda tt: np.stack([np.full_like(tt, side_sign * offset_m), y0 + speed * tt], 1)  # noqa: E731
    ml, mr = np.array([spacing / 2, 0.0]), np.array([-spacing / 2, 0.0])   # +x is the listener's left
    L, R = propagate(src, path, ml, t), propagate(src, path, mr, t)
    ref = np.sqrt(np.mean(L[np.abs(path(t)[:, 1]) < 10] ** 2) + 1e-12)     # siren level when close
    scale = ref / (np.sqrt(np.mean(noise_l ** 2)) + 1e-12) / 10 ** (snr_db / 20)
    return L + noise_l * scale, R + noise_r * scale, path, t


def evaluate(src_clips, noise_clips, spacing, snr_db, rng, trials=60):
    side_ok, side_n, appr_ok, appr_n = 0, 0, 0, 0
    for _ in range(trials):
        src = src_clips[rng.integers(len(src_clips))]
        if len(src) < 6 * SR:
            src = np.tile(src, int(np.ceil(6 * SR / len(src))))
        src = src[:int(10 * SR)]
        nl = np.resize(noise_clips[rng.integers(len(noise_clips))], len(src))
        nr = np.resize(noise_clips[rng.integers(len(noise_clips))], len(src))
        sign = rng.choice([-1, 1])
        L, R, path, t = scenario(src, nl, nr, sign, rng.uniform(3, 8), rng.uniform(8, 20), spacing, snr_db, rng)
        levels = []
        for end in range(LEVEL_N, len(src), int(STEP_S * SR)):
            levels.append(direction.band_level_db(L[end - LEVEL_N:end]))
            p = path(np.array([end / SR]))[0]
            true_angle = np.degrees(np.arctan2(abs(p[0]), abs(p[1]))) if p[1] != 0 else 90.0
            if true_angle >= 20:          # only where the siren is clearly to one side
                s, _, _ = direction.side(L[end - DIR_N:end], R[end - DIR_N:end], spacing)
                side_n += 1
                side_ok += s == ("left" if sign > 0 else "right")
            if len(levels) >= 6:
                d_now = np.hypot(*p)
                d_before = np.hypot(*path(np.array([end / SR - 3.0]))[0])
                change = d_now - d_before
                if abs(change) > 15:      # clearly approaching or receding over 3 s
                    a = direction.approach(levels[-6:], STEP_S)
                    appr_n += 1
                    appr_ok += a == ("approaching" if change < 0 else "moving away")
    return side_ok / max(side_n, 1), appr_ok / max(appr_n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    pos, neg = load_clips(args.data, "test")
    lines = ["Simulated: a siren from the test split passes 3 to 8 m to one side at 8 to 20 m/s, starting behind. "
             "Signal-to-noise is measured when the siren is closest. Side accuracy counts only moments when the "
             "siren is at least 20 degrees off the line of travel. Approach accuracy counts moments when the "
             "distance changed by more than 15 m over the last 3 s.", "",
             "| Microphone spacing | Signal-to-noise | Side correct | Approach or moving away correct |",
             "|---|---:|---:|---:|"]
    for spacing in (0.15, 0.20):
        for snr in (0, 10):
            s, a = evaluate(pos, neg, spacing, snr, rng)
            lines.append(f"| {spacing * 100:.0f} cm | {snr:+d} dB | {100 * s:.0f}% | {100 * a:.0f}% |")
            print(lines[-1], flush=True)
    md = "\n".join(lines) + "\n"
    (ROOT / "eval" / "results" / "direction.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
