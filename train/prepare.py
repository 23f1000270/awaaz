"""Resample the siren dataset to 16 kHz mono and split it without leaks.

The dataset (Asif et al., Scientific Data 2022, CC0) numbers its clips in the
order they were cut from longer source recordings, so neighbouring clips can
come from the same ambulance. We split by blocks of consecutive clip numbers,
so no source recording lands in both training and test.

    python train/prepare.py --src "data/raw" --out data/clips.npz
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from awaaz.features import SR, logmel

BLOCK = 40


def load(path: Path) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    peak = np.max(np.abs(x)) + 1e-9
    return (x / peak * 0.9).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)   # fail now, not after resampling every clip
    rng = np.random.default_rng(args.seed)
    audio, label, number, split = [], [], [], []
    for cls, folder, prefix in ((1, "Ambulance data", "ambulance"), (0, "Road Noises", "road")):
        files = sorted((args.src / folder).glob("*.wav"), key=lambda p: int(re.sub(r"\D", "", p.stem)))
        nums = [int(re.sub(r"\D", "", p.stem)) for p in files]
        blocks = sorted({(n - 1) // BLOCK for n in nums})
        order = rng.permutation(blocks)
        n_test = max(1, round(0.15 * len(blocks)))
        n_val = max(1, round(0.15 * len(blocks)))
        which = {b: "test" for b in order[:n_test]}
        which.update({b: "val" for b in order[n_test:n_test + n_val]})
        for p, n in zip(files, nums):
            audio.append((load(p) * 32767).astype(np.int16))
            label.append(cls)
            number.append(n)
            split.append(which.get((n - 1) // BLOCK, "train"))
        print(folder, len(files), "clips,", len(blocks), "blocks", flush=True)
    # How alike are neighbouring clips compared with random pairs? (justifies block splitting)
    prof = [logmel(a.astype(np.float32) / 32767).mean(0) for a in audio]
    prof = [(p - p.mean()) / (p.std() + 1e-9) for p in prof]
    near = [float(np.mean(prof[i] * prof[i + 1])) for i in range(len(prof) - 1) if label[i] == label[i + 1]]
    far = [float(np.mean(prof[i] * prof[j])) for i, j in rng.integers(0, len(prof), (2000, 2)) if label[i] == label[j]]
    print(f"spectral similarity: neighbouring clips {np.mean(near):.2f}, random clips of the same class {np.mean(far):.2f}")
    lengths = np.array([len(a) for a in audio])
    np.savez(args.out, audio=np.concatenate(audio), lengths=lengths, label=np.array(label),
             number=np.array(number), split=np.array(split))
    for s in ("train", "val", "test"):
        m = np.array(split) == s
        print(s, {c: int((np.array(label)[m] == c).sum()) for c in (0, 1)},
              f"{lengths[m].sum() / SR / 3600:.2f} h")


if __name__ == "__main__":
    main()
