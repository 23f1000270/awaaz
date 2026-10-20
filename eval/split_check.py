"""Why the split is by blocks of clips: neighbouring clips sound alike.

Compares the average log-mel spectrum of each clip with the next clip of the
same class. Then does the same for random pairs of the same class that are
more than 40 clips apart. Writes eval/results/split_check.md.

    python eval/split_check.py --data data/clips.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from awaaz.features import logmel  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    d = np.load(args.data)
    offs = np.concatenate([[0], np.cumsum(d["lengths"])])
    label, number, audio = d["label"], d["number"], d["audio"]   # read each array once
    order = np.lexsort((number, label))                 # by class, then clip number
    prof = {}
    for i in order:
        p = logmel(audio[offs[i]:offs[i + 1]].astype(np.float32) / 32767).mean(0)
        prof[i] = (p - p.mean()) / (p.std() + 1e-9)
    near = [float(np.mean(prof[a] * prof[b])) for a, b in zip(order, order[1:])
            if label[a] == label[b] and number[b] == number[a] + 1]
    rng = np.random.default_rng(args.seed)
    far = []
    while len(far) < 2000:
        a, b = rng.choice(order, 2, replace=False)
        if label[a] == label[b] and abs(int(number[a]) - int(number[b])) > 40:
            far.append(float(np.mean(prof[a] * prof[b])))
    md = ("Correlation between the average spectra of two clips (1 = identical shape).\n\n"
          "| Pair | Pairs | Mean correlation |\n|---|---:|---:|\n"
          f"| Consecutive clip numbers, same class | {len(near)} | {np.mean(near):.2f} |\n"
          f"| Random clips of the same class, more than 40 numbers apart | {len(far)} | {np.mean(far):.2f} |\n")
    (ROOT / "eval" / "results" / "split_check.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
