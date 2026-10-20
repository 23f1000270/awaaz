"""Train the siren classifier on the prepared clips.

Each epoch draws fresh one-second windows:
* positives: a siren clip, often mixed with road noise at -5 to +15 dB,
  sometimes through the car-cabin filter;
* negatives: road noise, often with synthetic horn bursts, sometimes through
  the cabin filter.

    python train/train.py --data data/clips.npz --out awaaz/weights.npz
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from awaaz import model as M
from awaaz import synth
from awaaz.features import FRAMES, N_MELS, SR, WINDOW_SAMPLES, logmel, normalise


class Clips:
    def __init__(self, path, split):
        d = np.load(path)
        offs = np.concatenate([[0], np.cumsum(d["lengths"])])
        audio = d["audio"]
        keep = d["split"] == split
        self.items = [(audio[offs[i]:offs[i + 1]].astype(np.float32) / 32767, int(d["label"][i]))
                      for i in np.nonzero(keep)[0]]
        self.pos = [a for a, y in self.items if y == 1]
        self.neg = [a for a, y in self.items if y == 0]


def crop(x, rng, n=WINDOW_SAMPLES):
    if len(x) <= n:
        return np.pad(x, (0, n - len(x)))
    s = rng.integers(0, len(x) - n)
    return x[s:s + n]


def example(clips: Clips, positive: bool, rng) -> np.ndarray:
    noise = crop(clips.neg[rng.integers(len(clips.neg))], rng)
    if positive:
        x = crop(clips.pos[rng.integers(len(clips.pos))], rng)
        if rng.random() < 0.7:
            x = synth.mix(x, noise, rng.uniform(-5, 15))
    else:
        x = noise
        if rng.random() < 0.4:
            x = synth.mix(noise, synth.horn(len(x) / SR, rng), rng.uniform(-15, 5))
    if rng.random() < 0.3:
        x = synth.cabin(x, rng) + 0.003 * synth.cabin(crop(clips.neg[rng.integers(len(clips.neg))], rng), rng)
    x = x * rng.uniform(0.05, 1.0)
    return normalise(logmel(x)[:FRAMES])


def batch(clips, n, rng):
    y = rng.random(n) < 0.5
    X = np.stack([example(clips, bool(v), rng) for v in y])
    return torch.from_numpy(X).transpose(1, 2)[:, None], torch.from_numpy(y.astype(np.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--per-epoch", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train, val = Clips(args.data, "train"), Clips(args.data, "val")
    vrng = np.random.default_rng(123)
    Xv, yv = batch(val, 3000, vrng)
    net = M.build()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=args.epochs * (args.per_epoch // 128))
    lossf = nn.BCEWithLogitsLoss()
    best, best_state = 1e9, None
    for ep in range(args.epochs):
        t0 = time.time()
        X, y = batch(train, args.per_epoch, rng)
        net.train()
        perm = torch.randperm(len(y))
        for i in range(0, len(y) - 127, 128):
            j = perm[i:i + 128]
            opt.zero_grad()
            loss = lossf(net(X[j])[:, 0], y[j])
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            pv = net(Xv)[:, 0]
            vl = float(lossf(pv, yv))
            acc = float(((pv > 0) == (yv > 0.5)).float().mean())
        print(f"epoch {ep + 1}: val loss {vl:.4f} acc {acc:.3f} ({time.time() - t0:.0f} s)", flush=True)
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    w = M.fold(net)
    # The folded NumPy reference must agree with PyTorch.
    with torch.no_grad():
        pt = torch.sigmoid(net(Xv[:20])[:, 0]).numpy()
    ref = np.array([M.forward(w, Xv[k, 0].T.numpy()) for k in range(20)])
    assert np.max(np.abs(pt - ref)) < 1e-4, np.max(np.abs(pt - ref))
    np.savez(args.out, **w)
    print("saved", args.out, sum(v.size for v in w.values()), "weights")


if __name__ == "__main__":
    main()
