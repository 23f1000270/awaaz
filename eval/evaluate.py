"""Score the trained detector on the held-out test split and harder variants.

Detection rule (the same on the device): a window is scored every 0.5 s and
an alarm needs two of the last three windows above the threshold. The
threshold is chosen on the validation split, never the test split: the
lowest one at which at most 5% of validation road-noise clips raise an alarm.

Writes eval/results/results.md and results.json.

    python eval/evaluate.py --data data/clips.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from awaaz import synth  # noqa: E402
from awaaz.features import SR, windows  # noqa: E402
from awaaz.model import CHANNELS  # noqa: E402

HOP_S = 0.5


def load_clips(path, split):
    d = np.load(path)
    offs = np.concatenate([[0], np.cumsum(d["lengths"])])
    keep = np.nonzero(d["split"] == split)[0]
    return ([d["audio"][offs[i]:offs[i + 1]].astype(np.float32) / 32767 for i in keep if d["label"][i] == 1],
            [d["audio"][offs[i]:offs[i + 1]].astype(np.float32) / 32767 for i in keep if d["label"][i] == 0])


class Scorer:
    def __init__(self, weights_path):
        self.w = {k: torch.from_numpy(v) for k, v in np.load(weights_path).items()}

    def __call__(self, X: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(X).transpose(1, 2)[:, None]
        with torch.no_grad():
            for i in range(len(CHANNELS)):
                x = F.relu(F.conv2d(x, self.w[f"conv{i}_w"], self.w[f"conv{i}_b"], padding=1))
                if i < len(CHANNELS) - 1:
                    x = F.max_pool2d(x, 2)
            z = x.mean(dim=(2, 3)) @ self.w["fc_w"] + self.w["fc_b"]
        return torch.sigmoid(z).numpy()


def clip_probs(scorer, x):
    return scorer(windows(x)) if len(x) else np.zeros(0)


def alarms(p, thr):
    """Indices of windows where the 2-of-3 rule fires."""
    hit = p >= thr
    fired = [i for i in range(len(p)) if hit[max(0, i - 2):i + 1].sum() >= 2]
    return fired


def onsets(p, thr):
    f = alarms(p, thr)
    return sum(1 for k, i in enumerate(f) if k == 0 or f[k - 1] != i - 1)


def choose_threshold(scorer, val_neg):
    probs = [clip_probs(scorer, x) for x in val_neg]
    for thr in np.arange(0.30, 0.995, 0.01):
        rate = np.mean([len(alarms(p, thr)) > 0 for p in probs])
        if rate <= 0.05:
            return float(round(thr, 2)), float(rate)
    return 0.99, float("nan")


def detect_rate(scorer, clips, thr):
    hits, delays = [], []
    for x in clips:
        f = alarms(clip_probs(scorer, x), thr)
        hits.append(bool(f))
        if f:
            delays.append((f[0] * 32 * 256 + 16640) / SR)   # time at which the alarm window ends
    return float(np.mean(hits)), (float(np.median(delays)) if delays else float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default=str(ROOT / "awaaz" / "weights.npz"))
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    scorer = Scorer(args.weights)
    val_pos, val_neg = load_clips(args.data, "val")
    test_pos, test_neg = load_clips(args.data, "test")
    thr, val_far = choose_threshold(scorer, val_neg)
    out = {"threshold": thr, "val_road_clip_false_alarm": val_far,
           "test_clips": {"siren": len(test_pos), "road": len(test_neg)}}

    # 1. The held-out recordings as they are.
    det, delay = detect_rate(scorer, test_pos, thr)
    neg_probs = [clip_probs(scorer, x) for x in test_neg]
    road_hours = sum(len(x) for x in test_neg) / SR / 3600
    out["as_recorded"] = {"siren_detected": det, "median_seconds_to_alarm": delay,
                          "road_clips_with_false_alarm": float(np.mean([len(alarms(p, thr)) > 0 for p in neg_probs])),
                          "false_alarms_per_hour": sum(onsets(p, thr) for p in neg_probs) / road_hours,
                          "road_hours": road_hours}
    from sklearn.metrics import roc_auc_score
    pw = np.concatenate([clip_probs(scorer, x) for x in test_pos])
    nw = np.concatenate(neg_probs)
    out["as_recorded"]["window_auc"] = float(roc_auc_score(np.r_[np.ones(len(pw)), np.zeros(len(nw))], np.r_[pw, nw]))

    def noise_like(n):
        x = test_neg[rng.integers(len(test_neg))]
        reps = int(np.ceil(n / len(x)))
        x = np.tile(x, reps)[:n]
        return x

    # 2. Sirens buried in other test road noise, open air and through a closed car.
    out["snr"] = {}
    for cabin in (False, True):
        for snr in (-10, -5, 0, 5, 10):
            mixed = []
            for x in test_pos:
                y = synth.mix(x, noise_like(len(x)), snr)
                if cabin:
                    y = synth.cabin(y, rng) + 0.003 * synth.cabin(noise_like(len(x)), rng)
                mixed.append(y)
            out["snr"][f"{'cabin' if cabin else 'open'} {snr:+d} dB"] = detect_rate(scorer, mixed, thr)[0]

    # 3. Synthetic Indian-pattern sirens the model never trained on.
    out["synthetic_sirens"] = {}
    for kind in ("wail", "yelp", "hilo"):
        for snr in (0, 10):
            clips = [synth.mix(synth.siren(kind, 6.0, rng), noise_like(int(6 * SR)), snr) for _ in range(40)]
            out["synthetic_sirens"][f"{kind} {snr:+d} dB"] = detect_rate(scorer, clips, thr)[0]

    # 4. Horns, the classic false alarm on Indian roads.
    hornc = [synth.mix(noise_like(int(8 * SR)), synth.horn(8.0, rng), -snr) for snr in rng.uniform(-10, 10, 120)]
    hp = [clip_probs(scorer, x) for x in hornc]
    out["horns"] = {"clips": len(hornc), "clips_with_false_alarm": float(np.mean([len(alarms(p, thr)) > 0 for p in hp]))}

    res = ROOT / "eval" / "results"
    res.mkdir(parents=True, exist_ok=True)
    (res / "results.json").write_text(json.dumps(out, indent=1))
    a = out["as_recorded"]
    md = [f"Threshold {thr} chosen on the validation split ({100 * val_far:.0f}% of validation road clips alarm).", "",
          f"**Held-out recordings** ({len(test_pos)} siren clips, {len(test_neg)} road clips, "
          f"{a['road_hours'] * 60:.0f} minutes of road noise)", "",
          "| Measure | Result |", "|---|---:|",
          f"| Siren clips detected | {100 * a['siren_detected']:.1f}% |",
          f"| Median time from start of clip to alarm | {a['median_seconds_to_alarm']:.1f} s |",
          f"| Road clips with a false alarm | {100 * a['road_clips_with_false_alarm']:.1f}% |",
          f"| False alarms per hour of road noise | {a['false_alarms_per_hour']:.1f} |",
          f"| Window-level ROC AUC | {a['window_auc']:.3f} |", "",
          "**Sirens mixed into other road noise** (detection rate)", "",
          "| Signal-to-noise | Open air | Inside a closed car (simulated) |", "|---|---:|---:|"]
    for snr in (-10, -5, 0, 5, 10):
        md.append(f"| {snr:+d} dB | {100 * out['snr'][f'open {snr:+d} dB']:.0f}% | {100 * out['snr'][f'cabin {snr:+d} dB']:.0f}% |")
    md += ["", "**Synthetic Indian siren patterns, never seen in training** (40 clips each)", "",
           "| Pattern | 0 dB | +10 dB |", "|---|---:|---:|"]
    for kind in ("wail", "yelp", "hilo"):
        md.append(f"| {kind} | {100 * out['synthetic_sirens'][f'{kind} +0 dB']:.0f}% | "
                  f"{100 * out['synthetic_sirens'][f'{kind} +10 dB']:.0f}% |")
    md += ["", f"**Horns** (synthetic two-tone horn bursts over road noise, {out['horns']['clips']} clips of 8 s): "
           f"{100 * out['horns']['clips_with_false_alarm']:.0f}% of clips raise a false alarm."]
    (res / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
