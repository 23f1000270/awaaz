"""A small CNN over one second of log-mel features: about 28,000 weights.

Kept deliberately plain (3x3 convolutions, ReLU, 2x2 max-pooling, global
average pooling, one linear layer) so that it can be reimplemented exactly
in a few dozen lines of C for the ESP32-S3 and of JavaScript for the browser.
Batch norm is folded into the convolutions before export.
"""

from __future__ import annotations

import numpy as np

CHANNELS = (16, 32, 32, 48)


def build():
    import torch.nn as nn

    layers, c_in = [], 1
    for i, c in enumerate(CHANNELS):
        layers += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU()]
        if i < len(CHANNELS) - 1:
            layers.append(nn.MaxPool2d(2))
        c_in = c
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(c_in, 1)]
    return nn.Sequential(*layers)


def fold(model) -> dict[str, np.ndarray]:
    """Fold each BatchNorm into the convolution before it. Returns float32 arrays:
    conv{i}_w (out, in, 3, 3), conv{i}_b (out), fc_w (in,), fc_b ()."""
    import torch.nn as nn

    out, i = {}, 0
    mods = list(model)
    for k, m in enumerate(mods):
        if isinstance(m, nn.Conv2d):
            bn = mods[k + 1]
            scale = (bn.weight / (bn.running_var + bn.eps).sqrt()).detach()
            w = m.weight.detach() * scale[:, None, None, None]
            b = bn.bias.detach() - bn.running_mean * scale
            out[f"conv{i}_w"] = w.numpy().astype(np.float32)
            out[f"conv{i}_b"] = b.numpy().astype(np.float32)
            i += 1
        elif isinstance(m, nn.Linear):
            out["fc_w"] = m.weight.detach().numpy()[0].astype(np.float32)
            out["fc_b"] = m.bias.detach().numpy().astype(np.float32)
    return out


def _conv3x3(x, w, b):
    """x (C, H, W), zero padding 1."""
    c, h, wd = x.shape
    xp = np.zeros((c, h + 2, wd + 2), np.float32)
    xp[:, 1:-1, 1:-1] = x
    out = np.zeros((w.shape[0], h, wd), np.float32)
    for dy in range(3):
        for dx in range(3):
            out += np.einsum("oc,chw->ohw", w[:, :, dy, dx], xp[:, dy:dy + h, dx:dx + wd])
    return out + b[:, None, None]


def forward(weights: dict[str, np.ndarray], window: np.ndarray) -> float:
    """Reference forward pass in NumPy for one (FRAMES, N_MELS) window. Returns P(siren)."""
    x = window.T[None].astype(np.float32)          # (1, mels, frames)
    for i in range(len(CHANNELS)):
        x = np.maximum(_conv3x3(x, weights[f"conv{i}_w"], weights[f"conv{i}_b"]), 0)
        if i < len(CHANNELS) - 1:
            c, h, w = x.shape
            x = x[:, :h // 2 * 2, :w // 2 * 2].reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))
    z = float(x.mean(axis=(1, 2)) @ weights["fc_w"] + weights["fc_b"][0])
    return 1.0 / (1.0 + np.exp(-z))
