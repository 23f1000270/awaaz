"""The Python reference, the C firmware code and the browser JavaScript must agree."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from awaaz import direction, synth
from awaaz.features import FRAMES, N_MELS, SR, WINDOW_SAMPLES, logmel, normalise, windows
from awaaz.model import forward

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / "build" / "host_test"
WEIGHTS = ROOT / "awaaz" / "weights.npz"
NODE = shutil.which("node")


@pytest.fixture(scope="session", autouse=True)
def host_binary():
    subprocess.run(["make", "-s", "build/host_test"], cwd=ROOT, check=True)


@pytest.fixture(scope="session")
def weights():
    return dict(np.load(WEIGHTS))


def run_c(*args):
    return subprocess.run([str(HOST), *map(str, args)], check=True, capture_output=True, text=True).stdout.split()


def run_js(*args):
    return subprocess.run([NODE, str(ROOT / "tests" / "js_check.mjs"), *map(str, args)], check=True,
                          capture_output=True, text=True).stdout.split()


def test_features_match_in_c(tmp_path):
    rng = np.random.default_rng(0)
    x = (0.3 * rng.standard_normal(SR) + synth.siren("wail", 1.0, rng) * 0.5).astype(np.float32)
    x.tofile(tmp_path / "a.f32")
    run_c("features", tmp_path / "a.f32", tmp_path / "c.f32")
    c = np.fromfile(tmp_path / "c.f32", np.float32).reshape(-1, N_MELS)
    py = logmel(x)
    assert c.shape == py.shape
    assert np.max(np.abs(c - py)) < 1e-4


@pytest.mark.skipif(NODE is None, reason="Node.js not installed")
def test_features_match_in_javascript(tmp_path):
    rng = np.random.default_rng(1)
    x = (0.2 * rng.standard_normal(SR)).astype(np.float32)
    x.tofile(tmp_path / "a.f32")
    run_js("features", tmp_path / "a.f32", tmp_path / "j.f32")
    j = np.fromfile(tmp_path / "j.f32", np.float32).reshape(-1, N_MELS)
    assert np.max(np.abs(j - logmel(x))) < 1e-4


def test_classifier_matches_everywhere(weights, tmp_path):
    rng = np.random.default_rng(2)
    for kind in ("wail", "yelp", None):
        x = 0.1 * rng.standard_normal(WINDOW_SAMPLES).astype(np.float32)
        if kind:
            x = synth.mix(synth.siren(kind, WINDOW_SAMPLES / SR, rng), x, 5)
        win = logmel(x)[:FRAMES]
        ref = forward(weights, normalise(win))
        win.astype(np.float32).tofile(tmp_path / "w.f32")
        assert abs(float(run_c("classify", tmp_path / "w.f32")[0]) - ref) < 1e-4
        if NODE:
            assert abs(float(run_js("classify", tmp_path / "w.f32")[0]) - ref) < 1e-4


def test_gcc_phat_matches_and_finds_the_side(tmp_path):
    rng = np.random.default_rng(3)
    s = synth.mix(synth.siren("yelp", 0.5, rng), 0.05 * rng.standard_normal(int(0.5 * SR)).astype(np.float32), 10)
    delay = 5                                   # samples: the left microphone hears it first
    left, right = s[delay:delay + 1024], s[:1024]
    tau, _ = direction.gcc_phat(left, right, 0.2 / 343)
    assert abs(tau * SR - delay) < 0.3
    np.concatenate([left, right]).astype(np.float32).tofile(tmp_path / "lr.f32")
    c_tau, c_peak = map(float, run_c("gcc", tmp_path / "lr.f32", 1024, 0.2 / 343))
    assert abs(c_tau - tau) < 2e-6 and 0.3 < c_peak <= 1.05
    assert direction.side(left, right, 0.2)[0] == "left"
    assert direction.side(right, left, 0.2)[0] == "right"


def test_band_level_and_trend(tmp_path):
    rng = np.random.default_rng(4)
    x = synth.siren("wail", 0.128, rng)
    x.astype(np.float32).tofile(tmp_path / "x.f32")
    assert abs(float(run_c("level", tmp_path / "x.f32")[0]) - direction.band_level_db(x)) < 1e-4
    # More samples than the C buffers hold: C uses the most recent 2048 and must not overrun.
    long = synth.siren("wail", 0.5, rng)
    long.astype(np.float32).tofile(tmp_path / "long.f32")
    assert abs(float(run_c("level", tmp_path / "long.f32")[0]) - direction.band_level_db(long[-2048:])) < 1e-4
    assert direction.approach([50, 51.5, 53, 54.6, 56, 57.4], 0.5) == "approaching"
    assert direction.approach([57, 55.5, 54, 52.6, 51, 49.4], 0.5) == "moving away"


def test_detects_a_siren_and_ignores_horns(weights):
    import json
    rng = np.random.default_rng(5)
    thr = json.loads((ROOT / "site" / "weights.json").read_text())["threshold"]
    noise = 0.05 * np.cumsum(rng.standard_normal(6 * SR)).astype(np.float32)
    noise -= np.convolve(noise, np.ones(400) / 400, mode="same")       # brown-ish traffic rumble
    siren = synth.mix(synth.siren("wail", 6.0, rng), noise, 5)
    p_siren = [forward(weights, w) for w in windows(siren)]
    p_horn = [forward(weights, w) for w in windows(synth.mix(noise, synth.horn(6.0, rng), 0))]
    assert np.mean(np.array(p_siren) > thr) > 0.7
    assert np.mean(np.array(p_horn) > thr) < 0.3


@pytest.mark.skipif(NODE is None, reason="Node.js not installed")
def test_browser_streaming_detector_raises_an_alarm(tmp_path):
    rng = np.random.default_rng(6)
    noise = 0.05 * rng.standard_normal(8 * SR).astype(np.float32)
    synth.mix(synth.siren("wail", 8.0, rng), noise, 8).astype(np.float32).tofile(tmp_path / "s.f32")
    alarms, decisions = map(int, run_js("stream", tmp_path / "s.f32"))
    assert decisions >= 12 and alarms >= decisions // 2
    noise.tofile(tmp_path / "n.f32")
    alarms, _ = map(int, run_js("stream", tmp_path / "n.f32"))
    assert alarms == 0
