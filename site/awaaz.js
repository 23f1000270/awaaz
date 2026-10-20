// Awaaz in the browser: the same log-mel features and CNN as awaaz/features.py,
// awaaz/model.py and csrc/awaaz.c. Tests run this file in Node and compare.

export const SR = 16000, NFFT = 512, HOP = 256, MELS = 40, FRAMES = 64;
const EPS = 1e-6;

const hann = new Float32Array(NFFT);
for (let i = 0; i < NFFT; i++) hann[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / NFFT);

const hzToMel = (f) => 2595 * Math.log10(1 + f / 700);
const melToHz = (m) => 700 * (10 ** (m / 2595) - 1);
const melLo = new Int32Array(MELS), melW = [];
{
  const lo = hzToMel(100), hi = hzToMel(6000), edges = [];
  for (let i = 0; i < MELS + 2; i++) edges.push(melToHz(lo + ((hi - lo) * i) / (MELS + 1)));
  for (let m = 0; m < MELS; m++) {
    const w = [];
    melLo[m] = -1;
    for (let b = 0; b <= NFFT / 2; b++) {
      const f = (b * SR) / NFFT;
      const v = Math.min((f - edges[m]) / (edges[m + 1] - edges[m]), (edges[m + 2] - f) / (edges[m + 2] - edges[m + 1]));
      if (v > 0) { if (melLo[m] < 0) melLo[m] = b; w.push(v); }
    }
    melW.push(Float32Array.from(w));
  }
}

const cosT = new Float32Array(NFFT / 2), sinT = new Float32Array(NFFT / 2);
for (let k = 0; k < NFFT / 2; k++) { cosT[k] = Math.cos((2 * Math.PI * k) / NFFT); sinT[k] = Math.sin((2 * Math.PI * k) / NFFT); }

function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const step = NFFT / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < len / 2; k++) {
        const wr = cosT[k * step], wi = -sinT[k * step], a = i + k, b = a + len / 2;
        const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
      }
    }
  }
}

const re = new Float32Array(NFFT), im = new Float32Array(NFFT);
/** Log-mel of one 512-sample frame (Float32Array), written into out[0..39]. */
export function logmelFrame(frame, out = new Float32Array(MELS)) {
  for (let i = 0; i < NFFT; i++) { re[i] = frame[i] * hann[i]; im[i] = 0; }
  fft(re, im);
  for (let m = 0; m < MELS; m++) {
    let s = 0;
    const w = melW[m], b0 = melLo[m];
    for (let j = 0; j < w.length; j++) s += (re[b0 + j] * re[b0 + j] + im[b0 + j] * im[b0 + j]) * w[j];
    out[m] = Math.log(s + EPS);
  }
  return out;
}

// ------------------------------------------------------------------ CNN

let W = null;
export async function loadWeights(source) {
  const j = typeof source === "string" ? await (await fetch(source)).json() : source;
  W = {};
  for (const [k, v] of Object.entries(j)) W[k] = k === "threshold" ? v : Float32Array.from(v.data);
  return W.threshold;
}

function convReluPool(input, cin, h, w, wt, bias, cout, pool) {
  const out = new Float32Array(pool ? cout * (h >> 1) * (w >> 1) : cout * h * w);
  const tmp = new Float32Array(h * w);
  for (let o = 0; o < cout; o++) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let s = bias[o];
        for (let c = 0; c < cin; c++) {
          const k = (o * cin + c) * 9, src = c * h * w;
          for (let dy = -1; dy <= 1; dy++) {
            const yy = y + dy;
            if (yy < 0 || yy >= h) continue;
            for (let dx = -1; dx <= 1; dx++) {
              const xx = x + dx;
              if (xx < 0 || xx >= w) continue;
              s += wt[k + (dy + 1) * 3 + dx + 1] * input[src + yy * w + xx];
            }
          }
        }
        tmp[y * w + x] = s > 0 ? s : 0;
      }
    }
    if (pool) {
      const h2 = h >> 1, w2 = w >> 1;
      for (let y = 0; y < h2; y++)
        for (let x = 0; x < w2; x++)
          out[(o * h2 + y) * w2 + x] = Math.max(tmp[2 * y * w + 2 * x], tmp[2 * y * w + 2 * x + 1],
            tmp[(2 * y + 1) * w + 2 * x], tmp[(2 * y + 1) * w + 2 * x + 1]);
    } else out.set(tmp, o * h * w);
  }
  return out;
}

/** P(siren) for frames: an array of FRAMES Float32Array(MELS), oldest first. */
export function classify(frames) {
  let mean = 0;
  for (const f of frames) for (let m = 0; m < MELS; m++) mean += f[m];
  mean /= FRAMES * MELS;
  const x = new Float32Array(MELS * FRAMES);
  for (let m = 0; m < MELS; m++) for (let t = 0; t < FRAMES; t++) x[m * FRAMES + t] = frames[t][m] - mean;
  let a = convReluPool(x, 1, 40, 64, W.conv0_w, W.conv0_b, 16, true);
  a = convReluPool(a, 16, 20, 32, W.conv1_w, W.conv1_b, 32, true);
  a = convReluPool(a, 32, 10, 16, W.conv2_w, W.conv2_b, 32, true);
  a = convReluPool(a, 32, 5, 8, W.conv3_w, W.conv3_b, 48, false);
  let z = W.fc_b[0];
  for (let o = 0; o < 48; o++) {
    let s = 0;
    for (let i = 0; i < 40; i++) s += a[o * 40 + i];
    z += W.fc_w[o] * (s / 40);
  }
  return 1 / (1 + Math.exp(-z));
}

/** Streaming detector: feed 16 kHz samples, get a decision every 0.5 s. */
export class Detector {
  constructor(threshold) {
    this.threshold = threshold;
    this.buf = new Float32Array(NFFT);
    this.pending = [];
    this.frames = [];
    this.hops = 0;
    this.recent = [0, 0, 0];
  }
  push(samples, onDecision) {
    for (const s of samples) this.pending.push(s);
    while (this.pending.length >= HOP) {
      const hop = this.pending.splice(0, HOP);
      this.buf.copyWithin(0, HOP);
      this.buf.set(hop, NFFT - HOP);
      this.frames.push(logmelFrame(this.buf));
      if (this.frames.length > FRAMES) this.frames.shift();
      if (++this.hops >= 32 && this.frames.length === FRAMES) {
        this.hops = 0;
        const p = classify(this.frames);
        this.recent = [this.recent[1], this.recent[2], p];
        const alarm = this.recent.filter((v) => v >= this.threshold).length >= 2;
        onDecision({ p, alarm, frame: this.frames[FRAMES - 1] });
      } else if (this.frames.length) {
        onDecision(null, this.frames[this.frames.length - 1]);
      }
    }
  }
}
