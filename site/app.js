import { Detector, MELS, SR, loadWeights } from "./awaaz.js";

const $ = (id) => document.getElementById(id);
const canvas = $("spec"), ctx = canvas.getContext("2d");
const COLS = 360;                               // about 5.8 s of history, one column per frame
const probs = new Float32Array(COLS).fill(NaN);
let threshold = 0.5, detector = null, audio = null, stopFns = [];

function setStatus(state, text, p) {
  $("status").dataset.state = state;
  $("status-text").textContent = text;
  $("prob").textContent = p === undefined ? "" : `P(siren) ${p.toFixed(2)}`;
}

let top = -10;                                  // running maximum of the log-mel values
function colour(v) {                            // log-mel value to a dark-to-amber colour over 9 units of ln(power), about 39 dB
  const t = Math.max(0, Math.min(1, (v - (top - 9)) / 9));
  return `rgb(${Math.round(255 * Math.min(1, t * 1.6))},${Math.round(200 * t * t)},${Math.round(60 * t)})`;
}

function drawFrame(frame) {
  const w = canvas.width, h = canvas.height, colW = w / COLS, strip = 56, specH = h - strip;
  top = Math.max(top - 0.02, ...frame);
  ctx.drawImage(canvas, -colW, 0);
  for (let m = 0; m < MELS; m++) {
    ctx.fillStyle = colour(frame[m]);
    ctx.fillRect(w - colW, specH - ((m + 1) * specH) / MELS, colW + 1, specH / MELS + 1);
  }
  probs.copyWithin(0, 1);
  probs[COLS - 1] = probs[COLS - 2];
  const y = (p) => h - 4 - (strip - 8) * p;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, specH, w, strip);
  ctx.lineWidth = 1;
  ctx.strokeStyle = "#555";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(0, y(threshold)); ctx.lineTo(w, y(threshold));
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#ff4d4d";
  ctx.beginPath();
  probs.forEach((p, i) => { if (!Number.isNaN(p)) ctx.lineTo(i * colW, y(p)); });
  ctx.stroke();
  ctx.fillStyle = "#9aa3ae";
  ctx.font = "12px JetBrains Mono, monospace";
  ctx.fillText("P(siren)", 6, specH + 14);
}

function feed(samples) {
  detector.push(samples, (d, frame) => {
    if (d) {
      probs[COLS - 1] = d.p;
      drawFrame(d.frame);
      setStatus(d.alarm ? "siren" : "listening", d.alarm ? "Siren" : "Listening", d.p);
    } else if (frame) drawFrame(frame);
  });
}

// Box-filter resampler to 16 kHz, for browsers that will not run a 16 kHz context
// on a microphone (Firefox, for one).
function resampler(fromRate) {
  const step = fromRate / SR;
  let pos = 0, carry = new Float32Array(0);
  return (chunk) => {
    const x = new Float32Array(carry.length + chunk.length);
    x.set(carry);
    x.set(chunk, carry.length);
    const out = [];
    while (pos + step <= x.length) {
      let s = 0, n = 0;
      for (let i = Math.floor(pos); i < Math.floor(pos + step); i++) { s += x[i]; n++; }
      out.push(n ? s / n : 0);
      pos += step;
    }
    const keep = Math.floor(pos);
    carry = x.slice(keep);
    pos -= keep;
    return Float32Array.from(out);
  };
}

function newContext(rate) {
  // Called synchronously inside the click handler, so the browser lets it start.
  try { return rate ? new AudioContext({ sampleRate: rate }) : new AudioContext(); } catch { return new AudioContext(); }
}

async function start(context) {
  audio = context;
  await audio.resume();
  threshold = await loadWeights("weights.json");
  detector = new Detector(threshold);
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  $("stop").disabled = false;
}

function stop() {
  stopFns.forEach((f) => f());
  stopFns = [];
  if (audio) audio.close();
  audio = null;
  $("stop").disabled = true;
  setStatus("idle", "Not listening");
}

$("mic").addEventListener("click", async () => {
  stop();
  let context = newContext(SR);
  try {
    const streamPromise = navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    await start(context);
    const stream = await streamPromise;
    let src;
    try {
      src = audio.createMediaStreamSource(stream);
    } catch {
      audio.close();                     // this browser cannot feed a 16 kHz context from the mic
      context = newContext(null);
      await start(context);
      src = audio.createMediaStreamSource(stream);
    }
    const toSR = audio.sampleRate === SR ? (x) => x : resampler(audio.sampleRate);
    const proc = audio.createScriptProcessor(1024, 1, 1);
    proc.onaudioprocess = (e) => feed(toSR(e.inputBuffer.getChannelData(0)));
    src.connect(proc);
    proc.connect(audio.destination);
    setStatus("listening", "Listening");
    stopFns.push(() => stream.getTracks().forEach((t) => t.stop()));
  } catch (e) {
    setStatus("idle", "Microphone not available: " + e.message);
  }
});

function testSignal(seconds) {
  // A wail siren that fades in, as if approaching, over brown-ish traffic noise.
  const n = seconds * SR, out = new Float32Array(n);
  let phase = 0, brown = 0;
  for (let i = 0; i < n; i++) {
    const t = i / SR;
    const f = 650 + 850 * 0.5 * (1 - Math.cos((2 * Math.PI * t) / 4.2));
    phase += (2 * Math.PI * f) / SR;
    const s = Math.sin(phase) + 0.45 * Math.sin(2 * phase) + 0.25 * Math.sin(3 * phase);
    const env = t < 3 ? 0 : Math.min(1, (t - 3) / 6);
    brown = 0.98 * brown + 0.2 * (Math.random() * 2 - 1);
    out[i] = 0.12 * env * s + 0.25 * brown;
  }
  return out;
}

$("test").addEventListener("click", async () => {
  stop();
  await start(newContext(SR));
  const sig = testSignal(14);
  const buf = audio.createBuffer(1, sig.length, SR);
  buf.copyToChannel(sig, 0);
  const node = audio.createBufferSource();
  node.buffer = buf;
  node.connect(audio.destination);
  node.start();
  setStatus("listening", "Listening");
  let pos = 0;
  const step = SR / 10;
  const timer = setInterval(() => {
    if (pos >= sig.length) { clearInterval(timer); return; }
    feed(sig.subarray(pos, pos + step));
    pos += step;
  }, 100);
  stopFns.push(() => { clearInterval(timer); try { node.stop(); } catch { /* already stopped */ } });
});

$("stop").addEventListener("click", stop);
