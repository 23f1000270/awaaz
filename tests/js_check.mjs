// Run site/awaaz.js in Node on numbers written by the Python tests.
//   node tests/js_check.mjs features in.f32 out.f32
//   node tests/js_check.mjs classify in.f32          (64 x 40 window)
//   node tests/js_check.mjs stream in.f32            (16 kHz audio) -> number of decisions with alarm
import { readFileSync, writeFileSync } from "node:fs";
import { classify, Detector, FRAMES, HOP, logmelFrame, MELS, NFFT, loadWeights } from "../site/awaaz.js";

const [mode, inPath, outPath] = process.argv.slice(2);
const buf = readFileSync(inPath);
const x = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
const weights = JSON.parse(readFileSync(new URL("../site/weights.json", import.meta.url)));
const threshold = await loadWeights(weights);

if (mode === "features") {
  const n = 1 + Math.floor((x.length - NFFT) / HOP);
  const out = new Float32Array(n * MELS);
  for (let i = 0; i < n; i++) out.set(logmelFrame(x.subarray(i * HOP, i * HOP + NFFT)), i * MELS);
  writeFileSync(outPath, Buffer.from(out.buffer));
} else if (mode === "classify") {
  const frames = [];
  for (let t = 0; t < FRAMES; t++) frames.push(x.subarray(t * MELS, (t + 1) * MELS));
  console.log(classify(frames).toFixed(7));
} else if (mode === "stream") {
  const d = new Detector(threshold);
  let alarms = 0, decisions = 0;
  for (let i = 0; i < x.length; i += 1600) d.push(x.subarray(i, i + 1600), (r) => { if (r) { decisions++; alarms += r.alarm ? 1 : 0; } });
  console.log(`${alarms} ${decisions}`);
}
