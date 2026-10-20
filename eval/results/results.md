Threshold 0.89 chosen on the validation split (5% of validation road clips alarm).

**Held-out recordings** (132 siren clips, 120 road clips, 20 minutes of road noise)

| Measure | Result |
|---|---:|
| Siren clips detected | 87.9% |
| Median time from start of clip to alarm | 1.6 s |
| Road clips with a false alarm | 0.0% |
| False alarms per hour of road noise | 0.0 |
| Window-level ROC AUC | 0.994 |

**Sirens mixed into other road noise** (detection rate)

| Signal-to-noise | Open air | Inside a closed car (simulated) |
|---|---:|---:|
| -10 dB | 30% | 34% |
| -5 dB | 63% | 64% |
| +0 dB | 89% | 83% |
| +5 dB | 92% | 92% |
| +10 dB | 92% | 92% |

**Synthetic Indian siren patterns, never seen in training** (40 clips each)

| Pattern | 0 dB | +10 dB |
|---|---:|---:|
| wail | 100% | 100% |
| yelp | 100% | 100% |
| hilo | 88% | 98% |

**Horns** (synthetic two-tone horn bursts over road noise, 120 clips of 8 s): 0% of clips raise a false alarm.
