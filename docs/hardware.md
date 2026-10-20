# Hardware and how we will test it

## Parts (about ₹1,050)

| Part | Qty | Approx. cost | Notes |
|---|---:|---:|---|
| ESP32-S3 development board (e.g. ESP32-S3-DevKitC-1) | 1 | ₹500 | runs detection, direction and alerts |
| INMP441 I2S MEMS microphone | 2 | ₹150 each | one set to the left slot (L/R to GND), one to the right (L/R to 3V3) |
| Active 5 V buzzer, NPN transistor (e.g. 2N2222) and 1 kΩ base resistor | 1 each | ₹40 | the sound |
| LED and resistor | 2 each | ₹10 | left and right lights |
| USB-C cable and a 12 V to USB car adapter | 1 | ₹200 | power |

## Wiring

| INMP441 | ESP32-S3 |
|---|---|
| SCK (both mics) | GPIO 4 |
| WS (both mics) | GPIO 5 |
| SD (both mics) | GPIO 6 |
| L/R | mic A to GND, mic B to 3V3 |
| VDD, GND | 3V3, GND |

Left LED on GPIO 1 and right LED on GPIO 2, each through a 330 Ω resistor. Put each LED on its side of the dashboard or helmet. When a siren is detected, the LED on its side lights up. Both light when it is straight ahead or behind. The buzzer runs from 5 V through an NPN transistor whose base is driven from GPIO 7 through a 1 kΩ resistor. A GPIO pin cannot power a buzzer directly.

Mount the two microphones 15 cm apart on a left-right line, facing outwards: at the ends of a small bar on the dashboard, or on each side of a helmet. **Mic A (L/R to GND) goes on the left, mic B (L/R to 3V3) on the right.** Swap them and every left becomes right. Set `MIC_SPACING_M` in `firmware/main/main.c` to the real spacing. Detection uses mic A alone.

## Build and flash

```bash
cd firmware
idf.py set-target esp32s3
idf.py -p PORT flash monitor
```

The console prints one line every half second, for example
`SIREN p=0.94 side=left angle=38 trend=+2.3dB/s approaching (… ms)`.
The time in brackets is how long the CNN took on the chip. It has not been measured yet. The network needs about 5.3 million multiply-adds per decision. The first hardware test will report the time.

## Test plan

Never stop or follow a real ambulance for a test. Record the ones that pass near a hospital, from a parked car or the roadside.

1. **Bench.** Play test-split siren clips from a speaker 2 m away at known angles (every 30°). Record detection, side and angle error. Then play road-noise clips and horn recordings for an hour to count false alarms.
2. **Parked car, windows up.** The same, with the speaker outside the car at 5, 10 and 20 m. This is the case the simulation calls "inside a closed car".
3. **Roadside log.** Leave the device on a parked car near a hospital for a few hours with a phone filming the road. Every alarm and every ambulance on the video is matched by hand: missed sirens, false alarms, warning time before the ambulance is in view.
4. **Riding.** Microphones on a helmet, riding at city speed. Wind noise is the open question.

Results will be added to the README, including the misses, with the minutes of audio behind every number.
