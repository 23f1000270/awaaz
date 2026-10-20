/* Awaaz on an ESP32-S3 with two INMP441 I2S microphones.
 *
 * Wiring (both microphones on one I2S bus, as left and right slots):
 *   SCK  -> GPIO 4      WS -> GPIO 5      SD -> GPIO 6
 *   mic A: L/R pin to GND (left slot), mounted on the vehicle's left
 *   mic B: L/R pin to 3V3 (right slot), mounted on the vehicle's right
 *   left LED -> GPIO 1, right LED -> GPIO 2 (each through a resistor)
 *   buzzer -> GPIO 7 via an NPN transistor
 * Mount the microphones MIC_SPACING_M apart on a left-right line, for
 * example at the two ends of a dashboard bar or on each side of a helmet.
 *
 * Two tasks, one per core, so listening never stops while the network runs:
 *   audio (core 0): reads the microphones and computes one log-mel frame
 *                   every 16 ms from the left microphone; every 0.512 s it
 *                   wakes the inference task.
 *   infer (core 1): copies the last second of frames and the latest audio,
 *                   runs the CNN, applies the two-of-three rule and, while
 *                   the alarm is on, works out the side and the approach.
 * Output: the buzzer sounds and the LED on the siren's side lights (both LEDs
 * when it is straight ahead or behind), plus a line on the serial console.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "awaaz.h"
#include "driver/gpio.h"
#include "driver/i2s_std.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

#define PIN_BCLK GPIO_NUM_4
#define PIN_WS GPIO_NUM_5
#define PIN_DIN GPIO_NUM_6
#define PIN_LED_LEFT GPIO_NUM_1
#define PIN_LED_RIGHT GPIO_NUM_2
#define PIN_BUZZ GPIO_NUM_7

#define MIC_SPACING_M 0.15f
#define DIR_SAMPLES 1024        /* 64 ms per channel for GCC-PHAT */
#define LEVEL_SAMPLES 2048      /* 128 ms for the siren-band level */
#define LEVELS 6                /* about 2.5 s of levels for the trend */
#define STEP_S 0.512f           /* 32 hops of 256 samples between decisions */

static const char *TAG = "awaaz";
static i2s_chan_handle_t rx;
static SemaphoreHandle_t lock;
static TaskHandle_t infer_handle;

/* Written by the audio task under `lock`. */
static float window[AWZ_FRAMES][AWZ_MELS];
static float left[LEVEL_SAMPLES], right[LEVEL_SAMPLES];
static int frames_ready;

/* The inference task's own copies. */
static float win_copy[AWZ_FRAMES][AWZ_MELS];
static float l_copy[LEVEL_SAMPLES], r_copy[LEVEL_SAMPLES];

static void push(float *buf, int len, const float *in, int n) {
    memmove(buf, buf + n, sizeof(float) * (len - n));
    memcpy(buf + len - n, in, sizeof(float) * n);
}

static void i2s_start(void) {
    i2s_chan_config_t chan = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    chan.dma_desc_num = 6;
    chan.dma_frame_num = 256;          /* 96 ms of buffering for the audio task */
    ESP_ERROR_CHECK(i2s_new_channel(&chan, NULL, &rx));
    i2s_std_config_t cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(AWZ_SR),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {.mclk = I2S_GPIO_UNUSED, .bclk = PIN_BCLK, .ws = PIN_WS, .dout = I2S_GPIO_UNUSED,
                     .din = PIN_DIN, .invert_flags = {0}},
    };
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx, &cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx));
}

static void audio_task(void *arg) {
    static int32_t raw[AWZ_HOP * 2];
    static float mono[AWZ_NFFT];
    float l[AWZ_HOP], r[AWZ_HOP], m[AWZ_HOP], frame[AWZ_MELS];
    int hops = 0;
    for (;;) {
        size_t got = 0;
        if (i2s_channel_read(rx, raw, sizeof raw, &got, portMAX_DELAY) != ESP_OK || got != sizeof raw) continue;
        for (int i = 0; i < AWZ_HOP; i++) {
            l[i] = (float)(raw[2 * i] >> 8) / 8388608.0f;       /* 24-bit samples in 32-bit slots */
            r[i] = (float)(raw[2 * i + 1] >> 8) / 8388608.0f;
            m[i] = l[i];   /* features from one microphone: mixing two 15 cm apart would comb-filter the siren */
        }
        push(mono, AWZ_NFFT, m, AWZ_HOP);
        awz_logmel_frame(mono, frame);
        xSemaphoreTake(lock, portMAX_DELAY);
        memmove(window[0], window[1], sizeof(float) * AWZ_MELS * (AWZ_FRAMES - 1));
        memcpy(window[AWZ_FRAMES - 1], frame, sizeof frame);
        push(left, LEVEL_SAMPLES, l, AWZ_HOP);
        push(right, LEVEL_SAMPLES, r, AWZ_HOP);
        if (frames_ready < AWZ_FRAMES) frames_ready++;
        int full = frames_ready == AWZ_FRAMES;
        xSemaphoreGive(lock);
        if (++hops >= 32 && full) {                             /* every 0.5 s */
            hops = 0;
            xTaskNotifyGive(infer_handle);
        }
    }
}

static void infer_task(void *arg) {
    float recent[3] = {0}, levels[LEVELS] = {0};
    int nlev = 0;
    const float thr = awz_threshold();
    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        xSemaphoreTake(lock, portMAX_DELAY);
        memcpy(win_copy, window, sizeof window);
        memcpy(l_copy, left, sizeof left);
        memcpy(r_copy, right, sizeof right);
        xSemaphoreGive(lock);

        int64_t t0 = esp_timer_get_time();
        float p = awz_classify((const float (*)[AWZ_MELS])win_copy);
        int64_t ms = (esp_timer_get_time() - t0) / 1000;
        recent[0] = recent[1];
        recent[1] = recent[2];
        recent[2] = p;
        int alarm = (recent[0] >= thr) + (recent[1] >= thr) + (recent[2] >= thr) >= 2;

        memmove(levels, levels + 1, sizeof(float) * (LEVELS - 1));
        levels[LEVELS - 1] = awz_band_level_db(l_copy, LEVEL_SAMPLES);
        if (nlev < LEVELS) nlev++;
        gpio_set_level(PIN_BUZZ, alarm);
        if (!alarm) {
            gpio_set_level(PIN_LED_LEFT, 0);
            gpio_set_level(PIN_LED_RIGHT, 0);
            printf("quiet p=%.2f (%lld ms)\n", p, (long long)ms);
            continue;
        }
        float peak = 0.0f, max_tau = MIC_SPACING_M / 343.0f;
        float tau = awz_gcc_phat(l_copy + LEVEL_SAMPLES - DIR_SAMPLES, r_copy + LEVEL_SAMPLES - DIR_SAMPLES,
                                 DIR_SAMPLES, max_tau, &peak);
        float s = tau / max_tau;
        s = s > 1.0f ? 1.0f : (s < -1.0f ? -1.0f : s);
        float angle = asinf(s) * 57.29578f;
        const char *side = fabsf(angle) < 10.0f ? "ahead/behind" : (angle > 0 ? "left" : "right");
        gpio_set_level(PIN_LED_LEFT, angle > -10.0f);    /* both lights when ahead or behind */
        gpio_set_level(PIN_LED_RIGHT, angle < 10.0f);
        float slope = awz_trend(levels + LEVELS - nlev, nlev, STEP_S);
        const char *motion = slope > 1.0f ? "approaching" : (slope < -1.0f ? "moving away" : "steady");
        printf("SIREN p=%.2f side=%s angle=%.0f trend=%+.1fdB/s %s (%lld ms)\n", p, side, angle, slope, motion,
               (long long)ms);
    }
}

void app_main(void) {
    awz_init();
    lock = xSemaphoreCreateMutex();
    const gpio_num_t outputs[] = {PIN_LED_LEFT, PIN_LED_RIGHT, PIN_BUZZ};
    for (int i = 0; i < 3; i++) {
        gpio_reset_pin(outputs[i]);
        gpio_set_direction(outputs[i], GPIO_MODE_OUTPUT);
        gpio_set_level(outputs[i], 0);
    }
    i2s_start();
    ESP_LOGI(TAG, "listening: 16 kHz, two microphones %.0f cm apart", MIC_SPACING_M * 100);
    xTaskCreatePinnedToCore(infer_task, "infer", 6144, NULL, 5, &infer_handle, 1);
    xTaskCreatePinnedToCore(audio_task, "audio", 6144, NULL, 10, NULL, 0);
}
