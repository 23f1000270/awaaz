/* Awaaz: siren detection and direction on a microcontroller.
 *
 * Plain C99 with no dependencies, so the same file builds for the ESP32-S3
 * firmware and for the host tests that compare it against the Python
 * reference (awaaz/features.py, awaaz/model.py, awaaz/direction.py).
 */
#ifndef AWAAZ_H
#define AWAAZ_H

#ifdef __cplusplus
extern "C" {
#endif

#define AWZ_SR 16000
#define AWZ_NFFT 512
#define AWZ_HOP 256
#define AWZ_MELS 40
#define AWZ_FRAMES 64
#define AWZ_MAX_FFT 2048

/* Call once before anything else: builds the window, twiddles and mel filters. */
void awz_init(void);

/* In-place complex FFT of size n (a power of two, at most AWZ_MAX_FFT). */
void awz_fft(float *re, float *im, int n, int inverse);

/* Log-mel of one 512-sample frame. */
void awz_logmel_frame(const float *frame, float out[AWZ_MELS]);

/* P(siren) for one window of AWZ_FRAMES log-mel frames, oldest first.
 * The window is mean-normalised inside, as in the Python reference. */
float awz_classify(const float window[AWZ_FRAMES][AWZ_MELS]);

/* Decision threshold for one window, chosen on the validation split. */
float awz_threshold(void);

/* GCC-PHAT delay of right relative to left in seconds (positive: the left
 * microphone heard it first). n samples per channel, n * 2 <= AWZ_MAX_FFT. */
float awz_gcc_phat(const float *left, const float *right, int n, float max_tau, float *peak);

/* Siren-band (400 to 4000 Hz) level in dB of n samples. Beyond AWZ_MAX_FFT
 * samples, only the most recent AWZ_MAX_FFT are used. */
float awz_band_level_db(const float *x, int n);

/* Least-squares slope in dB per second of `count` levels spaced step_s apart. */
float awz_trend(const float *levels_db, int count, float step_s);

#ifdef __cplusplus
}
#endif
#endif
