#include "awaaz.h"

#include <math.h>
#include <string.h>

#include "awaaz_weights.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float hann[AWZ_NFFT];
static float cos_t[AWZ_MAX_FFT / 2], sin_t[AWZ_MAX_FFT / 2];
/* Mel filters stored sparsely: for each band its first bin, its length and an
 * offset into one packed array of weights (about 2 weights per FFT bin). */
#define MEL_PACKED 640
static int mel_lo[AWZ_MELS], mel_len[AWZ_MELS], mel_off[AWZ_MELS];
static float mel_w[MEL_PACKED];
/* Shared scratch for the direction code (used one call at a time). */
static float scratch[4][AWZ_MAX_FFT];

static double hz_to_mel(double f) { return 2595.0 * log10(1.0 + f / 700.0); }
static double mel_to_hz(double m) { return 700.0 * (pow(10.0, m / 2595.0) - 1.0); }

void awz_init(void) {
    for (int i = 0; i < AWZ_NFFT; i++) hann[i] = (float)(0.5 - 0.5 * cos(2.0 * M_PI * i / AWZ_NFFT));
    for (int k = 0; k < AWZ_MAX_FFT / 2; k++) {
        cos_t[k] = (float)cos(2.0 * M_PI * k / AWZ_MAX_FFT);
        sin_t[k] = (float)sin(2.0 * M_PI * k / AWZ_MAX_FFT);
    }
    double lo = hz_to_mel(100.0), hi = hz_to_mel(6000.0), edges[AWZ_MELS + 2];
    for (int i = 0; i < AWZ_MELS + 2; i++) edges[i] = mel_to_hz(lo + (hi - lo) * i / (AWZ_MELS + 1));
    int used = 0;
    for (int m = 0; m < AWZ_MELS; m++) {
        mel_lo[m] = -1;
        mel_len[m] = 0;
        mel_off[m] = used;
        for (int b = 0; b <= AWZ_NFFT / 2 && used < MEL_PACKED; b++) {
            double f = (double)b * AWZ_SR / AWZ_NFFT;
            double up = (f - edges[m]) / (edges[m + 1] - edges[m]);
            double down = (edges[m + 2] - f) / (edges[m + 2] - edges[m + 1]);
            double w = up < down ? up : down;
            if (w > 0.0) {
                if (mel_lo[m] < 0) mel_lo[m] = b;
                mel_w[used++] = (float)w;
                mel_len[m]++;
            }
        }
    }
}

void awz_fft(float *re, float *im, int n, int inverse) {
    for (int i = 1, j = 0; i < n; i++) {           /* bit reversal */
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        int step = AWZ_MAX_FFT / len;
        for (int i = 0; i < n; i += len) {
            for (int k = 0; k < len / 2; k++) {
                float wr = cos_t[k * step], wi = inverse ? sin_t[k * step] : -sin_t[k * step];
                int a = i + k, b = i + k + len / 2;
                float xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
                re[b] = re[a] - xr; im[b] = im[a] - xi;
                re[a] += xr; im[a] += xi;
            }
        }
    }
    if (inverse)
        for (int i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

void awz_logmel_frame(const float *frame, float out[AWZ_MELS]) {
    static float re[AWZ_NFFT], im[AWZ_NFFT], pw[AWZ_NFFT / 2 + 1];
    for (int i = 0; i < AWZ_NFFT; i++) { re[i] = frame[i] * hann[i]; im[i] = 0.0f; }
    awz_fft(re, im, AWZ_NFFT, 0);
    for (int b = 0; b <= AWZ_NFFT / 2; b++) pw[b] = re[b] * re[b] + im[b] * im[b];
    for (int m = 0; m < AWZ_MELS; m++) {
        float s = 0.0f;
        for (int j = 0; j < mel_len[m]; j++) s += pw[mel_lo[m] + j] * mel_w[mel_off[m] + j];
        out[m] = logf(s + 1e-6f);
    }
}

/* ---------------------------------------------------------------- the CNN */

/* conv 3x3 (zero padding) + ReLU, then 2x2 max-pool if pool is set.
 * in: (cin, h, w); out: (cout, h/2, w/2) or (cout, h, w). */
static void conv_relu_pool(const float *in, int cin, int h, int w, const float *wt, const float *bias,
                           int cout, int pool, float *out) {
    static float tmp[AWZ_MELS * AWZ_FRAMES];
    for (int o = 0; o < cout; o++) {
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                float s = bias[o];
                for (int c = 0; c < cin; c++) {
                    const float *k = wt + ((o * cin + c) * 9);
                    const float *src = in + c * h * w;
                    for (int dy = -1; dy <= 1; dy++) {
                        int yy = y + dy;
                        if (yy < 0 || yy >= h) continue;
                        for (int dx = -1; dx <= 1; dx++) {
                            int xx = x + dx;
                            if (xx < 0 || xx >= w) continue;
                            s += k[(dy + 1) * 3 + (dx + 1)] * src[yy * w + xx];
                        }
                    }
                }
                tmp[y * w + x] = s > 0.0f ? s : 0.0f;
            }
        if (pool) {
            int h2 = h / 2, w2 = w / 2;
            for (int y = 0; y < h2; y++)
                for (int x = 0; x < w2; x++) {
                    float a = tmp[(2 * y) * w + 2 * x], b = tmp[(2 * y) * w + 2 * x + 1];
                    float c = tmp[(2 * y + 1) * w + 2 * x], d = tmp[(2 * y + 1) * w + 2 * x + 1];
                    float m = a > b ? a : b;
                    m = m > c ? m : c;
                    out[(o * h2 + y) * w2 + x] = m > d ? m : d;
                }
        } else {
            memcpy(out + o * h * w, tmp, sizeof(float) * h * w);
        }
    }
}

float awz_classify(const float window[AWZ_FRAMES][AWZ_MELS]) {
    static float a[16 * 20 * 32], b[32 * 10 * 16];
    static float in[AWZ_MELS * AWZ_FRAMES];
    double mean = 0.0;
    for (int t = 0; t < AWZ_FRAMES; t++)
        for (int m = 0; m < AWZ_MELS; m++) mean += window[t][m];
    mean /= AWZ_FRAMES * AWZ_MELS;
    for (int m = 0; m < AWZ_MELS; m++)          /* (mels, frames), as in the reference */
        for (int t = 0; t < AWZ_FRAMES; t++) in[m * AWZ_FRAMES + t] = window[t][m] - (float)mean;
    conv_relu_pool(in, 1, 40, 64, conv0_w, conv0_b, 16, 1, a);    /* -> 16 x 20 x 32 */
    conv_relu_pool(a, 16, 20, 32, conv1_w, conv1_b, 32, 1, b);    /* -> 32 x 10 x 16 */
    conv_relu_pool(b, 32, 10, 16, conv2_w, conv2_b, 32, 1, a);    /* -> 32 x 5 x 8 */
    conv_relu_pool(a, 32, 5, 8, conv3_w, conv3_b, 48, 0, b);      /* -> 48 x 5 x 8 */
    float z = fc_b[0];
    for (int o = 0; o < 48; o++) {
        float s = 0.0f;
        for (int i = 0; i < 40; i++) s += b[o * 40 + i];
        z += fc_w[o] * (s / 40.0f);
    }
    return 1.0f / (1.0f + expf(-z));
}

float awz_threshold(void) { return AWZ_THRESHOLD; }

/* ---------------------------------------------------------- direction */

float awz_gcc_phat(const float *left, const float *right, int n, float max_tau, float *peak) {
    float *lr = scratch[0], *li = scratch[1], *rr = scratch[2], *ri = scratch[3];
    int N = 1;
    while (N < 2 * n) N <<= 1;
    if (N > AWZ_MAX_FFT) N = AWZ_MAX_FFT;
    memset(lr, 0, sizeof(float) * N); memset(li, 0, sizeof(float) * N);
    memset(rr, 0, sizeof(float) * N); memset(ri, 0, sizeof(float) * N);
    for (int i = 0; i < n && i < N; i++) { lr[i] = left[i]; rr[i] = right[i]; }
    awz_fft(lr, li, N, 0);
    awz_fft(rr, ri, N, 0);
    int in_band = 0;
    for (int k = 0; k < N; k++) {
        int kk = k <= N / 2 ? k : N - k;              /* frequency bin of k */
        float f = (float)kk * AWZ_SR / N;
        float gr = rr[k] * lr[k] + ri[k] * li[k];      /* R * conj(L) */
        float gi = ri[k] * lr[k] - rr[k] * li[k];
        float mag = sqrtf(gr * gr + gi * gi) + 1e-12f;
        if (f < 400.0f || f > 4000.0f) { gr = 0.0f; gi = 0.0f; }
        else if (k <= N / 2) in_band++;
        lr[k] = gr / mag; li[k] = gi / mag;
    }
    awz_fft(lr, li, N, 1);
    int m = (int)ceilf(max_tau * AWZ_SR), best = -m;
    float bv = -1e30f;
    for (int lag = -m; lag <= m; lag++) {
        float v = lr[(lag + N) % N];
        if (v > bv) { bv = v; best = lag; }
    }
    float lag = (float)best;
    if (best > -m && best < m) {
        float a = lr[(best - 1 + N) % N], b = bv, c = lr[(best + 1 + N) % N];
        float den = a - 2.0f * b + c;
        if (den != 0.0f) lag += 0.5f * (a - c) / den;
    }
    if (peak) *peak = in_band ? bv * N / (2.0f * in_band) : 0.0f;
    return lag / AWZ_SR;
}

float awz_band_level_db(const float *x, int n) {
    float *re = scratch[0], *im = scratch[1];
    if (n > AWZ_MAX_FFT) {          /* keep the most recent samples that fit */
        x += n - AWZ_MAX_FFT;
        n = AWZ_MAX_FFT;
    }
    int N = 1;
    while (N < n) N <<= 1;
    memset(re, 0, sizeof(float) * N);
    memset(im, 0, sizeof(float) * N);
    for (int i = 0; i < n; i++) {
        float w = n > 1 ? (float)(0.5 - 0.5 * cos(2.0 * M_PI * i / (n - 1))) : 1.0f;  /* numpy.hanning */
        re[i] = x[i] * w;
    }
    awz_fft(re, im, N, 0);
    double p = 0.0;
    for (int k = 0; k <= N / 2; k++) {
        float f = (float)k * AWZ_SR / N;
        if (f >= 400.0f && f <= 4000.0f) p += (double)re[k] * re[k] + (double)im[k] * im[k];
    }
    return (float)(10.0 * log10(p / n + 1e-12));
}

float awz_trend(const float *levels_db, int count, float step_s) {
    if (count < 3) return 0.0f;
    double st = 0, sl = 0, stt = 0, stl = 0;
    for (int i = 0; i < count; i++) {
        double t = i * step_s;
        st += t; sl += levels_db[i]; stt += t * t; stl += t * levels_db[i];
    }
    double den = count * stt - st * st;
    return den != 0.0 ? (float)((count * stl - st * sl) / den) : 0.0f;
}
