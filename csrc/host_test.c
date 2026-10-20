/* Host harness: runs the C pipeline on numbers written by the Python tests.
 *
 *   host_test features in.f32 out.f32     audio -> log-mel frames
 *   host_test classify in.f32             64x40 window -> P(siren)
 *   host_test gcc in.f32 n max_tau        left[n] then right[n] -> tau peak
 *   host_test level in.f32                samples -> band level in dB
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "awaaz.h"

static float *read_all(const char *path, long *count) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    float *buf = malloc(bytes);
    if (fread(buf, 1, bytes, f) != (size_t)bytes) { perror("read"); exit(1); }
    fclose(f);
    *count = bytes / (long)sizeof(float);
    return buf;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: see source\n"); return 2; }
    awz_init();
    long n;
    float *x = read_all(argv[2], &n);
    if (!strcmp(argv[1], "features") && argc == 4) {
        int frames = 1 + (int)((n - AWZ_NFFT) / AWZ_HOP);
        FILE *o = fopen(argv[3], "wb");
        float out[AWZ_MELS];
        for (int i = 0; i < frames; i++) {
            awz_logmel_frame(x + (long)i * AWZ_HOP, out);
            fwrite(out, sizeof(float), AWZ_MELS, o);
        }
        fclose(o);
    } else if (!strcmp(argv[1], "classify")) {
        printf("%.7f\n", awz_classify((const float (*)[AWZ_MELS])x));
    } else if (!strcmp(argv[1], "gcc") && argc == 5) {
        int len = atoi(argv[3]);
        float peak, tau = awz_gcc_phat(x, x + len, len, (float)atof(argv[4]), &peak);
        printf("%.9g %.6f\n", tau, peak);
    } else if (!strcmp(argv[1], "level")) {
        printf("%.6f\n", awz_band_level_db(x, (int)n));
    } else {
        fprintf(stderr, "unknown mode\n");
        return 2;
    }
    free(x);
    return 0;
}
