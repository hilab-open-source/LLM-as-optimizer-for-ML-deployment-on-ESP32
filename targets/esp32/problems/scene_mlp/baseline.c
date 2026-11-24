/* Baseline: int8 ResNet-MLP, straightforward C. One sample per call.
 * Weights are the const arrays declared in problem.h (they live in flash). */
#include "problem.h"

static void dense(const int8_t *W, const int32_t *b, const int8_t *x, int8_t *y,
                  int n_out, int n_in, int relu) {
    for (int o = 0; o < n_out; o++) {
        int32_t acc = b[o];
        for (int i = 0; i < n_in; i++)
            acc += (int32_t)W[o * n_in + i] * (int32_t)x[i];
        acc >>= 7;
        if (relu && acc < 0) acc = 0;
        if (acc > 127) acc = 127;
        if (acc < -128) acc = -128;
        y[o] = (int8_t)acc;
    }
}

static void res_block(const int8_t *Wa, const int32_t *ba, const int8_t *Wb, const int32_t *bb,
                      int8_t *x) {
    int8_t h[HIDDEN_SIZE];
    dense(Wa, ba, x, h, HIDDEN_SIZE, HIDDEN_SIZE, 1);
    for (int o = 0; o < HIDDEN_SIZE; o++) {
        int32_t acc = bb[o];
        for (int i = 0; i < HIDDEN_SIZE; i++)
            acc += (int32_t)Wb[o * HIDDEN_SIZE + i] * (int32_t)h[i];
        acc = (acc >> 7) + x[o];             /* residual added AFTER the requant */
        if (acc < 0) acc = 0;
        if (acc > 127) acc = 127;
        x[o] = (int8_t)acc;
    }
}

void run_inference(const int8_t *input, int8_t *output) {
    int8_t x[HIDDEN_SIZE];
    dense(&W_stem[0][0], b_stem, input, x, HIDDEN_SIZE, INPUT_SIZE, 1);
    res_block(&W1a[0][0], b1a, &W1b[0][0], b1b, x);
    res_block(&W2a[0][0], b2a, &W2b[0][0], b2b, x);
    dense(&W_out[0][0], b_out, x, output, OUTPUT_SIZE, HIDDEN_SIZE, 0);
}
