/* Baseline: int8 GEMV, y = bias + W x, int32 accumulate. One row at a time. */
#include "problem.h"

void run_inference(const int8_t *input, int32_t *output) {
    for (int m = 0; m < M_ROWS; m++) {
        int32_t acc = gemv_bias[m];
        for (int n = 0; n < N_COLS; n++)
            acc += (int32_t)gemv_W[m][n] * (int32_t)input[n];
        output[m] = acc;
    }
}
