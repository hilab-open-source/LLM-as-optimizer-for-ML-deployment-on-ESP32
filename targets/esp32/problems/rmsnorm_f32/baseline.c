/* Baseline: RMSNorm, two passes over x in float32. */
#include <math.h>
#include "problem.h"

void run_inference(const float *input, float *output) {
    float ss = 0.0f;
    for (int i = 0; i < D_MODEL; i++)
        ss += input[i] * input[i];
    float r = 1.0f / sqrtf(ss / (float)D_MODEL + RMS_EPS);
    for (int i = 0; i < D_MODEL; i++)
        output[i] = input[i] * r * rms_gamma[i];
}
