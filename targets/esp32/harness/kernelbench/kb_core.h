/* KernelBench core: correctness gate + timing, shared by the ESP32 sketch
 * (kernelbench.ino) and the host precheck (host_main.c). Fixed; candidates never edit it.
 *
 * Before including this file the platform must define:
 *   KB_TICK_T                  unsigned tick type (uint32_t cycles on ESP32, uint64_t ns on host)
 *   KB_TICK_UNIT               "cycles" or "ns"
 *   KB_PLATFORM                "esp32" or "host"
 *   static KB_TICK_T kb_ticks(void);
 *   static void kb_emit(const char *line);           prints one line
 *   static void kb_device(char *buf, size_t n);      human-readable device description
 *   static int  kb_cpu_mhz(void);                    0 if unknown
 */
#pragma once
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "problem.h"
#include "kb_data.h"

#define KB_GUARD 16

static in_t  kb_in[INPUT_SIZE] __attribute__((aligned(16)));
static out_t kb_out[OUTPUT_SIZE + KB_GUARD] __attribute__((aligned(16)));
static KB_TICK_T kb_samples[KB_ITERS];
static char kb_line[640];

static int kb_close(out_t got, out_t exp) {
#if KB_CHECK_EXACT
    return got == exp;
#else
    double d = fabs((double)got - (double)exp);
    return d <= KB_ATOL + KB_RTOL * fabs((double)exp);   /* NaN fails */
#endif
}

static void kb_fill_sentinel(void) {
    memset(kb_out, 0x5A, sizeof kb_out);
}

static int kb_guard_ok(void) {
    const unsigned char *g = (const unsigned char *)&kb_out[OUTPUT_SIZE];
    for (size_t i = 0; i < KB_GUARD * sizeof(out_t); i++)
        if (g[i] != 0x5A) return 0;
    return 1;
}

/* Runs every test vector; returns the number of wrong output elements (-1 if the
 * kernel wrote past the end of the output buffer) and describes the first one. */
static int kb_check_all(char *first, size_t n) {
    int bad = 0;
    first[0] = 0;
    for (int v = 0; v < NUM_VECTORS; v++) {
        memcpy(kb_in, KB_INPUTS[v], sizeof kb_in);
        kb_fill_sentinel();
        run_inference(kb_in, kb_out);
        if (!kb_guard_ok()) {
            snprintf(first, n, "vector %d: kernel wrote past output[OUTPUT_SIZE-1]", v);
            return -1;
        }
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            if (!kb_close(kb_out[i], KB_EXPECTED[v][i])) {
                if (!bad)
                    snprintf(first, n, "vector %d output[%d]: got %.9g expected %.9g",
                             v, i, (double)kb_out[i], (double)KB_EXPECTED[v][i]);
                bad++;
            }
        }
    }
    return bad;
}

static int kb_cmp(const void *a, const void *b) {
    KB_TICK_T x = *(const KB_TICK_T *)a, y = *(const KB_TICK_T *)b;
    return x < y ? -1 : x > y;
}

static void kb_fail(const char *stage, int mismatches, const char *first) {
    snprintf(kb_line, sizeof kb_line,
             "@@RESULT {\"ok\":false,\"stage\":\"%s\",\"mismatches\":%d,\"first_mismatch\":\"%s\"}",
             stage, mismatches, first);
    kb_emit(kb_line);
}

static void kb_run(void) {
    char first[200], dev[120];
    kb_device(dev, sizeof dev);

    int bad = kb_check_all(first, sizeof first);
    if (bad) { kb_fail("correctness", bad, first); return; }

    for (int i = 0; i < KB_WARMUP; i++) {
        memcpy(kb_in, KB_INPUTS[i % NUM_VECTORS], sizeof kb_in);
        run_inference(kb_in, kb_out);
    }
    for (int i = 0; i < KB_ITERS; i++) {
        memcpy(kb_in, KB_INPUTS[i % NUM_VECTORS], sizeof kb_in);   /* not timed */
        KB_TICK_T t0 = kb_ticks();
        run_inference(kb_in, kb_out);
        KB_TICK_T t1 = kb_ticks();
        kb_samples[i] = (KB_TICK_T)(t1 - t0);                       /* wrap-safe */
    }

    /* the timed calls must not have broken anything (state corruption, caching bugs) */
    bad = kb_check_all(first, sizeof first);
    if (bad) { kb_fail("correctness", bad, first); return; }

    qsort(kb_samples, KB_ITERS, sizeof(KB_TICK_T), kb_cmp);
    snprintf(kb_line, sizeof kb_line,
             "@@RESULT {\"ok\":true,\"stage\":\"ok\",\"platform\":\"%s\",\"unit\":\"%s\","
             "\"ticks_median\":%.1f,\"ticks_min\":%.1f,\"ticks_max\":%.1f,\"iters\":%d,"
             "\"cpu_mhz\":%d,\"device\":\"%s\"}",
             KB_PLATFORM, KB_TICK_UNIT, (double)kb_samples[KB_ITERS / 2], (double)kb_samples[0],
             (double)kb_samples[KB_ITERS - 1], KB_ITERS, kb_cpu_mhz(), dev);
    kb_emit(kb_line);
}
