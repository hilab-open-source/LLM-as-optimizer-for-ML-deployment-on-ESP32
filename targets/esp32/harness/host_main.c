/* Host precheck: the same gate as the board (kb_core.h), compiled with the Mac's
 * cc. It catches wrong kernels in about a second, before spending a ~30 s flash.
 * Its timing is in host nanoseconds and says nothing about the ESP32. */
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define KB_TICK_T uint64_t
#define KB_TICK_UNIT "ns"
#define KB_PLATFORM "host"

static KB_TICK_T kb_ticks(void) {
#ifdef __APPLE__
    return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);   /* finer than CLOCK_MONOTONIC on macOS */
#endif
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
static void kb_emit(const char *line) { puts(line); fflush(stdout); }
static int kb_cpu_mhz(void) { return 0; }
static void kb_device(char *buf, size_t n) { snprintf(buf, n, "host CPU (precheck only)"); }

#include "kb_core.h"

int main(void) {
    kb_run();
    return 0;
}
