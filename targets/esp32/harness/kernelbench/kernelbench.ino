// KernelBench for ESP32 (fixed harness; candidates never edit it).
//
// The optimizer copies problem.h, kb_data.h, problem_data.c and the candidate's
// candidate.c next to this file, compiles with arduino-cli, flashes, then talks over
// serial: the board prints "@@READY" until the host sends 'g', then runs the gate +
// timing in kb_core.h and prints one "@@RESULT {json}" line.
#include <Arduino.h>

#define KB_TICK_T uint32_t
#define KB_TICK_UNIT "cycles"
#define KB_PLATFORM "esp32"

static KB_TICK_T kb_ticks(void) { return ESP.getCycleCount(); }
static void kb_emit(const char *line) { Serial.println(line); Serial.flush(); }
static int kb_cpu_mhz(void) { return (int)getCpuFrequencyMhz(); }
static void kb_device(char *buf, size_t n) {
    snprintf(buf, n, "%s rev%d, %d MHz, %u KB free heap",
             ESP.getChipModel(), (int)ESP.getChipRevision(), (int)getCpuFrequencyMhz(),
             (unsigned)(ESP.getFreeHeap() / 1024));
}

#include "kb_core.h"

void setup() {
    Serial.begin(115200);
    uint32_t last = 0;
    for (;;) {                                   // handshake: wait for the host's 'g'
        if (Serial.available() && Serial.read() == 'g') break;
        if (millis() - last > 250) { Serial.println("@@READY"); last = millis(); }
        delay(5);
    }
    kb_run();
    Serial.println("@@DONE");
}

void loop() { delay(1000); }
