"""ESP32 target: C kernels compiled with arduino-cli, flashed to an ESP32 over USB,
checked against the answer key and timed in CPU cycles on the chip.

Every candidate first goes through a host precheck (the same gate compiled with
this computer's cc, about a second), so wrong kernels are rejected without a flash.
Without --port only the precheck runs, and scores are host nanoseconds: useful for
testing the pipeline, meaningless as ESP32 performance.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from core.prompting import compose_prompt
from core.types import EvalResult, Problem

HERE = Path(__file__).parent
PROBLEMS = HERE / "problems"
HARNESS = HERE / "harness"
SKETCH = HARNESS / "kernelbench"
BUILD = HERE / ".build"

HEADER = """\
You are an embedded performance engineer optimizing a C kernel for the ESP32 family
of microcontrollers. The classic ESP32 is a dual-core Xtensa LX6 at 240 MHz: 520 KB
internal SRAM, no data cache for SRAM (1-cycle-ish loads), const data and code in external
SPI flash behind a 32 KB cache (a cache miss costs tens to hundreds of cycles), a
single-precision FPU, 32-bit multiply and the MAC16 16x16 multiply-accumulate unit, and
zero-overhead loops (LOOP instruction). There is no general SIMD on the LX6. The ESP32-S3
(LX7) adds 128-bit PIE SIMD instructions (ee.vmulas.*); use them only behind
`#if CONFIG_IDF_TARGET_ESP32S3` with a portable fallback.

The candidate is ONE C11 file (candidate.c) compiled by arduino-esp32 (GCC, -Os by
default) together with problem.h (shown below). It must #include "problem.h" and define
run_inference with the exact signature declared there. It is also compiled and checked on
a desktop machine with cc before every flash, so code outside `#if` guards must be portable C:
use int8_t/int32_t (never plain char for data), no undefined behaviour.

Your ONLY goal is to minimize the measured cycles per call while staying correct.
Allowed levers include: `#pragma GCC optimize("O2")` / `__attribute__((optimize("O3")))`,
IRAM_ATTR on hot functions, copying weights from flash into a static SRAM buffer
(SRAM is limited: keep copies well under ~150 KB), loop unrolling and blocking, 32-bit
packed loads of int8 data, keeping accumulators in registers, restructuring the math as long
as the result is identical. Not allowed: caching or looking up outputs by input,
reading the test vectors, or anything that would break on unseen inputs.
"""


def _sh(cmd: list[str], timeout: int = 600, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)


def _parse_result(text: str) -> dict | None:
    for line in reversed(text.splitlines()):
        if line.startswith("@@RESULT "):
            return json.loads(line[len("@@RESULT "):])
    return None


def _compile_errors(text: str, limit: int = 1500) -> str:
    """Keep the compiler's error lines about the candidate; drop the long command lines."""
    lines = [l for l in text.splitlines() if "candidate.c" in l or "error" in l.lower() or "undefined" in l.lower() or "run_inference" in l]
    lines = [re.sub(r"^.*/(candidate\.c)", r"\1", l) for l in lines if len(l) < 400]
    return ("\n".join(lines) or text[-limit:])[:limit]


class ESP32Target:
    name = "esp32"
    max_parallel_eval = 1        # one board, and timing is contention-sensitive

    def __init__(self, port: str | None = None, fqbn: str | None = None, **_):
        self.port = port
        self.fqbn = fqbn or os.environ.get("ESP32_FQBN", "esp32:esp32:esp32")
        self.results_key = "esp32" if port else "esp32_host"
        self.device_desc: str | None = None

    # ---- problems ----------------------------------------------------------
    def load_problem(self, name: str) -> Problem:
        d = PROBLEMS / name
        if not (d / "problem_data.c").exists():
            raise SystemExit(f"{d}/problem_data.c missing: run  python -m targets.esp32.make_problems {name}")
        spec = json.loads((d / "spec.json").read_text())
        metric = spec["metric_note"] if self.port else \
            "host nanoseconds per call (precheck machine, NOT the ESP32; pipeline testing only)"
        arch = (d / "arch.txt").read_text() + "\nproblem.h (fixed; #include it, do not repeat it):\n```c\n" \
            + (d / "problem.h").read_text() + "```"
        return Problem(name=name, dir=d, baseline_code=(d / "baseline.c").read_text(), arch_text=arch,
                       metric_name=metric, language="c", entry_symbol="run_inference", extra={"spec": spec})

    def build_prompt(self, problem, parent, failures, leaderboard, baseline_score) -> str:
        if self.port:
            where = f"Measurements are taken on: {self.device_desc or 'an ESP32 board'} (board: {self.fqbn})."
        else:
            where = ("NOTE: no board is attached in this run, so timings come from the host precheck "
                     "machine. Still optimize for the ESP32 as described above.")
        return compose_prompt(HEADER + "\n" + where, problem, parent, failures, leaderboard, baseline_score)

    # ---- evaluation --------------------------------------------------------
    def evaluate(self, problem: Problem, code: str) -> EvalResult:
        host = self._eval_host(problem, code)
        if host.get("stage") == "compile" and self.port:
            pass  # may use ESP32-only intrinsics/headers; let the board decide
        elif not host.get("ok") or not self.port:
            if not host.get("ok") and host.get("stage") == "correctness":
                host["error_prefix"] = "host precheck: "
            return self._to_result(host)
        return self._to_result(self._eval_device(problem, code))

    def _to_result(self, rep: dict) -> EvalResult:
        stage = rep.get("stage", "run")
        metrics = {k: rep.get(k) for k in ("ticks_median", "ticks_min", "ticks_max", "unit", "iters",
                                            "cpu_mhz", "device", "platform", "compile_s", "flash_s")
                   if rep.get(k) is not None}
        if not rep.get("ok"):
            msg = rep.get("error") or stage
            if stage == "correctness":
                msg = f"{rep.get('error_prefix', '')}{rep.get('mismatches')} wrong outputs; first: {rep.get('first_mismatch')}"
            return EvalResult(correct=False, score=float("inf"), msg=str(msg)[:1500], stage=stage, metrics=metrics)
        t = float(rep["ticks_median"])
        if rep.get("platform") == "esp32":
            self.device_desc = rep.get("device")
            us = t / rep["cpu_mhz"] if rep.get("cpu_mhz") else None
            msg = f"{t:.0f} cycles" + (f" = {us:.1f} us" if us else "") + f" [device: {rep.get('device')}]"
        else:
            msg = f"{t:.0f} ns on the host precheck [device: host, not the ESP32]"
        return EvalResult(correct=True, score=t, msg=msg, stage="ok", metrics=metrics)

    # host: the same gate (kb_core.h) built with cc
    def _eval_host(self, problem: Problem, code: str) -> dict:
        cc = os.environ.get("CC", "cc")
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "candidate.c"
            src.write_text(code)
            exe = Path(tmp) / "kb_host"
            cmd = [cc, "-O2", "-std=gnu11", "-w", "-I", str(problem.dir), "-I", str(SKETCH),
                   str(src), str(problem.dir / "problem_data.c"), str(HARNESS / "host_main.c"),
                   "-o", str(exe), "-lm"]
            r = _sh(cmd, timeout=180)
            if r.returncode != 0:
                return {"ok": False, "stage": "compile", "error": "host cc: " + _compile_errors(r.stderr)}
            try:
                r = _sh([str(exe)], timeout=60)
            except subprocess.TimeoutExpired:
                return {"ok": False, "stage": "run", "error": "host precheck timed out after 60 s (infinite loop?)"}
            rep = _parse_result(r.stdout)
            if rep is None:
                return {"ok": False, "stage": "run",
                        "error": f"host precheck crashed (rc={r.returncode}; a negative rc is a signal, usually an "
                                 f"out-of-bounds access or stack overflow): {(r.stderr or r.stdout)[-600:]}"}
            return rep

    # device: arduino-cli compile + upload, then the serial handshake
    def _eval_device(self, problem: Problem, code: str) -> dict:
        sk = BUILD / self.fqbn.replace(":", "_") / problem.name / "kernelbench"
        bp = sk.parent / "build"
        sk.mkdir(parents=True, exist_ok=True)
        for f in ("kernelbench.ino", "kb_core.h"):
            shutil.copy2(SKETCH / f, sk / f)
        for f in ("problem.h", "kb_data.h", "problem_data.c"):
            if not (sk / f).exists() or (sk / f).read_bytes() != (problem.dir / f).read_bytes():
                shutil.copy2(problem.dir / f, sk / f)
        (sk / "candidate.c").write_text(code)

        t0 = time.time()
        try:
            r = _sh(["arduino-cli", "compile", "--fqbn", self.fqbn, "--build-path", str(bp), str(sk)], timeout=900)
        except subprocess.TimeoutExpired:
            return {"ok": False, "stage": "compile", "error": "arduino-cli compile timed out"}
        if r.returncode != 0:
            return {"ok": False, "stage": "compile", "error": _compile_errors(r.stderr + r.stdout)}
        compile_s = time.time() - t0

        t0 = time.time()
        try:
            r = _sh(["arduino-cli", "upload", "--fqbn", self.fqbn, "-p", self.port,
                     "--input-dir", str(bp), str(sk)], timeout=300)
        except subprocess.TimeoutExpired:
            r = None
        if r is None or r.returncode != 0:
            # infrastructure, not the kernel's fault; recorded as a 'run' failure with this message
            raise RuntimeError(f"upload to {self.port} failed: {(r.stderr if r else 'timeout')[-800:]}")
        flash_s = time.time() - t0

        rep = self._serial_run(timeout=120)
        rep.update(compile_s=round(compile_s, 1), flash_s=round(flash_s, 1))
        return rep

    def _serial_run(self, timeout: float) -> dict:
        import serial  # pyserial
        s = serial.Serial()
        s.port, s.baudrate, s.timeout = self.port, 115200, 0.2
        s.dtr = s.rts = False            # do not hold the chip in reset / bootloader on open
        s.open()
        buf, sent, t_end = "", False, time.time() + timeout
        try:
            while time.time() < t_end:
                buf += s.read(4096).decode("utf-8", "replace")
                if not sent and "@@READY" in buf:
                    s.write(b"g"); s.flush()
                    sent, t_end = True, time.time() + timeout
                if "Guru Meditation" in buf or "Backtrace:" in buf:
                    time.sleep(0.5)
                    buf += s.read(4096).decode("utf-8", "replace")
                    i = buf.find("Guru Meditation")
                    return {"ok": False, "stage": "run", "error": "kernel crashed the ESP32: " + buf[max(i, 0):][:600]}
                rep = _parse_result(buf)
                if rep is not None:
                    return rep
                if not sent and time.time() > t_end - timeout + 20:
                    return {"ok": False, "stage": "run",
                            "error": f"no @@READY from the board in 20 s; serial said: {buf[-400:]!r}"}
            return {"ok": False, "stage": "run",
                    "error": f"no result within {timeout:.0f} s (kernel hang or too slow); last output: {buf[-300:]!r}"}
        finally:
            s.close()


def make_target(port: str | None = None, fqbn: str | None = None, **_) -> ESP32Target:
    return ESP32Target(port=port, fqbn=fqbn)
