# LLM kernel optimizer for the ESP32

An LLM rewrites a small machine-learning kernel over and over to make it faster on an
ESP32 microcontroller. Every candidate is compiled, flashed to the board, checked against
a frozen **answer key** (bit-exact for integer kernels, tolerance-gated for float) and
timed in CPU cycles on the chip. Wrong kernels are rejected, so the only way for the
optimizer to win is to be faster. The idea comes from
[TTT-Discover](https://github.com/test-time-training/discover)
([arXiv 2601.16175](https://arxiv.org/abs/2601.16175)); this branch applies it to C
kernels on a microcontroller. (The `main` branch is the same optimizer for iPhone GPUs.)

```
              ┌─────────────┐  prompt: ESP32 hardware + problem + baseline + current best
              │    LLM      │          + leaderboard + recent failures
              └──────┬──────┘
                     │ N candidate kernels (C) per step
                     ▼
core/loop.py ─► extract code ─► host precheck ─► arduino-cli compile + flash ─► serial ─► archive
                                (cc on your laptop,   (ESP32 over USB)     "@@RESULT {...}"  results/esp32/...
                                 same gate, ~1 s)
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python -m targets.esp32.make_problems                     # writes the problems (no training needed)
python -m core.cli --problem mnist_mlp --fake-llm --steps 1 --candidates 4 --host   # whole pipeline, no board, zero tokens
cp .env.example .env      # add your ANTHROPIC_API_KEY
```

Then plug in a board and run for real:

```bash
arduino-cli board list                                    # find the port, e.g. /dev/cu.usbserial-0001 or COM5
python -m core.cli --problem mnist_mlp --port /dev/cu.usbserial-0001 --steps 5 --candidates 4
python -m core.report results/esp32/mnist_mlp             # summary in results/esp32/mnist_mlp/report.md
```

## Setup

You need:

- **Python 3.10+** and `pip install -r requirements.txt` (`torch`/`datasets` only if you retrain the models).
- **arduino-cli** with the ESP32 core:
  ```bash
  brew install arduino-cli                  # or see arduino.github.io/arduino-cli
  arduino-cli config init
  arduino-cli config add board_manager.additional_urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
  arduino-cli core update-index && arduino-cli core install esp32:esp32
  ```
- **A C compiler** on your laptop (`cc`; Xcode Command Line Tools on macOS, `gcc` on Linux).
  It runs the host precheck. On Windows, use WSL or set `CC` to a MinGW gcc.
- **An ESP32 dev board** on USB. The default board id is `esp32:esp32:esp32` (classic
  ESP32, e.g. ESP32-DevKitC / WROOM-32). For an ESP32-S3 use `--fqbn esp32:esp32:esp32s3`.
  You can put `ESP32_PORT` and `ESP32_FQBN` in `.env` instead of passing them every time.
- **An LLM key.** With `ANTHROPIC_API_KEY` set the loop calls the Claude API (default model
  `claude-opus-5-5`; `--effort low..max` sets thinking depth, `--model` overrides). Without
  it, the `TOOLCHAIN_*` variables select any OpenAI-compatible endpoint.

## How a candidate is evaluated

1. **Extract** the C code from the LLM's answer (`core/extract.py`).
2. **Host precheck.** The candidate, the problem's data and the harness core are compiled
   with your laptop's `cc` and run. A kernel with wrong outputs, an out-of-bounds write
   (the output buffer has guard bytes) or a crash is rejected here in about a second,
   without flashing. A kernel that only compiles for the ESP32 (for example S3 SIMD
   intrinsics) skips this step.
3. **Compile and flash** with `arduino-cli` (a fixed sketch, `targets/esp32/harness/kernelbench/`,
   plus the candidate as `candidate.c`). Incremental builds take 5–20 s, flashing 10–20 s.
4. **Gate and time on the chip.** The board waits for the host's go signal, runs every
   test vector and compares against the answer key, times `KB_ITERS` calls with the CPU
   cycle counter (input copy not timed), runs the gate again, and prints one
   `@@RESULT {...}` line over serial. The score is the **median cycles per call**.
5. A candidate that would become the new best is measured a second time and scored with
   the worse of the two runs, so timing noise cannot pose as an improvement.

Runs are **resumable**: re-running with the same results dir replays `trace.jsonl` and
continues. `--select best` is plain hill climbing; the default `puct` samples parents from
the top few solutions with an exploration bonus. Without a board (`--host`, or no
`--port`), steps 3–4 are skipped and the score is host nanoseconds, stored under
`results/esp32_host/`. That mode is for testing the pipeline; it says nothing about ESP32
speed, and the LLM is told so.

## Problems

| problem | what | work per call | gate |
|---|---|---|---|
| `mnist_mlp` | int8 ResNet-MLP 256→64→64→64→10 (16×16 digit classifier) | 33k MACs, 34 KB weights | bit-exact |
| `scene_mlp` | int8 ResNet-MLP 1024→128→128→128→20 | 199k MACs, 197 KB weights | bit-exact |
| `gemv_int8` | 512×512 int8 W · int8 x → int32 | 262k MACs, 256 KB weights | bit-exact |
| `rmsnorm_f32` | RMSNorm over 1024 floats with learned scale | 1024 elements | atol 1e-4, rtol 1e-3 |

Each problem lives in `targets/esp32/problems/<name>/`:

```
problem.h       the contract the LLM sees: sizes, weight arrays, run_inference() signature
arch.txt        what the kernel computes and the rules (shown to the LLM)
baseline.c      the starting kernel: correct, deliberately plain C
spec.json       sizes, dtypes, gate, metric
kb_data.h       harness only: number of test vectors, gate mode, timing iterations
problem_data.c  weights + test inputs + expected outputs as C const arrays (generated, git-ignored)
```

The kernel always has the form

```c
#include "problem.h"
void run_inference(const in_t *input, out_t *output) { ... }   // one sample per call
```

Weights are `const` arrays, so the linker puts them in **flash**, read through a 32 KB
cache. Much of the speedup on the bigger problems comes from how that flash is read, or
from copying weights into SRAM once.

**Adding a problem.** Write a generator function in `targets/esp32/make_problems.py`: a numpy
reference produces the answer key, `write_problem()` writes the files, and a baseline in
`targets/esp32/templates/` is the kernel to beat. To add an MLP model, write a
`models/<name>/train.py` like the two shipped ones and add one line to `ALL`.

## Layout

```
core/                       the loop: sampling, code extraction, archive/resume, parent
                            selection, failure feedback, versions, report, CLI (target-agnostic)
targets/esp32/
  target.py                 prompt (ESP32 hardware description) + host and board evaluators
  make_problems.py          generates the problems from numpy references
  templates/                baseline kernels
  problems/<name>/          one directory per problem (see above)
  harness/kernelbench/      the fixed Arduino sketch + kb_core.h (gate + timing, shared with the host)
  harness/host_main.c       host precheck entry point
models/                     int8 ResNet-MLPs (trained here; `python -m models.mnist.train`)
results/esp32/<problem>/    run.log, trace.jsonl, versions/, best.json, report.md
```

## Results

Per problem under `results/esp32/<problem>/`:

- `run.log`: timestamped console log of every run.
- `trace.jsonl`: one line per candidate: stage (`extract`, `compile`, `correctness`, `run`,
  `flaky`, `ok`), score, message, token usage, the code, the LLM's explanation and its raw response.
- `versions/`: one documented `.c` file per candidate (`v001_step1_<id>.c`; rejected ones end in
  `_FAILED_<stage>`). Each starts with its parent, the measured cycles, the speedup vs. baseline
  and vs. parent, and the LLM's own account of what it changed and why it should be faster
  (written before it was measured). `versions/CHANGELOG.md` is the same history on one page.
- `best.json`: the fastest correct kernel and the baseline score.
- `report.md`: from `python -m core.report results/esp32/<problem>`.

## Troubleshooting

- **`no @@READY from the board`**: wrong port, or the serial monitor in the Arduino IDE has
  the port open (close it). On an ESP32-S3 connected through its native USB port, set
  *USB CDC On Boot* to Enabled in the FQBN: `--fqbn esp32:esp32:esp32s3:CDCOnBoot=cdc`.
- **`upload ... failed`**: hold the BOOT button while flashing starts, or try a different
  USB cable (many are charge-only). Check that `arduino-cli upload` works with a blink sketch.
- **`kernel crashed the ESP32`**: the board printed a Guru Meditation error (usually an
  out-of-bounds access or a stack overflow from big local arrays; the Arduino loop task has
  8 KB of stack).
- **A candidate passes on the host but fails on the board**: usually undefined behaviour
  or type-width assumptions. The gate on the chip is the one that counts.

## Exercises

1. Run `mnist_mlp` for five steps. Read `versions/CHANGELOG.md`: which ideas did the LLM try
   (IRAM placement, copying weights to SRAM, `-O2`/`-O3` pragmas, unrolling, packed 32-bit
   loads)? Which ones did the measurement confirm, and which claims were wrong?
2. Compare `mnist_mlp` (34 KB of weights, which nearly fits in the 32 KB flash cache) with
   `gemv_int8` (256 KB). Why do different tricks win on each?
3. Run the same problem on an ESP32 and an ESP32-S3 (`--fqbn esp32:esp32:esp32s3`). Does the
   LLM use the S3's SIMD instructions, and how much do they help?
4. The baseline compiles at the Arduino default `-Os`. How much of the final speedup comes
   from compiler flags alone? Write that version by hand and measure it.
5. Loosen the `rmsnorm_f32` gate to 1e-2 and look for candidates that pass only because of the
   slack (for example a fast inverse square root approximation). Why is a tight gate the right default?
6. The gate can be cheated by a kernel that remembers outputs for inputs it has seen. Design a
   harness change that would catch this (hint: test vectors the kernel never sees until after timing).

## License

MIT.
