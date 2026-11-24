"""Entry point:  python -m core.cli --problem mnist_mlp --steps 10 --candidates 8 --port /dev/cu.usbserial-0001

Target: ESP32 C kernels, compiled with arduino-cli, flashed over USB and timed in CPU
cycles on the chip. Without --port only the host precheck runs (pipeline testing)."""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.llm import FakeSampler, make_sampler   # noqa: E402
from core.loop import optimize                          # noqa: E402


def load_target(name: str, **kw):
    mod = importlib.import_module(f"targets.{name}.target")
    return mod.make_target(**kw)


class _Tee:
    """Mirror console output into results/<...>/run.log with a timestamp per line."""
    def __init__(self, console, logfile):
        self.console, self.f, self.at_line_start = console, logfile, True

    def write(self, s):
        self.console.write(s)
        for piece in s.splitlines(keepends=True):
            if self.at_line_start and piece.strip():
                self.f.write(time.strftime("[%Y-%m-%d %H:%M:%S] "))
            self.f.write(piece)
            self.at_line_start = piece.endswith("\n")
        self.f.flush()

    def flush(self):
        self.console.flush(); self.f.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


def main(argv=None):
    ap = argparse.ArgumentParser(description="LLM kernel optimizer (TTT-Discover style, frozen-policy loop)")
    ap.add_argument("--target", default="esp32", choices=["esp32"])
    ap.add_argument("--problem", required=True, help="kernel problem under targets/esp32/problems")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--select", default="puct", choices=["puct", "best"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--fake-llm", action="store_true", help="echo the baseline instead of calling the API (pipeline test)")
    ap.add_argument("--model", default=None, help="LLM model id (default: TOOLCHAIN_MODEL)")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=1.0, help="OpenAI-compat backend only")
    ap.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"],
                    help="Claude backend: thinking depth (default ANTHROPIC_EFFORT or high)")
    ap.add_argument("--port", default=os.environ.get("ESP32_PORT"), help="ESP32 serial port, e.g. /dev/cu.usbserial-0001 or COM5 "
                    "(see `arduino-cli board list`); omit to run the host precheck only")
    ap.add_argument("--host", action="store_true", help="ignore --port/ESP32_PORT; host precheck only")
    ap.add_argument("--fqbn", default=None, help="arduino-cli board id (default ESP32_FQBN or esp32:esp32:esp32; "
                    "ESP32-S3: esp32:esp32:esp32s3)")
    ap.add_argument("--results", default=None, help="results dir (default results/<target>/<problem>)")
    ap.add_argument("--skip-baseline", action="store_true", help="do not measure the baseline first")
    ap.add_argument("--no-confirm", action="store_true", help="skip the re-measurement of new-best candidates")
    args = ap.parse_args(argv)

    if args.host:
        args.port = None
    target = load_target(args.target, port=args.port, fqbn=args.fqbn)
    problem = target.load_problem(args.problem)
    key = getattr(target, "results_key", target.name)
    results_dir = Path(args.results) if args.results else REPO / "results" / key / problem.name
    results_dir.mkdir(parents=True, exist_ok=True)
    logfile = open(results_dir / "run.log", "a")
    sys.stdout, sys.stderr = _Tee(sys.__stdout__, logfile), _Tee(sys.__stderr__, logfile)
    print(f"=== run started: {' '.join(sys.argv[1:]) or '(no args)'}")
    print(f"Log: {results_dir / 'run.log'}")

    baseline_score = None
    if not args.skip_baseline:
        # Same protocol as candidates: measure twice, keep the worse, so the reported
        # speedup is never inflated by a slow baseline run.
        print("Measuring baseline (twice)...", flush=True)
        if hasattr(target, "report_status"):
            target.report_status({"problem": problem.name, "title": problem.arch_text.strip().splitlines()[0][:160],
                                  "phase": "measuring baseline", "step_first": 1, "step_last": args.steps,
                                  "cands_per_step": args.candidates, "done": 0, "total": args.steps * args.candidates})
        runs = [target.evaluate(problem, problem.baseline_code) for _ in range(2)]
        bad = [r for r in runs if not r.correct]
        if bad:
            raise SystemExit(f"Baseline failed its own answer key [{bad[0].stage}]: {bad[0].msg}")
        baseline_score = max(r.score for r in runs)
        print(f"  baseline: {baseline_score:.1f}  (runs: {runs[0].score:.1f} / {runs[1].score:.1f})  {runs[0].msg}\n")

    sampler = (FakeSampler(problem.baseline_code, problem.language) if args.fake_llm
               else make_sampler(model=args.model, max_tokens=args.max_tokens,
                                 temperature=args.temperature, effort=args.effort))
    best = optimize(target, problem, sampler, results_dir, steps=args.steps,
                    candidates_per_step=args.candidates, select=args.select, seed=args.seed,
                    baseline_score=baseline_score, confirm_best=not args.no_confirm)
    if best and baseline_score:
        print(f"Best {best.score:.1f} vs baseline {baseline_score:.1f} "
              f"= {baseline_score / best.score:.2f}x speedup. Saved to {results_dir / 'best.json'}")


if __name__ == "__main__":
    main()
