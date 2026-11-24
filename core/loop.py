"""The optimization loop: sample N candidates per step from selected parents,
evaluate, archive, repeat. Resumable. Target-agnostic."""
from __future__ import annotations

import concurrent.futures as cf
import random
import time
from pathlib import Path

from core.archive import Archive
from core.extract import extract_code, extract_rationale
from core.types import Candidate, EvalResult, Problem, FAIL
from core.versions import VersionWriter


class Target:
    """Interface targets implement (duck-typed; see targets/*/target.py)."""
    name: str = "base"
    max_parallel_eval: int = 1

    def load_problem(self, name: str) -> Problem: ...
    def build_prompt(self, problem: Problem, parent, failures, leaderboard, baseline_score) -> str: ...
    def evaluate(self, problem: Problem, code: str) -> EvalResult: ...


def _evaluate_sample(target: Target, problem: Problem, text: str, finish_reason: str | None):
    t0 = time.time()
    code, how = extract_code(text, problem.entry_symbol)
    if code is None:
        res = EvalResult.fail("extract", f"{how}" + (" (response truncated at max_tokens)"
                                                     if finish_reason == "length" else ""))
        code = ""
    elif finish_reason == "length" and problem.entry_symbol not in code:
        res = EvalResult.fail("extract", "response truncated at max_tokens before the kernel was complete")
    else:
        try:
            res = target.evaluate(problem, code)
        except Exception as exc:  # evaluator bug or infra failure -> recorded, not fatal
            res = EvalResult.fail("run", f"evaluator exception: {exc!r}")
    res.metrics.setdefault("extract", how)
    return code, res, time.time() - t0


def _push(target, state: dict) -> None:
    fn = getattr(target, "report_status", None)
    if fn:
        fn(dict(state))


def optimize(target: Target, problem: Problem, sampler, results_dir: Path,
             steps: int = 20, candidates_per_step: int = 8, select: str = "puct",
             seed: int | None = None, baseline_score: float | None = None,
             on_candidate=None, confirm_best: bool = True) -> Candidate | None:
    archive = Archive(results_dir)
    versions = VersionWriter(results_dir, problem, baseline_score, archive)
    rng = random.Random(seed)
    start_step = archive.last_step + 1
    if archive.cands:
        print(f"Resuming from trace: {archive.summary()}")
    print(f"[{target.name}/{problem.name}] steps {start_step}..{start_step + steps - 1}, "
          f"{candidates_per_step} candidates/step, parent selection={select}")
    print(f"LLM: {sampler.describe()}")
    print(f"Trace: {archive.trace_path}")
    state = {"problem": problem.name, "title": problem.arch_text.strip().splitlines()[0][:160],
             "llm": sampler.describe(), "phase": "starting", "step": start_step, "step_first": start_step,
             "step_last": start_step + steps - 1, "cand": 0, "cands_per_step": candidates_per_step,
             "done": 0, "total": steps * candidates_per_step,
             "ok": sum(1 for c in archive.cands if c.usable), "failed": sum(1 for c in archive.cands if not c.usable),
             "baseline": baseline_score, "best": archive.best.score if archive.best else None,
             "best_step": archive.best.step if archive.best else None, "last": None}
    _push(target, state)
    print(f"Versions: {versions.dir}  (one documented file per candidate + CHANGELOG.md)\n")

    for step in range(start_step, start_step + steps):
        # one parent per candidate so lineages diversify within a step
        parents = [archive.select_parent(select, rng=rng) for _ in range(candidates_per_step)]
        failures = archive.recent_failures(3)
        board = archive.leaderboard(3)
        prompts = [target.build_prompt(problem, p, failures, board, baseline_score) for p in parents]

        state.update(step=step, cand=0, phase=f"sampling {candidates_per_step} candidates from the LLM")
        _push(target, state)
        print(f"[step {step}] sampling {candidates_per_step} candidates "
              f"(parents: {sorted({(p.score if p else None) for p in parents}, key=lambda s: (s is None, s))})")
        with cf.ThreadPoolExecutor(max_workers=min(candidates_per_step, 8)) as pool:
            futs = [pool.submit(sampler.sample, pr) for pr in prompts]
            samples = []
            for i, f in enumerate(futs):
                try:
                    samples.append(f.result())
                except Exception as exc:
                    print(f"  sample {i + 1} failed: {exc}")
                    samples.append(None)

        print(f"  evaluating ({target.max_parallel_eval} parallel)...", flush=True)
        state.update(cand=1, phase="evaluating on device")
        _push(target, state)
        jobs = [(i, s) for i, s in enumerate(samples) if s is not None]
        with cf.ThreadPoolExecutor(max_workers=max(1, target.max_parallel_eval)) as pool:
            futs = {pool.submit(_evaluate_sample, target, problem, s.text, s.finish_reason): (i, s)
                    for i, s in jobs}
            for f in cf.as_completed(futs):
                i, s = futs[f]
                code, res, wall = f.result()
                if getattr(s, "usage", None):
                    res.metrics.setdefault("usage", s.usage)  # tokens per candidate, kept in the trace
                res.metrics["rationale"] = extract_rationale(s.text)  # the LLM's "what changed and why"
                # Confirmation re-measurement: a candidate that would become the new
                # best is measured a second time and scored with the WORSE of the two
                # medians, so timing noise cannot masquerade as an improvement.
                cur_best = archive.best
                if confirm_best and res.correct and (cur_best is None or res.score < cur_best.score):
                    try:
                        res2 = target.evaluate(problem, code)
                    except Exception as exc:
                        res2 = EvalResult.fail("run", f"evaluator exception on re-measure: {exc!r}")
                    if not res2.correct:
                        res = EvalResult(correct=False, score=FAIL, stage="flaky",
                                         msg=f"passed once then failed on re-measure [{res2.stage}]: {res2.msg}",
                                         metrics={**res.metrics, "first_score": res.score})
                    else:
                        res.metrics.update({"score_run1": res.score, "score_run2": res2.score})
                        res.score = max(res.score, res2.score)
                        res.msg = f"{res.score:.1f} (confirmed: {res.metrics['score_run1']:.1f} / {res2.score:.1f}) " + res.msg
                c = Candidate(id=Candidate.new_id(), step=step,
                              parent_id=parents[i].id if parents[i] else None,
                              code=code, raw_response=s.text, correct=res.correct,
                              score=None if res.score == FAIL else res.score,
                              msg=res.msg, stage=res.stage, metrics=res.metrics,
                              finish_reason=s.finish_reason, wall_s=wall)
                archive.add(c)
                vpath = versions.record(c)
                tag = f"{c.score:.1f}" if c.usable else f"FAIL[{c.stage}]"
                print(f"  cand {step}.{i + 1} [{vpath.stem.split('_')[0]}]: {tag}  {c.msg[:110]}", flush=True)
                if on_candidate:
                    on_candidate(c)
                state["done"] += 1
                state["ok" if c.usable else "failed"] += 1
                state.update(cand=min(i + 2, candidates_per_step), last=f"{step}.{i + 1}: {tag}",
                             best=archive.best.score if archive.best else None,
                             best_step=archive.best.step if archive.best else None,
                             phase="evaluating on device" if state["done"] % candidates_per_step else "step finished")
                _push(target, state)
        b = archive.best
        print(f"  -> best so far: {b.score:.1f} (step {b.step})\n" if b else "  -> no usable candidate yet\n")
        archive.save_best({"target": target.name, "problem": problem.name, "baseline_score": baseline_score,
                           "metric": problem.metric_name, "language": problem.language})

    state.update(phase="done", cand=candidates_per_step)
    _push(target, state)
    print("Done.", archive.summary())
    return archive.best
