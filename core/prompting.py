"""Shared prompt assembly: baseline, parent solution, leaderboard and recent
failures. Targets supply the hardware/contract header and the per-problem arch."""
from __future__ import annotations

from core.types import Candidate, Problem


def _head_comment(code: str, max_lines: int = 12) -> str:
    """LLM kernels usually open with a comment summarizing the tricks used —
    that is enough context for 'this approach failed' without pasting the file."""
    lines = code.strip().splitlines()[:max_lines]
    return "\n".join(lines)


def compose_prompt(header: str, problem: Problem, parent: Candidate | None,
                   failures: list[Candidate], leaderboard: list[Candidate],
                   baseline_score: float | None) -> str:
    lang = problem.language
    parts = [header.rstrip(), "", problem.arch_text.strip(), "",
             f"Metric: {problem.metric_name} (lower is better)."]
    if baseline_score is not None:
        parts.append(f"The baseline below measures {baseline_score:.1f}.")
    parts += ["", "Baseline implementation (correct, unoptimized):", f"```{lang}", problem.baseline_code.rstrip(), "```"]

    if parent is not None:
        parts += ["", f"Current solution to improve — it measured {parent.score:.1f} and is bit-exact. "
                      "Study it and produce a strictly faster version that stays correct:",
                  f"```{lang}", parent.code.rstrip(), "```"]
        if parent.metrics:
            interesting = {k: v for k, v in parent.metrics.items()
                           if isinstance(v, (int, float, str)) and k not in ("code",)}
            if interesting:
                parts.append("Measurement details for the current solution: " +
                             ", ".join(f"{k}={v}" for k, v in list(interesting.items())[:8]))
    if leaderboard:
        parts += ["", "Leaderboard so far (score, step):"]
        parts += [f"  {c.score:.1f}  (step {c.step})" for c in leaderboard]
    if failures:
        parts += ["", "Recent attempts that FAILED — do not repeat these mistakes:"]
        for c in failures:
            parts += [f"--- failed at stage '{c.stage}': {c.msg[:600].strip()}",
                      f"```{lang}", _head_comment(c.code), "```"]
    prev = "the current solution above" if parent is not None else "the baseline"
    parts += ["", "Think about the hardware first, then write the complete kernel.",
              "Begin your answer with a section headed '## Changes vs. previous version': 3-10 bullet "
              f"points stating exactly what you changed relative to {prev} and why each change should "
              "reduce the measured time on this hardware. This section is stored verbatim as the documentation "
              "of the new version, so be specific (memory traffic, flash vs. SRAM, loop overhead, instruction count...).",
              f"Then wrap the final, complete source in a single ```{lang} block."]
    return "\n".join(parts)
