"""Summarize an optimization run as a Markdown report.

    python -m core.report results/esp32/mnist_mlp [--out report.md]

Reads trace.jsonl (every candidate) and best.json (baseline + winner) and reports the
speedup, the best-so-far trajectory per step, failure breakdown and token spend.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from core.extract import extract_rationale


def load(results_dir: Path):
    cands = [json.loads(l) for l in (results_dir / "trace.jsonl").read_text().splitlines() if l.strip()]
    best = json.loads((results_dir / "best.json").read_text()) if (results_dir / "best.json").exists() else {}
    return cands, best


def _device(cands: list[dict]) -> str:
    for c in cands:
        m = re.search(r"\[device: (.*?)\]", c.get("msg", ""))
        if m:
            return m.group(1)
    return "unknown device"


def _tokens(c: dict) -> tuple[int, int]:
    u = (c.get("metrics") or {}).get("usage") or {}
    return int(u.get("input_tokens") or u.get("prompt_tokens") or 0), int(u.get("output_tokens") or u.get("completion_tokens") or 0)


def render(results_dir: Path) -> str:
    cands, best = load(results_dir)
    base = best.get("baseline_score")
    problem = best.get("problem") or results_dir.name
    ok = [c for c in cands if c.get("correct") and c.get("score") is not None]
    steps = sorted({c["step"] for c in cands})
    tin = sum(_tokens(c)[0] for c in cands)
    tout = sum(_tokens(c)[1] for c in cands)
    metric = best.get("metric", "")
    lang = best.get("language", "c")

    L = [f"# Optimization report: `{problem}`", ""]
    L += [f"- **Device:** {_device(cands)}",
          f"- **Metric:** {metric or 'lower is better'}",
          f"- **Candidates:** {len(cands)} over {len(steps)} step(s); {len(ok)} correct",
          f"- **Generated:** {time.strftime('%Y-%m-%d %H:%M')}", ""]

    if base and best.get("score"):
        sp = base / best["score"]
        L += ["## Result", "",
              "| | score | speedup |", "|---|---:|---:|",
              f"| baseline | {base:.1f} | 1.00x |",
              f"| best (candidate {best['id']}, step {best['step']}) | {best['score']:.1f} | **{sp:.2f}x** |",
              "", f"Latency reduced by **{(1 - 1 / sp) * 100:.1f}%**.", ""]
    else:
        L += ["## Result", "", "No correct candidate beat the baseline (or baseline not recorded).", ""]

    L += ["## Best-so-far by step", "", "| step | candidates | correct | best this step | best so far | speedup |",
          "|---:|---:|---:|---:|---:|---:|"]
    running = None
    for s in steps:
        cs = [c for c in cands if c["step"] == s]
        oks = [c["score"] for c in cs if c.get("correct") and c.get("score") is not None]
        step_best = min(oks) if oks else None
        if step_best is not None:
            running = step_best if running is None else min(running, step_best)
        L.append(f"| {s} | {len(cs)} | {len(oks)} | {f'{step_best:.1f}' if step_best else '—'} | "
                 f"{f'{running:.1f}' if running else '—'} | {f'{base / running:.2f}x' if (running and base) else '—'} |")
    L.append("")

    L += ["## Candidates", "", "| id | parent | stage | score | vs baseline | tokens in/out | note |", "|---|---|---|---:|---:|---:|---|"]
    for c in cands:
        ti, to = _tokens(c)
        sc = c.get("score")
        vs = f"{base / sc:.2f}x" if (sc and base and c.get("correct")) else "—"
        note = (c.get("msg") or "").split(" [device")[0] if c.get("correct") else (c.get("msg") or "")[:90].replace("|", "/").replace("\n", " ")
        L.append(f"| {c['id']} | {c.get('parent_id') or '—'} | {c.get('stage')} | {f'{sc:.1f}' if (sc and c.get('correct')) else '—'} | {vs} | {ti}/{to} | {note} |")
    L.append("")

    L += ["## What changed at each improvement", ""]
    running = None
    for v, c in enumerate(cands, 1):
        if not (c.get("correct") and c.get("score") is not None):
            continue
        if running is not None and c["score"] >= running:
            continue
        prev = running if running is not None else base
        running = c["score"]
        vfiles = sorted((results_dir / "versions").glob(f"v{v:03d}_*")) if (results_dir / "versions").exists() else []
        link = f" · [{vfiles[0].name}](versions/{vfiles[0].name})" if vfiles else ""
        why = (c.get("metrics") or {}).get("rationale") or extract_rationale(c.get("raw_response", ""))
        L += [f"### v{v:03d} (step {c['step']}): {c['score']:.1f}"
              + (f", {prev / c['score']:.2f}x over the previous best" if prev else "") + link, ""]
        L += ["> " + l if l.strip() else ">" for l in (why or "(no explanation given)").splitlines()] + [""]

    stages: dict[str, int] = {}
    for c in cands:
        k = "ok" if c.get("correct") else c.get("stage", "?")
        stages[k] = stages.get(k, 0) + 1
    L += ["## Failure breakdown", "", ", ".join(f"{k}: {v}" for k, v in sorted(stages.items())), ""]
    L += ["## Token spend", "", f"Input {tin:,} / output {tout:,} tokens across {len(cands)} LLM calls"
          + (f" ({tin // len(cands):,} in / {tout // len(cands):,} out per call)." if cands else "."), ""]
    if best.get("code"):
        L += ["## Best kernel", "", f"```{lang}", best["code"].rstrip(), "```", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default: <results_dir>/report.md")
    a = ap.parse_args(argv)
    md = render(a.results_dir)
    out = a.out or (a.results_dir / "report.md")
    out.write_text(md)
    print(md.split("## Candidates")[0])
    print(f"[report written to {out}]")


if __name__ == "__main__":
    main()
