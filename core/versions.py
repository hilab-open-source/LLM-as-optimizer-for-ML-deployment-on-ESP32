"""Documented code versions: one file per candidate plus a CHANGELOG.

results/<target>/<problem>/versions/
    v000_baseline.c
    v001_step1_2b6e6e59.c                  header: parent, measurements, why it is faster
    v002_step1_836c86c2_FAILED_compile.c
    CHANGELOG.md                            the same, in reading order

Rebuild from an existing trace (no tokens):  python -m core.versions results/esp32/mnist_mlp
"""
from __future__ import annotations

import json
import re
import shutil
import textwrap
import time
from pathlib import Path

from core.extract import extract_rationale
from core.types import Candidate, Problem

EXT = {"c": ".c", "cpp": ".cpp", "metal": ".metal", "cuda": ".cu"}


def version_filename(v: int, c: Candidate, ext: str) -> str:
    tail = "" if c.usable else f"_FAILED_{c.stage}"
    return f"v{v:03d}_step{c.step}_{c.id[:8]}{tail}{ext}"


def _comment(text: str, width: int = 92) -> str:
    out = []
    for para in (text or "").strip().splitlines():
        if not para.strip():
            out.append("//")
            continue
        indent = len(para) - len(para.lstrip())
        for l in textwrap.wrap(para.strip(), width=width - indent) or [""]:
            out.append("// " + " " * indent + l)
    return "\n".join(out) if out else "// (no explanation was given in the response)"


def _device(msg: str) -> str:
    m = re.search(r"\[device: (.*?)\]", msg or "")
    return m.group(1) if m else ""


class VersionWriter:
    def __init__(self, results_dir: Path, problem: Problem, baseline_score: float | None, archive):
        self.dir = results_dir / "versions"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.problem = problem
        self.base = baseline_score
        self.ext = EXT.get(problem.language, "." + problem.language)
        self.changelog = self.dir / "CHANGELOG.md"
        self.by_id: dict[str, tuple[int, Candidate]] = {}
        self.n = 0
        self.best: float | None = None
        self._write_baseline()
        for c in archive.cands:  # resuming: keep numbering continuous, do not rewrite old files
            self.n += 1
            self.by_id[c.id] = (self.n, c)
            if c.usable:
                self.best = c.score if self.best is None else min(self.best, c.score)

    # ---- files -------------------------------------------------------------
    def _write_baseline(self) -> None:
        p = self.dir / f"v000_baseline{self.ext}"
        if not p.exists():
            meas = f"{self.base:.1f}" if self.base else "(not measured in this run)"
            hdr = [f"// v000  baseline  problem={self.problem.name}", f"// measured: {meas}",
                   "// Starting point. Every later version documents its changes relative to its parent."]
            p.write_text("\n".join(hdr) + "\n\n" + self.problem.baseline_code.rstrip() + "\n")
        if not self.changelog.exists():
            self.changelog.write_text(
                f"# Optimization log: {self.problem.name}\n\n"
                f"One entry per candidate kernel, in the order they were produced. "
                f"Metric: {self.problem.metric_name} (lower is better). Files are next to this changelog.\n\n"
                f"## v000 baseline\n\n- measured {self.base:.1f}\n\n" if self.base else
                f"# Optimization log: {self.problem.name}\n\n## v000 baseline\n\n")

    def record(self, c: Candidate) -> Path:
        self.n += 1
        v = self.n
        self.by_id[c.id] = (v, c)
        pv, pc = self.by_id.get(c.parent_id or "", (0, None))
        parent_score = pc.score if (pc and pc.usable) else self.base
        parent_label = f"v{pv:03d}" + (f" ({pc.id})" if pc else " (baseline)")
        new_best = bool(c.usable and (self.best is None or c.score < self.best))
        if new_best:
            self.best = c.score

        hdr = [f"// v{v:03d}  step {c.step}  candidate {c.id}  parent: {parent_label}"]
        if c.usable:
            line = f"// measured: {c.score:.1f}"
            if self.base:
                line += f"  = {self.base / c.score:.2f}x vs baseline"
            if parent_score:
                line += f", {parent_score / c.score:.2f}x vs parent ({parent_score:.1f})"
            hdr.append(line + ("   <-- new best" if new_best else ""))
            if _device(c.msg):
                hdr.append(f"// device: {_device(c.msg)}")
        else:
            hdr.append(f"// REJECTED at stage '{c.stage}': {(c.msg or '').strip()[:400]}".replace("\n", " "))
        u = (c.metrics or {}).get("usage") or {}
        if u:
            hdr.append(f"// llm tokens: {u.get('input_tokens', '?')} in / {u.get('output_tokens', '?')} out")
        rationale = (c.metrics or {}).get("rationale") or extract_rationale(c.raw_response)
        hdr += ["//", "// Why this version should be faster than its parent (the LLM's own account):",
                _comment(rationale)]
        body = c.code.rstrip() if c.code.strip() else "// (no code could be extracted from the response)"
        path = self.dir / version_filename(v, c, self.ext)
        path.write_text("\n".join(hdr) + "\n\n" + body + "\n")
        self._append_changelog(v, c, parent_label, parent_score, new_best, rationale, path.name)
        return path

    def _append_changelog(self, v, c, parent_label, parent_score, new_best, rationale, fname) -> None:
        L = [f"## v{v:03d} — step {c.step}, candidate `{c.id}` (parent {parent_label})" + ("  **new best**" if new_best else ""), ""]
        if c.usable:
            line = f"- **{c.score:.1f}**"
            if self.base:
                line += f" = {self.base / c.score:.2f}x vs baseline"
            if parent_score:
                line += f", {parent_score / c.score:.2f}x vs parent"
            if _device(c.msg):
                line += f" · {_device(c.msg)}"
            L.append(line + f" · [{fname}]({fname})")
        else:
            L.append(f"- **REJECTED** at stage `{c.stage}`: {(c.msg or '').strip()[:300]} · [{fname}]({fname})")
        L += ["- **Why (per the LLM):**", ""]
        L += ["  > " + l if l.strip() else "  >" for l in (rationale or "(no explanation given)").splitlines()]
        L.append("")
        with open(self.changelog, "a") as f:
            f.write("\n".join(L) + "\n")


# ---- rebuild from a trace ---------------------------------------------------
def rebuild(results_dir: Path) -> Path:
    from core.archive import Archive
    from core.cli import load_target
    best = json.loads((results_dir / "best.json").read_text()) if (results_dir / "best.json").exists() else {}
    target = load_target(best.get("target", "esp32"), port="rebuild" if results_dir.parent.name == "esp32" else None)
    problem = target.load_problem(best.get("problem") or results_dir.name)
    archive = Archive(results_dir)
    vdir = results_dir / "versions"
    if vdir.exists():
        shutil.rmtree(vdir)
    cands, archive.cands = archive.cands, []
    vw = VersionWriter(results_dir, problem, best.get("baseline_score"), archive)
    for c in cands:
        vw.record(c)
    return vdir


if __name__ == "__main__":
    import sys
    d = rebuild(Path(sys.argv[1]))
    print(f"rebuilt {len(list(d.glob('v*')))} version files in {d}")
