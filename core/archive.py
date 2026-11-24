"""Append-only candidate archive with resume and parent selection.

trace.jsonl is the single source of truth: one line per evaluated candidate,
written immediately. On start, an existing trace is replayed so a crashed or
interrupted run continues from the last completed step instead of overwriting.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from core.types import Candidate


class Archive:
    def __init__(self, results_dir: Path):
        self.dir = Path(results_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.dir / "trace.jsonl"
        self.cands: list[Candidate] = []
        self.visits: dict[str, int] = {}     # parent id -> times used as parent
        if self.trace_path.exists():
            for line in self.trace_path.read_text().splitlines():
                if line.strip():
                    c = Candidate.from_dict(json.loads(line))
                    self.cands.append(c)
                    if c.parent_id:
                        self.visits[c.parent_id] = self.visits.get(c.parent_id, 0) + 1

    # ---- persistence -------------------------------------------------------
    def add(self, c: Candidate) -> None:
        self.cands.append(c)
        if c.parent_id:
            self.visits[c.parent_id] = self.visits.get(c.parent_id, 0) + 1
        with open(self.trace_path, "a") as f:
            f.write(json.dumps(c.to_dict()) + "\n")

    def save_best(self, extra: dict | None = None) -> Path | None:
        b = self.best
        if b is None:
            return None
        out = {"id": b.id, "step": b.step, "score": b.score, "metrics": b.metrics,
               "msg": b.msg, "code": b.code}
        if extra:
            out.update(extra)
        p = self.dir / "best.json"
        p.write_text(json.dumps(out, indent=2))
        return p

    # ---- queries -----------------------------------------------------------
    @property
    def last_step(self) -> int:
        return max((c.step for c in self.cands), default=0)

    @property
    def usable(self) -> list[Candidate]:
        return sorted((c for c in self.cands if c.usable), key=lambda c: c.score)

    @property
    def best(self) -> Candidate | None:
        u = self.usable
        return u[0] if u else None

    def recent_failures(self, n: int = 3, since_step: int | None = None) -> list[Candidate]:
        fails = [c for c in self.cands if not c.usable and c.code]
        if since_step is not None:
            fails = [c for c in fails if c.step >= since_step]
        return fails[-n:]

    def leaderboard(self, n: int = 3) -> list[Candidate]:
        return self.usable[:n]

    def summary(self) -> dict:
        stages: dict[str, int] = {}
        for c in self.cands:
            k = "ok" if c.usable else c.stage
            stages[k] = stages.get(k, 0) + 1
        return {"candidates": len(self.cands), "steps": self.last_step,
                "by_stage": stages, "best": self.best.score if self.best else None}

    # ---- parent selection --------------------------------------------------
    def select_parent(self, mode: str = "puct", top_k: int = 4, c_explore: float = 1.0,
                      rng: random.Random | None = None) -> Candidate | None:
        """Choose which prior solution the next sample should improve on.

        best : always the current best (plain hill climbing, the old behaviour).
        puct : among the top_k usable states, score = normalized quality +
               c_explore * sqrt(ln(total_visits+1) / (visits+1)), then softmax-sample.
               This keeps a few good-but-not-best lineages alive, in the spirit
               of TTT-Discover's state sampler, without needing the RL stack.
        """
        u = self.usable
        if not u:
            return None
        if mode == "best" or len(u) == 1:
            return u[0]
        rng = rng or random.Random()
        top = u[:top_k]
        lo, hi = top[0].score, top[-1].score
        total = sum(self.visits.get(c.id, 0) for c in top)
        prefs = []
        for c in top:
            quality = 1.0 if hi == lo else (hi - c.score) / (hi - lo)      # 1 = best
            explore = c_explore * math.sqrt(math.log(total + 1) / (self.visits.get(c.id, 0) + 1))
            prefs.append(quality + explore)
        m = max(prefs)
        w = [math.exp(2.0 * (p - m)) for p in prefs]
        return rng.choices(top, weights=w, k=1)[0]
