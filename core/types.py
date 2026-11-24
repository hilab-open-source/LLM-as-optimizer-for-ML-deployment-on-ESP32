from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

FAIL = float("inf")


@dataclass
class Problem:
    """One optimization problem: a baseline kernel + its answer key, loaded by a target."""
    name: str
    dir: Path
    baseline_code: str
    arch_text: str
    metric_name: str              # e.g. "ESP32 CPU cycles per inference (median of 31)"
    language: str                 # fence language shown to the LLM ("c", ...)
    entry_symbol: str             # must appear in extracted code ("run_inference")
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    correct: bool
    score: float                  # lower is better; FAIL when not usable
    msg: str
    stage: str = "ok"             # extract | compile | run | correctness | ok
    metrics: dict[str, Any] = field(default_factory=dict)
    stdout: str = ""

    @staticmethod
    def fail(stage: str, msg: str, **metrics) -> "EvalResult":
        return EvalResult(correct=False, score=FAIL, msg=msg, stage=stage, metrics=metrics)


@dataclass
class Candidate:
    """One evaluated LLM sample. Persisted as one JSON line in trace.jsonl."""
    id: str
    step: int
    parent_id: str | None
    code: str
    raw_response: str
    correct: bool
    score: float | None           # None == failed
    msg: str
    stage: str
    metrics: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    wall_s: float = 0.0
    created_at: float = field(default_factory=time.time)

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:12]

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Candidate":
        known = {f for f in Candidate.__dataclass_fields__}
        return Candidate(**{k: v for k, v in d.items() if k in known})

    @property
    def usable(self) -> bool:
        return self.correct and self.score is not None
