"""Generic LLM-driven kernel optimization loop (target-agnostic).

Targets (targets/<name>/target.py) provide: problem loading, prompt building,
and an evaluator that returns an EvalResult. The core owns sampling, code
extraction, the candidate archive, parent selection, failure feedback,
append-only tracing and resume.
"""
from core.types import Candidate, EvalResult, Problem
from core.archive import Archive

__all__ = ["Candidate", "EvalResult", "Problem", "Archive"]
