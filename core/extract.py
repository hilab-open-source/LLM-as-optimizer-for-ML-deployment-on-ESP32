"""Robust code extraction from LLM responses.

An earlier version only accepted a fence spelled exactly ```c, which threw
away 45 of 50 candidates in one run. This accepts any fence language (or none),
prefers the last block that defines the entry symbol, and falls back to the
longest block, then to an unfenced response that still contains the entry.
"""
from __future__ import annotations

import re

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+#.-]*)[^\n]*\n(.*?)(?:\n```|$)", re.DOTALL)


def extract_code(response: str, entry_symbol: str) -> tuple[str | None, str]:
    """Return (code, how). `how` explains which rule fired (kept in the trace)."""
    if not response or not response.strip():
        return None, "empty response"
    blocks = [(m.group(1).lower(), m.group(2).strip()) for m in _FENCE.finditer(response)]
    blocks = [(lang, body) for lang, body in blocks if body]
    with_entry = [b for b in blocks if entry_symbol in b[1]]
    if with_entry:
        lang, body = with_entry[-1]
        return body, f"last fenced block containing {entry_symbol} (lang={lang or 'none'})"
    if blocks:
        lang, body = max(blocks, key=lambda b: len(b[1]))
        return body, f"longest fenced block (lang={lang or 'none'}; entry symbol missing)"
    if entry_symbol in response:
        return response.strip(), "unfenced response containing entry symbol"
    return None, "no code block and no entry symbol in response"


_RATIONALE_HEAD = re.compile(
    r"(?im)^[ \t]*(?:#+[ \t]*|\*\*)?[ \t]*(?:changes?|what changed|what I changed|why (?:it|this)(?: is|'s)? faster|optimizations? (?:made|applied))\b")


def extract_rationale(response: str, max_chars: int = 3000) -> str:
    """The LLM's own explanation of what it changed and why: the prose outside code
    fences, starting at a 'Changes ...' heading when there is one."""
    if not response:
        return ""
    prose = _FENCE.sub("", response)
    m = _RATIONALE_HEAD.search(prose)
    text = prose[m.start():] if m else prose
    lines = [l.rstrip() for l in text.strip().splitlines()]
    if m and lines and _RATIONALE_HEAD.match(lines[0]) and len(lines[0]) < 80:
        lines = lines[1:]  # the heading itself is implied by where the text is stored
    # drop lead-in / trailing filler such as "Here is the complete kernel:"
    while lines and re.match(r"(?i)^\s*(here(?:'s| is)\b.*|the (?:complete|final|full) (?:kernel|source|code).*:?)\s*$", lines[-1]):
        lines.pop()
    out = "\n".join(lines).strip()
    return out[:max_chars] + (" […]" if len(out) > max_chars else "")
