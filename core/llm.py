"""LLM samplers: the Claude API (native SDK, used when ANTHROPIC_API_KEY is set),
an OpenAI-compatible chat endpoint, and a fake sampler for pipeline tests."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Sample:
    text: str
    finish_reason: str | None
    model: str
    usage: dict


class OpenAICompatSampler:
    def __init__(self, model: str | None = None, max_tokens: int | None = None,
                 temperature: float = 1.0, retries: int = 2):
        base = (os.environ.get("ANTHROPIC_BASE_URL")
                or os.environ.get("TOOLCHAIN_BASE_URL", "https://llms.toolchain.ai"))
        self.base_url = base.rstrip("/") + "/v1"
        self.api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("TOOLCHAIN_API_KEY", "")
        self.model = model or os.environ.get("TOOLCHAIN_MODEL", "claude-sonnet-4-6")
        # 16k default: a fully unrolled kernel plus explanation easily exceeds the old 4k cap.
        self.max_tokens = max_tokens or int(os.environ.get("TOOLCHAIN_MAX_TOKENS", "16000"))
        self.temperature = temperature
        self.retries = retries
        self.timeout = httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=600.0)
        if not self.api_key:
            raise RuntimeError("No API key: set TOOLCHAIN_API_KEY (or ANTHROPIC_AUTH_TOKEN) in .env")

    def describe(self) -> str:
        return f"{self.model} @ {self.base_url} (max_tokens={self.max_tokens}, T={self.temperature})"

    def sample(self, prompt: str) -> Sample:
        last = None
        for attempt in range(self.retries + 1):
            try:
                r = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "Content-Type": "application/json"},
                    json={"model": self.model,
                          "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": self.max_tokens,
                          "temperature": self.temperature},
                    timeout=self.timeout,
                )
                r.raise_for_status()
                j = r.json()
                ch = j["choices"][0]
                return Sample(text=ch["message"]["content"] or "",
                              finish_reason=ch.get("finish_reason"),
                              model=j.get("model", self.model), usage=j.get("usage", {}))
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(2 * (attempt + 1))
        raise last


class FakeSampler:
    """Returns the baseline (and trivially perturbed variants) — exercises the
    whole loop, evaluator and trace without spending tokens."""
    def __init__(self, baseline_code: str, language: str):
        self.baseline = baseline_code
        self.language = language
        self.n = 0

    def describe(self) -> str:
        return "fake sampler (baseline echo)"

    def sample(self, prompt: str) -> Sample:
        self.n += 1
        variants = [
            self.baseline,
            "// variant: identical to baseline, comment added\n" + self.baseline,
            "I could not finish.",                                  # no code -> extract failure
            self.baseline.replace("run_inference", "run_inferenc"),  # compile failure
        ]
        code = variants[(self.n - 1) % len(variants)]
        text = ("## Changes vs. previous version\n"
                f"- (fake sampler, variant {(self.n - 1) % len(variants) + 1}) echo of the baseline; "
                "no real optimization, this exercises the pipeline only.\n\n"
                f"Here is my optimized kernel.\n```{self.language}\n{code}\n```\n")
        return Sample(text=text, finish_reason="stop", model="fake", usage={})


class AnthropicSampler:
    """Claude via the official SDK. Reads ANTHROPIC_API_KEY from the environment / .env.
    Thinking is always on for Claude 5-series models; depth is controlled by `effort`
    (low|medium|high|xhigh|max), not temperature (which these models reject)."""
    def __init__(self, model: str | None = None, max_tokens: int | None = None,
                 effort: str | None = None, retries: int = 2):
        import anthropic  # local import so the OpenAI-compat path needs no SDK
        self.client = anthropic.Anthropic(max_retries=retries, timeout=900.0)
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-opus-5-5")
        self.max_tokens = max_tokens or int(os.environ.get("ANTHROPIC_MAX_TOKENS", "16000"))
        self.effort = effort or os.environ.get("ANTHROPIC_EFFORT", "high")

    def describe(self) -> str:
        return f"{self.model} @ Claude API (max_tokens={self.max_tokens}, effort={self.effort})"

    def sample(self, prompt: str) -> Sample:
        # Streaming keeps long kernel outputs clear of HTTP timeouts. `fallbacks: default`
        # re-runs a safety-declined request on another model inside the same call.
        with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            extra_body={"fallbacks": "default"},
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            msg = stream.get_final_message()
        text = "".join(b.text for b in msg.content if b.type == "text")
        if msg.stop_reason == "refusal":
            text = ""  # -> recorded as an extract failure, not a crash
        finish = {"end_turn": "stop", "max_tokens": "length"}.get(msg.stop_reason, msg.stop_reason)
        u = msg.usage
        usage = {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                 "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0}
        return Sample(text=text, finish_reason=finish, model=msg.model, usage=usage)


def make_sampler(model: str | None = None, max_tokens: int | None = None,
                 temperature: float = 1.0, effort: str | None = None):
    """Claude API when ANTHROPIC_API_KEY is set, otherwise the OpenAI-compatible endpoint."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicSampler(model=model, max_tokens=max_tokens, effort=effort)
    return OpenAICompatSampler(model=model, max_tokens=max_tokens, temperature=temperature)
