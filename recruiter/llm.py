"""LLM provider abstraction.

Every provider implements ``complete_json(system, user, schema)`` and returns a
Python dict. Agents never talk to a vendor SDK directly, so swapping Ollama for
an external API (or for the offline rules mode) is a one-line config change.

Providers:
    * ``OllamaProvider``  - local inference, the default when Ollama is running.
    * ``OpenAIProvider``  - optional, needs OPENAI_API_KEY and the ``openai`` package.
    * ``AnthropicProvider`` - optional, needs ANTHROPIC_API_KEY and ``anthropic``.
    * ``None`` (no provider) - agents fall back to deterministic rule-based logic.
"""

from __future__ import annotations

import json
import os
import platform
import re
import time
from dataclasses import dataclass, field
from typing import Any

import requests
from pydantic import BaseModel, ValidationError

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


class LLMError(RuntimeError):
    pass


@dataclass
class LLMCallLog:
    provider: str
    model: str
    duration_ms: float
    ok: bool
    error: str = ""


class LLMProvider:
    name = "base"

    def __init__(self, model: str, temperature: float = 0.1, timeout: int = 300):
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.calls: list[LLMCallLog] = []

    @property
    def label(self) -> str:
        return f"{self.name}:{self.model}"

    def _raw_json(self, system: str, user: str, schema: dict[str, Any]) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            text = self._raw_json(system, user, schema)
            data = extract_json(text)
        except Exception as exc:
            self.calls.append(LLMCallLog(self.name, self.model, _ms(start), False, str(exc)[:200]))
            raise LLMError(str(exc)) from exc
        self.calls.append(LLMCallLog(self.name, self.model, _ms(start), True))
        return data


class OllamaProvider(LLMProvider):
    name = "ollama"

    def _raw_json(self, system: str, user: str, schema: dict[str, Any]) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            # Ollama >= 0.5 accepts a JSON schema here and constrains decoding to it.
            "format": schema or "json",
            # Small context + output cap keeps CPU-only inference tolerable.
            "options": {"temperature": self.temperature, "num_ctx": 4096, "num_predict": 1200},
            "keep_alive": "15m",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["message"]["content"]


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", **kw: Any):
        super().__init__(model, **kw)
        from openai import OpenAI  # optional dependency

        self._client = OpenAI()

    def _raw_json(self, system: str, user: str, schema: dict[str, Any]) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system + "\nRespond with JSON only."},
                {"role": "user", "content": user},
            ],
            timeout=self.timeout,
        )
        return resp.choices[0].message.content or ""


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str = "claude-haiku-5-5", **kw: Any):
        super().__init__(model, **kw)
        import anthropic  # optional dependency

        self._client = anthropic.Anthropic()

    def _raw_json(self, system: str, user: str, schema: dict[str, Any]) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=self.temperature,
            system=system + "\nRespond with a single JSON object and nothing else.",
            messages=[{"role": "user", "content": user + "\n\nJSON schema:\n" + json.dumps(schema)}],
            timeout=self.timeout,
        )
        return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")


# --------------------------------------------------------------------------- #
# Structured-output helpers
# --------------------------------------------------------------------------- #
def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of an LLM reply (handles ``` fences and chatter)."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise LLMError("no JSON object in model output")
        candidate = text[start : end + 1]
        # Common small-model slips: trailing commas.
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        value = json.loads(candidate)
    if not isinstance(value, dict):
        raise LLMError("model output is not a JSON object")
    return value


def structured_call(
    provider: LLMProvider,
    system: str,
    user: str,
    model_cls: type[BaseModel],
    max_attempts: int = 2,
) -> BaseModel:
    """Call the provider and validate into ``model_cls``, re-prompting once with the validation error."""
    schema = model_cls.model_json_schema()
    last_error = ""
    prompt = user
    for _ in range(max_attempts):
        data = provider.complete_json(system, prompt, schema)
        try:
            return model_cls.model_validate(data)
        except ValidationError as exc:
            last_error = str(exc)[:1500]
            prompt = (
                f"{user}\n\nYour previous answer failed validation:\n{last_error}\n"
                "Return corrected JSON that matches the schema exactly."
            )
    raise LLMError(f"structured output failed validation: {last_error}")


# --------------------------------------------------------------------------- #
# Environment detection
# --------------------------------------------------------------------------- #
@dataclass
class Environment:
    os: str
    python: str
    cpu: str
    gpu: str
    ollama_running: bool
    ollama_models: list[str] = field(default_factory=list)
    openai_key: bool = False
    anthropic_key: bool = False

    def provider_options(self) -> list[str]:
        options = ["rules"]
        options += [f"ollama:{m}" for m in self.ollama_models]
        if self.openai_key:
            options.append("openai:gpt-4o-mini")
        if self.anthropic_key:
            options.append("anthropic:claude-haiku-5-5")
        return options

    def recommended_provider(self) -> str:
        for preferred in ("llama3.2", "qwen2.5", "llama3.1", "mistral", "phi3", "llama3"):
            for m in self.ollama_models:
                if m.startswith(preferred):
                    return f"ollama:{m}"
        if self.ollama_models:
            return f"ollama:{self.ollama_models[0]}"
        return "rules"


def ollama_models(timeout: float = 2.0) -> list[str]:
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=timeout)
        resp.raise_for_status()
        return sorted(m["name"] for m in resp.json().get("models", []))
    except Exception:
        return []


def _detect_gpu() -> str:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "none detected (CPU inference)"


def detect_environment() -> Environment:
    models = ollama_models()
    return Environment(
        os=f"{platform.system()} {platform.release()}",
        python=platform.python_version(),
        cpu=platform.processor() or platform.machine(),
        gpu=_detect_gpu(),
        ollama_running=bool(models) or _ollama_up(),
        ollama_models=models,
        openai_key=bool(os.environ.get("OPENAI_API_KEY")),
        anthropic_key=bool(os.environ.get("ANTHROPIC_API_KEY")),
    )


def _ollama_up() -> bool:
    try:
        return requests.get(f"{OLLAMA_URL}/api/version", timeout=1.5).ok
    except Exception:
        return False


def make_provider(spec: str | None, temperature: float = 0.1) -> LLMProvider | None:
    """Build a provider from a spec like ``ollama:llama3.2:latest`` or ``rules``."""
    if not spec or spec == "rules":
        return None
    kind, _, model = spec.partition(":")
    if kind == "ollama":
        return OllamaProvider(model or "llama3.2", temperature=temperature)
    if kind == "openai":
        return OpenAIProvider(model or "gpt-4o-mini", temperature=temperature)
    if kind == "anthropic":
        return AnthropicProvider(model or "claude-haiku-5-5", temperature=temperature)
    raise ValueError(f"unknown provider spec: {spec}")


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
