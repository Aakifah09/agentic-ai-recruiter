"""Common agent behaviour: timing, tracing and LLM-with-fallback execution."""

from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

from recruiter.llm import LLMError, LLMProvider
from recruiter.schemas import AgentStep

log = logging.getLogger(__name__)
T = TypeVar("T")


class Agent:
    """Base class. Subclasses implement ``_run_rules`` and optionally ``_run_llm``.

    ``run`` tries the LLM path when a provider is configured and falls back to
    the deterministic rules path on any provider or validation error, recording
    what happened in a trace step. The workflow therefore always completes, even
    with Ollama switched off.
    """

    name = "agent"
    description = ""

    def __init__(self, provider: LLMProvider | None = None):
        self.provider = provider
        self.last_step: AgentStep | None = None

    def _execute(self, llm_fn: Callable[[], T] | None, rules_fn: Callable[[], T], detail: str = "") -> T:
        start = time.perf_counter()
        step = AgentStep(agent=self.name, mode="rules", detail=detail)
        result: T
        if self.provider is not None and llm_fn is not None:
            try:
                result = llm_fn()
                step.mode = "hybrid"
            except (LLMError, Exception) as exc:  # noqa: BLE001 - any LLM failure must degrade gracefully
                log.warning("%s: LLM path failed, using rules: %s", self.name, exc)
                step.status = "fallback"
                step.detail = f"{detail} | LLM failed ({type(exc).__name__}: {str(exc)[:120]}); used rules".strip(" |")
                result = rules_fn()
        else:
            result = rules_fn()
        step.duration_ms = round((time.perf_counter() - start) * 1000, 1)
        self.last_step = step
        return result
