import json
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recruiter.llm import LLMProvider  # noqa: E402

SAMPLES = ROOT / "data" / "samples"


class FakeProvider(LLMProvider):
    """Scripted LLM: ``responder(system, user)`` returns a dict (or raises)."""

    name = "fake"

    def __init__(self, responder: Callable[[str, str], Any]):
        super().__init__("fake-model")
        self.responder = responder
        self.prompts: list[tuple[str, str]] = []

    def _raw_json(self, system: str, user: str, schema: dict[str, Any]) -> str:
        self.prompts.append((system, user))
        out = self.responder(system, user)
        return out if isinstance(out, str) else json.dumps(out)


@pytest.fixture
def fake_provider():
    return FakeProvider


@pytest.fixture(scope="session")
def jobs() -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in sorted((SAMPLES / "jobs").glob("*.txt"))}


@pytest.fixture(scope="session")
def resumes() -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in sorted((SAMPLES / "resumes").glob("*.txt"))}
