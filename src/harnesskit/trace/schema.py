"""Trace schema (design doc §8.1): what a harness run captures.

A Trajectory is the unit every scorer in the eval engine consumes. Adapters
are responsible for producing one; nothing downstream ever looks at
framework-native run objects.
"""
from __future__ import annotations

from enum import Enum
from time import time

from pydantic import BaseModel, Field


class StepType(str, Enum):
    llm_call = "llm_call"
    tool_call = "tool_call"
    guardrail_event = "guardrail_event"
    compaction_event = "compaction_event"


class Step(BaseModel):
    step_type: StepType
    input: str | None = None
    output: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_result: str | None = None
    note: str | None = None  # e.g. which guardrail fired, or the compaction strategy applied


class Trajectory(BaseModel):
    harness_name: str
    harness_version: str
    model_id: str
    adapter: str
    input: str
    started_at: float = Field(default_factory=time)
    steps: list[Step] = Field(default_factory=list)
    final_output: str | None = None
    stopped_reason: str | None = None  # which termination condition fired

    @property
    def turns(self) -> int:
        return sum(1 for s in self.steps if s.step_type == StepType.llm_call)

    @property
    def tool_calls(self) -> list[Step]:
        return [s for s in self.steps if s.step_type == StepType.tool_call]

    @property
    def total_cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.steps)

    @property
    def total_tokens(self) -> int:
        return sum(s.tokens_in + s.tokens_out for s in self.steps)

    @property
    def duration_ms(self) -> int:
        return sum(s.duration_ms for s in self.steps)

    @property
    def duplicate_tool_calls(self) -> int:
        seen: set[tuple[str, str]] = set()
        dupes = 0
        for s in self.tool_calls:
            key = (s.tool_name or "", str(s.tool_args or {}))
            if key in seen:
                dupes += 1
            seen.add(key)
        return dupes


# Blended $/1M-token rates, used only where a real provider rate isn't known.
# Adapters that call a real API should compute cost_usd per-step directly.
FALLBACK_PRICE_PER_1M = {
    "claude-sonnet-5": 6.0,
    "claude-opus-5": 20.0,
    "claude-haiku-4-5-20251001": 1.2,
}


def estimate_cost_usd(model_id: str, tokens_in: int, tokens_out: int) -> float:
    rate = FALLBACK_PRICE_PER_1M.get(model_id, 6.0)
    return (tokens_in + tokens_out) / 1_000_000 * rate
