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


class CostStatus(str, Enum):
    """Provenance of a Step's `cost_usd`. Never let the absence of a number
    silently render as $0 — that's indistinguishable from a genuinely free
    call and hides real spend."""

    observed = "observed"  # actual cost reported by the provider/billing response
    estimated = "estimated"  # computed from a local pricing table, not billed truth
    known_zero = "known_zero"  # verified free (e.g. free-tier, cached-hit) call
    unavailable = "unavailable"  # model isn't in the pricing table and no observed cost exists


class AttemptStatus(str, Enum):
    """A call that failed or was retried still consumed tokens and may still
    have cost money — it must not be omitted from the trace just because it
    didn't produce a usable output."""

    ok = "ok"
    failed = "failed"
    retried = "retried"


class Step(BaseModel):
    step_type: StepType
    input: str | None = None
    output: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    tokens_observed: bool = True
    """False when tokens_in/tokens_out are themselves estimated (e.g. a
    tokenizer approximation) rather than read off a provider usage response."""
    cost_usd: float | None = None
    """None means "unavailable" — no data, not zero. See CostStatus."""
    cost_status: CostStatus | None = None
    attempt_status: AttemptStatus = AttemptStatus.ok
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
        """Sum of known costs only. A step with `cost_usd is None`
        (unavailable) contributes nothing here — see `has_unavailable_cost`
        to detect when that means this total understates the real cost."""
        return sum(s.cost_usd for s in self.steps if s.cost_usd is not None)

    @property
    def has_unavailable_cost(self) -> bool:
        """True if any step's cost is unknown — `total_cost_usd` for this
        trajectory should not be trusted as a complete bill."""
        return any(s.cost_status == CostStatus.unavailable for s in self.steps)

    @property
    def total_tokens(self) -> int:
        return sum((s.tokens_in or 0) + (s.tokens_out or 0) for s in self.steps)

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


# Versioned $/1M-token pricing table, used only where a real provider rate
# isn't known. Adapters that receive an actual billed cost from the provider
# should use CostStatus.observed instead of calling this at all. Bump
# PRICING_TABLE_VERSION whenever rates change, so a stored trajectory's cost
# can be understood against the table that produced it.
PRICING_TABLE_VERSION = "2026-09"
PRICE_PER_1M_USD: dict[str, float] = {
    "claude-sonnet-5": 6.0,
    "claude-opus-5": 20.0,
    "claude-haiku-4-5-20251001": 1.2,
}


def estimate_cost_usd(model_id: str, tokens_in: int, tokens_out: int) -> tuple[float | None, CostStatus]:
    """Estimate cost from the versioned pricing table.

    Returns `(None, CostStatus.unavailable)` for a model not in the table —
    callers must not treat that as $0. A model explicitly priced at 0.0 in
    the table is reported as `(0.0, CostStatus.known_zero)`, distinguishing
    "actually free" from "we don't know."
    """
    rate = PRICE_PER_1M_USD.get(model_id)
    if rate is None:
        return None, CostStatus.unavailable
    if rate == 0.0:
        return 0.0, CostStatus.known_zero
    return (tokens_in + tokens_out) / 1_000_000 * rate, CostStatus.estimated
