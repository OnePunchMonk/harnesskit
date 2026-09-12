"""Callback adapter (issue #9, P1) — "bring your own agent" without
rewriting its internal loop.

Many real agents are not built on harnesskit's own loop strategies at all:
they're a plain Python entrypoint, e.g. `def run_agent(task: str) -> str`,
that internally does whatever it does (a hand-rolled loop, a third-party
framework, a hosted service call) and returns a final answer. This adapter
wraps such a callable as a `HarnessAdapter` so it can be observed, scored,
and compared like any other harness — without harnesskit ever touching its
internals.

This is deliberately the thinnest possible adapter: it calls the callback
once per `run()`, times it wall-clock, and records whatever the callback
returns. It CANNOT:

- enforce the wrapped agent's own timeout or max-turns (there are no turns
  to enforce — the callback is one opaque call from harnesskit's point of
  view);
- enforce the wrapped agent's own spend limit (harnesskit has no visibility
  into what the callback itself may have spent internally, e.g. on its own
  provider calls);
- record individual tool calls or intermediate steps (the callback is a
  black box; only its final output and wall-clock duration are observed).

See `feature_status()` below for the explicit controlled/observed/unavailable
labeling of each of these, consumed by `harness inspect` and by
`testing.conformance`-style diagnostics.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from harnesskit.adapters.base import AdapterCapabilities, FeatureStatus, RunnableAgent
from harnesskit.format.spec import HarnessSpec
from harnesskit.trace.schema import AttemptStatus, CostStatus, Step, StepType, Trajectory

# A callback agent's entrypoint: (input_text, context) -> output_text. The
# `context` dict is optional extra state a caller's agent might want (e.g.
# conversation history); a callback that only needs the input text can
# ignore its second argument entirely and still be wrapped (see
# `wrap_simple_callback`).
CallbackFn = Callable[[str, dict[str, Any]], str]


@dataclass
class CallbackAdapter:
    """Wraps `callback` as a `HarnessAdapter`. `model_id`/`runtime_name` are
    free-form labels for the trace/trajectory since a callback-wrapped agent
    may not have (or disclose) a single "model" at all."""

    callback: CallbackFn
    model_id: str = "external:callback"
    runtime_name: str = "callback"

    def supports(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            loop_strategies={"custom"},
            tool_sources=set(),
            hook_points=set(),
            memory_backends={"none"},
            runtime=self.runtime_name,
        )

    def feature_status(self) -> dict[str, FeatureStatus]:
        """Explicit controlled/observed/unavailable labeling for this
        adapter. `harness inspect` (or any capability-diagnostic caller)
        should surface this alongside `supports()` so a developer doesn't
        mistake "we can see the output" for "we enforce the limit"."""
        return {
            "termination.timeout": FeatureStatus.unavailable,
            "termination.budget_exhaustion": FeatureStatus.unavailable,
            "loop.max_turns": FeatureStatus.unavailable,
            "trace.tool_calls": FeatureStatus.unavailable,
            "trace.wall_clock_duration": FeatureStatus.observed,
            "trace.final_output": FeatureStatus.observed,
            "cost.enforcement": FeatureStatus.unavailable,
        }

    def build(self, spec: HarnessSpec) -> RunnableAgent:
        return RunnableAgent(spec=spec, handle=self.callback)

    def run(self, agent: RunnableAgent, input: str) -> Trajectory:
        spec = agent.spec
        callback = agent.handle
        trajectory = Trajectory(
            harness_name=spec.metadata.name,
            harness_version=spec.metadata.version,
            model_id=self.model_id,
            adapter=self.runtime_name,
            input=input,
        )
        t0 = time.time()
        try:
            output = callback(input, {})
            attempt_status = AttemptStatus.ok
            note = (
                "observed: wall-clock duration and returned output only. "
                "harnesskit does not enforce this callback's own internal "
                "timeout or spend limits — see CallbackAdapter.feature_status()."
            )
        except Exception as e:  # noqa: BLE001 — a raising callback must stay visible, not crash the suite
            output = None
            attempt_status = AttemptStatus.failed
            note = f"callback raised: {e}"
        duration_ms = int((time.time() - t0) * 1000)

        trajectory.steps.append(
            Step(
                step_type=StepType.llm_call,
                input=input,
                output=output,
                cost_usd=None,
                cost_status=CostStatus.unavailable,
                attempt_status=attempt_status,
                duration_ms=duration_ms,
                note=note,
            )
        )
        trajectory.final_output = output
        trajectory.stopped_reason = "callback_returned" if attempt_status == AttemptStatus.ok else "callback_error"
        return trajectory


def wrap_simple_callback(fn: Callable[[str], str]) -> CallbackFn:
    """Adapt a single-argument callback `(input) -> output` (the common case
    for "just call my existing agent's entrypoint function") to the
    `CallbackFn` shape `CallbackAdapter` expects."""

    def _wrapped(input: str, _context: dict[str, Any]) -> str:
        return fn(input)

    return _wrapped
