"""Unit tests for the callback adapter (issue #9, P1 rest)."""
from __future__ import annotations

from harnesskit.adapters.base import FeatureStatus, inspect_support
from harnesskit.adapters.callback_adapter import CallbackAdapter, wrap_simple_callback
from harnesskit.testing.conformance import make_spec
from harnesskit.trace.schema import AttemptStatus, CostStatus


def test_wraps_a_plain_function_without_rewriting_its_loop():
    def existing_agent(question: str) -> str:
        return f"answer to: {question}"

    adapter = CallbackAdapter(callback=wrap_simple_callback(existing_agent))
    spec = make_spec()
    agent = adapter.build(spec)
    trajectory = adapter.run(agent, "what is 2+2?")

    assert trajectory.final_output == "answer to: what is 2+2?"
    assert trajectory.stopped_reason == "callback_returned"
    assert len(trajectory.steps) == 1
    assert trajectory.steps[0].attempt_status == AttemptStatus.ok


def test_records_wall_clock_duration_and_marks_cost_unavailable():
    adapter = CallbackAdapter(callback=wrap_simple_callback(lambda q: "ok"))
    spec = make_spec()
    agent = adapter.build(spec)
    trajectory = adapter.run(agent, "hi")

    step = trajectory.steps[0]
    assert step.duration_ms >= 0
    assert step.cost_usd is None
    assert step.cost_status == CostStatus.unavailable


def test_raising_callback_is_surfaced_not_crashed():
    def flaky(_question: str, _context: dict) -> str:
        raise RuntimeError("kaboom")

    adapter = CallbackAdapter(callback=flaky)
    spec = make_spec()
    agent = adapter.build(spec)
    trajectory = adapter.run(agent, "hi")

    assert trajectory.final_output is None
    assert trajectory.stopped_reason == "callback_error"
    assert trajectory.steps[0].attempt_status == AttemptStatus.failed
    assert "kaboom" in trajectory.steps[0].note


def test_feature_status_marks_limits_unavailable_not_silently_enforced():
    adapter = CallbackAdapter(callback=wrap_simple_callback(lambda q: q))
    status = adapter.feature_status()

    assert status["termination.timeout"] == FeatureStatus.unavailable
    assert status["termination.budget_exhaustion"] == FeatureStatus.unavailable
    assert status["cost.enforcement"] == FeatureStatus.unavailable
    # what it CAN do is explicitly "observed", not "controlled" — harnesskit
    # sees the callback's output/duration but does not enforce anything.
    assert status["trace.final_output"] == FeatureStatus.observed
    assert status["trace.wall_clock_duration"] == FeatureStatus.observed


def test_declared_capabilities_are_narrow_and_checkable_via_inspect_support():
    adapter = CallbackAdapter(callback=wrap_simple_callback(lambda q: q))
    spec = make_spec(tools=[])
    findings = inspect_support(spec, adapter)
    # a plain callback adapter supports none of the loop strategies the
    # default fixture spec declares (loop.type defaults to "react" in
    # HarnessSpec, while CallbackAdapter only claims "custom").
    assert any(f.field == "loop.type" for f in findings)
