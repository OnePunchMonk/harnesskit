"""Parallel eval case execution (issue #4 item 10): `run_suite(..., max_workers=N)`
must be measurably faster than serial for I/O-bound cases and produce
identical, order-independent results.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from harnesskit.adapters.base import AdapterCapabilities, RunnableAgent
from harnesskit.eval.engine import CaseStatus, run_suite
from harnesskit.format.spec import HarnessSpec
from harnesskit.parser import load_harness
from harnesskit.trace.schema import Step, StepType, Trajectory

EXAMPLE = Path(__file__).parent.parent / "examples" / "react-web-researcher"


@dataclass
class _SlowAdapter:
    delay_s: float
    fail_inputs: frozenset[str] = frozenset()

    def supports(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            loop_strategies={"react", "plan_execute", "rewoo", "code_action", "custom"},
            tool_sources={"builtin", "mcp", "custom"},
            hook_points=set(),
            memory_backends={"none", "in_memory", "persistent"},
        )

    def build(self, spec: HarnessSpec) -> RunnableAgent:
        return RunnableAgent(spec=spec, handle=None)

    def run(self, agent: RunnableAgent, input: str) -> Trajectory:
        time.sleep(self.delay_s)
        if input in self.fail_inputs:
            raise RuntimeError(f"simulated failure for {input!r}")
        return Trajectory(
            harness_name="react-web-researcher",
            harness_version="0.1.0",
            model_id="claude-sonnet-5",
            adapter="slow",
            input=input,
            steps=[Step(step_type=StepType.llm_call, tokens_in=10, tokens_out=5, cost_usd=0.0001)],
            final_output=input,
            stopped_reason="tag_emitted:answer",
        )


def test_parallel_is_faster_than_serial_and_matches_results():
    result = load_harness(EXAMPLE)
    n_cases = len(result.spec.eval.cases)
    assert n_cases >= 2

    delay = 0.05
    serial_adapter = _SlowAdapter(delay_s=delay)
    parallel_adapter = _SlowAdapter(delay_s=delay)

    t0 = time.monotonic()
    serial_suite = run_suite(result.spec, serial_adapter, max_workers=1)
    serial_elapsed = time.monotonic() - t0

    t0 = time.monotonic()
    parallel_suite = run_suite(result.spec, parallel_adapter, max_workers=n_cases)
    parallel_elapsed = time.monotonic() - t0

    assert parallel_elapsed < serial_elapsed * 0.7

    serial_by_id = {r.case.id: r for r in serial_suite.results}
    parallel_by_id = {r.case.id: r for r in parallel_suite.results}
    assert serial_by_id.keys() == parallel_by_id.keys()
    for case_id in serial_by_id:
        assert serial_by_id[case_id].passed == parallel_by_id[case_id].passed


def test_parallel_execution_still_records_per_case_errors_as_errored():
    result = load_harness(EXAMPLE)
    failing_input = result.spec.eval.cases[0].input
    adapter = _SlowAdapter(delay_s=0.01, fail_inputs=frozenset({failing_input}))

    suite = run_suite(result.spec, adapter, max_workers=4)

    assert len(suite.results) == len(result.spec.eval.cases)
    failed = [r for r in suite.results if r.case.input == failing_input]
    assert len(failed) == 1
    assert failed[0].status == CaseStatus.errored
