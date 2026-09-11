"""Adapter conformance suite (issue #6): proves adapters behave the way their
declared capabilities claim, using scripted clients — no API key, no network.
"""
from __future__ import annotations

import pytest
from typer.testing import CliRunner

from harnesskit.adapters.base import AdapterCapabilities, RunnableAgent
from harnesskit.adapters.raw_api import RawAPIAdapter
from harnesskit.cli.main import app
from harnesskit.testing.conformance import RAW_API_CASES, generate_matrix, run_case
from harnesskit.trace.schema import Trajectory


def test_conformance_cli_command_passes_and_is_json_serializable():
    result = CliRunner().invoke(app, ["conformance", "--json"])
    assert result.exit_code == 0, result.output
    payload = __import__("json").loads(result.output)
    assert len(payload) == len(RAW_API_CASES)
    assert all(r["status"] == "pass" for r in payload)


@pytest.mark.parametrize("case", RAW_API_CASES, ids=[c.id for c in RAW_API_CASES])
def test_raw_api_adapter_conformance(case):
    result = run_case(RawAPIAdapter(), case, runtime="raw_api")
    assert result.status == "pass", result.detail


def test_conformance_matrix_reports_all_raw_api_cases_passing():
    adapter = RawAPIAdapter()
    results = [run_case(adapter, case, runtime="raw_api") for case in RAW_API_CASES]
    matrix = generate_matrix(results)
    assert all(r.status == "pass" for r in results)
    assert "raw_api" in matrix
    assert "❌" not in matrix and "⚠️" not in matrix


class BrokenMaxTurnsAdapter:
    """Deliberately ignores loop.max_turns: runs one extra turn past the cap.

    Used to prove the conformance suite actually fails a broken adapter
    rather than only ever passing well-behaved ones.
    """

    def supports(self) -> AdapterCapabilities:
        return AdapterCapabilities(loop_strategies={"react"}, tool_sources={"custom"}, hook_points=set(), memory_backends={"in_memory"}, runtime="broken")

    def build(self, spec):  # pragma: no cover - conformance bypasses build()
        raise NotImplementedError

    def run(self, agent: RunnableAgent, input: str) -> Trajectory:
        spec = agent.spec
        client = agent.handle["client"]
        callbacks = agent.handle["callbacks"]
        trajectory = Trajectory(
            harness_name=spec.metadata.name,
            harness_version=spec.metadata.version,
            model_id=spec.model.model_id,
            adapter="broken",
            input=input,
        )
        from harnesskit.trace.schema import Step, StepType

        # off-by-one: loops max_turns + 1 times instead of max_turns
        for _ in range(spec.loop.max_turns + 1):
            response = client.messages.create(model=spec.model.model_id, max_tokens=1, system="", messages=[])
            tool_blocks = [b for b in response.content if b.type == "tool_use"]
            trajectory.steps.append(Step(step_type=StepType.llm_call))
            if not tool_blocks:
                trajectory.stopped_reason = "no_tool_use"
                return trajectory
            for block in tool_blocks:
                callback = callbacks.get(block.name)
                result = str(callback(**(block.input or {}))) if callback else "error: no callback"
                trajectory.steps.append(Step(step_type=StepType.tool_call, tool_name=block.name, tool_args=block.input, tool_result=result))
        trajectory.stopped_reason = "max_turns"
        return trajectory


def test_broken_adapter_fails_max_turns_conformance():
    case = next(c for c in RAW_API_CASES if c.id == "max_turn_termination")
    result = run_case(BrokenMaxTurnsAdapter(), case, runtime="broken")
    assert result.status in ("fail", "error"), result
    assert result.status != "pass"
