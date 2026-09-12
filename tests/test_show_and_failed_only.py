"""`harness show` and `harness eval --failed-only` (issue #4 item 13): the
step-by-step debugging view for a failing eval case, reused between the two
commands via `render_trajectory`.
"""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from harnesskit.cli.main import app
from harnesskit.eval.mock_adapter import MockAdapter
from harnesskit.parser import load_harness
from harnesskit.trace.schema import Step, StepType, Trajectory

EXAMPLE = Path(__file__).parent.parent / "examples" / "react-web-researcher"


def _trajectory(input_text: str, final_output: str, stopped_reason: str) -> Trajectory:
    return Trajectory(
        harness_name="react-web-researcher",
        harness_version="0.1.0",
        model_id="claude-sonnet-5",
        adapter="mock",
        input=input_text,
        steps=[
            Step(step_type=StepType.llm_call, input=input_text, output="searching...", tokens_in=100, tokens_out=20, cost_usd=0.001),
            Step(step_type=StepType.tool_call, tool_name="search", tool_args={"query": "capital of Brazil"}, tool_result="Brasília"),
        ],
        final_output=final_output,
        stopped_reason=stopped_reason,
    )


def test_show_renders_tool_name_result_and_stopped_reason(tmp_path):
    traj = _trajectory("what is the capital of Brazil?", "Brasília", "tag_emitted:answer")
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"schema_version": 2, "trajectory": json.loads(traj.model_dump_json())}))

    result = CliRunner().invoke(app, ["show", str(path)])

    assert result.exit_code == 0, result.output
    assert "search" in result.output
    assert "Brasília" in result.output
    assert "tag_emitted:answer" in result.output


def test_show_baseline_style_file_requires_case_when_multiple(tmp_path):
    traj_a = _trajectory("q1", "a1", "tag_emitted:answer")
    traj_b = _trajectory("q2", "a2", "tag_emitted:answer")
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"a": json.loads(traj_a.model_dump_json()), "b": json.loads(traj_b.model_dump_json())}))

    result = CliRunner().invoke(app, ["show", str(path)])
    assert result.exit_code == 1
    assert "--case" in result.output

    result = CliRunner().invoke(app, ["show", str(path), "--case", "b"])
    assert result.exit_code == 0, result.output
    assert "a2" in result.output


def test_eval_failed_only_shows_detail_only_for_failing_case(monkeypatch, tmp_path):
    import shutil

    from harnesskit.cli import main as cli_main

    harness_copy = tmp_path / "harness_copy"
    shutil.copytree(EXAMPLE, harness_copy)

    result_load = load_harness(harness_copy)
    cases = result_load.spec.eval.cases
    assert len(cases) >= 2
    passing_case, failing_case = cases[0], cases[1]

    def _passing_output(case):
        if case.expected_output_contains:
            return " ".join(case.expected_output_contains)
        return case.ground_truth or "ok"

    passing_traj = _trajectory(passing_case.input, _passing_output(passing_case), "tag_emitted:answer")
    failing_traj = _trajectory(failing_case.input, "definitely-not-the-right-answer-xyz", "tag_emitted:answer")

    trajectories_by_input = {passing_case.input: passing_traj, failing_case.input: failing_traj}
    monkeypatch.setitem(cli_main.ADAPTERS, "test-mock", lambda: MockAdapter(trajectories_by_input=trajectories_by_input))

    result = CliRunner().invoke(cli_main.app, ["eval", str(harness_copy), "--adapter", "test-mock", "--failed-only"])

    assert failing_case.id in result.output
    assert "search" in result.output  # tool_call detail rendered for the failing case
    # The passing case's id appears in the summary table but not with its own step-detail block.
    assert f"== {failing_case.id} ==" in result.output
    assert f"== {passing_case.id} ==" not in result.output
