"""`harness replay` (issue #5): re-scoring a saved baseline offline, with no
adapter or provider client ever constructed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from harnesskit.adapters import ADAPTERS
from harnesskit.cli.main import app
from harnesskit.eval import replay_suite
from harnesskit.parser import load_harness
from harnesskit.trace.schema import Step, StepType, Trajectory

EXAMPLE = Path(__file__).parent.parent / "examples" / "react-web-researcher"
FIXTURE_BASELINE = EXAMPLE / "eval" / "fixtures" / "fixture-demo.baseline.json"


def test_fixture_baseline_replays_one_pass_one_understandable_failure():
    result = CliRunner().invoke(app, ["replay", str(EXAMPLE), "--baseline", str(FIXTURE_BASELINE)])
    assert result.exit_code == 1, result.output  # elevation-lookup's exact_output mismatch is a real failure
    assert "capital-lookup" in result.output
    assert "elevation-lookup" in result.output
    assert "missing=0" in result.output


def test_replay_never_touches_adapter_construction(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("replay must not construct a provider adapter")

    for adapter_cls in ADAPTERS.values():
        monkeypatch.setattr(adapter_cls, "build", _boom)
        monkeypatch.setattr(adapter_cls, "run", _boom)

    result = CliRunner().invoke(app, ["replay", str(EXAMPLE), "--baseline", str(FIXTURE_BASELINE)])
    assert "must not construct" not in result.output  # i.e. _boom was never reached
    assert result.exit_code == 1  # still the expected scoring failure, not a crash


def test_missing_case_trajectory_is_reported_and_blocks_by_default(tmp_path):
    result_load = load_harness(EXAMPLE)
    only_one = {"capital-lookup": _trivial_trajectory("capital-lookup")}
    partial_path = tmp_path / "partial.json"
    partial_path.write_text(json.dumps({k: json.loads(v.model_dump_json()) for k, v in only_one.items()}))

    suite = replay_suite(result_load.spec, only_one)
    assert suite.missing_case_ids == ["elevation-lookup"]

    result = CliRunner().invoke(app, ["replay", str(EXAMPLE), "--baseline", str(partial_path)])
    assert result.exit_code == 1
    assert "missing=1" in result.output
    assert "elevation-lookup" in result.output
    assert "refusing to report a complete result" in result.output


def test_missing_case_trajectory_with_partial_flag_scores_what_exists(tmp_path):
    only_one = {"capital-lookup": _trivial_trajectory("capital-lookup")}
    partial_path = tmp_path / "partial.json"
    partial_path.write_text(json.dumps({k: json.loads(v.model_dump_json()) for k, v in only_one.items()}))

    result = CliRunner().invoke(app, ["replay", str(EXAMPLE), "--baseline", str(partial_path), "--partial"])
    assert "capital-lookup" in result.output
    assert "missing=1" in result.output


def _trivial_trajectory(case_id: str) -> Trajectory:
    return Trajectory(
        harness_name="react-web-researcher",
        harness_version="0.1.0",
        model_id="claude-sonnet-5",
        adapter="fixture",
        input="x",
        steps=[Step(step_type=StepType.tool_call, tool_name="search", tool_args={})],
        final_output="The capital of Brazil is Brasília.",
        stopped_reason="tag_emitted:answer",
    )
