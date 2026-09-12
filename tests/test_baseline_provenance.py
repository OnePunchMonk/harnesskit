"""Baseline provenance (issue #4 item 15): a baseline records what it was
captured with, and `harness eval --compare` warns loudly on drift instead of
silently reporting a difference as a regression.
"""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from harnesskit.cli.main import app
from harnesskit.eval.mock_adapter import MockAdapter
from harnesskit.trace import current_provenance, load_baseline_provenance
from harnesskit.trace import save_baseline as save_baseline_fn
from harnesskit.parser import load_harness
from harnesskit.trace.schema import Step, StepType, Trajectory

EXAMPLE = Path(__file__).parent.parent / "examples" / "react-web-researcher"


def _trajectory(input_text: str, model_id: str = "claude-sonnet-5") -> Trajectory:
    return Trajectory(
        harness_name="react-web-researcher",
        harness_version="0.1.0",
        model_id=model_id,
        adapter="mock",
        input=input_text,
        steps=[Step(step_type=StepType.llm_call, tokens_in=100, tokens_out=20, cost_usd=0.001)],
        final_output="ok",
        stopped_reason="tag_emitted:answer",
    )


def test_save_baseline_records_provenance(tmp_path):
    result = load_harness(EXAMPLE)
    case = result.spec.eval.cases[0]
    prov = current_provenance(tmp_path, model_id="claude-sonnet-5", adapter="raw_api", harness_name="react-web-researcher", harness_version="0.1.0")

    save_baseline_fn(tmp_path, "v1", {case.id: _trajectory(case.input)}, provenance=prov)

    loaded_prov = load_baseline_provenance(tmp_path, "v1")
    assert not loaded_prov.is_legacy
    assert loaded_prov.model_id == "claude-sonnet-5"
    assert loaded_prov.adapter == "raw_api"


def test_legacy_baseline_reports_no_provenance_without_crashing(tmp_path):
    # Pre-provenance format: schema_version + baselines, no "provenance" key.
    case_id = "capital-lookup"
    path = tmp_path / ".harness" / "baselines" / "legacy.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 2, "baselines": {case_id: json.loads(_trajectory("q").model_dump_json())}}))

    prov = load_baseline_provenance(tmp_path, "legacy")
    assert prov.is_legacy


def test_compare_warns_loudly_on_model_mismatch(tmp_path, monkeypatch):
    from harnesskit.cli import main as cli_main

    result_load = load_harness(EXAMPLE)
    case = result_load.spec.eval.cases[0]

    import shutil

    harness_copy = tmp_path / "harness_copy"
    shutil.copytree(EXAMPLE, harness_copy)

    old_prov = current_provenance(harness_copy, model_id="claude-haiku-4-5-20251001", adapter="test-mock", harness_name="react-web-researcher", harness_version="0.1.0")
    baseline_trajectories = {c.id: _trajectory(c.input, model_id="claude-haiku-4-5-20251001") for c in result_load.spec.eval.cases}
    save_baseline_fn(harness_copy, "v1", baseline_trajectories, provenance=old_prov)

    current_trajectories = {c.id: _trajectory(c.input, model_id="claude-sonnet-5") for c in result_load.spec.eval.cases}
    monkeypatch.setitem(cli_main.ADAPTERS, "test-mock", lambda: MockAdapter(trajectories_by_input={c.input: current_trajectories[c.id] for c in result_load.spec.eval.cases}))

    result = CliRunner().invoke(app, ["eval", str(harness_copy), "--adapter", "test-mock", "--compare", "v1"])

    assert "captured under a different environment" in result.output
    assert "claude-haiku-4-5-20251001" in result.output
    assert "claude-sonnet-5" in result.output


def test_compare_legacy_baseline_reports_no_provenance(tmp_path, monkeypatch):
    from harnesskit.cli import main as cli_main
    import shutil

    result_load = load_harness(EXAMPLE)
    trajectories = {c.id: _trajectory(c.input) for c in result_load.spec.eval.cases}

    harness_copy = tmp_path / "harness_copy"
    shutil.copytree(EXAMPLE, harness_copy)
    baselines_dir = harness_copy / ".harness" / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)
    # Legacy no-wrapper format.
    (baselines_dir / "legacy.json").write_text(
        json.dumps({cid: json.loads(t.model_dump_json()) for cid, t in trajectories.items()})
    )

    monkeypatch.setitem(cli_main.ADAPTERS, "test-mock", lambda: MockAdapter(trajectories_by_input={c.input: trajectories[c.id] for c in result_load.spec.eval.cases}))

    result = CliRunner().invoke(app, ["eval", str(harness_copy), "--adapter", "test-mock", "--compare", "legacy"])

    assert "no provenance recorded (legacy baseline)" in result.output
