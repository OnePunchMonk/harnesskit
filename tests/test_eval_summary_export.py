from pathlib import Path

from harnesskit.eval import run_suite
from harnesskit.eval.mock_adapter import MockAdapter
from harnesskit.packaging import export_bundle, preview_bundle
from harnesskit.parser import load_harness
from harnesskit.trace import save_baseline
from harnesskit.trace.schema import Step, StepType, Trajectory

EXAMPLE = Path(__file__).parent.parent / "examples" / "react-web-researcher"


def _trajectory(input_text: str) -> Trajectory:
    return Trajectory(
        harness_name="react-web-researcher",
        harness_version="0.1.0",
        model_id="claude-sonnet-5",
        adapter="mock",
        input=input_text,
        steps=[Step(step_type=StepType.llm_call, tokens_in=100, tokens_out=20, cost_usd=0.001)],
        final_output="ok",
        stopped_reason="tag_emitted:answer",
    )


def test_export_embeds_self_reported_eval_summary(tmp_path):
    result = load_harness(EXAMPLE)
    trajectories = {c.id: _trajectory(c.input) for c in result.spec.eval.cases}
    adapter = MockAdapter(trajectories_by_input={c.input: t for c, t in zip(result.spec.eval.cases, trajectories.values())})
    suite = run_suite(result.spec, adapter)
    save_baseline(EXAMPLE, "test-summary", {r.case.id: r.trajectory for r in suite.results})

    try:
        bundle_path = tmp_path / "out.harn"
        bundle = export_bundle(EXAMPLE, bundle_path, include_eval_summary="test-summary")

        assert bundle.eval_summary is not None
        assert bundle.eval_summary["self_reported"] is True
        assert bundle.eval_summary["case_count"] == len(result.spec.eval.cases)

        preview = preview_bundle(bundle_path)
        assert preview.manifest["eval_summary"]["self_reported"] is True
    finally:
        (EXAMPLE / ".harness" / "baselines" / "test-summary.json").unlink(missing_ok=True)


def test_export_without_summary_flag_has_none(tmp_path):
    bundle_path = tmp_path / "out.harn"
    bundle = export_bundle(EXAMPLE, bundle_path)
    assert bundle.eval_summary is None
