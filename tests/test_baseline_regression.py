from pathlib import Path

from harnesskit.eval import check_regression, compare, run_suite
from harnesskit.eval.mock_adapter import MockAdapter
from harnesskit.parser import load_harness
from harnesskit.trace import list_baselines, load_baseline
from harnesskit.trace import save_baseline as save_baseline_fn
from harnesskit.trace.schema import Step, StepType, Trajectory

EXAMPLE = Path(__file__).parent.parent / "examples" / "react-web-researcher"


def _trajectory(input_text: str, turns: int) -> Trajectory:
    steps = []
    for _ in range(turns):
        steps.append(Step(step_type=StepType.llm_call, tokens_in=200, tokens_out=50, cost_usd=0.002))
    return Trajectory(
        harness_name="react-web-researcher",
        harness_version="0.1.0",
        model_id="claude-sonnet-5",
        adapter="mock",
        input=input_text,
        steps=steps,
        final_output="The capital is Brasília.",
        stopped_reason="tag_emitted:answer",
    )


def test_save_and_load_baseline_roundtrip(tmp_path):
    result = load_harness(EXAMPLE)
    case = result.spec.eval.cases[0]
    traj = _trajectory(case.input, turns=3)

    save_baseline_fn(tmp_path, "v1", {case.id: traj})
    assert list_baselines(tmp_path) == ["v1"]

    loaded = load_baseline(tmp_path, "v1")
    assert loaded[case.id].turns == 3


def test_check_regression_flags_turn_increase():
    result = load_harness(EXAMPLE)
    case = result.spec.eval.cases[0]

    baseline_adapter = MockAdapter(trajectories_by_input={case.input: _trajectory(case.input, turns=3)})
    current_adapter = MockAdapter(trajectories_by_input={case.input: _trajectory(case.input, turns=9)})

    baseline_suite = run_suite(result.spec, baseline_adapter, sample=1)
    current_suite = run_suite(result.spec, current_adapter, sample=1)

    reg = check_regression(baseline_suite, current_suite, tolerances={"avg_turns": 2})
    assert not reg.ok
    assert reg.regressions[0].metric == "avg_turns"


def test_check_regression_ok_within_tolerance():
    result = load_harness(EXAMPLE)
    case = result.spec.eval.cases[0]

    baseline_adapter = MockAdapter(trajectories_by_input={case.input: _trajectory(case.input, turns=3)})
    current_adapter = MockAdapter(trajectories_by_input={case.input: _trajectory(case.input, turns=4)})

    baseline_suite = run_suite(result.spec, baseline_adapter, sample=1)
    current_suite = run_suite(result.spec, current_adapter, sample=1)

    reg = check_regression(baseline_suite, current_suite, tolerances={"avg_turns": 2})
    assert reg.ok
