from pathlib import Path

from harnesskit.eval import CaseResult, SuiteComparisonError, SuiteResult, compare, run_suite
from harnesskit.eval.scorers import Score
from harnesskit.format.spec import EvalCase
from harnesskit.eval.mock_adapter import MockAdapter
from harnesskit.parser import load_harness
from harnesskit.trace.schema import Step, StepType, Trajectory

EXAMPLE = Path(__file__).parent.parent / "examples" / "react-web-researcher"


def _good_trajectory(input_text: str) -> Trajectory:
    return Trajectory(
        harness_name="react-web-researcher",
        harness_version="0.1.0",
        model_id="claude-sonnet-5",
        adapter="mock",
        input=input_text,
        steps=[
            Step(step_type=StepType.llm_call, tokens_in=200, tokens_out=50, cost_usd=0.002),
            Step(step_type=StepType.tool_call, tool_name="search", tool_args={"query": "x"}),
            Step(step_type=StepType.llm_call, tokens_in=250, tokens_out=30, cost_usd=0.002),
        ],
        final_output="The capital is Brasília.",
        stopped_reason="tag_emitted:answer",
    )


def _looping_trajectory(input_text: str) -> Trajectory:
    steps = [Step(step_type=StepType.llm_call, tokens_in=200, tokens_out=50, cost_usd=0.002)]
    for _ in range(4):
        steps.append(Step(step_type=StepType.tool_call, tool_name="search", tool_args={"query": "x"}))
        steps.append(Step(step_type=StepType.llm_call, tokens_in=200, tokens_out=50, cost_usd=0.002))
    return Trajectory(
        harness_name="react-web-researcher",
        harness_version="0.1.0",
        model_id="claude-sonnet-5",
        adapter="mock",
        input=input_text,
        steps=steps,
        final_output=None,
        stopped_reason="max_turns",
    )


def test_run_suite_passes_on_good_trajectory():
    result = load_harness(EXAMPLE)
    case = result.spec.eval.cases[0]
    adapter = MockAdapter(trajectories_by_input={case.input: _good_trajectory(case.input)})
    suite = run_suite(result.spec, adapter, sample=1)
    assert suite.pass_rate == 1.0
    assert suite.results[0].trajectory.duplicate_tool_calls == 0


def test_run_suite_fails_and_flags_duplicates_on_looping_trajectory():
    result = load_harness(EXAMPLE)
    case = result.spec.eval.cases[0]
    adapter = MockAdapter(trajectories_by_input={case.input: _looping_trajectory(case.input)})
    suite = run_suite(result.spec, adapter, sample=1)
    assert suite.pass_rate == 0.0
    assert suite.results[0].trajectory.duplicate_tool_calls == 3


def test_compare_reports_pass_rate_delta():
    result = load_harness(EXAMPLE)
    case = result.spec.eval.cases[0]

    good_adapter = MockAdapter(trajectories_by_input={case.input: _good_trajectory(case.input)})
    bad_adapter = MockAdapter(trajectories_by_input={case.input: _looping_trajectory(case.input)})

    suite_a = run_suite(result.spec, bad_adapter, sample=1)
    suite_b = run_suite(result.spec, good_adapter, sample=1)
    ab_result = compare(suite_a, suite_b)

    pass_rate_delta = next(d for d in ab_result.deltas if d.metric == "pass_rate")
    assert pass_rate_delta.delta == 1.0


def _suite_with_outcomes(name: str, outcomes: list[bool]) -> SuiteResult:
    trajectory = _good_trajectory("input")
    return SuiteResult(
        harness_name=name,
        results=[
            CaseResult(
                case=EvalCase(id=f"case-{index}", input="input"),
                trajectory=trajectory,
                scores=[Score("test", float(passed), passed, "test")],
            )
            for index, passed in enumerate(outcomes)
        ],
    )


def test_compare_bootstraps_paired_case_outcomes():
    # The outcomes are identical for every case. A paired bootstrap has zero
    # uncertainty even though independent resampling would report a wide CI.
    baseline = _suite_with_outcomes("baseline", [False, False, True, True])
    current = _suite_with_outcomes("current", [False, False, True, True])

    assert compare(baseline, current).pass_rate_ci == (0.0, 0.0)


def test_compare_rejects_mismatched_case_sets():
    baseline = _suite_with_outcomes("baseline", [True])
    current = _suite_with_outcomes("current", [True, True])

    try:
        compare(baseline, current)
    except SuiteComparisonError as error:
        assert "same case IDs" in str(error)
    else:
        raise AssertionError("expected mismatched suites to be rejected")
