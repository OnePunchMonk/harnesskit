"""Scorer framework (design doc §5.2).

A Scorer takes a Trajectory + the EvalCase it was run for, and returns a
Score in [0.0, 1.0] with a threshold verdict and an explanation. Deterministic
scorers run first, are free, and cover most of what matters; llm_judge is the
one scorer here that costs a model call, reserved for semantic cases.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from harnesskit.format.spec import EvalCase
from harnesskit.trace.schema import Trajectory


@dataclass
class Score:
    name: str
    value: float  # 0.0-1.0
    passed: bool
    explanation: str


Scorer = Callable[[Trajectory, EvalCase], Score]


class UnsupportedScoringModeError(ValueError):
    """Raised before execution for scoring modes without a configured judge."""


def validate_scoring_mode(case: EvalCase) -> None:
    if case.scoring_mode.value in {"semantic_match", "llm_judge"}:
        raise UnsupportedScoringModeError(
            f"case '{case.id}' uses scoring_mode='{case.scoring_mode.value}', which requires an external judge "
            "and is not implemented; use exact_match or a trajectory mode"
        )


def has_case_assertions(case: EvalCase) -> bool:
    """Whether a case has an outcome, trajectory, or resource assertion."""
    return bool(
        case.ground_truth is not None
        or case.expected_output_contains
        or case.expected_tools
        or case.max_turns is not None
        or case.max_cost_usd is not None
    )


def score_tool_presence(trajectory: Trajectory, case: EvalCase) -> Score:
    called = {s.tool_name for s in trajectory.tool_calls}
    expected = set(case.expected_tools)
    if not expected:
        return Score("tool_presence", 1.0, True, "no expected tools declared")
    hit = expected & called
    value = len(hit) / len(expected)
    return Score("tool_presence", value, value == 1.0, f"expected {expected}, called {called}")


def score_trajectory_order(trajectory: Trajectory, case: EvalCase) -> Score:
    if case.scoring_mode.value not in ("trajectory_exact", "trajectory_in_order", "trajectory_any_order"):
        return Score("trajectory_order", 1.0, True, "not applicable to this scoring_mode")
    called_order = [s.tool_name for s in trajectory.tool_calls]
    expected = case.expected_tools
    if case.scoring_mode.value == "trajectory_exact":
        passed = called_order == expected
    elif case.scoring_mode.value == "trajectory_in_order":
        # Expected tools appear as a subsequence; extra calls are allowed.
        it = iter(called_order)
        passed = all(t in it for t in expected)
    else:
        passed = set(expected).issubset(called_order)
    return Score("trajectory_order", 1.0 if passed else 0.0, passed, f"called order: {called_order}")


def score_output_contains(trajectory: Trajectory, case: EvalCase) -> Score:
    if not case.expected_output_contains:
        return Score("output_contains", 1.0, True, "no expected substrings declared")
    output = trajectory.final_output or ""
    hits = [s for s in case.expected_output_contains if s in output]
    value = len(hits) / len(case.expected_output_contains)
    return Score("output_contains", value, value == 1.0, f"found {hits} of {case.expected_output_contains}")


def score_exact_output(trajectory: Trajectory, case: EvalCase) -> Score:
    if case.scoring_mode.value != "exact_match" or case.ground_truth is None:
        return Score("exact_output", 1.0, True, "not applicable")
    output = trajectory.final_output or ""
    passed = output == case.ground_truth
    return Score("exact_output", 1.0 if passed else 0.0, passed, "exact output match" if passed else "output differs from ground truth")


def score_step_budget(trajectory: Trajectory, case: EvalCase) -> Score:
    limit = case.max_turns
    if limit is None:
        return Score("step_budget", 1.0, True, "no max_turns declared")
    passed = trajectory.turns <= limit
    return Score("step_budget", 1.0 if passed else 0.0, passed, f"{trajectory.turns} turns (limit {limit})")


def score_cost_budget(trajectory: Trajectory, case: EvalCase) -> Score:
    limit = case.max_cost_usd
    if limit is None:
        return Score("cost_budget", 1.0, True, "no max_cost_usd declared")
    passed = trajectory.total_cost_usd <= limit
    return Score("cost_budget", 1.0 if passed else 0.0, passed, f"${trajectory.total_cost_usd:.4f} (limit ${limit:.2f})")


def score_loop_health(trajectory: Trajectory, case: EvalCase) -> Score:
    dupes = trajectory.duplicate_tool_calls
    passed = dupes == 0
    return Score("loop_health", 1.0 if passed else max(0.0, 1.0 - dupes * 0.25), passed, f"{dupes} duplicate tool calls")


DETERMINISTIC_SCORERS: list[Scorer] = [
    score_tool_presence,
    score_trajectory_order,
    score_output_contains,
    score_exact_output,
    score_step_budget,
    score_cost_budget,
    score_loop_health,
]


def score_trajectory(trajectory: Trajectory, case: EvalCase) -> list[Score]:
    validate_scoring_mode(case)
    if not has_case_assertions(case):
        return []
    return [scorer(trajectory, case) for scorer in DETERMINISTIC_SCORERS]
