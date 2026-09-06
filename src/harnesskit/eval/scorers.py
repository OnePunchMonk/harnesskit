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


def score_tool_presence(trajectory: Trajectory, case: EvalCase) -> Score:
    called = {s.tool_name for s in trajectory.tool_calls}
    expected = set(case.expected_tools)
    if not expected:
        return Score("tool_presence", 1.0, True, "no expected tools declared")
    hit = expected & called
    value = len(hit) / len(expected)
    return Score("tool_presence", value, value == 1.0, f"expected {expected}, called {called}")


def score_trajectory_order(trajectory: Trajectory, case: EvalCase) -> Score:
    if case.scoring_mode.value not in ("trajectory_exact", "trajectory_in_order"):
        return Score("trajectory_order", 1.0, True, "not applicable to this scoring_mode")
    called_order = [s.tool_name for s in trajectory.tool_calls]
    expected = case.expected_tools
    if case.scoring_mode.value == "trajectory_exact":
        passed = called_order == expected
    else:  # in_order: expected tools appear as a subsequence, extra calls allowed
        it = iter(called_order)
        passed = all(t in it for t in expected)
    return Score("trajectory_order", 1.0 if passed else 0.0, passed, f"called order: {called_order}")


def score_output_contains(trajectory: Trajectory, case: EvalCase) -> Score:
    if not case.expected_output_contains:
        return Score("output_contains", 1.0, True, "no expected substrings declared")
    output = trajectory.final_output or ""
    hits = [s for s in case.expected_output_contains if s in output]
    value = len(hits) / len(case.expected_output_contains)
    return Score("output_contains", value, value == 1.0, f"found {hits} of {case.expected_output_contains}")


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
    score_step_budget,
    score_cost_budget,
    score_loop_health,
]


def score_trajectory(trajectory: Trajectory, case: EvalCase) -> list[Score]:
    return [scorer(trajectory, case) for scorer in DETERMINISTIC_SCORERS]
