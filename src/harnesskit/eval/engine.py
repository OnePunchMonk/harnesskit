"""Eval Engine (design doc §5) — runs a harness against its eval suite and
produces scores. The "training loss curve": fast, local, pass/fail feedback
on every harness change, plus the A/B comparison that's the actual wedge.
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from enum import Enum

from harnesskit.adapters.base import HarnessAdapter
from harnesskit.eval.scorers import Score, score_trajectory, validate_scoring_mode
from harnesskit.format.spec import EvalCase, HarnessSpec
from harnesskit.trace.schema import Trajectory


class CaseStatus(str, Enum):
    """Explicit case outcome, kept in `SuiteResult.results` regardless of
    value so a missing/errored/cancelled/unscored case is never filtered out
    before a success-rate or coverage ratio is computed."""

    ok = "ok"
    errored = "errored"
    cancelled = "cancelled"
    unscored = "unscored"


@dataclass
class CaseResult:
    case: EvalCase
    trajectory: Trajectory | None
    scores: list[Score]
    status: CaseStatus = CaseStatus.ok
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == CaseStatus.ok and bool(self.scores) and all(s.passed for s in self.scores)

    @property
    def is_scored(self) -> bool:
        return bool(self.scores)

    @property
    def mean_score(self) -> float:
        return statistics.mean(s.value for s in self.scores) if self.scores else 0.0


@dataclass
class SuiteResult:
    harness_name: str
    results: list[CaseResult] = field(default_factory=list)
    missing_case_ids: list[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Denominator is every attempted case, including errored/cancelled/
        unscored ones (they simply count as not-passed) — never just the
        subset that happened to score cleanly."""
        return sum(1 for r in self.results if r.passed) / len(self.results) if self.results else 0.0

    @property
    def errored_case_ids(self) -> list[str]:
        return [r.case.id for r in self.results if r.status == CaseStatus.errored]

    @property
    def unscored_case_ids(self) -> list[str]:
        return [r.case.id for r in self.results if r.status == CaseStatus.unscored]

    @property
    def avg_turns(self) -> float:
        turns = [r.trajectory.turns for r in self.results if r.trajectory is not None]
        return statistics.mean(turns) if turns else 0.0

    @property
    def unavailable_cost_case_ids(self) -> list[str]:
        """Cases whose trajectory contains at least one step with an
        unavailable (unknown) cost. `avg_cost_usd` excludes these rather than
        silently averaging their missing cost in as $0 — check this list to
        know when the average is incomplete."""
        return [r.case.id for r in self.results if r.trajectory is not None and r.trajectory.has_unavailable_cost]

    @property
    def avg_cost_usd(self) -> float | None:
        known = [
            r.trajectory.total_cost_usd
            for r in self.results
            if r.trajectory is not None and not r.trajectory.has_unavailable_cost
        ]
        return statistics.mean(known) if known else None

    @property
    def total_duplicate_tool_calls(self) -> int:
        return sum(r.trajectory.duplicate_tool_calls for r in self.results if r.trajectory is not None)


def run_case(spec: HarnessSpec, case: EvalCase, adapter: HarnessAdapter, agent) -> CaseResult:
    try:
        trajectory = adapter.run(agent, case.input)
    except Exception as e:  # noqa: BLE001 — a failing case must stay visible, not crash the whole suite
        return CaseResult(case=case, trajectory=None, scores=[], status=CaseStatus.errored, error=str(e))
    scores = score_trajectory(trajectory, case)
    status = CaseStatus.ok if scores else CaseStatus.unscored
    return CaseResult(case=case, trajectory=trajectory, scores=scores, status=status)


def run_suite(spec: HarnessSpec, adapter: HarnessAdapter, sample: int | None = None) -> SuiteResult:
    """Run every eval case (or `sample` of them, for a cheap dev-loop pass)."""
    cases = spec.eval.cases
    if sample is not None:
        cases = cases[:sample]
    for case in cases:
        validate_scoring_mode(case)
    agent = adapter.build(spec)
    results = [run_case(spec, case, adapter, agent) for case in cases]
    return SuiteResult(harness_name=spec.metadata.name, results=results)


def replay_suite(spec: HarnessSpec, trajectories: dict[str, Trajectory]) -> SuiteResult:
    """Re-score cached trajectories without calling the model again — the
    cheapest eval mode, used when only a scorer changed (design doc §5.3).

    A case with no matching trajectory is recorded in `missing_case_ids`
    rather than silently dropped, so a caller can't mistake a partial replay
    for a complete one (`SuiteResult.pass_rate` is computed over `results`
    only, which — without that check — would look like a clean 100%)."""
    results = []
    missing_case_ids = []
    for case in spec.eval.cases:
        trajectory = trajectories.get(case.id)
        if trajectory is None:
            missing_case_ids.append(case.id)
            continue
        results.append(CaseResult(case=case, trajectory=trajectory, scores=score_trajectory(trajectory, case)))
    return SuiteResult(harness_name=spec.metadata.name, results=results, missing_case_ids=missing_case_ids)


@dataclass
class MetricDelta:
    metric: str
    a: float | None
    b: float | None

    @property
    def delta(self) -> float | None:
        """None when either side's value is unavailable (e.g. avg_cost_usd
        with no cases of known cost) — never silently treated as 0."""
        if self.a is None or self.b is None:
            return None
        return self.b - self.a


@dataclass
class ABResult:
    name_a: str
    name_b: str
    deltas: list[MetricDelta]
    pass_rate_ci: tuple[float, float]  # 95% bootstrap CI on (b.pass_rate - a.pass_rate)


class SuiteComparisonError(ValueError):
    """Raised when two eval results cannot support a case-level comparison."""


def _results_by_case_id(suite: SuiteResult) -> dict[str, CaseResult]:
    results = {result.case.id: result for result in suite.results}
    if len(results) != len(suite.results):
        raise SuiteComparisonError(
            f"{suite.harness_name} contains duplicate eval case IDs; comparisons require unique IDs"
        )
    return results


def _paired_results(a: SuiteResult, b: SuiteResult) -> list[tuple[CaseResult, CaseResult]]:
    """Return results aligned by case ID, rejecting partial or duplicate suites.

    Aggregate metrics can make a comparison of different case sets look valid.
    Requiring the same IDs makes both A/B confidence intervals and regression
    gates reflect changes to the harness rather than changes to the benchmark.
    """
    a_by_id = _results_by_case_id(a)
    b_by_id = _results_by_case_id(b)
    if a_by_id.keys() != b_by_id.keys():
        only_a = sorted(a_by_id.keys() - b_by_id.keys())
        only_b = sorted(b_by_id.keys() - a_by_id.keys())
        details = []
        if only_a:
            details.append(f"only in {a.harness_name}: {', '.join(only_a)}")
        if only_b:
            details.append(f"only in {b.harness_name}: {', '.join(only_b)}")
        raise SuiteComparisonError("eval suites must contain the same case IDs (" + "; ".join(details) + ")")
    return [(result, b_by_id[result.case.id]) for result in a.results]


def _bootstrap_pass_rate_delta_ci(a: SuiteResult, b: SuiteResult, n_resamples: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    pairs = _paired_results(a, b)
    if not pairs:
        return (0.0, 0.0)
    # Resample case-level deltas, not each suite independently. Independent
    # sampling destroys the within-case pairing and overstates uncertainty.
    outcome_deltas = [float(current.passed) - float(baseline.passed) for baseline, current in pairs]
    deltas = []
    for _ in range(n_resamples):
        deltas.append(statistics.mean(rng.choice(outcome_deltas) for _ in outcome_deltas))
    deltas.sort()
    lo = deltas[int(0.025 * n_resamples)]
    hi = deltas[int(0.975 * n_resamples)]
    return (lo, hi)


def compare(a: SuiteResult, b: SuiteResult) -> ABResult:
    """Paired A/B comparison (design doc §5.4): same metrics, both directions,
    with a bootstrap confidence interval on the pass-rate delta since eval
    suites are typically small-N."""
    _paired_results(a, b)
    deltas = [
        MetricDelta("pass_rate", a.pass_rate, b.pass_rate),
        MetricDelta("avg_turns", a.avg_turns, b.avg_turns),
        MetricDelta("avg_cost_usd", a.avg_cost_usd, b.avg_cost_usd),
        MetricDelta("duplicate_tool_calls", a.total_duplicate_tool_calls, b.total_duplicate_tool_calls),
    ]
    ci = _bootstrap_pass_rate_delta_ci(a, b)
    return ABResult(name_a=a.harness_name, name_b=b.harness_name, deltas=deltas, pass_rate_ci=ci)


@dataclass
class RegressionResult:
    regressions: list[MetricDelta]
    ok: bool


def check_regression(baseline: SuiteResult, current: SuiteResult, tolerances: dict[str, float]) -> RegressionResult:
    """CI gate (design doc §5.5): fail if any metric regresses beyond its
    configured tolerance. `tolerances` maps metric name -> max allowed drop
    (for pass_rate) or max allowed increase (for cost/turns)."""
    ab = compare(baseline, current)
    regressions = []
    worse_when_lower = {"pass_rate"}
    for d in ab.deltas:
        tol = tolerances.get(d.metric)
        if tol is None or d.delta is None:
            continue
        if d.metric in worse_when_lower and d.delta < -tol:
            regressions.append(d)
        elif d.metric not in worse_when_lower and d.delta > tol:
            regressions.append(d)
    return RegressionResult(regressions=regressions, ok=not regressions)
