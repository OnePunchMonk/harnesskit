"""Trainer: bounded, evidence-driven search over a harness's trainable parameters.

One ``fit()``:

1. Split eval cases into train / val / test (declared ``split:`` on every
   case, or a seeded hash of case ids when none is declared).
2. Evaluate the starting harness on train and val.
3. For each step: build evidence from the current best's *train* results,
   ask the proposer for candidates, materialize each as its own harness
   directory (rejecting any that touch frozen components), evaluate it on
   train, and — if train did not get worse — on val. A candidate is accepted
   only if it beats the current best on val.
4. Evaluate the original and the selected harness once on the untouched test
   split and compare them case-by-case (paired bootstrap CI).

The verdict is ``improved`` only when the test gain is positive with a CI
lower bound above zero; a positive but uncertain gain is ``inconclusive``,
and anything else is ``no_supported_improvement``. Every candidate —
accepted, rejected, invalid, duplicate — and every cost (evaluation and
proposal) is recorded in the report.

Budget: ``max_cost_usd`` is checked before each evaluation against spend so
far plus an estimate of the next evaluation (the largest observed so far).
That is an admission check on estimates, not a guaranteed bill cap: the
final evaluation can still overrun, and any overrun is recorded. When a cap
is set but some cost is unavailable (e.g. a callback agent that reports no
cost), training stops rather than pretend the cap is enforced — pass
``allow_unmetered=True`` to proceed and have the report say so.
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from harnesskit.adapters.base import HarnessAdapter, check_support
from harnesskit.eval.engine import ABResult, SuiteResult, compare, run_suite
from harnesskit.format.spec import HarnessSpec
from harnesskit.train.evidence import collect_evidence
from harnesskit.train.module import CandidateRejected, HarnessModule
from harnesskit.train.proposers import Proposer
from harnesskit.trace.schema import CostStatus

AdapterFactory = Callable[[HarnessSpec], HarnessAdapter]
SPLITS = ("train", "val", "test")


class TrainError(ValueError):
    """Training cannot start (e.g. no trainable parameters, empty split)."""


@dataclass
class SplitPlan:
    train: list[str]
    val: list[str]
    test: list[str]
    source: str  # "declared" or "hashed(seed=N)"


def plan_splits(spec: HarnessSpec, seed: int = 0, fractions: tuple[float, float] = (0.5, 0.25)) -> SplitPlan:
    cases = spec.eval.cases
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        raise TrainError("eval case ids must be unique to train")
    declared = [c.split for c in cases if c.split is not None]
    if declared and len(declared) != len(cases):
        missing = [c.id for c in cases if c.split is None]
        raise TrainError(f"either every eval case declares a split or none does; missing: {', '.join(missing)}")
    if declared:
        plan = SplitPlan(*([c.id for c in cases if c.split == s] for s in SPLITS), source="declared")
    else:
        ranked = sorted(ids, key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())
        n_train = max(1, round(len(ranked) * fractions[0]))
        n_val = max(1, round(len(ranked) * fractions[1]))
        plan = SplitPlan(ranked[:n_train], ranked[n_train : n_train + n_val], ranked[n_train + n_val :], source=f"hashed(seed={seed})")
    for name in SPLITS:
        if not getattr(plan, name):
            raise TrainError(f"the {name} split is empty; training needs at least one case in each of train/val/test")
    return plan


@dataclass
class TrainBudget:
    max_steps: int = 3
    candidates_per_step: int = 4
    max_candidates: int | None = None
    max_cost_usd: float | None = None
    allow_unmetered: bool = False


@dataclass
class SpendLedger:
    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(self, kind: str, label: str, cost_usd: float | None, status: str) -> None:
        self.entries.append({"kind": kind, "label": label, "cost_usd": cost_usd, "cost_status": status})

    def total(self, kind: str | None = None) -> float:
        return sum(e["cost_usd"] or 0.0 for e in self.entries if kind is None or e["kind"] == kind)

    @property
    def unknown_entries(self) -> int:
        return sum(1 for e in self.entries if e["cost_status"] == CostStatus.unavailable.value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "known_total_usd": self.total(),
            "evaluation_usd": self.total("evaluation"),
            "proposal_usd": self.total("proposal"),
            "entries_with_unavailable_cost": self.unknown_entries,
            "complete": self.unknown_entries == 0,
            "entries": self.entries,
        }


@dataclass
class SplitScore:
    pass_rate: float
    mean_score: float
    n: int
    cost_usd: float | None

    @classmethod
    def of(cls, suite: SuiteResult) -> SplitScore:
        means = [r.mean_score for r in suite.results]
        unknown = bool(suite.unavailable_cost_case_ids)
        cost = None if unknown else sum(r.trajectory.total_cost_usd for r in suite.results if r.trajectory)
        return cls(suite.pass_rate, statistics.mean(means) if means else 0.0, len(suite.results), cost)

    def key(self) -> tuple[float, float]:
        return (self.pass_rate, self.mean_score)


@dataclass
class CandidateRecord:
    id: str
    step: int
    parent: str
    proposer: str
    edits: dict[str, Any]
    rationale: str
    status: str  # accepted | rejected | invalid | duplicate
    reason: str
    diff: str = ""
    directory: str | None = None
    train: SplitScore | None = None
    val: SplitScore | None = None


@dataclass
class TrainResult:
    harness: str
    verdict: str
    verdict_reason: str
    stop_reason: str
    splits: SplitPlan
    initial_state: dict[str, Any]
    best_state: dict[str, Any]
    best_candidate: str
    best_directory: str
    best_diff: str
    initial_train: SplitScore
    initial_val: SplitScore
    best_train: SplitScore
    best_val: SplitScore
    test_initial: SuiteResult | None
    test_best: SuiteResult | None
    test_comparison: ABResult | None
    candidates: list[CandidateRecord]
    spend: SpendLedger
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        def suite(s: SuiteResult | None) -> dict | None:
            if s is None:
                return None
            return {
                "pass_rate": s.pass_rate,
                "cases": {r.case.id: {"passed": r.passed, "status": r.status.value} for r in s.results},
            }

        comparison = None
        if self.test_comparison is not None:
            comparison = {
                "deltas": {d.metric: {"initial": d.a, "best": d.b, "delta": d.delta} for d in self.test_comparison.deltas},
                "pass_rate_delta_ci95": list(self.test_comparison.pass_rate_ci),
            }
        return {
            "report_version": 1,
            "harness": self.harness,
            "verdict": self.verdict,
            "verdict_reason": self.verdict_reason,
            "stop_reason": self.stop_reason,
            "splits": asdict(self.splits),
            "initial_state": self.initial_state,
            "best_state": self.best_state,
            "best_candidate": self.best_candidate,
            "best_directory": self.best_directory,
            "best_diff": self.best_diff,
            "initial": {"train": asdict(self.initial_train), "val": asdict(self.initial_val)},
            "best": {"train": asdict(self.best_train), "val": asdict(self.best_val)},
            "test": {"initial": suite(self.test_initial), "best": suite(self.test_best), "comparison": comparison},
            "candidates": [asdict(c) for c in self.candidates],
            "spend": self.spend.to_dict(),
            "notes": self.notes,
        }


def _state_key(state: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(state, sort_keys=True, default=str).encode()).hexdigest()


def _subset(spec: HarnessSpec, ids: list[str]) -> HarnessSpec:
    wanted = set(ids)
    cases = [c for c in spec.eval.cases if c.id in wanted]
    subset = spec.model_copy(update={"eval": spec.eval.model_copy(update={"cases": cases})})
    subset.source_dir = spec.source_dir
    return subset


class Trainer:
    def __init__(
        self,
        module: HarnessModule,
        adapter_factory: AdapterFactory,
        proposer: Proposer,
        work_dir: str | Path,
        budget: TrainBudget | None = None,
        seed: int = 0,
        min_improvement: float = 0.0,
        max_workers: int = 1,
    ):
        self.module = module
        self.adapter_factory = adapter_factory
        self.proposer = proposer
        self.work_dir = Path(work_dir)
        self.budget = budget or TrainBudget()
        self.seed = seed
        self.min_improvement = min_improvement
        self.max_workers = max_workers
        self.ledger = SpendLedger()
        self.notes: list[str] = []
        self._largest_eval_cost = 0.0

    # -- budget --------------------------------------------------------
    def _admit(self, label: str) -> str | None:
        """Return a stop reason if the next evaluation should not start."""
        cap = self.budget.max_cost_usd
        if cap is None:
            return None
        if self.ledger.unknown_entries and not self.budget.allow_unmetered:
            return (
                f"cost cap ${cap:.2f} cannot be enforced: {self.ledger.unknown_entries} recorded cost(s) are "
                "unavailable (pass allow_unmetered to continue without an enforced cap)"
            )
        projected = self.ledger.total() + self._largest_eval_cost
        if projected > cap:
            return f"budget: {label} would bring estimated spend to ${projected:.4f} > cap ${cap:.2f}"
        return None

    def _evaluate(self, module: HarnessModule, ids: list[str], label: str) -> SuiteResult:
        spec = _subset(module.spec, ids)
        suite = run_suite(spec, self.adapter_factory(module.spec), max_workers=self.max_workers)
        score = SplitScore.of(suite)
        status = CostStatus.unavailable if score.cost_usd is None else CostStatus.estimated
        self.ledger.add("evaluation", label, score.cost_usd, status.value)
        self._largest_eval_cost = max(self._largest_eval_cost, score.cost_usd or 0.0)
        return suite

    def _better(self, cand: SplitScore, best: SplitScore) -> bool:
        if cand.pass_rate > best.pass_rate + self.min_improvement:
            return True
        return self.min_improvement == 0 and cand.pass_rate == best.pass_rate and cand.mean_score > best.mean_score + 1e-9

    # -- main loop -----------------------------------------------------
    def fit(self) -> TrainResult:
        trainable = [name for name, _ in self.module.named_parameters()]
        if not trainable:
            raise TrainError("the harness declares no trainable parameters (see `trainable:` in harness.yaml)")
        plan = plan_splits(self.module.spec, self.seed)
        if self.work_dir.exists() and any(self.work_dir.iterdir()):
            raise TrainError(f"work directory is not empty: {self.work_dir}")
        (self.work_dir / "candidates").mkdir(parents=True, exist_ok=True)
        rng = random.Random(self.seed)
        warnings = check_support(self.module.spec, self.adapter_factory(self.module.spec))
        self.notes += [f"adapter: {w}" for w in warnings]

        initial = self.module
        init_train = self._evaluate(initial, plan.train, "initial/train")
        init_val = self._evaluate(initial, plan.val, "initial/val")
        best, best_id = initial, "initial"
        best_train_suite = init_train
        best_train, best_val = SplitScore.of(init_train), SplitScore.of(init_val)
        initial_train, initial_val = best_train, best_val
        seen = {_state_key(initial.state_dict()): "initial"}
        records: list[CandidateRecord] = []
        stop_reason = f"completed {self.budget.max_steps} step(s)"
        counter = 0

        for step in range(1, self.budget.max_steps + 1):
            evidence = collect_evidence(best_train_suite, split="train")
            n = self.budget.candidates_per_step
            if self.budget.max_candidates is not None:
                n = min(n, self.budget.max_candidates - counter)
            if n <= 0:
                stop_reason = f"reached max_candidates={self.budget.max_candidates}"
                break
            batch = self.proposer.propose(best.parameters(), evidence, n, rng)
            self.ledger.add("proposal", f"step{step}/{self.proposer.name}", batch.cost_usd, batch.cost_status.value)
            self.notes += [f"step {step}: {note}" for note in batch.notes]
            if not batch.proposals:
                stop_reason = f"proposer returned no proposals at step {step}"
                break
            stopped = None
            for proposal in batch.proposals:
                counter += 1
                cid = f"c{counter:03d}"
                record = CandidateRecord(cid, step, best_id, self.proposer.name, proposal.edits, proposal.rationale, "invalid", "")
                records.append(record)
                try:
                    merged = best.validate_state(proposal.edits)
                except CandidateRejected as e:
                    record.reason = str(e)
                    continue
                record.diff = "".join(c.render() for c in best.diff(merged))
                key = _state_key(merged)
                if key == _state_key(best.state_dict()):
                    record.reason = "no-op: proposal does not change any parameter"
                    continue
                if key in seen:
                    record.status, record.reason = "duplicate", f"same state as {seen[key]}; not re-evaluated"
                    continue
                stopped = self._admit(f"candidate {cid}")
                if stopped:
                    record.status, record.reason = "rejected", "not evaluated: " + stopped
                    break
                try:
                    candidate = best.materialize(merged, self.work_dir / "candidates" / cid)
                except CandidateRejected as e:
                    record.reason = str(e)
                    continue
                seen[key] = cid
                record.directory = str(candidate.directory)
                train_suite = self._evaluate(candidate, plan.train, f"{cid}/train")
                record.train = SplitScore.of(train_suite)
                if record.train.key() < best_train.key():
                    record.status, record.reason = "rejected", "train objective decreased; val not evaluated"
                    continue
                stopped = self._admit(f"candidate {cid} val")
                if stopped:
                    record.status, record.reason = "rejected", "val not evaluated: " + stopped
                    break
                record.val = SplitScore.of(self._evaluate(candidate, plan.val, f"{cid}/val"))
                if self._better(record.val, best_val):
                    record.status, record.reason = "accepted", f"val {best_val.pass_rate:.2f} -> {record.val.pass_rate:.2f}"
                    best, best_id, best_train_suite = candidate, cid, train_suite
                    best_train, best_val = record.train, record.val
                else:
                    record.status, record.reason = "rejected", "no val improvement over current best"
                if self.budget.max_candidates is not None and counter >= self.budget.max_candidates:
                    break
            if stopped:
                stop_reason = stopped
                break
            if self.budget.max_candidates is not None and counter >= self.budget.max_candidates:
                stop_reason = f"reached max_candidates={self.budget.max_candidates}"
                break

        return self._finish(plan, initial, best, best_id, initial_train, initial_val, best_train, best_val, records, stop_reason)

    def _finish(self, plan, initial, best, best_id, initial_train, initial_val, best_train, best_val, records, stop_reason) -> TrainResult:
        test_initial = test_best = comparison = None
        best_diff = "".join(c.render() for c in initial.diff(best.state_dict()))
        best_dir = self.work_dir / "best"
        if best_id == "initial":
            evaluated = sum(1 for r in records if r.train is not None)
            reason = (
                f"no candidate beat the starting harness on val ({evaluated} evaluated)"
                if evaluated
                else f"no candidate was evaluated ({stop_reason})"
            )
            verdict = "no_supported_improvement"
        else:
            shutil.copytree(best.directory, best_dir, ignore=shutil.ignore_patterns(".harness", "__pycache__"))
            (self.work_dir / "best.diff").write_text(best_diff)
            stopped = self._admit("final test evaluation")
            if stopped is None:
                test_initial = self._evaluate(initial, plan.test, "initial/test")
            stopped = stopped or self._admit("final test evaluation")
            if stopped is None:
                test_best = self._evaluate(best, plan.test, f"{best_id}/test")
            if test_initial is None or test_best is None:
                verdict, reason = "holdout_not_evaluated", stopped or "test evaluation skipped"
            else:
                comparison = compare(test_initial, test_best)
                delta = test_best.pass_rate - test_initial.pass_rate
                lo, hi = comparison.pass_rate_ci
                if delta > 0 and lo > 0:
                    verdict, reason = "improved", f"test pass rate {test_initial.pass_rate:.2f} -> {test_best.pass_rate:.2f}, 95% CI [{lo:+.2f}, {hi:+.2f}]"
                elif delta > 0:
                    verdict, reason = "inconclusive", f"test pass rate {test_initial.pass_rate:.2f} -> {test_best.pass_rate:.2f} but 95% CI [{lo:+.2f}, {hi:+.2f}] includes 0 (n={len(plan.test)})"
                else:
                    verdict, reason = "no_supported_improvement", f"val gain did not hold on test: {test_initial.pass_rate:.2f} -> {test_best.pass_rate:.2f}"
        if self.ledger.unknown_entries:
            self.notes.append(f"{self.ledger.unknown_entries} spend entries have unavailable cost; known_total_usd understates spend")
        result = TrainResult(
            harness=self.module.spec.metadata.name,
            verdict=verdict,
            verdict_reason=reason,
            stop_reason=stop_reason,
            splits=plan,
            initial_state=initial.state_dict(),
            best_state=best.state_dict(),
            best_candidate=best_id,
            best_directory=str(best_dir if best_id != "initial" else initial.directory),
            best_diff=best_diff,
            initial_train=initial_train,
            initial_val=initial_val,
            best_train=best_train,
            best_val=best_val,
            test_initial=test_initial,
            test_best=test_best,
            test_comparison=comparison,
            candidates=records,
            spend=self.ledger,
            notes=self.notes,
        )
        (self.work_dir / "train_report.json").write_text(json.dumps(result.to_dict(), indent=2, default=str))
        return result
