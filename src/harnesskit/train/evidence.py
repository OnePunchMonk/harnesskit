"""Evidence: what an optimizer is allowed to learn from.

In the ``requires_grad`` analogy this is the "backward pass" — but it is not a
gradient. It is a structured record of how the current harness behaved on
the *train* split: which cases failed, what was expected, what the harness
actually produced, and which checks rejected it. Proposers turn it into
candidate edits. Validation and test cases never appear here; the trainer
only ever builds evidence from a train-split ``SuiteResult``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from harnesskit.eval.engine import CaseResult, SuiteResult


def _clip(text: str | None, limit: int) -> str | None:
    if text is None or len(text) <= limit:
        return text
    return text[:limit] + f"… [{len(text) - limit} more chars]"


@dataclass
class CaseEvidence:
    case_id: str
    input: str
    passed: bool
    status: str
    expected: dict[str, Any]
    output: str | None
    failed_checks: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    stopped_reason: str | None = None
    error: str | None = None


@dataclass
class Evidence:
    split: str
    pass_rate: float
    cases: list[CaseEvidence]

    @property
    def failures(self) -> list[CaseEvidence]:
        return [c for c in self.cases if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {"split": self.split, "pass_rate": self.pass_rate, "cases": [asdict(c) for c in self.cases]}


def _case_evidence(result: CaseResult, max_chars: int) -> CaseEvidence:
    case = result.case
    expected: dict[str, Any] = {}
    if case.expected_output_contains:
        expected["output_contains"] = case.expected_output_contains
    if case.ground_truth is not None:
        expected["exact_output"] = case.ground_truth
    if case.expected_tools:
        expected["tools"] = case.expected_tools
    if case.max_turns is not None:
        expected["max_turns"] = case.max_turns
    traj = result.trajectory
    return CaseEvidence(
        case_id=case.id,
        input=_clip(case.input, max_chars) or "",
        passed=result.passed,
        status=result.status.value,
        expected=expected,
        output=_clip(traj.final_output, max_chars) if traj else None,
        failed_checks=[f"{s.name}: {s.explanation}" for s in result.scores if not s.passed],
        tool_calls=[s.tool_name or "?" for s in traj.tool_calls] if traj else [],
        stopped_reason=traj.stopped_reason if traj else None,
        error=_clip(result.error, max_chars),
    )


def collect_evidence(suite: SuiteResult, split: str = "train", max_chars: int = 800) -> Evidence:
    """Summarize a suite result for a proposer. Only call with train results."""
    return Evidence(
        split=split,
        pass_rate=suite.pass_rate,
        cases=[_case_evidence(r, max_chars) for r in suite.results],
    )
