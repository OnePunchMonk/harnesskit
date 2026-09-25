"""harnesskit.train — declare which parts of a harness are trainable
(``requires_grad``), then search for better values with evidence from eval
runs, selecting on a validation split and reporting on a held-out test split.

Experimental: this API may change between versions. See
``harnesskit.train.trainer`` for the loop, its budget semantics, and verdicts.
"""
from harnesskit.train.evidence import CaseEvidence, Evidence, collect_evidence
from harnesskit.train.module import CandidateRejected, HarnessModule, ParamChange
from harnesskit.train.params import Parameter, ParameterError
from harnesskit.train.proposers import (
    DEFAULT_PROPOSER_PROMPT,
    FunctionProposer,
    HarnessProposer,
    Proposal,
    ProposalBatch,
    Proposer,
    RandomSearchProposer,
    ScriptedProposer,
    default_proposer_spec,
    parse_proposals,
)
from harnesskit.train.trainer import (
    CandidateRecord,
    SplitPlan,
    SplitScore,
    SpendLedger,
    TrainBudget,
    Trainer,
    TrainError,
    TrainResult,
    plan_splits,
)

__all__ = [
    "HarnessModule",
    "Parameter",
    "ParamChange",
    "ParameterError",
    "CandidateRejected",
    "Evidence",
    "CaseEvidence",
    "collect_evidence",
    "Proposer",
    "Proposal",
    "ProposalBatch",
    "RandomSearchProposer",
    "ScriptedProposer",
    "FunctionProposer",
    "HarnessProposer",
    "DEFAULT_PROPOSER_PROMPT",
    "default_proposer_spec",
    "parse_proposals",
    "Trainer",
    "TrainBudget",
    "TrainResult",
    "TrainError",
    "CandidateRecord",
    "SplitPlan",
    "SplitScore",
    "SpendLedger",
    "plan_splits",
]
