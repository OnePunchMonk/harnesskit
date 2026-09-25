"""Proposers: turn evidence into candidate parameter edits (the "optimizer
step" of the analogy).

- ``RandomSearchProposer`` — equal-budget random search over bounded
  numeric/choice parameters, one parameter per candidate. This is the
  baseline any smarter proposer has to beat.
- ``ScriptedProposer`` — a fixed list of edits: manual candidates ("what I
  would have tried by hand") and deterministic tests.
- ``HarnessProposer`` — a harness that proposes edits to another harness.
  It runs any ``HarnessAdapter`` (e.g. ``RawAPIAdapter`` for a model call)
  on a prompt built from the parameters and train evidence, and parses JSON
  edits from the final output. Because the proposer is itself a harness,
  its own system prompt can be declared trainable and optimized with the
  same ``Trainer`` — the recursive case.

Every proposer reports what its proposals cost, so optimizer spend is
counted alongside evaluation spend.
"""
from __future__ import annotations

import json
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from harnesskit.adapters.base import HarnessAdapter, RunnableAgent
from harnesskit.format.spec import (
    HarnessSpec,
    LoopConfig,
    LoopStrategyType,
    MemoryConfig,
    Metadata,
    ModelConfig,
    ParameterKind,
    ScaffoldConfig,
)
from harnesskit.train.evidence import Evidence
from harnesskit.train.params import Parameter
from harnesskit.trace.schema import CostStatus, Trajectory


@dataclass
class Proposal:
    edits: dict[str, Any]
    rationale: str = ""


@dataclass
class ProposalBatch:
    proposals: list[Proposal]
    cost_usd: float | None = 0.0
    cost_status: CostStatus = CostStatus.known_zero
    notes: list[str] = field(default_factory=list)
    trajectories: list[Trajectory] = field(default_factory=list)


class Proposer(Protocol):
    name: str

    def propose(
        self,
        parameters: list[Parameter],
        evidence: Evidence,
        n: int,
        rng: random.Random,
    ) -> ProposalBatch:
        """Return up to ``n`` proposals. ``parameters`` carry current values."""
        ...


def _sample(param: Parameter, rng: random.Random) -> Any:
    d = param.decl
    if d.kind == ParameterKind.choice:
        options = [c for c in d.choices or [] if not (c == param.value and type(c) is type(param.value))]
        return rng.choice(options) if options else None
    if d.min is None or d.max is None:
        return None
    if d.kind == ParameterKind.int:
        options = [v for v in range(int(d.min), int(d.max) + 1) if v != param.value]
        return rng.choice(options) if options else None
    return rng.uniform(d.min, d.max)


@dataclass
class RandomSearchProposer:
    """Mutate one randomly chosen searchable parameter per proposal. Text
    parameters and numeric parameters without both ``min`` and ``max`` are not
    searchable and are skipped (reported in ``notes``)."""

    name: str = "random"

    def propose(self, parameters, evidence, n, rng) -> ProposalBatch:
        searchable = [p for p in parameters if p.requires_grad and _sample(p, random.Random(0)) is not None]
        skipped = [p.name for p in parameters if p.requires_grad and p not in searchable]
        notes = [f"random search skips non-searchable parameter(s): {', '.join(skipped)}"] if skipped else []
        if not searchable:
            return ProposalBatch([], notes=notes + ["no searchable parameters"])
        proposals: list[Proposal] = []
        for _ in range(n * 10):  # bounded retries so a batch has no repeated edits
            if len(proposals) == n:
                break
            p = rng.choice(searchable)
            edits = {p.name: _sample(p, rng)}
            if all(existing.edits != edits for existing in proposals):
                proposals.append(Proposal(edits, rationale=f"random: resample {p.name}"))
        return ProposalBatch(proposals, notes=notes)


@dataclass
class ScriptedProposer:
    """Yields the given proposals in order, ``n`` at a time, then nothing."""

    proposals: list[Proposal]
    name: str = "scripted"
    _cursor: int = 0

    def propose(self, parameters, evidence, n, rng) -> ProposalBatch:
        batch = self.proposals[self._cursor : self._cursor + n]
        self._cursor += len(batch)
        return ProposalBatch(list(batch))


@dataclass
class FunctionProposer:
    """Wrap ``fn(parameters, evidence, n, rng) -> list[Proposal]`` (zero cost)."""

    fn: Callable[[list[Parameter], Evidence, int, random.Random], list[Proposal]]
    name: str = "function"

    def propose(self, parameters, evidence, n, rng) -> ProposalBatch:
        return ProposalBatch(list(self.fn(parameters, evidence, n, rng))[:n])


DEFAULT_PROPOSER_PROMPT = """\
You improve an AI agent harness by editing its trainable parameters.
You receive the parameters (name, kind, constraints, current value) and
evidence from the harness's training cases: inputs, expected checks, the
harness's actual output, and which checks failed.

Propose edits that address the failure patterns in general. Do not
hard-code answers to individual training cases: candidates are selected on
separate validation cases and judged on held-out test cases you never see.
Only edit the listed parameters, respect each parameter's constraints, and
keep each proposal small and focused.

Reply with only a JSON object of the form
{"proposals": [{"rationale": "...", "edits": {"<parameter name>": <new value>}}]}
"""


def build_proposer_input(parameters: list[Parameter], evidence: Evidence, n: int, max_chars: int = 4000) -> str:
    params = []
    for p in parameters:
        if not p.requires_grad:
            continue
        d = p.describe()
        d.pop("requires_grad", None)
        if isinstance(d["value"], str) and len(d["value"]) > max_chars:
            d["value"] = d["value"][:max_chars] + "… [truncated]"
        params.append(d)
    return (
        f"Return up to {n} proposals.\n\n"
        f"PARAMETERS:\n{json.dumps(params, indent=2)}\n\n"
        f"TRAIN EVIDENCE (pass rate {evidence.pass_rate:.0%}; failures first):\n"
        + json.dumps(
            [c.__dict__ for c in sorted(evidence.cases, key=lambda c: c.passed)],
            indent=2,
        )
    )


def parse_proposals(text: str | None) -> tuple[list[Proposal], list[str]]:
    """Parse ``{"proposals": [...]}`` from a model's output, tolerating
    surrounding prose. Malformed output yields no proposals and a note."""
    if not text:
        return [], ["proposer returned no output"]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return [], ["proposer output contained no JSON object"]
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        return [], [f"proposer output was not valid JSON: {e}"]
    raw = data.get("proposals") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return [], ["proposer JSON had no 'proposals' list"]
    proposals, notes = [], []
    for i, item in enumerate(raw):
        if isinstance(item, dict) and isinstance(item.get("edits"), dict) and item["edits"]:
            proposals.append(Proposal(dict(item["edits"]), str(item.get("rationale", ""))))
        else:
            notes.append(f"ignored malformed proposal #{i}")
    return proposals, notes


def default_proposer_spec(
    model_id: str = "claude-sonnet-5", provider: str = "anthropic", source_dir: Path | None = None
) -> HarnessSpec:
    """A minimal single-turn, tool-free proposer harness. Adapters expect a
    loaded harness to have a ``source_dir``; an empty temporary directory is
    used unless one is given."""
    spec = HarnessSpec(
        metadata=Metadata(name="harnesskit-proposer", version="0.1.0"),
        model=ModelConfig(provider=provider, model_id=model_id, requires_tool_use=False),
        loop=LoopConfig(type=LoopStrategyType.react, max_turns=1),
        memory=MemoryConfig(session="none"),
        scaffold=ScaffoldConfig(system_prompt=DEFAULT_PROPOSER_PROMPT, system_prompt_is_file=False),
    )
    spec.source_dir = source_dir or Path(tempfile.mkdtemp(prefix="harnesskit-proposer-"))
    return spec


@dataclass
class HarnessProposer:
    """Run a proposer harness through ``adapter`` once per ``propose`` call."""

    spec: HarnessSpec
    adapter: HarnessAdapter
    name: str = "harness"
    _agent: RunnableAgent | None = None

    @classmethod
    def default(cls, adapter: HarnessAdapter, model_id: str = "claude-sonnet-5") -> HarnessProposer:
        return cls(default_proposer_spec(model_id), adapter)

    def propose(self, parameters, evidence, n, rng) -> ProposalBatch:
        if self._agent is None:
            self._agent = self.adapter.build(self.spec)
        try:
            trajectory = self.adapter.run(self._agent, build_proposer_input(parameters, evidence, n))
        except Exception as e:  # noqa: BLE001 — a failed proposal call is recorded, not fatal
            return ProposalBatch([], cost_usd=None, cost_status=CostStatus.unavailable, notes=[f"proposer run failed: {e}"])
        proposals, notes = parse_proposals(trajectory.final_output)
        unknown = trajectory.has_unavailable_cost
        return ProposalBatch(
            proposals[:n],
            cost_usd=None if unknown else trajectory.total_cost_usd,
            cost_status=CostStatus.unavailable if unknown else CostStatus.estimated,
            notes=notes,
            trajectories=[trajectory],
        )
