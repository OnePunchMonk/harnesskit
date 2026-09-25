"""harnesskit.train: trainable declarations, frozen-component enforcement,
split hygiene, budget admission, and the offline trainable_qa example."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from harnesskit.adapters.callback_adapter import CallbackAdapter, wrap_simple_callback
from harnesskit.cli.main import app
from harnesskit.parser import load_harness
from harnesskit.train import (
    CandidateRejected,
    FunctionProposer,
    HarnessModule,
    HarnessProposer,
    ParameterError,
    Proposal,
    RandomSearchProposer,
    ScriptedProposer,
    TrainBudget,
    Trainer,
    TrainError,
    parse_proposals,
    plan_splits,
)
from harnesskit.train.loading import resolve_adapter_factory
from harnesskit.trace.schema import CostStatus

EXAMPLE = Path(__file__).parent.parent / "examples" / "trainable_qa"
FACTORY = f"custom:{EXAMPLE / 'agent.py'}:make_adapter"


def make_harness(tmp_path: Path, trainable: list[dict], cases: list[dict] | None = None, files: dict[str, str] | None = None) -> Path:
    d = tmp_path / "h"
    d.mkdir()
    (d / "prompt.md").write_text("Be helpful.\n")
    (d / "tools").mkdir()
    (d / "tools" / "search.json").write_text(
        json.dumps({"name": "search", "description": "Search.", "input_schema": {"type": "object", "properties": {}}})
    )
    (d / "config.json").write_text(json.dumps({"mode": "a", "n": 1}))
    (d / "eval").mkdir()
    (d / "eval" / "notes.md").write_text("frozen\n")
    for rel, text in (files or {}).items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(text)
    raw = {
        "schema_version": 1,
        "metadata": {"name": "t"},
        "loop": {"type": "custom", "max_turns": 5},
        "tools": [{"name": "search", "source": "custom", "ref": "tools/search.json", "permissions": ["network"]}],
        "trainable": trainable,
        "eval": {"cases": cases or [{"id": "c1", "input": "x", "expected_output_contains": ["y"]}]},
        "scaffold": {"system_prompt": "prompt.md"},
    }
    (d / "harness.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    return d


# -- declarations ---------------------------------------------------------

def test_example_named_parameters_and_state_dict():
    module = HarnessModule(EXAMPLE)
    names = [n for n, _ in module.named_parameters()]
    assert names == ["answer_mode", "strip_stopwords", "min_overlap"]  # frozen system_prompt excluded
    assert [n for n, _ in module.named_parameters(trainable_only=False)][-1] == "system_prompt"
    state = module.state_dict()
    assert state["strip_stopwords"] is False and state["min_overlap"] == 0


@pytest.mark.parametrize(
    "decl, message",
    [
        ({"name": "p", "target": "file:harness.yaml"}, "frozen"),
        ({"name": "p", "target": "file:eval/notes.md"}, "frozen"),
        ({"name": "p", "target": "file:../outside.md"}, "outside"),
        ({"name": "p", "target": "file:tools/search.json"}, "tool schema"),
        ({"name": "p", "target": "json:tools/search.json#/input_schema/type", "kind": "choice", "choices": ["object"]}, "description"),
        ({"name": "p", "target": "json:config.json#/missing", "kind": "choice", "choices": ["a"]}, "not found"),
        ({"name": "p", "target": "json:config.json#/mode", "kind": "choice"}, "choices"),
        ({"name": "p", "target": "tools", "kind": "text"}, "unsupported"),
        ({"name": "p", "target": "loop.max_turns", "kind": "text"}, "kind"),
    ],
)
def test_invalid_declarations_are_rejected(tmp_path, decl, message):
    d = make_harness(tmp_path, [decl])
    with pytest.raises(ParameterError, match=message):
        HarnessModule(d)


def test_tool_description_is_trainable_but_schema_is_not(tmp_path):
    d = make_harness(tmp_path, [{"name": "desc", "target": "json:tools/search.json#/description"}])
    module = HarnessModule(d)
    cand = module.materialize({"desc": "Search the local corpus."}, tmp_path / "c1")
    schema = json.loads((cand.directory / "tools" / "search.json").read_text())
    assert schema["description"] == "Search the local corpus."
    assert schema["name"] == "search" and schema["input_schema"] == {"type": "object", "properties": {}}
    assert cand.spec.tools[0].permissions == ["network"]


# -- materialize / frozen enforcement ------------------------------------

def test_materialize_writes_each_target_kind(tmp_path):
    d = make_harness(
        tmp_path,
        [
            {"name": "prompt", "target": "scaffold.system_prompt"},
            {"name": "turns", "target": "loop.max_turns", "kind": "int", "min": 1, "max": 10},
            {"name": "skill", "target": "file:skills/style.md"},
            {"name": "mode", "target": "json:config.json#/mode", "kind": "choice", "choices": ["a", "b"]},
        ],
        files={"skills/style.md": "terse\n"},
    )
    module = HarnessModule(d)
    cand = module.materialize({"prompt": "Be precise.\n", "turns": 3, "skill": "verbose\n", "mode": "b"}, tmp_path / "c1")
    assert (cand.directory / "prompt.md").read_text() == "Be precise.\n"
    assert cand.spec.scaffold.system_prompt == "Be precise.\n"
    assert cand.spec.loop.max_turns == 3
    assert (cand.directory / "skills" / "style.md").read_text() == "verbose\n"
    assert json.loads((cand.directory / "config.json").read_text())["mode"] == "b"
    # the source harness is untouched
    assert module.state_dict()["turns"] == 5 and (d / "prompt.md").read_text() == "Be helpful.\n"
    assert {c.name for c in module.diff(cand.state_dict())} == {"prompt", "turns", "skill", "mode"}


@pytest.mark.parametrize(
    "state, message",
    [
        ({"frozen_prompt": "changed"}, "frozen"),
        ({"nope": 1}, "unknown"),
        ({"turns": 99}, "above max"),
        ({"turns": True}, "number"),
        ({"mode": "c"}, "one of"),
    ],
)
def test_invalid_states_are_rejected_and_leave_no_directory(tmp_path, state, message):
    d = make_harness(
        tmp_path,
        [
            {"name": "frozen_prompt", "target": "scaffold.system_prompt", "requires_grad": False},
            {"name": "turns", "target": "loop.max_turns", "kind": "int", "min": 1, "max": 10},
            {"name": "mode", "target": "json:config.json#/mode", "kind": "choice", "choices": ["a", "b"]},
        ],
    )
    with pytest.raises(CandidateRejected, match=message):
        HarnessModule(d).materialize(state, tmp_path / "c1")
    assert not (tmp_path / "c1").exists()


def test_frozen_check_catches_changes_outside_targets(tmp_path):
    d = make_harness(tmp_path, [{"name": "mode", "target": "json:config.json#/mode", "kind": "choice", "choices": ["a", "b"]}])
    module = HarnessModule(d)
    cand = module.materialize({"mode": "b"}, tmp_path / "c1")
    (cand.directory / "eval" / "notes.md").write_text("tampered\n")
    with pytest.raises(CandidateRejected, match="frozen file"):
        module._check_frozen(HarnessModule(cand.directory))
    raw = yaml.safe_load((cand.directory / "harness.yaml").read_text())
    raw["tools"][0]["permissions"] = ["network", "shell"]
    (cand.directory / "harness.yaml").write_text(yaml.safe_dump(raw))
    (cand.directory / "eval" / "notes.md").write_text("frozen\n")
    with pytest.raises(CandidateRejected, match="frozen part of the harness spec"):
        module._check_frozen(HarnessModule(cand.directory))


# -- splits ----------------------------------------------------------------

def test_plan_splits_declared_hashed_and_errors(tmp_path):
    spec = load_harness(EXAMPLE).spec
    plan = plan_splits(spec)
    assert plan.source == "declared" and len(plan.train) == len(plan.val) == len(plan.test) == 4

    cases = [{"id": f"c{i}", "input": "x", "expected_output_contains": ["y"]} for i in range(8)]
    hashed_spec = load_harness(make_harness(tmp_path, [], cases)).spec
    a, b = plan_splits(hashed_spec, seed=1), plan_splits(hashed_spec, seed=1)
    assert a == b and a.source == "hashed(seed=1)"
    assert sorted(a.train + a.val + a.test) == sorted(c["id"] for c in cases)

    partial = hashed_spec.model_copy(deep=True)
    partial.eval.cases[0].split = "train"
    with pytest.raises(TrainError, match="every eval case"):
        plan_splits(partial)
    no_test = hashed_spec.model_copy(deep=True)
    for c in no_test.eval.cases:
        c.split = "train"
    with pytest.raises(TrainError, match="split is empty"):
        plan_splits(no_test)


# -- the training loop -----------------------------------------------------

def _trainer(tmp_path, proposer, budget=None, factory=None, name="work"):
    return Trainer(
        HarnessModule(EXAMPLE),
        factory or resolve_adapter_factory(FACTORY),
        proposer,
        tmp_path / name,
        budget=budget or TrainBudget(max_steps=4, candidates_per_step=3),
    )


def test_random_search_improves_example_on_held_out_test(tmp_path):
    result = _trainer(tmp_path, RandomSearchProposer()).fit()
    assert result.verdict == "improved"
    assert result.best_state["answer_mode"] == "best_sentence"
    assert result.test_initial.pass_rate < result.test_best.pass_rate
    report = json.loads((tmp_path / "work" / "train_report.json").read_text())
    assert report["verdict"] == "improved" and report["candidates"]
    best = tmp_path / "work" / "best"
    assert json.loads((best / "agent_config.json").read_text()) == {k: result.best_state[k] for k in ("answer_mode", "strip_stopwords", "min_overlap")}
    assert "+best_sentence" in (tmp_path / "work" / "best.diff").read_text()
    # every evaluated candidate (accepted or not) is kept for review
    for c in result.candidates:
        if c.train is not None:
            assert Path(c.directory).is_dir()
    # unavailable callback cost is reported, never shown as $0 spend
    assert not report["spend"]["complete"]


def test_proposer_only_ever_sees_train_cases(tmp_path):
    seen: list[str] = []
    splits = plan_splits(load_harness(EXAMPLE).spec)

    def spy(params, evidence, n, rng):
        assert evidence.split == "train"
        seen.extend(c.case_id for c in evidence.cases)
        return [Proposal({"answer_mode": "best_sentence"})]

    _trainer(tmp_path, FunctionProposer(spy), TrainBudget(max_steps=2, candidates_per_step=1)).fit()
    assert seen and set(seen) <= set(splits.train)


def test_invalid_and_worse_candidates_yield_no_supported_improvement(tmp_path):
    proposer = ScriptedProposer([
        Proposal({"system_prompt": "hack"}),  # frozen
        Proposal({"eval_cases": []}),  # not a parameter
        Proposal({"min_overlap": 9}),  # out of bounds
        Proposal({"min_overlap": 0}),  # no-op
        Proposal({"strip_stopwords": True}),  # evaluated, no val gain
    ])
    result = _trainer(tmp_path, proposer, TrainBudget(max_steps=1, candidates_per_step=5)).fit()
    statuses = [c.status for c in result.candidates]
    assert statuses == ["invalid", "invalid", "invalid", "invalid", "rejected"]
    assert all(c.train is None for c in result.candidates[:4])  # invalid proposals cost no evaluation
    assert result.verdict == "no_supported_improvement"
    assert result.test_initial is None and result.test_best is None  # test split untouched
    assert not (tmp_path / "work" / "best").exists()


def test_cost_cap_with_unavailable_cost_stops_unless_allowed(tmp_path):
    proposer = ScriptedProposer([Proposal({"answer_mode": "best_sentence"})])
    result = _trainer(tmp_path, proposer, TrainBudget(max_steps=1, candidates_per_step=1, max_cost_usd=1.0)).fit()
    assert "cannot be enforced" in result.stop_reason
    assert result.candidates[0].train is None

    proposer = ScriptedProposer([Proposal({"answer_mode": "best_sentence"})])
    result = _trainer(tmp_path, proposer, TrainBudget(1, 1, max_cost_usd=1.0, allow_unmetered=True), name="w2").fit()
    assert result.candidates[0].status == "accepted"


class _CostedAdapter:
    """Wrap an adapter and stamp an estimated per-call cost onto its trajectories."""

    def __init__(self, inner, cost):
        self.inner, self.cost = inner, cost

    def supports(self):
        return self.inner.supports()

    def build(self, spec):
        return self.inner.build(spec)

    def run(self, agent, input):
        t = self.inner.run(agent, input)
        for s in t.steps:
            s.cost_usd, s.cost_status = self.cost, CostStatus.estimated
        return t


def test_cost_cap_admission_stops_before_overspending(tmp_path):
    inner = resolve_adapter_factory(FACTORY)
    factory = lambda spec: _CostedAdapter(inner(spec), 0.01)  # noqa: E731 — 4 cases -> $0.04 per split eval
    proposer = RandomSearchProposer()
    result = _trainer(tmp_path, proposer, TrainBudget(max_steps=5, candidates_per_step=3, max_cost_usd=0.13), factory).fit()
    assert result.stop_reason.startswith("budget")
    spend = result.spend.to_dict()
    assert spend["complete"] and spend["known_total_usd"] <= 0.13 + 1e-9


# -- proposers ---------------------------------------------------------------

def test_parse_proposals_tolerates_prose_and_rejects_garbage():
    ok, notes = parse_proposals('Sure!\n{"proposals": [{"rationale": "r", "edits": {"a": 1}}, {"edits": {}}]}\nDone.')
    assert [p.edits for p in ok] == [{"a": 1}] and notes == ["ignored malformed proposal #1"]
    for bad in (None, "no json here", "{not json}", '{"other": 1}'):
        proposals, notes = parse_proposals(bad)
        assert proposals == [] and notes


def test_harness_proposer_runs_a_harness_and_records_cost(tmp_path):
    prompts: list[str] = []

    def fake_model(prompt: str) -> str:
        prompts.append(prompt)
        return json.dumps({"proposals": [{"rationale": "answer from the matching sentence", "edits": {"answer_mode": "best_sentence"}}]})

    proposer = HarnessProposer.default(CallbackAdapter(wrap_simple_callback(fake_model)))
    result = _trainer(tmp_path, proposer, TrainBudget(max_steps=1, candidates_per_step=2)).fit()
    assert result.candidates[0].status == "accepted"
    assert "PARAMETERS" in prompts[0] and "answer_mode" in prompts[0]
    test_inputs = [c.input for c in load_harness(EXAMPLE).spec.eval.cases if c.split != "train"]
    assert not any(inp in prompts[0] for inp in test_inputs)  # val/test inputs never reach the proposer
    proposal_entries = [e for e in result.spend.entries if e["kind"] == "proposal"]
    assert proposal_entries[0]["cost_status"] == "unavailable"  # callback proposer cost is unknown, not $0


# -- CLI -----------------------------------------------------------------------

def test_cli_params_and_train(tmp_path):
    runner = CliRunner()
    r = runner.invoke(app, ["params", str(EXAMPLE), "--json"])
    assert r.exit_code == 0, r.output
    assert [p["name"] for p in json.loads(r.output)][:3] == ["answer_mode", "strip_stopwords", "min_overlap"]

    out = tmp_path / "work"
    r = runner.invoke(app, ["train", str(EXAMPLE), "--adapter", FACTORY, "--steps", "4", "--candidates", "3", "--out", str(out), "--json"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)["verdict"] == "improved"

    r = runner.invoke(app, ["train", str(EXAMPLE), "--adapter", FACTORY, "--out", str(out)])
    assert r.exit_code == 1 and "not empty" in r.output


def test_training_does_not_modify_the_source_harness(tmp_path):
    copy = tmp_path / "src"
    shutil.copytree(EXAMPLE, copy)
    before = {p.relative_to(copy): p.read_bytes() for p in copy.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    Trainer(HarnessModule(copy), resolve_adapter_factory(FACTORY), RandomSearchProposer(), tmp_path / "w").fit()
    after = {p.relative_to(copy): p.read_bytes() for p in copy.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    assert before == after


def test_trained_harness_exports_with_its_trained_values(tmp_path):
    from harnesskit.packaging import export_bundle, import_bundle

    result = _trainer(tmp_path, RandomSearchProposer()).fit()
    bundle = tmp_path / "best.harn"
    export_bundle(Path(result.best_directory), bundle)
    imported = tmp_path / "imported"
    import_bundle(bundle, imported)
    assert HarnessModule(imported).state_dict() == result.best_state


def test_default_llm_proposer_runs_through_raw_api_adapter_offline(tmp_path):
    from harnesskit.adapters.base import RunnableAgent
    from harnesskit.adapters.raw_api import RawAPIAdapter
    from harnesskit.testing.fakes import FakeAnthropicClient, FakeResponse, text_block

    reply = '{"proposals": [{"rationale": "use the matching sentence", "edits": {"answer_mode": "best_sentence"}}]}'
    client = FakeAnthropicClient([FakeResponse([text_block(reply)])])

    class ScriptedRawAPI(RawAPIAdapter):
        def build(self, spec):
            assert spec.source_dir is not None
            return RunnableAgent(spec=spec, handle={"client": client, "tool_schemas": [], "callbacks": {}})

    proposer = HarnessProposer.default(ScriptedRawAPI())
    result = _trainer(tmp_path, proposer, TrainBudget(max_steps=1, candidates_per_step=1)).fit()
    assert result.candidates[0].status == "accepted"
    assert client.calls[0]["system"].startswith("You improve an AI agent harness")
    entry = [e for e in result.spend.entries if e["kind"] == "proposal"][0]
    assert entry["cost_status"] == "estimated" and entry["cost_usd"] > 0  # proposal spend is counted
