"""The 15-minute local debugging walkthrough (issue #9, P1).

Run directly: `python examples/local_qa_recipe/run_recipe.py`
Or import `run_all()` from a test for a machine-checkable version of each
step's artifact/signal.

Six steps, each a function below:

1. run_baseline            — run the shipped offline recipe.
2. inspect_failure          — look at the intentionally-failed case's steps.
3. promote_to_regression    — save it as a reviewed regression case + baseline.
4. change_scorer_and_replay — replay cached trajectories under a looser
                               scorer; score changes, no new agent calls.
5. change_harness_and_block — changing the wrapped agent requires a fresh
                               run to measure any behavioral effect; replay
                               alone must say so, not silently reuse old data.
6. compare_and_export       — compare two fresh runs, export the winner as a
                               bundle, and re-import it to prove round-trip.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from harnesskit.adapters.callback_adapter import CallbackAdapter, wrap_simple_callback
from harnesskit.eval.engine import CaseResult, SuiteResult, compare, run_case
from harnesskit.eval.scorers import score_trajectory
from harnesskit.packaging.bundle import export_bundle, import_bundle
from harnesskit.parser import load_harness
from harnesskit.trace.store import save_baseline

RECIPE_DIR = Path(__file__).parent
if str(RECIPE_DIR) not in sys.path:
    sys.path.insert(0, str(RECIPE_DIR))

from agent import ImprovedQAAgent, NaiveQAAgent  # noqa: E402 — must follow sys.path setup


class HarnessChangeRequiresFreshRunError(Exception):
    """Raised when a replay is attempted after something that affects
    behavior (here: the wrapped agent's own version) changed since the
    trajectories being replayed were recorded. Replay only re-scores cached
    trajectories — it cannot measure the effect of a change it never ran."""


def build_suite_result(spec, adapter: CallbackAdapter) -> SuiteResult:
    """Like `eval.engine.run_suite`, but keeps this file self-contained about
    what "one case's tool/result events" means for a callback-wrapped agent
    (see `inspect_failure`)."""
    agent = adapter.build(spec)
    results = [run_case(spec, case, adapter, agent) for case in spec.eval.cases]
    return SuiteResult(harness_name=spec.metadata.name, results=results)


def step1_run_baseline(spec, wrapped_agent: NaiveQAAgent) -> SuiteResult:
    adapter = CallbackAdapter(callback=wrap_simple_callback(wrapped_agent.answer_question))
    suite = build_suite_result(spec, adapter)
    print(f"[1] baseline run: {sum(r.passed for r in suite.results)}/{len(suite.results)} cases passed")
    return suite


def step2_inspect_failure(suite: SuiteResult, wrapped_agent: NaiveQAAgent) -> CaseResult:
    failing = [r for r in suite.results if not r.passed]
    assert failing, "expected at least one intentionally-failed case in this fixture"
    failure = failing[0]
    # CallbackAdapter only observes the wrapped agent's final output and
    # wall-clock duration (see its module docstring) — it has no visibility
    # into the agent's internal lookup. The agent's own `call_log` is what
    # lets us inspect the "tool/result" event here; a fully opaque callback
    # would only offer the final output.
    log_entry = next((e for e in wrapped_agent.call_log if e["question"] == failure.case.input), None)
    print(f"[2] inspecting failed case '{failure.case.id}':")
    print(f"    input: {failure.case.input!r}")
    print(f"    expected_output_contains: {failure.case.expected_output_contains}")
    print(f"    actual output: {failure.trajectory.final_output!r}")
    print(f"    wrapped-agent lookup event: {log_entry}")
    return failure


def step3_promote_to_regression(harness_dir: Path, failure: CaseResult) -> Path:
    """A regression case here already carries explicit expected state
    (`expected_output_contains`) — "promoting" it means recording its
    trajectory as a named, non-overwritable baseline so future changes are
    compared against this exact reviewed failure rather than re-derived."""
    baseline_path = save_baseline(
        harness_dir, f"regression-{failure.case.id}", {failure.case.id: failure.trajectory}, overwrite=True
    )
    print(f"[3] promoted '{failure.case.id}' to a regression baseline at {baseline_path}")
    return baseline_path


def step4_change_scorer_and_replay(spec, suite: SuiteResult, wrapped_agent: NaiveQAAgent) -> tuple[SuiteResult, int]:
    """Loosen the nepal_capital case's assertion and re-score the *same*
    cached trajectories — no new call to the wrapped agent."""
    calls_before = len(wrapped_agent.call_log)
    trajectories_by_id = {r.case.id: r.trajectory for r in suite.results}

    loosened_spec = copy.deepcopy(spec)
    for case in loosened_spec.eval.cases:
        if case.id == "nepal_capital":
            # Loosen from an exact fact match to "did it at least try to
            # mention Nepal" — a deliberately different, weaker bar.
            case.expected_output_contains = ["Nepal"]

    results = []
    for case in loosened_spec.eval.cases:
        trajectory = trajectories_by_id[case.id]
        results.append(CaseResult(case=case, trajectory=trajectory, scores=score_trajectory(trajectory, case)))
    replayed = SuiteResult(harness_name=loosened_spec.metadata.name, results=results)

    calls_after = len(wrapped_agent.call_log)
    assert calls_after == calls_before, "replay must not invoke the wrapped agent again"
    print(
        f"[4] replay with a looser scorer: nepal_capital passed="
        f"{[r for r in replayed.results if r.case.id == 'nepal_capital'][0].passed} "
        f"(was {[r for r in suite.results if r.case.id == 'nepal_capital'][0].passed}); "
        f"agent calls unchanged at {calls_after}"
    )
    return replayed, calls_after


def step5_change_harness_and_block(recorded_agent_version: str, current_agent: NaiveQAAgent) -> str:
    """Changing the wrapped agent (v1 -> v2) is a harness-affecting change.
    Replaying old trajectories cannot measure its effect; this must be
    reported explicitly rather than silently reusing stale data."""
    try:
        if current_agent.version != recorded_agent_version:
            raise HarnessChangeRequiresFreshRunError(
                f"wrapped agent changed ({recorded_agent_version!r} -> {current_agent.version!r}); "
                "replay only re-scores cached trajectories from the old agent and cannot measure "
                "this change's behavioral effect — run the suite fresh instead."
            )
        return "no change detected; replay remains valid"
    except HarnessChangeRequiresFreshRunError as e:
        print(f"[5] blocked as expected: {e}")
        return str(e)


def step6_compare_and_export(spec, harness_dir: Path) -> tuple[SuiteResult, SuiteResult, Path]:
    v1_agent = NaiveQAAgent()
    v2_agent = ImprovedQAAgent()
    v1_suite = build_suite_result(spec, CallbackAdapter(callback=wrap_simple_callback(v1_agent.answer_question), model_id="external:callback:v1"))
    v2_suite = build_suite_result(spec, CallbackAdapter(callback=wrap_simple_callback(v2_agent.answer_question), model_id="external:callback:v2"))

    ab = compare(v1_suite, v2_suite)
    print(f"[6] fresh v1 vs v2 comparison: pass_rate {v1_suite.pass_rate:.2f} -> {v2_suite.pass_rate:.2f}")

    save_baseline(harness_dir, "v2-winner", {r.case.id: r.trajectory for r in v2_suite.results}, overwrite=True)
    bundle_path = harness_dir / ".harness" / "exports" / "v2-winner.harn"
    export_bundle(harness_dir, bundle_path)
    print(f"    exported winning config to {bundle_path}")

    with TemporaryDirectory() as tmp:
        reimport_dir = Path(tmp) / "reimported"
        import_bundle(bundle_path, reimport_dir)
        reloaded = load_harness(reimport_dir)
        assert reloaded.spec.metadata.name == spec.metadata.name
        print(f"    re-imported bundle at {reimport_dir}, spec verified")

    return v1_suite, v2_suite, bundle_path


def run_all(harness_dir: Path = RECIPE_DIR) -> dict:
    spec = load_harness(harness_dir).spec

    agent_v1 = NaiveQAAgent()
    baseline_suite = step1_run_baseline(spec, agent_v1)
    failure = step2_inspect_failure(baseline_suite, agent_v1)
    baseline_path = step3_promote_to_regression(harness_dir, failure)
    replayed_suite, calls_after_replay = step4_change_scorer_and_replay(spec, baseline_suite, agent_v1)
    block_message = step5_change_harness_and_block(agent_v1.version, ImprovedQAAgent())
    v1_suite, v2_suite, bundle_path = step6_compare_and_export(spec, harness_dir)

    return {
        "baseline_suite": baseline_suite,
        "failure": failure,
        "baseline_path": baseline_path,
        "replayed_suite": replayed_suite,
        "calls_after_replay": calls_after_replay,
        "block_message": block_message,
        "v1_suite": v1_suite,
        "v2_suite": v2_suite,
        "bundle_path": bundle_path,
    }


if __name__ == "__main__":
    run_all()
