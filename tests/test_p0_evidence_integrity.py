"""Regression tests for the P0 acceptance criteria in issue #9:

- repeated/concurrent writes cannot replace previous evidence
- altered/incompatible baselines cannot silently compare
- an extra unlisted bundle file is rejected before extraction
- a path-traversal bundle member is rejected before extraction
- an unknown bill cannot pass as $0, and that status survives aggregation
- missing/errored cases stay visible in the denominator, not silently dropped
- the small public `import harnesskit` API surface works
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

import harnesskit
from harnesskit.eval.engine import CaseResult, CaseStatus, SuiteResult, run_case
from harnesskit.packaging.bundle import BundleValidationError, export_bundle, import_bundle
from harnesskit.parser import load_harness
from harnesskit.trace.schema import CostStatus, Step, StepType, Trajectory
from harnesskit.trace.store import IncompatibleArtifactError, save_baseline, save_trajectory

EXAMPLE = Path(__file__).parent.parent / "examples" / "react-web-researcher"


def _trajectory(cost_usd: float | None = 0.001, cost_status: CostStatus | None = CostStatus.estimated) -> Trajectory:
    return Trajectory(
        harness_name="react-web-researcher",
        harness_version="0.1.0",
        model_id="claude-sonnet-5",
        adapter="mock",
        input="hi",
        steps=[
            Step(
                step_type=StepType.llm_call,
                tokens_in=10,
                tokens_out=10,
                cost_usd=cost_usd,
                cost_status=cost_status,
            )
        ],
        final_output="done",
    )


# --- collision-resistant, non-overwriting run storage ----------------------


def test_frozen_clock_runs_never_collide(tmp_path, monkeypatch):
    import harnesskit.trace.store as store

    monkeypatch.setattr(store, "time", lambda: 1_700_000_000.123)

    path_a = save_trajectory(tmp_path, _trajectory())
    path_b = save_trajectory(tmp_path, _trajectory())

    assert path_a != path_b
    assert path_a.exists() and path_b.exists()
    assert len(list(store.list_trajectories(tmp_path))) == 2


# --- named baseline overwrite protection ------------------------------------


def test_save_baseline_without_overwrite_raises_then_succeeds_with_flag(tmp_path):
    save_baseline(tmp_path, "v1", {"case-a": _trajectory()})

    with pytest.raises(FileExistsError):
        save_baseline(tmp_path, "v1", {"case-a": _trajectory(cost_usd=0.5)})

    # unchanged after the rejected write
    from harnesskit.trace.store import load_baseline

    assert load_baseline(tmp_path, "v1")["case-a"].total_cost_usd == 0.001

    save_baseline(tmp_path, "v1", {"case-a": _trajectory(cost_usd=0.5)}, overwrite=True)
    assert load_baseline(tmp_path, "v1")["case-a"].total_cost_usd == 0.5


# --- legacy artifact compatibility -------------------------------------------


def test_legacy_flat_baseline_still_loads(tmp_path):
    from harnesskit.trace.store import baselines_dir, load_baseline

    legacy_path = baselines_dir(tmp_path) / "legacy.json"
    legacy_payload = {"case-a": json.loads(_trajectory().model_dump_json())}
    legacy_path.write_text(json.dumps(legacy_payload))

    loaded = load_baseline(tmp_path, "legacy")
    assert loaded["case-a"].total_cost_usd == 0.001


def test_future_schema_version_baseline_raises_actionable_error(tmp_path):
    from harnesskit.trace.store import baselines_dir, load_baseline

    path = baselines_dir(tmp_path) / "toonew.json"
    path.write_text(json.dumps({"schema_version": 999, "baselines": {}}))

    with pytest.raises(IncompatibleArtifactError, match="schema_version=999"):
        load_baseline(tmp_path, "toonew")


# --- bundle import validation -------------------------------------------------


def test_bundle_with_unlisted_extra_file_is_rejected_before_extraction(tmp_path):
    bundle_path = tmp_path / "out.harn"
    export_bundle(EXAMPLE, bundle_path)

    tampered_path = tmp_path / "extra.harn"
    with zipfile.ZipFile(bundle_path) as src, zipfile.ZipFile(tampered_path, "w") as dst:
        for item in src.infolist():
            dst.writestr(item, src.read(item.filename))
        dst.writestr("extra_unlisted_file.txt", "surprise")

    target = tmp_path / "imported-extra"
    with pytest.raises(BundleValidationError, match="present in bundle but not listed in manifest"):
        import_bundle(tampered_path, target)

    assert not target.exists() or not any(target.iterdir())


def test_bundle_with_path_traversal_member_is_rejected(tmp_path):
    bundle_path = tmp_path / "out.harn"
    export_bundle(EXAMPLE, bundle_path)

    evil_path = tmp_path / "evil.harn"
    with zipfile.ZipFile(bundle_path) as src, zipfile.ZipFile(evil_path, "w") as dst:
        manifest = json.loads(src.read("manifest.json"))
        for item in src.infolist():
            if item.filename == "manifest.json":
                continue
            dst.writestr(item, src.read(item.filename))
        # smuggle a traversal member and list it in the manifest so the
        # member-set check alone wouldn't catch it -- only the path-safety
        # check should.
        traversal_name = "../../evil"
        manifest["checksums"][traversal_name] = "0" * 64
        dst.writestr(traversal_name, "pwned")
        dst.writestr("manifest.json", json.dumps(manifest))

    target = tmp_path / "imported-evil"
    with pytest.raises(BundleValidationError, match="Unsafe member path"):
        import_bundle(evil_path, target)

    assert not target.exists() or not any(target.iterdir())


def test_bundle_missing_manifest_entry_rejected(tmp_path):
    bundle_path = tmp_path / "out.harn"
    export_bundle(EXAMPLE, bundle_path)

    trimmed_path = tmp_path / "trimmed.harn"
    with zipfile.ZipFile(bundle_path) as src, zipfile.ZipFile(trimmed_path, "w") as dst:
        manifest = json.loads(src.read("manifest.json"))
        manifest["checksums"]["a_file_that_does_not_exist.txt"] = "0" * 64
        for item in src.infolist():
            if item.filename == "manifest.json":
                continue
            dst.writestr(item, src.read(item.filename))
        dst.writestr("manifest.json", json.dumps(manifest))

    with pytest.raises(BundleValidationError, match="listed in manifest but missing from bundle"):
        import_bundle(trimmed_path, tmp_path / "imported-trimmed")


# --- unknown cost stays "unavailable", never $0 -------------------------------


def test_unknown_model_cost_is_unavailable_not_zero():
    from harnesskit.trace.schema import estimate_cost_usd

    cost, status = estimate_cost_usd("some-model-not-in-pricing-table", 100, 50)
    assert cost is None
    assert status == CostStatus.unavailable


def test_unavailable_cost_survives_into_suite_aggregation():
    result = load_harness(EXAMPLE)
    case_a = result.spec.eval.cases[0]

    known_traj = _trajectory(cost_usd=0.01, cost_status=CostStatus.estimated)
    unknown_traj = _trajectory(cost_usd=None, cost_status=CostStatus.unavailable)

    results = [
        CaseResult(case=case_a, trajectory=known_traj, scores=[]),
        CaseResult(case=case_a, trajectory=unknown_traj, scores=[]),
    ]
    suite = SuiteResult(harness_name="test", results=results)

    # the unavailable case must not be silently averaged in as $0 -- it's
    # excluded from avg_cost_usd and separately surfaced.
    assert suite.avg_cost_usd == 0.01
    assert suite.unavailable_cost_case_ids == [case_a.id]
    assert unknown_traj.total_cost_usd == 0.0  # would look like "free" without has_unavailable_cost
    assert unknown_traj.has_unavailable_cost is True


def test_suite_with_no_known_cost_reports_avg_cost_as_none_not_zero():
    result = load_harness(EXAMPLE)
    case_a = result.spec.eval.cases[0]
    unknown_traj = _trajectory(cost_usd=None, cost_status=CostStatus.unavailable)
    suite = SuiteResult(harness_name="test", results=[CaseResult(case=case_a, trajectory=unknown_traj, scores=[])])
    assert suite.avg_cost_usd is None


# --- errored/missing cases stay in the denominator ----------------------------


class _FlakyAdapter:
    def __init__(self, ok_input: str):
        self._ok_input = ok_input

    def build(self, spec):
        return object()

    def run(self, agent, input: str) -> Trajectory:
        if input != self._ok_input:
            raise RuntimeError("simulated adapter failure")
        return _trajectory()


def test_errored_case_stays_visible_in_denominator_not_dropped():
    result = load_harness(EXAMPLE)
    cases = result.spec.eval.cases
    assert len(cases) >= 2

    adapter = _FlakyAdapter(ok_input=cases[0].input)
    results = [run_case(result.spec, case, adapter, agent=None) for case in cases]
    suite = SuiteResult(harness_name="test", results=results)

    # the errored case is still counted -- present in results, and lowers
    # pass_rate rather than vanishing from the denominator.
    assert len(suite.results) == len(cases)
    errored = [r for r in suite.results if r.status == CaseStatus.errored]
    assert len(errored) == len(cases) - 1
    assert suite.errored_case_ids
    assert suite.pass_rate < 1.0
    # pass_rate's denominator is every attempted case, not just the ones
    # that ran cleanly:
    assert suite.pass_rate == sum(1 for r in suite.results if r.passed) / len(cases)


# --- the small public API surface --------------------------------------------


def test_public_api_exports_load_and_load_baseline(tmp_path):
    result = harnesskit.load_harness(EXAMPLE)
    assert result.spec.metadata.name == "react-web-researcher"

    harnesskit.save_baseline(tmp_path, "pub", {"case-a": _trajectory()})
    loaded = harnesskit.load_baseline(tmp_path, "pub")
    assert loaded["case-a"].total_cost_usd == 0.001
    assert harnesskit.list_baselines(tmp_path) == ["pub"]


def test_public_api_exports_run_replay_compare():
    result = harnesskit.load_harness(EXAMPLE)
    case = result.spec.eval.cases[0]

    class _Mock:
        def build(self, spec):
            return object()

        def run(self, agent, input):
            return _trajectory()

    suite_a = harnesskit.run_suite(result.spec, _Mock(), sample=1)
    suite_b = harnesskit.replay_suite(result.spec, {case.id: _trajectory()})
    ab = harnesskit.compare(suite_a, suite_b)
    assert ab.deltas


def test_public_api_exports_bundle_functions(tmp_path):
    bundle_path = tmp_path / "out.harn"
    result = harnesskit.export_bundle(EXAMPLE, bundle_path)
    assert result.path.exists()
    preview = harnesskit.preview_bundle(bundle_path)
    assert "harness.yaml" in preview.files
    target = tmp_path / "imported"
    harnesskit.import_bundle(bundle_path, target)
    assert (target / "harness.yaml").exists()
