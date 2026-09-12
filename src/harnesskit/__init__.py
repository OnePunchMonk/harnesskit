"""harnesskit: a small, deliberate public API on top of the CLI internals.

`import harnesskit` supports:

- ``load_harness`` / ``HarnessLoadError`` — load and validate a harness
  directory (``harness.yaml`` plus its eval suite) into a ``HarnessSpec``.
- ``Trajectory`` / ``Step`` — the trace schema every adapter and scorer
  speaks; ``CostStatus`` / ``AttemptStatus`` distinguish observed, estimated,
  known-zero, and unavailable cost so an unknown bill is never reported as
  ``$0``.
- ``save_trajectory`` / ``load_trajectory`` — record and load a single run
  (collision-resistant run ids, atomic writes).
- ``save_baseline`` / ``load_baseline`` / ``list_baselines`` — import/replace
  a named baseline explicitly (``overwrite=True`` required to replace one)
  and load legacy or current-format baselines interchangeably.
- ``run_suite`` — supported execution: run a harness's eval suite fresh
  against an adapter.
- ``replay_suite`` — scorer-only replay of previously recorded trajectories,
  with missing cases reported rather than silently dropped.
- ``compare`` / ``check_regression`` — paired, case-ID-aligned comparison
  between two suite results (e.g. a baseline vs. a current run).
- ``export_bundle`` / ``import_bundle`` / ``preview_bundle`` — the ``.harn``
  packaging/import boundary. Import validates the exact archive member set
  against its checksum manifest, rejects duplicates/unsafe paths/symlinks,
  and checks the bundle format version, all before extracting anything.

This is a thin, curated re-export layer — it does not duplicate any logic.
Everything else in ``harnesskit.*`` (CLI command wiring, adapters, the
linter, scaffold generators, the parser's internals) is CLI-only or
internal and not covered by this compatibility surface: it may change
without notice between versions. Import from the relevant
``harnesskit.<subpackage>`` module directly if you need it, but don't build
long-term integrations on it the way you can on the names below.
"""
from __future__ import annotations

from harnesskit.eval import CaseResult, CaseStatus, SuiteResult, check_regression, compare, replay_suite, run_suite
from harnesskit.packaging import (
    BundleResult,
    BundleValidationError,
    ImportPreview,
    export_bundle,
    import_bundle,
    preview_bundle,
)
from harnesskit.parser import HarnessLoadError, load_harness
from harnesskit.trace import (
    AttemptStatus,
    BaselineProvenance,
    CostStatus,
    IncompatibleArtifactError,
    Step,
    Trajectory,
    current_provenance,
    list_baselines,
    list_trajectories,
    load_baseline,
    load_baseline_provenance,
    load_trajectory,
    save_baseline,
    save_trajectory,
)

__all__ = [
    # load/validate
    "load_harness",
    "HarnessLoadError",
    # trace schema
    "Trajectory",
    "Step",
    "CostStatus",
    "AttemptStatus",
    # record/import a trajectory
    "save_trajectory",
    "load_trajectory",
    "list_trajectories",
    "save_baseline",
    "load_baseline",
    "list_baselines",
    "IncompatibleArtifactError",
    "BaselineProvenance",
    "current_provenance",
    "load_baseline_provenance",
    # supported execution / replay / compare
    "run_suite",
    "replay_suite",
    "compare",
    "check_regression",
    "SuiteResult",
    "CaseResult",
    "CaseStatus",
    # bundle import boundary
    "export_bundle",
    "import_bundle",
    "preview_bundle",
    "BundleResult",
    "ImportPreview",
    "BundleValidationError",
]
