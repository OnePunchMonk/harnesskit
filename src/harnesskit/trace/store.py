"""Trace storage (design doc §8.4): plain JSON files under .harness/runs/.

Deliberately not a database — traces are small, local, and need to be
diffable and replayable, not queried.

Two correctness properties this module guarantees (HK-04):

1. Writes are atomic. A run/baseline file is written to a sibling temp file
   and then moved into place with `os.replace`, which is atomic on both
   POSIX and Windows. A crash mid-write can never leave a corrupted or
   half-written file at the destination path.
2. Auto-generated run IDs are collision-resistant. Two runs started within
   the same millisecond (e.g. under a frozen/mocked clock in tests, or a
   fast CI loop) get distinct IDs and therefore distinct files — neither
   silently replaces the other. Named baselines go further: writing to an
   existing baseline name requires an explicit `overwrite=True`, so a
   baseline that a comparison depends on can't be clobbered by accident.
"""
from __future__ import annotations

import json
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from time import time

from harnesskit.trace.schema import Trajectory

# Bumped whenever the on-disk run/baseline JSON shape changes in a way that
# requires migration. Files saved before this field existed (schema_version
# key absent entirely) are the original flat format and are treated as
# version 1 for backward compatibility — they still load.
CURRENT_SCHEMA_VERSION = 2
MIN_SUPPORTED_SCHEMA_VERSION = 1


class IncompatibleArtifactError(ValueError):
    """Raised when a stored run or baseline is a schema version this build
    of harnesskit does not know how to read. Actionable by design — the
    message says which version was found and what this build supports —
    instead of an opaque JSON/pydantic validation crash deeper in the stack.
    """


def runs_dir(harness_dir: Path) -> Path:
    d = harness_dir / ".harness" / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _new_run_id() -> str:
    """Collision-resistant run id: millisecond timestamp (kept for sort
    order and human readability) plus a random suffix. Two runs started in
    the same millisecond — including under a frozen clock, as in tests —
    still get distinct ids."""
    return f"{int(time() * 1000)}-{uuid.uuid4().hex[:12]}"


def _atomic_write_text(path: Path, text: str) -> None:
    """Write via temp-file-then-rename so a crash mid-write, or a concurrent
    writer to the same path, can never leave a torn/partial file at `path`."""
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp_path.write_text(text)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _check_schema_version(version: int, what: str) -> None:
    if version > CURRENT_SCHEMA_VERSION:
        raise IncompatibleArtifactError(
            f"{what} was saved by a newer harnesskit (schema_version={version}); "
            f"this build only understands up to schema_version={CURRENT_SCHEMA_VERSION}. "
            "Upgrade harnesskit to read it."
        )
    if version < MIN_SUPPORTED_SCHEMA_VERSION:
        raise IncompatibleArtifactError(f"{what} has an unsupported schema_version={version}.")


def save_trajectory(harness_dir: Path, trajectory: Trajectory, run_id: str | None = None) -> Path:
    """Save a run. An auto-generated run_id is always collision-resistant
    and never overwrites a previous run's file. An explicit `run_id` (e.g.
    a case id, used to save one trajectory per case during an eval sweep)
    intentionally may overwrite an existing file for that same id — that's
    the "latest run" contract for .harness/runs/, distinct from the
    non-overwriting named-baseline contract below."""
    run_id = run_id or _new_run_id()
    path = runs_dir(harness_dir) / f"{run_id}.json"
    payload = {"schema_version": CURRENT_SCHEMA_VERSION, "trajectory": json.loads(trajectory.model_dump_json())}
    _atomic_write_text(path, json.dumps(payload, indent=2))
    return path


def load_trajectory(path: Path) -> Trajectory:
    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and "schema_version" in raw:
        _check_schema_version(raw["schema_version"], f"Run artifact {path}")
        return Trajectory.model_validate(raw["trajectory"])
    # Legacy format (pre-schema_version): the file *is* the trajectory JSON.
    return Trajectory.model_validate(raw)


def list_trajectories(harness_dir: Path) -> list[Path]:
    return sorted(runs_dir(harness_dir).glob("*.json"))


def baselines_dir(harness_dir: Path) -> Path:
    d = harness_dir / ".harness" / "baselines"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class BaselineProvenance:
    """What the baseline was captured with — compared against the current
    run's own values so a regression gate can't mistake environment drift
    (a different model, adapter, harness version, or git sha) for a real
    harness regression (issue #4 item 15)."""

    model_id: str | None
    adapter: str | None
    harness_name: str | None
    harness_version: str | None
    git_sha: str | None
    saved_at: float | None

    @property
    def is_legacy(self) -> bool:
        """True for a pre-provenance baseline: nothing was recorded, so a
        mismatch check must not silently claim a match."""
        return (
            self.model_id is None
            and self.adapter is None
            and self.harness_name is None
            and self.harness_version is None
            and self.git_sha is None
            and self.saved_at is None
        )


def _git_sha(harness_dir: Path) -> str | None:
    """Best-effort `git rev-parse HEAD` in the harness's directory. Returns
    None (never raises) when the directory isn't inside a git repo, or git
    isn't available at all."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(harness_dir),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def current_provenance(harness_dir: Path, model_id: str, adapter: str, harness_name: str, harness_version: str) -> BaselineProvenance:
    """Build the provenance record for a run about to be saved as a baseline
    (or compared against one)."""
    return BaselineProvenance(
        model_id=model_id,
        adapter=adapter,
        harness_name=harness_name,
        harness_version=harness_version,
        git_sha=_git_sha(harness_dir),
        saved_at=time(),
    )


def save_baseline(
    harness_dir: Path,
    name: str,
    trajectories_by_case_id: dict[str, Trajectory],
    overwrite: bool = False,
    provenance: BaselineProvenance | None = None,
) -> Path:
    """Snapshot a completed eval run as a named baseline: {case_id: trajectory}.

    Stored separately from .harness/runs/ (which holds the *latest* run and
    gets overwritten every `harness eval`) so a baseline survives future runs
    and can be diffed against with `harness eval --compare <name>`.

    A named baseline is evidence other comparisons depend on, so writing to
    an existing name without `overwrite=True` raises `FileExistsError`
    rather than silently replacing it.

    `provenance` (model id, adapter, harness name+version, git sha, timestamp)
    is persisted alongside the trajectories so a future `--compare` can warn
    loudly when the environment that captured the baseline doesn't match the
    current run's.
    """
    path = baselines_dir(harness_dir) / f"{name}.json"
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Baseline '{name}' already exists at {path}. Pass overwrite=True "
            "(or `--overwrite-baseline` on the CLI) to replace it explicitly."
        )
    payload = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "baselines": {case_id: json.loads(t.model_dump_json()) for case_id, t in trajectories_by_case_id.items()},
        "provenance": (
            {
                "model_id": provenance.model_id,
                "adapter": provenance.adapter,
                "harness_name": provenance.harness_name,
                "harness_version": provenance.harness_version,
                "git_sha": provenance.git_sha,
                "saved_at": provenance.saved_at,
            }
            if provenance is not None
            else None
        ),
    }
    _atomic_write_text(path, json.dumps(payload, indent=2))
    return path


def load_baseline(harness_dir: Path, name: str) -> dict[str, Trajectory]:
    path = baselines_dir(harness_dir) / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No baseline named '{name}' at {path}")
    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and "schema_version" in raw:
        _check_schema_version(raw["schema_version"], f"Baseline '{name}'")
        data = raw["baselines"]
    else:
        # Legacy format: {case_id: trajectory} directly, no wrapper.
        data = raw
    return {case_id: Trajectory.model_validate(t) for case_id, t in data.items()}


def load_baseline_provenance(harness_dir: Path, name: str) -> BaselineProvenance:
    """Load just the provenance record for a baseline. A legacy baseline (no
    schema_version wrapper, or a wrapper with no "provenance" key) reports an
    all-None `BaselineProvenance` (`.is_legacy` is True) rather than raising
    or silently pretending to match the current run."""
    path = baselines_dir(harness_dir) / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No baseline named '{name}' at {path}")
    raw = json.loads(path.read_text())
    prov = raw.get("provenance") if isinstance(raw, dict) else None
    if not prov:
        return BaselineProvenance(model_id=None, adapter=None, harness_name=None, harness_version=None, git_sha=None, saved_at=None)
    return BaselineProvenance(
        model_id=prov.get("model_id"),
        adapter=prov.get("adapter"),
        harness_name=prov.get("harness_name"),
        harness_version=prov.get("harness_version"),
        git_sha=prov.get("git_sha"),
        saved_at=prov.get("saved_at"),
    )


def list_baselines(harness_dir: Path) -> list[str]:
    return sorted(p.stem for p in baselines_dir(harness_dir).glob("*.json"))
