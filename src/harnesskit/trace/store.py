"""Trace storage (design doc §8.4): plain JSON files under .harness/runs/.

Deliberately not a database — traces are small, local, and need to be
diffable and replayable, not queried.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import time

from harnesskit.trace.schema import Trajectory


def runs_dir(harness_dir: Path) -> Path:
    d = harness_dir / ".harness" / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_trajectory(harness_dir: Path, trajectory: Trajectory, run_id: str | None = None) -> Path:
    run_id = run_id or f"{int(time() * 1000)}"
    path = runs_dir(harness_dir) / f"{run_id}.json"
    path.write_text(trajectory.model_dump_json(indent=2))
    return path


def load_trajectory(path: Path) -> Trajectory:
    return Trajectory.model_validate_json(path.read_text())


def list_trajectories(harness_dir: Path) -> list[Path]:
    return sorted(runs_dir(harness_dir).glob("*.json"))


def baselines_dir(harness_dir: Path) -> Path:
    d = harness_dir / ".harness" / "baselines"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_baseline(harness_dir: Path, name: str, trajectories_by_case_id: dict[str, Trajectory]) -> Path:
    """Snapshot a completed eval run as a named baseline: {case_id: trajectory}.

    Stored separately from .harness/runs/ (which holds the *latest* run and
    gets overwritten every `harness eval`) so a baseline survives future runs
    and can be diffed against with `harness eval --compare <name>`.
    """
    path = baselines_dir(harness_dir) / f"{name}.json"
    payload = {case_id: json.loads(t.model_dump_json()) for case_id, t in trajectories_by_case_id.items()}
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_baseline(harness_dir: Path, name: str) -> dict[str, Trajectory]:
    path = baselines_dir(harness_dir) / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No baseline named '{name}' at {path}")
    payload = json.loads(path.read_text())
    return {case_id: Trajectory.model_validate(data) for case_id, data in payload.items()}


def list_baselines(harness_dir: Path) -> list[str]:
    return sorted(p.stem for p in baselines_dir(harness_dir).glob("*.json"))
