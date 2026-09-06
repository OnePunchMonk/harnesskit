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
