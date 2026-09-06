"""Loads a harness directory into a typed, fully-resolved HarnessSpec.

Responsibilities (design doc §3):
  - file resolution: turn `path:`-style references into actual file contents
  - schema versioning: dispatch on schema_version for future migrations
  - partial harness support: warn, don't error, on missing optional pieces
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import ValidationError

from harnesskit.format.spec import HarnessSpec


class HarnessLoadError(Exception):
    pass


@dataclass
class LoadResult:
    spec: HarnessSpec
    warnings: list[str] = field(default_factory=list)


def load_harness(directory: str | Path) -> LoadResult:
    directory = Path(directory)
    manifest_path = directory / "harness.yaml"
    if not manifest_path.exists():
        raise HarnessLoadError(f"No harness.yaml found in {directory}")

    raw = yaml.safe_load(manifest_path.read_text()) or {}

    version = raw.get("schema_version", 1)
    if version != 1:
        raise HarnessLoadError(
            f"Unsupported schema_version={version}. This build of harnesskit only understands version 1."
        )

    warnings: list[str] = []

    # Resolve system_prompt: if scaffold.system_prompt_is_file (default True),
    # treat the value as a path relative to the harness directory.
    scaffold = raw.get("scaffold", {})
    if scaffold.get("system_prompt_is_file", True) and "system_prompt" in scaffold:
        prompt_path = directory / scaffold["system_prompt"]
        if prompt_path.exists():
            scaffold = {**scaffold, "system_prompt": prompt_path.read_text()}
        else:
            raise HarnessLoadError(f"scaffold.system_prompt file not found: {prompt_path}")
    raw["scaffold"] = scaffold

    if not raw.get("eval", {}).get("cases") and not raw.get("eval", {}).get("cases_file"):
        warnings.append("No eval cases defined — `harness eval` will have nothing to run yet.")

    try:
        spec = HarnessSpec.model_validate(raw)
    except ValidationError as e:
        raise HarnessLoadError(f"Invalid harness.yaml:\n{e}") from e

    spec.source_dir = directory

    # Verify referenced tool files resolve.
    for tool in spec.tools:
        if tool.source.value == "custom":
            tool_path = directory / tool.ref
            if not tool_path.exists():
                warnings.append(f"Tool '{tool.name}' references missing file: {tool.ref}")

    return LoadResult(spec=spec, warnings=warnings)
