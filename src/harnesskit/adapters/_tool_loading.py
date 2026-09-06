"""Shared tool-callback loading convention, used by every adapter that
supports source="custom" tools: a tool declared at `tools/<name>.json` may
have a sibling `tools/<name>.py` exposing `def run(**kwargs) -> str`.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


def load_tool_callback(harness_dir: Path, tool_ref: str):
    py_path = (harness_dir / tool_ref).with_suffix(".py")
    if not py_path.exists():
        return None
    spec = importlib.util.spec_from_file_location(py_path.stem, py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, "run", None)
