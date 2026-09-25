"""Resolve adapter factories and proposers from CLI-style strings.

``custom:<file.py>:<function>`` imports ``file.py`` and calls
``function(spec) -> HarnessAdapter`` for every candidate. This executes the
named file as trusted local code; it is not a sandbox.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from harnesskit.adapters import ADAPTERS
from harnesskit.parser import load_harness
from harnesskit.train.proposers import HarnessProposer, Proposal, RandomSearchProposer, ScriptedProposer
from harnesskit.train.trainer import AdapterFactory


def _load_attr(ref: str, kind: str):
    path_str, sep, attr = ref.rpartition(":")
    if not sep or not path_str or not attr:
        raise ValueError(f"expected {kind} 'custom:<file.py>:<name>', got 'custom:{ref}'")
    path = Path(path_str).resolve()
    if not path.is_file():
        raise ValueError(f"{kind} file not found: {path}")
    module_name = f"_harnesskit_custom_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))  # let the file import its siblings
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    finally:
        sys.path.remove(str(path.parent))
    if not hasattr(module, attr):
        raise ValueError(f"{path} has no attribute '{attr}'")
    return getattr(module, attr)


def resolve_adapter_factory(name: str) -> AdapterFactory:
    if name.startswith("custom:"):
        return _load_attr(name[len("custom:"):], "adapter factory")
    adapter_cls = ADAPTERS.get(name)
    if adapter_cls is None:
        raise ValueError(f"unknown adapter '{name}'; use one of {', '.join(ADAPTERS)} or custom:<file.py>:<factory>")
    return lambda spec: adapter_cls()


def resolve_proposer(name: str, adapter: str = "raw_api", model_id: str = "claude-sonnet-5"):
    """``random`` | ``scripted:<file.json>`` | ``llm`` | ``harness:<dir>``."""
    if name == "random":
        return RandomSearchProposer()
    if name.startswith("scripted:"):
        items = json.loads(Path(name[len("scripted:"):]).read_text())
        return ScriptedProposer([Proposal(dict(i["edits"]), str(i.get("rationale", ""))) for i in items])
    if name == "llm":
        return HarnessProposer.default(resolve_adapter_factory(adapter)(None), model_id=model_id)
    if name.startswith("harness:"):
        spec = load_harness(name[len("harness:"):]).spec
        return HarnessProposer(spec, resolve_adapter_factory(adapter)(spec))
    raise ValueError(f"unknown proposer '{name}'; use random, scripted:<file.json>, llm, or harness:<dir>")
