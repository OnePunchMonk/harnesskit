"""HarnessModule: a harness directory viewed as a model with named parameters.

The analogy to ``torch.nn.Module`` is deliberate but limited:
``named_parameters()`` lists the declared trainable components,
``state_dict()`` reads their current values, and ``materialize(state, dir)``
writes a new candidate harness directory with those values applied. There
are no numeric gradients: an optimizer proposes new values from recorded
evidence (see ``harnesskit.train.proposers``), and every candidate is a real
harness directory that can be linted, run by any adapter, diffed, and
exported like any other.

``materialize`` is the enforcement point for "frozen means frozen": after
writing a candidate it reloads it and rejects the candidate if any part of
the spec or any file outside the declared trainable targets changed.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from harnesskit.format.spec import HarnessSpec, ParameterKind
from harnesskit.parser import HarnessLoadError, load_harness
from harnesskit.train.params import (
    Parameter,
    ParameterError,
    check_declaration,
    check_value,
    json_pointer_get,
    json_pointer_set,
    resolve_in_dir,
)

_COPY_IGNORE = shutil.ignore_patterns(".harness", "__pycache__", "*.pyc")


def _read_raw(directory: Path) -> dict:
    return yaml.safe_load((directory / "harness.yaml").read_text()) or {}


def _prompt_file(raw: dict) -> str | None:
    scaffold = raw.get("scaffold", {})
    if scaffold.get("system_prompt_is_file", True) and "system_prompt" in scaffold:
        return Path(scaffold["system_prompt"]).as_posix()
    return None


def _file_digests(directory: Path) -> dict[str, str]:
    digests = {}
    for path in sorted(directory.rglob("*")):
        rel = path.relative_to(directory)
        if not path.is_file() or rel.parts[0] in (".harness",) or "__pycache__" in rel.parts or path.suffix == ".pyc":
            continue
        digests[rel.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def _frozen_spec_view(spec: HarnessSpec, trainable_spec_paths: set[str]) -> dict:
    data = spec.model_dump(mode="json", exclude={"source_dir"})
    for dotted in trainable_spec_paths:
        section, key = dotted.split(".")
        data.get(section, {}).pop(key, None)
    return data


class CandidateRejected(ParameterError):
    """A proposed state is invalid or would change a frozen component."""


@dataclass
class ParamChange:
    name: str
    old: Any
    new: Any

    def render(self) -> str:
        if isinstance(self.old, str) and isinstance(self.new, str):
            lines = difflib.unified_diff(
                self.old.splitlines(),
                self.new.splitlines(),
                fromfile=f"a/{self.name}",
                tofile=f"b/{self.name}",
                lineterm="",
            )
            return "\n".join(lines) + "\n"
        return f"--- a/{self.name}\n+++ b/{self.name}\n-{json.dumps(self.old)}\n+{json.dumps(self.new)}\n"


class HarnessModule:
    """A loaded harness directory plus its trainable parameters."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.spec: HarnessSpec = load_harness(self.directory).spec
        self._raw = _read_raw(self.directory)
        self._parameters: list[Parameter] = []
        seen: set[str] = set()
        for decl in self.spec.trainable:
            if decl.name in seen:
                raise ParameterError(f"duplicate trainable parameter name '{decl.name}'")
            seen.add(decl.name)
            target = check_declaration(decl, self.spec, self.directory)
            self._parameters.append(Parameter(decl=decl, target=target, value=self._read(target)))
        targets = [p.decl.target for p in self._parameters]
        if len(set(targets)) != len(targets):
            raise ParameterError("two trainable parameters point at the same target")

    # -- the nn.Module-like surface ---------------------------------------
    def parameters(self) -> list[Parameter]:
        return list(self._parameters)

    def named_parameters(self, trainable_only: bool = True) -> list[tuple[str, Parameter]]:
        return [(p.name, p) for p in self._parameters if p.requires_grad or not trainable_only]

    def state_dict(self) -> dict[str, Any]:
        return {p.name: p.value for p in self._parameters}

    def parameter(self, name: str) -> Parameter:
        for p in self._parameters:
            if p.name == name:
                return p
        raise KeyError(name)

    # -- reading / writing values ----------------------------------------
    def _read(self, target) -> Any:
        if target.kind == "spec":
            section, key = target.path.split(".")
            return getattr(getattr(self.spec, section), key)
        path = resolve_in_dir(self.directory, target.path)
        if target.kind == "file":
            return path.read_text()
        return json_pointer_get(json.loads(path.read_text()), target.pointer)

    def _write(self, directory: Path, raw: dict, param: Parameter, value: Any) -> None:
        target = param.target
        if target.kind == "spec":
            section, key = target.path.split(".")
            prompt_file = _prompt_file(raw)
            if target.path == "scaffold.system_prompt" and prompt_file is not None:
                resolve_in_dir(directory, prompt_file).write_text(value)
            else:
                raw.setdefault(section, {})[key] = value
            return
        path = resolve_in_dir(directory, target.path)
        if target.kind == "file":
            path.write_text(value)
            return
        doc = json.loads(path.read_text())
        json_pointer_set(doc, target.pointer, value)
        path.write_text(json.dumps(doc, indent=2) + "\n")

    def resolve_edits(self, edits: dict[str, Any]) -> dict[str, Any]:
        """Turn incremental text operations into full values.

        A text parameter's edit may be a full replacement string or an
        operation (ACE-style incremental update, which avoids rewriting and
        eroding a long prompt): ``{"op": "append", "text": ...}``,
        ``{"op": "prepend", "text": ...}``, or
        ``{"op": "replace", "old": ..., "new": ...}`` where ``old`` must occur
        exactly once.
        """
        resolved = dict(edits)
        current = self.state_dict()
        for name, edit in edits.items():
            if not isinstance(edit, dict) or name not in current:
                continue
            param = self.parameter(name)
            if param.decl.kind != ParameterKind.text:
                raise CandidateRejected(f"'{name}': text operations only apply to text parameters")
            value, op = current[name], edit.get("op")
            if op in ("append", "prepend") and isinstance(edit.get("text"), str):
                sep = "" if not value or value.endswith("\n") else "\n"
                resolved[name] = value + sep + edit["text"] if op == "append" else edit["text"] + "\n" + value
            elif op == "replace" and isinstance(edit.get("old"), str) and isinstance(edit.get("new"), str):
                count = value.count(edit["old"]) if edit["old"] else 0
                if count != 1:
                    raise CandidateRejected(f"'{name}': replace target must occur exactly once (found {count})")
                resolved[name] = value.replace(edit["old"], edit["new"])
            else:
                raise CandidateRejected(f"'{name}': unsupported text operation {edit!r}")
        return resolved

    def validate_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """Return the full candidate state, or raise CandidateRejected."""
        current = self.state_dict()
        unknown = sorted(set(state) - set(current))
        if unknown:
            raise CandidateRejected(f"unknown parameter(s): {', '.join(unknown)}")
        merged = {**current, **state}
        for p in self._parameters:
            new = merged[p.name]
            if new == current[p.name] and type(new) is type(current[p.name]):
                continue
            if not p.requires_grad:
                raise CandidateRejected(f"parameter '{p.name}' is frozen (requires_grad: false)")
            try:
                check_value(p.decl, new)
            except ParameterError as e:
                raise CandidateRejected(str(e)) from e
            if p.decl.kind == ParameterKind.int:
                merged[p.name] = int(new)
        return merged

    def diff(self, state: dict[str, Any]) -> list[ParamChange]:
        current = self.state_dict()
        return [
            ParamChange(name, current[name], value)
            for name, value in state.items()
            if name in current and value != current[name]
        ]

    def materialize(self, state: dict[str, Any], out_dir: str | Path) -> HarnessModule:
        """Write a candidate harness with ``state`` applied to ``out_dir``.

        Raises CandidateRejected (and removes ``out_dir``) when the state is
        invalid or the written candidate differs from this harness anywhere
        outside the declared trainable targets.
        """
        merged = self.validate_state(state)
        out_dir = Path(out_dir)
        if out_dir.exists():
            raise FileExistsError(f"candidate directory already exists: {out_dir}")
        shutil.copytree(self.directory, out_dir, ignore=_COPY_IGNORE)
        try:
            raw = _read_raw(out_dir)
            changed_spec = False
            for p in self._parameters:
                if merged[p.name] == p.value and type(merged[p.name]) is type(p.value):
                    continue
                self._write(out_dir, raw, p, merged[p.name])
                changed_spec |= p.target.kind == "spec" and not (
                    p.target.path == "scaffold.system_prompt" and _prompt_file(raw) is not None
                )
            if changed_spec:
                (out_dir / "harness.yaml").write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
            candidate = HarnessModule(out_dir)
            self._check_frozen(candidate)
        except (ParameterError, HarnessLoadError, OSError, ValueError) as e:
            shutil.rmtree(out_dir, ignore_errors=True)
            if isinstance(e, CandidateRejected):
                raise
            raise CandidateRejected(f"candidate failed to materialize: {e}") from e
        return candidate

    def _check_frozen(self, candidate: HarnessModule) -> None:
        live = [p for p in self._parameters if p.requires_grad]
        spec_paths = {p.target.path for p in live if p.target.kind == "spec"}
        allowed_files = {p.target.path for p in live if p.target.kind in ("file", "json")}
        prompt_file = _prompt_file(self._raw)
        if prompt_file is not None:
            if "scaffold.system_prompt" in spec_paths:
                allowed_files.add(prompt_file)
            if prompt_file in allowed_files:
                spec_paths.add("scaffold.system_prompt")
        if spec_paths - {"scaffold.system_prompt"} or (prompt_file is None and "scaffold.system_prompt" in spec_paths):
            allowed_files.add("harness.yaml")
        allowed_files = {Path(f).as_posix() for f in allowed_files}

        if _frozen_spec_view(self.spec, spec_paths) != _frozen_spec_view(candidate.spec, spec_paths):
            raise CandidateRejected("candidate changed a frozen part of the harness spec")
        before, after = _file_digests(self.directory), _file_digests(candidate.directory)
        changed = {f for f in before.keys() | after.keys() if before.get(f) != after.get(f)}
        illegal = sorted(changed - allowed_files)
        if illegal:
            raise CandidateRejected(f"candidate changed frozen file(s): {', '.join(illegal)}")
