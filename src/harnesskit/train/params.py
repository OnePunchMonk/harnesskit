"""Trainable harness parameters: which components an optimizer may edit.

A harness declares its trainable components under ``trainable:`` in
``harness.yaml``. Each entry is the harness analogue of a tensor with
``requires_grad=True``; every component not declared (tools, permissions,
guardrails, termination, eval cases, ...) is frozen and a candidate that
changes it is rejected (see ``HarnessModule.materialize``).

Supported ``target`` forms:

- ``scaffold.system_prompt`` (text): the system prompt, written back to its
  prompt file (or inline in ``harness.yaml`` when not file-backed).
- ``loop.max_turns``, ``loop.max_tool_calls``, ``context.budget_tokens``
  (int), ``model.temperature`` (float), ``model.model_id`` (choice): a field
  in ``harness.yaml``.
- ``file:<relative path>`` (text): the whole contents of a file inside the
  harness directory — a skill, a prompt fragment, or a code component.
- ``json:<relative path>#<JSON pointer>``: one value inside a JSON file,
  e.g. ``json:agent_config.json#/answer_mode``. For a custom tool's schema
  file only ``.../description`` pointers are allowed, so a tool's
  description can be trained while its name and input schema stay fixed.

Never trainable: ``harness.yaml`` as a file, the eval ``cases_file``,
anything under ``eval/`` or ``.harness/``, whole tool schema files, and any
path resolving outside the harness directory.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harnesskit.format.spec import HarnessSpec, ParameterKind, TrainableParam

SPEC_FIELD_TARGETS: dict[str, ParameterKind] = {
    "scaffold.system_prompt": ParameterKind.text,
    "loop.max_turns": ParameterKind.int,
    "loop.max_tool_calls": ParameterKind.int,
    "context.budget_tokens": ParameterKind.int,
    "model.temperature": ParameterKind.float,
    "model.model_id": ParameterKind.choice,
}

PROTECTED_DIRS = ("eval", ".harness")


class ParameterError(ValueError):
    """A trainable declaration or a proposed value is invalid."""


@dataclass(frozen=True)
class Target:
    kind: str  # "spec" | "file" | "json"
    path: str  # dotted spec path, or a path relative to the harness directory
    pointer: str = ""  # JSON pointer for kind == "json"


def parse_target(target: str) -> Target:
    if target.startswith("file:"):
        return Target("file", target[len("file:"):])
    if target.startswith("json:"):
        path, sep, pointer = target[len("json:"):].partition("#")
        if not sep or not pointer.startswith("/"):
            raise ParameterError(f"json target '{target}' needs a JSON pointer, e.g. json:config.json#/key")
        return Target("json", path, pointer)
    if target in SPEC_FIELD_TARGETS:
        return Target("spec", target)
    raise ParameterError(
        f"unsupported trainable target '{target}'; use one of {', '.join(SPEC_FIELD_TARGETS)}, "
        "file:<path> or json:<path>#<pointer>"
    )


def _pointer_tokens(pointer: str) -> list[str]:
    return [t.replace("~1", "/").replace("~0", "~") for t in pointer.split("/")[1:]]


def _walk(doc: Any, tokens: list[str]) -> Any:
    node = doc
    for token in tokens:
        if isinstance(node, list):
            node = node[int(token)]
        elif isinstance(node, dict):
            node = node[token]
        else:
            raise KeyError(token)
    return node


def json_pointer_get(doc: Any, pointer: str) -> Any:
    return _walk(doc, _pointer_tokens(pointer))


def json_pointer_set(doc: Any, pointer: str, value: Any) -> None:
    """Replace an existing value; never creates new keys."""
    tokens = _pointer_tokens(pointer)
    parent = _walk(doc, tokens[:-1])
    last = tokens[-1]
    if isinstance(parent, list):
        parent[int(last)] = value
    elif isinstance(parent, dict):
        if last not in parent:
            raise KeyError(pointer)
        parent[last] = value
    else:
        raise KeyError(pointer)


def _protected_files(spec: HarnessSpec) -> set[str]:
    protected = {"harness.yaml"}
    if spec.eval.cases_file:
        protected.add(Path(spec.eval.cases_file).as_posix())
    return protected


def _tool_schema_files(spec: HarnessSpec) -> set[str]:
    return {Path(t.ref).as_posix() for t in spec.tools if t.source.value == "custom"}


def resolve_in_dir(directory: Path, relpath: str) -> Path:
    resolved = (directory / relpath).resolve()
    root = directory.resolve()
    if resolved != root and root not in resolved.parents:
        raise ParameterError(f"trainable path '{relpath}' resolves outside the harness directory")
    return resolved


def check_declaration(decl: TrainableParam, spec: HarnessSpec, directory: Path) -> Target:
    """Validate one ``trainable:`` entry against the harness it belongs to."""
    target = parse_target(decl.target)
    if target.kind == "spec":
        expected = SPEC_FIELD_TARGETS[target.path]
        if decl.kind != expected:
            raise ParameterError(f"parameter '{decl.name}': target {target.path} has kind '{expected.value}', not '{decl.kind.value}'")
    else:
        rel = Path(target.path).as_posix()
        path = resolve_in_dir(directory, target.path)
        if rel in _protected_files(spec) or rel.split("/")[0] in PROTECTED_DIRS:
            raise ParameterError(f"parameter '{decl.name}': '{rel}' is frozen and can never be trainable")
        if not path.is_file():
            raise ParameterError(f"parameter '{decl.name}': file not found: {rel}")
        if target.kind == "file":
            if decl.kind != ParameterKind.text:
                raise ParameterError(f"parameter '{decl.name}': file targets must have kind 'text'")
            if rel in _tool_schema_files(spec):
                raise ParameterError(
                    f"parameter '{decl.name}': a tool schema file cannot be trained whole; "
                    f"use json:{rel}#/description so the tool's name and input schema stay fixed"
                )
        else:
            if rel in _tool_schema_files(spec) and not target.pointer.endswith("/description"):
                raise ParameterError(
                    f"parameter '{decl.name}': only description fields of tool schema '{rel}' may be trained"
                )
            try:
                json_pointer_get(json.loads(path.read_text()), target.pointer)
            except (KeyError, IndexError, ValueError) as e:
                raise ParameterError(f"parameter '{decl.name}': pointer {target.pointer} not found in {rel}") from e
    if decl.kind == ParameterKind.choice and not decl.choices:
        raise ParameterError(f"parameter '{decl.name}': kind 'choice' requires a non-empty 'choices' list")
    if decl.min is not None and decl.max is not None and decl.min > decl.max:
        raise ParameterError(f"parameter '{decl.name}': min > max")
    return target


def check_value(decl: TrainableParam, value: Any) -> None:
    """Raise ParameterError if ``value`` violates the declaration's kind or bounds."""
    kind = decl.kind
    if kind == ParameterKind.text:
        if not isinstance(value, str):
            raise ParameterError(f"'{decl.name}' expects text, got {type(value).__name__}")
        if decl.max_chars is not None and len(value) > decl.max_chars:
            raise ParameterError(f"'{decl.name}' exceeds max_chars={decl.max_chars} ({len(value)} chars)")
        return
    if kind == ParameterKind.choice:
        # Compare types too, so True never matches a choice of 1.
        if not any(value == c and type(value) is type(c) for c in decl.choices or []):
            raise ParameterError(f"'{decl.name}' must be one of {decl.choices}, got {value!r}")
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParameterError(f"'{decl.name}' expects a number, got {value!r}")
    if kind == ParameterKind.int and not float(value).is_integer():
        raise ParameterError(f"'{decl.name}' expects an integer, got {value!r}")
    if decl.min is not None and value < decl.min:
        raise ParameterError(f"'{decl.name}'={value} is below min={decl.min}")
    if decl.max is not None and value > decl.max:
        raise ParameterError(f"'{decl.name}'={value} is above max={decl.max}")


@dataclass
class Parameter:
    """A declared trainable component together with its current value."""

    decl: TrainableParam
    target: Target
    value: Any

    @property
    def name(self) -> str:
        return self.decl.name

    @property
    def requires_grad(self) -> bool:
        return self.decl.requires_grad

    def describe(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "target": self.decl.target,
            "kind": self.decl.kind.value,
            "requires_grad": self.requires_grad,
            "value": self.value,
        }
        for key in ("description", "choices", "min", "max", "max_chars"):
            if getattr(self.decl, key) is not None:
                d[key] = getattr(self.decl, key)
        return d
