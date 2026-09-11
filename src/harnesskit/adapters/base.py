"""Adapter interface (design doc §7.1) — the translation layer between a
framework-agnostic HarnessSpec and a runnable agent in some target runtime.

Deliberately the thinnest module in the toolkit: it composes with existing
frameworks (Pydantic AI, LangGraph, raw provider APIs), it never implements
loop strategies or tool dispatch itself — that would be scope creep into
framework territory (design doc §13).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Protocol

from harnesskit.format.spec import HarnessSpec
from harnesskit.trace.schema import Trajectory


@dataclass
class AdapterCapabilities:
    loop_strategies: set[str]
    tool_sources: set[str]
    hook_points: set[str]
    memory_backends: set[str]
    runtime: str = "unknown"


class SupportStatus(str, Enum):
    supported = "supported"
    unsupported = "unsupported"


@dataclass(frozen=True)
class SupportFinding:
    field: str
    requested: str
    status: SupportStatus
    reason: str
    runtime: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class RunnableAgent:
    spec: HarnessSpec
    handle: Any  # adapter-specific: a client, a graph, whatever the framework needs


class HarnessAdapter(Protocol):
    def build(self, spec: HarnessSpec) -> RunnableAgent:
        """Convert a harness spec into a runnable agent."""
        ...

    def run(self, agent: RunnableAgent, input: str) -> Trajectory:
        """Execute the agent on an input, returning a full trajectory."""
        ...

    def supports(self) -> AdapterCapabilities:
        """Declare which harness features this adapter supports."""
        ...


def inspect_support(spec: HarnessSpec, adapter: HarnessAdapter) -> list[SupportFinding]:
    """Return adapter-specific findings without initializing a provider or tool."""
    caps = adapter.supports()
    findings = []

    def add(field: str, requested: str, supported: bool, reason: str) -> None:
        findings.append(
            SupportFinding(
                field=field,
                requested=requested,
                status=SupportStatus.supported if supported else SupportStatus.unsupported,
                reason=reason,
                runtime=caps.runtime,
            )
        )

    if spec.loop.type.value not in caps.loop_strategies:
        add(
            "loop.type",
            spec.loop.type.value,
            False,
            f"supports: {', '.join(sorted(caps.loop_strategies))}",
        )
    for tool in spec.tools:
        if tool.source.value not in caps.tool_sources:
            add(f"tools.{tool.name}.source", tool.source.value, False, "tool source is not implemented")
    if spec.memory.session not in caps.memory_backends:
        add(
            "memory.session",
            spec.memory.session,
            False,
            f"supports: {', '.join(sorted(caps.memory_backends))}",
        )
    for hook in spec.hooks:
        if hook.point.value not in caps.hook_points:
            add(f"hooks.{hook.point.value}", hook.point.value, False, "hook is ignored at runtime")
    return findings


def check_support(spec: HarnessSpec, adapter: HarnessAdapter) -> list[str]:
    """Return compatible human-readable warnings for unsupported findings."""
    return [
        f"{finding.field}='{finding.requested}' is unsupported by {finding.runtime}: {finding.reason}"
        for finding in inspect_support(spec, adapter)
        if finding.status is SupportStatus.unsupported
    ]
