"""Adapter interface (design doc §7.1) — the translation layer between a
framework-agnostic HarnessSpec and a runnable agent in some target runtime.

Deliberately the thinnest module in the toolkit: it composes with existing
frameworks (Pydantic AI, LangGraph, raw provider APIs), it never implements
loop strategies or tool dispatch itself — that would be scope creep into
framework territory (design doc §13).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    # Issue #3 item 1 ("visibility half"): the format declares guardrails,
    # compaction, model routing, max_tool_calls and five termination types,
    # but historically only RawAPIAdapter.run() honoured max_turns,
    # tag_emitted and explicit_tool — silently. These fields let an adapter
    # state HONESTLY which of those it actually enforces at runtime, so
    # `inspect_support`/`check_support` can surface the rest as visible gaps
    # instead of a harness looking "clean" while half its declared safety
    # net does nothing.
    enforced_guardrails: set[str] = field(default_factory=set)
    enforced_termination_types: set[str] = field(default_factory=set)
    enforced_compaction_strategies: set[str] = field(default_factory=set)
    supports_routing: bool = False
    supports_max_tool_calls: bool = False


class SupportStatus(str, Enum):
    supported = "supported"
    unsupported = "unsupported"


class FeatureStatus(str, Enum):
    """Finer-grained than `SupportStatus`: whether a feature is actually
    enforced by harnesskit's own execution (`controlled`), merely visible in
    the trace without harnesskit being able to enforce it (`observed`), or
    not present at all for this adapter (`unavailable`).

    Introduced for adapters that wrap externally-run agents (e.g.
    `CallbackAdapter`): such an adapter can *observe* wall-clock duration and
    a returned output, but cannot *control* the wrapped agent's internal
    timeouts or spend limits — those are `unavailable` through this path,
    never silently reported as if harnesskit enforces them.
    """

    controlled = "controlled"
    observed = "observed"
    unavailable = "unavailable"


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

    # Issue #3 item 1: declared-but-unenforced gaps. These are a different
    # flavor from the above — the feature loads and lints fine, it just
    # silently doesn't run — so the reason text says so explicitly rather
    # than "not implemented"/"is ignored".
    def unenforced(field_name: str, requested: str) -> None:
        add(
            field_name,
            requested,
            False,
            f"declared but not enforced at runtime by adapter '{caps.runtime}'",
        )

    for guardrail in spec.guardrails:
        if guardrail.name not in caps.enforced_guardrails:
            unenforced(f"guardrails.{guardrail.name}", guardrail.check)
    for term in spec.termination:
        if term.type not in caps.enforced_termination_types:
            unenforced("termination", term.type)
    if spec.context.compaction.value != "none" and spec.context.compaction.value not in caps.enforced_compaction_strategies:
        unenforced("context.compaction", spec.context.compaction.value)
    if spec.model.routing and not caps.supports_routing:
        unenforced("model.routing", ", ".join(sorted(str(k) for k in spec.model.routing)))
    if spec.loop.max_tool_calls is not None and not caps.supports_max_tool_calls:
        unenforced("loop.max_tool_calls", str(spec.loop.max_tool_calls))

    return findings


def check_support(spec: HarnessSpec, adapter: HarnessAdapter) -> list[str]:
    """Return compatible human-readable warnings for unsupported findings."""
    return [
        f"{finding.field}='{finding.requested}' is unsupported by {finding.runtime}: {finding.reason}"
        for finding in inspect_support(spec, adapter)
        if finding.status is SupportStatus.unsupported
    ]
