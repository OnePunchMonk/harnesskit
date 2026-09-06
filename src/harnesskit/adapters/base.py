"""Adapter interface (design doc §7.1) — the translation layer between a
framework-agnostic HarnessSpec and a runnable agent in some target runtime.

Deliberately the thinnest module in the toolkit: it composes with existing
frameworks (Pydantic AI, LangGraph, raw provider APIs), it never implements
loop strategies or tool dispatch itself — that would be scope creep into
framework territory (design doc §13).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from harnesskit.format.spec import HarnessSpec
from harnesskit.trace.schema import Trajectory


@dataclass
class AdapterCapabilities:
    loop_strategies: set[str]
    tool_sources: set[str]
    hook_points: set[str]
    memory_backends: set[str]


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


def check_support(spec: HarnessSpec, adapter: HarnessAdapter) -> list[str]:
    """Returns human-readable warnings for spec features the adapter can't express."""
    caps = adapter.supports()
    warnings = []
    if spec.loop.type.value not in caps.loop_strategies:
        warnings.append(
            f"loop.type='{spec.loop.type.value}' is not supported by this adapter "
            f"(supports: {', '.join(sorted(caps.loop_strategies))})"
        )
    for tool in spec.tools:
        if tool.source.value not in caps.tool_sources:
            warnings.append(f"tool '{tool.name}' has source='{tool.source.value}', unsupported by this adapter")
    return warnings
