"""A scripted adapter for tests and cheap dev-loop iteration (design doc §5.3
'mocked tools'): no LLM calls, no side effects — just replays a fixed
Trajectory per case id so scorer logic and CLI plumbing can be exercised
without a network call or an API key.
"""
from __future__ import annotations

from dataclasses import dataclass

from harnesskit.adapters.base import AdapterCapabilities, RunnableAgent
from harnesskit.format.spec import HarnessSpec
from harnesskit.trace.schema import Trajectory


@dataclass
class MockAdapter:
    trajectories_by_input: dict[str, Trajectory]

    def supports(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            loop_strategies={"react", "plan_execute", "rewoo", "code_action", "custom"},
            tool_sources={"builtin", "mcp", "custom"},
            hook_points={"pre_tool", "post_tool", "pre_turn", "post_turn", "on_error"},
            memory_backends={"none", "in_memory", "persistent"},
        )

    def build(self, spec: HarnessSpec) -> RunnableAgent:
        return RunnableAgent(spec=spec, handle=None)

    def run(self, agent: RunnableAgent, input: str) -> Trajectory:
        if input not in self.trajectories_by_input:
            raise KeyError(f"MockAdapter has no scripted trajectory for input: {input!r}")
        return self.trajectories_by_input[input]
