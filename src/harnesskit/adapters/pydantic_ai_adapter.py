"""Pydantic AI v2 adapter (design doc §7.3) — the first "real framework"
adapter, built second on purpose (raw_api came first): this is the one that
exposes which parts of the format were accidentally raw-API-shaped instead
of genuinely framework-agnostic.

Feature mapping, and where it's honest about not mapping cleanly:
  - loop.type == "react" -> Pydantic AI's own agent loop (it always runs a
    ReAct-shaped tool loop; there's no separate "plan_execute" mode to map
    to, so `supports()` only declares "react").
  - loop.max_turns -> UsageLimits(request_limit=...), the closest analog to
    a turn cap that Pydantic AI exposes.
  - tools (source="custom") -> pydantic_ai.Tool wrapping the same
    tools/<name>.py callback convention the raw-API adapter uses, so a
    harness's tools work unmodified across both adapters.
  - termination.explicit_tool / tag_emitted -> NOT enforced mid-run here.
    Pydantic AI's loop decides for itself when to stop calling tools and
    return; there's no hook to cut it off early on a specific tool call or
    tag the way the raw-API adapter's hand-rolled loop can. This is exactly
    the kind of format/adapter mismatch check_support() exists to surface.
  - memory.session == "persistent" -> NOT supported; declared honestly in
    supports() rather than silently ignored.
"""
from __future__ import annotations

import time

from harnesskit.adapters._tool_loading import load_tool_callback
from harnesskit.adapters.base import AdapterCapabilities, RunnableAgent
from harnesskit.format.spec import HarnessSpec
from harnesskit.trace.schema import CostStatus, Step, StepType, Trajectory, estimate_cost_usd


class PydanticAIAdapter:
    def supports(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            loop_strategies={"react"},
            tool_sources={"custom"},
            hook_points=set(),
            memory_backends={"none", "in_memory"},
            runtime="pydantic_ai",
            # run() below only enforces max_turns, via
            # UsageLimits(request_limit=...). explicit_tool/tag_emitted are
            # NOT enforced mid-run (see module docstring) — Pydantic AI's own
            # loop decides when to stop. No guardrail, no compaction
            # strategy, no model.routing and no loop.max_tool_calls are
            # enforced either.
            enforced_termination_types={"max_turns"},
            enforced_guardrails=set(),
            enforced_compaction_strategies=set(),
            supports_routing=False,
            supports_max_tool_calls=False,
        )

    def build(self, spec: HarnessSpec) -> RunnableAgent:
        try:
            from pydantic_ai import Agent, Tool
        except ImportError as e:
            raise ImportError("PydanticAIAdapter requires the 'pydantic-ai' package: pip install pydantic-ai") from e

        assert spec.source_dir is not None, "spec must be loaded via harnesskit.parser.load_harness"

        import json

        tools = []
        for tool_def in spec.tools:
            callback = load_tool_callback(spec.source_dir, tool_def.ref)
            if callback is None:
                continue  # declared but not wired to a callback — nothing to hand pydantic-ai
            schema_path = spec.source_dir / tool_def.ref
            description = json.loads(schema_path.read_text()).get("description")
            tools.append(Tool(function=callback, name=tool_def.name, description=description))

        agent = Agent(
            f"anthropic:{spec.model.model_id}",
            system_prompt=spec.scaffold.system_prompt,
            tools=tools,
        )
        return RunnableAgent(spec=spec, handle=agent)

    def run(self, agent: RunnableAgent, input: str) -> Trajectory:
        from pydantic_ai.exceptions import UsageLimitExceeded
        from pydantic_ai.usage import UsageLimits

        spec = agent.spec

        trajectory = Trajectory(
            harness_name=spec.metadata.name,
            harness_version=spec.metadata.version,
            model_id=spec.model.model_id,
            adapter="pydantic_ai",
            input=input,
        )

        t0 = time.time()
        try:
            result = agent.handle.run_sync(input, usage_limits=UsageLimits(request_limit=spec.loop.max_turns))
        except UsageLimitExceeded:
            # Pydantic AI raises rather than returning a partial result when the
            # request cap is hit; there's no run() hook to intervene mid-loop the
            # way the raw-API adapter's hand-rolled loop can, so no per-turn
            # steps survive to attach here — see the module docstring.
            trajectory.stopped_reason = "max_turns"
            return trajectory
        total_duration_ms = int((time.time() - t0) * 1000)

        messages = result.all_messages()
        for i, message in enumerate(messages):
            if message.kind == "response":
                usage = message.usage
                observed_cost = getattr(usage, "cost", None)
                if observed_cost is not None:
                    cost, cost_status = float(observed_cost), CostStatus.observed
                else:
                    cost, cost_status = estimate_cost_usd(
                        spec.model.model_id, usage.input_tokens, usage.output_tokens
                    )
                text = "\n".join(p.content for p in message.parts if p.part_kind == "text")
                trajectory.steps.append(
                    Step(
                        step_type=StepType.llm_call,
                        output=text,
                        tokens_in=usage.input_tokens,
                        tokens_out=usage.output_tokens,
                        cost_usd=cost,
                        cost_status=cost_status,
                        duration_ms=total_duration_ms // max(1, len([m for m in messages if m.kind == "response"])),
                    )
                )
                for part in message.parts:
                    if part.part_kind == "tool-call":
                        trajectory.steps.append(
                            Step(step_type=StepType.tool_call, tool_name=part.tool_name, tool_args=part.args)
                        )
            elif message.kind == "request":
                for part in message.parts:
                    if part.part_kind == "tool-return":
                        # attach the result to the most recent matching tool_call step
                        for step in reversed(trajectory.steps):
                            if step.step_type == StepType.tool_call and step.tool_name == part.tool_name and step.tool_result is None:
                                step.tool_result = str(part.content)
                                break

        trajectory.final_output = str(result.output)
        trajectory.stopped_reason = "agent_returned"
        return trajectory
