"""Raw-API adapter (design doc §7.5) — no framework, just direct calls to the
model provider's API. This is the simplest adapter and the one that proves
the harness format isn't framework-dependent: if a harness runs here, it
isn't secretly relying on a framework feature the format doesn't express.

Tool execution convention for source="custom": a tool declared at
`tools/<name>.json` (its JSON-Schema input_schema) may have a sibling
`tools/<name>.py` exposing `def run(**kwargs) -> str`. Tools with no sibling
callback are declared to the model but any call to them returns an error
string — useful for linting/dry-run without wiring real side effects yet.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from harnesskit.adapters._tool_loading import load_tool_callback
from harnesskit.adapters.base import AdapterCapabilities, RunnableAgent
from harnesskit.format.spec import HarnessSpec, TerminationCondition
from harnesskit.trace.schema import Step, StepType, Trajectory, estimate_cost_usd


class MissingAPIKeyError(Exception):
    pass


class RawAPIAdapter:
    """Implements HarnessAdapter for loop.type == 'react' against Anthropic's API."""

    def supports(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            loop_strategies={"react"},
            tool_sources={"custom"},
            hook_points=set(),
            memory_backends={"in_memory", "none"},
        )

    def build(self, spec: HarnessSpec) -> RunnableAgent:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("RawAPIAdapter requires the 'anthropic' package: pip install anthropic") from e

        assert spec.source_dir is not None, "spec must be loaded via harnesskit.parser.load_harness"
        client = anthropic.Anthropic()

        tool_schemas = []
        callbacks = {}
        for tool in spec.tools:
            schema_path = spec.source_dir / tool.ref
            schema = json.loads(schema_path.read_text())
            tool_schemas.append(schema)
            callbacks[tool.name] = load_tool_callback(spec.source_dir, tool.ref)

        return RunnableAgent(spec=spec, handle={"client": client, "tool_schemas": tool_schemas, "callbacks": callbacks})

    def run(self, agent: RunnableAgent, input: str) -> Trajectory:
        spec = agent.spec
        client = agent.handle["client"]
        tool_schemas = agent.handle["tool_schemas"]
        callbacks = agent.handle["callbacks"]

        system_prompt = spec.scaffold.system_prompt.replace("{{task}}", input)
        messages: list[dict] = [{"role": "user", "content": input}]
        trajectory = Trajectory(
            harness_name=spec.metadata.name,
            harness_version=spec.metadata.version,
            model_id=spec.model.model_id,
            adapter="raw_api",
            input=input,
        )

        stop_tags = [t.value for t in spec.termination if t.type == "tag_emitted"]
        explicit_tools = [t.value for t in spec.termination if t.type == "explicit_tool"]
        max_turns = spec.loop.max_turns
        last_text_blocks: list[str] = []

        for turn in range(max_turns):
            t0 = time.time()
            create_kwargs = dict(
                model=spec.model.model_id,
                max_tokens=1024,
                system=system_prompt,
                messages=messages,
                tools=tool_schemas or None,
            )
            try:
                response = client.messages.create(temperature=spec.model.temperature, **create_kwargs)
            except TypeError:
                # some SDK versions have dropped `temperature` from create() entirely
                response = client.messages.create(**create_kwargs)
            duration_ms = int((time.time() - t0) * 1000)
            cost = estimate_cost_usd(spec.model.model_id, response.usage.input_tokens, response.usage.output_tokens)

            text_blocks = [b.text for b in response.content if b.type == "text"]
            last_text_blocks = text_blocks
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

            trajectory.steps.append(
                Step(
                    step_type=StepType.llm_call,
                    input=json.dumps(messages[-1]),
                    output="\n".join(text_blocks),
                    tokens_in=response.usage.input_tokens,
                    tokens_out=response.usage.output_tokens,
                    cost_usd=cost,
                    duration_ms=duration_ms,
                )
            )
            messages.append({"role": "assistant", "content": response.content})

            full_text = "\n".join(text_blocks)
            for tag in stop_tags:
                if f"<{tag}>" in full_text:
                    trajectory.final_output = full_text
                    trajectory.stopped_reason = f"tag_emitted:{tag}"
                    return trajectory

            if not tool_use_blocks:
                trajectory.final_output = full_text
                trajectory.stopped_reason = "no_tool_use" if not explicit_tools else "idle"
                return trajectory

            tool_results = []
            for block in tool_use_blocks:
                if block.name in explicit_tools:
                    trajectory.final_output = json.dumps(block.input)
                    trajectory.stopped_reason = f"explicit_tool:{block.name}"
                    return trajectory

                callback = callbacks.get(block.name)
                t_tool = time.time()
                if callback is None:
                    result_text = f"error: tool '{block.name}' has no callback wired up"
                else:
                    try:
                        result_text = str(callback(**block.input))
                    except Exception as e:  # noqa: BLE001 — surfaced to the model, not raised
                        result_text = f"error: {e}"
                tool_duration_ms = int((time.time() - t_tool) * 1000)

                trajectory.steps.append(
                    Step(
                        step_type=StepType.tool_call,
                        tool_name=block.name,
                        tool_args=block.input,
                        tool_result=result_text,
                        duration_ms=tool_duration_ms,
                    )
                )
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result_text})

            messages.append({"role": "user", "content": tool_results})

        trajectory.stopped_reason = "max_turns"
        trajectory.final_output = "\n".join(last_text_blocks) or None
        return trajectory
