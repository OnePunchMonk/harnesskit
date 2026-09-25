"""Importable adapter conformance suite (issue #6).

`check_support` (see `harnesskit.adapters.base`) only compares declarations —
it never proves an adapter actually behaves the way it claims. This module
runs the same small set of deterministic scenarios against a real
`HarnessAdapter.run()` implementation, using scripted clients and fake tools
(see `harnesskit.testing.fakes`) so no API key or network access is needed.

Minimal usage for a third-party adapter:

    from harnesskit.testing.conformance import RAW_API_CASES, run_case

    def test_my_adapter_conformance():
        adapter = MyRawAPIStyleAdapter()
        for case in RAW_API_CASES:
            result = run_case(adapter, case)
            assert result.status == "pass", result.detail

A case's `.build()` returns a `RunnableAgent` wired to a scripted client, not
a real one — conformance tests the run loop's contract, not provider I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from harnesskit.adapters.base import HarnessAdapter, RunnableAgent
from harnesskit.format.spec import (
    HarnessSpec,
    LoopConfig,
    Metadata,
    ScaffoldConfig,
    TerminationCondition,
    ToolDef,
    ToolSource,
)
from harnesskit.testing.fakes import (
    FakeAnthropicClient,
    FakeResponse,
    echo_tool,
    raising_tool,
    text_block,
    tool_use_block,
)
from harnesskit.trace.schema import StepType, Trajectory

Status = Literal["pass", "fail", "error", "not_applicable"]


def make_spec(
    *,
    max_turns: int = 5,
    termination: list[TerminationCondition] | None = None,
    tools: list[ToolDef] | None = None,
) -> HarnessSpec:
    """A minimal in-memory HarnessSpec for conformance fixtures — no harness.yaml needed."""
    return HarnessSpec(
        metadata=Metadata(name="conformance-fixture"),
        loop=LoopConfig(max_turns=max_turns),
        tools=tools or [],
        termination=termination or [],
        scaffold=ScaffoldConfig(system_prompt="You are a conformance test agent.", system_prompt_is_file=False),
    )


@dataclass(frozen=True)
class ConformanceCase:
    id: str
    capability: str
    description: str
    build: Callable[[], tuple[RunnableAgent, str]]
    check: Callable[[Trajectory], None]


@dataclass
class ScenarioResult:
    case_id: str
    runtime: str
    status: Status
    detail: str = ""


def run_case(adapter: HarnessAdapter, case: ConformanceCase, *, runtime: str = "") -> ScenarioResult:
    runtime = runtime or getattr(adapter, "runtime", adapter.__class__.__name__)
    try:
        agent, input_text = case.build()
    except Exception as e:  # noqa: BLE001 — a build failure is itself a result, not a crash
        return ScenarioResult(case.id, runtime, "error", f"case setup failed: {e}")
    try:
        trajectory = adapter.run(agent, input_text)
    except Exception as e:  # noqa: BLE001
        return ScenarioResult(case.id, runtime, "error", f"adapter.run raised: {e}")
    try:
        case.check(trajectory)
    except AssertionError as e:
        return ScenarioResult(case.id, runtime, "fail", str(e))
    return ScenarioResult(case.id, runtime, "pass")


# ---------------------------------------------------------------------------
# raw_api-shaped cases: any adapter whose RunnableAgent.handle exposes an
# anthropic-Messages-API-shaped client under handle["client"] can reuse these
# by swapping in its own build_agent() equivalent.
# ---------------------------------------------------------------------------

_NOOP_TOOL = ToolDef(name="noop", source=ToolSource.custom, ref="tools/noop.json")
_FLAKY_TOOL = ToolDef(name="flaky", source=ToolSource.custom, ref="tools/flaky.json")


def _raw_api_agent(spec: HarnessSpec, script: list[FakeResponse], callbacks: dict) -> RunnableAgent:
    client = FakeAnthropicClient(script)
    tool_schemas = [{"name": t.name, "description": "", "input_schema": {"type": "object", "properties": {}}} for t in spec.tools]
    return RunnableAgent(spec=spec, handle={"client": client, "tool_schemas": tool_schemas, "callbacks": callbacks})


def _build_max_turns() -> tuple[RunnableAgent, str]:
    spec = make_spec(max_turns=3, tools=[_NOOP_TOOL])
    script = [FakeResponse(content=[tool_use_block("noop")]) for _ in range(3)]
    return _raw_api_agent(spec, script, {"noop": echo_tool()}), "go"


def _check_max_turns(t: Trajectory) -> None:
    assert t.stopped_reason == "max_turns", f"expected max_turns termination, got {t.stopped_reason!r}"
    assert t.turns == 3, f"expected exactly 3 llm_call steps, got {t.turns}"


def _build_explicit_tool_termination() -> tuple[RunnableAgent, str]:
    spec = make_spec(
        max_turns=5,
        tools=[ToolDef(name="submit", source=ToolSource.custom, ref="tools/submit.json")],
        termination=[TerminationCondition(type="explicit_tool", value="submit")],
    )
    script = [FakeResponse(content=[tool_use_block("submit", {"answer": "42"})])]
    return _raw_api_agent(spec, script, {}), "go"


def _check_explicit_tool_termination(t: Trajectory) -> None:
    assert t.stopped_reason == "explicit_tool:submit", f"expected explicit_tool termination, got {t.stopped_reason!r}"
    assert t.tool_calls == [], "the terminating tool call itself should not be recorded as an executed tool step"


def _build_tag_emitted_termination() -> tuple[RunnableAgent, str]:
    spec = make_spec(max_turns=5, termination=[TerminationCondition(type="tag_emitted", value="done")])
    script = [FakeResponse(content=[text_block("finishing up <done>result</done>")])]
    return _raw_api_agent(spec, script, {}), "go"


def _check_tag_emitted_termination(t: Trajectory) -> None:
    assert t.stopped_reason == "tag_emitted:done", f"expected tag_emitted termination, got {t.stopped_reason!r}"
    assert t.final_output and "<done>" in t.final_output


def _build_tool_result_recording() -> tuple[RunnableAgent, str]:
    spec = make_spec(max_turns=5, tools=[_NOOP_TOOL])
    script = [
        FakeResponse(content=[tool_use_block("noop", {"x": 1})]),
        FakeResponse(content=[text_block("done")]),
    ]
    return _raw_api_agent(spec, script, {"noop": echo_tool()}), "go"


def _check_tool_result_recording(t: Trajectory) -> None:
    calls = t.tool_calls
    assert len(calls) == 1, f"expected exactly one tool_call step, got {len(calls)}"
    step = calls[0]
    assert step.tool_name == "noop"
    assert step.tool_args == {"x": 1}
    assert step.tool_result is not None and "ok:" in step.tool_result, "tool result must be recorded on the step"


def _build_tool_exception_is_surfaced_not_raised() -> tuple[RunnableAgent, str]:
    spec = make_spec(max_turns=5, tools=[_FLAKY_TOOL])
    script = [
        FakeResponse(content=[tool_use_block("flaky")]),
        FakeResponse(content=[text_block("recovered")]),
    ]
    return _raw_api_agent(spec, script, {"flaky": raising_tool("kaboom")}), "go"


def _check_tool_exception_is_surfaced_not_raised(t: Trajectory) -> None:
    calls = t.tool_calls
    assert len(calls) == 1
    assert calls[0].tool_result is not None and "kaboom" in calls[0].tool_result, (
        "a raising tool callback must produce an error result on the trace, not crash the run"
    )
    assert t.stopped_reason == "no_tool_use"


def _build_missing_callback_reported_as_error_result() -> tuple[RunnableAgent, str]:
    spec = make_spec(max_turns=5, tools=[_NOOP_TOOL])
    script = [
        FakeResponse(content=[tool_use_block("noop")]),
        FakeResponse(content=[text_block("done")]),
    ]
    return _raw_api_agent(spec, script, {}), "go"  # no callback registered for "noop"


def _check_missing_callback_reported_as_error_result(t: Trajectory) -> None:
    calls = t.tool_calls
    assert len(calls) == 1
    assert calls[0].tool_result is not None and "no callback" in calls[0].tool_result


def _build_well_formed_trace() -> tuple[RunnableAgent, str]:
    spec = make_spec(max_turns=5, tools=[_NOOP_TOOL])
    script = [
        FakeResponse(content=[tool_use_block("noop", {"a": 1})]),
        FakeResponse(content=[text_block("final answer")]),
    ]
    return _raw_api_agent(spec, script, {"noop": echo_tool()}), "go"


def _check_well_formed_trace(t: Trajectory) -> None:
    assert t.final_output == "final answer"
    assert t.stopped_reason == "no_tool_use"
    step_types = [s.step_type for s in t.steps]
    assert step_types == [StepType.llm_call, StepType.tool_call, StepType.llm_call], step_types


RAW_API_CASES: list[ConformanceCase] = [
    ConformanceCase(
        "max_turn_termination",
        "loop.max_turns",
        "run() stops exactly at loop.max_turns and reports stopped_reason='max_turns'",
        _build_max_turns,
        _check_max_turns,
    ),
    ConformanceCase(
        "explicit_tool_termination",
        "termination.explicit_tool",
        "calling the declared explicit-tool ends the run without executing that tool",
        _build_explicit_tool_termination,
        _check_explicit_tool_termination,
    ),
    ConformanceCase(
        "tag_emitted_termination",
        "termination.tag_emitted",
        "emitting the declared stop tag in text ends the run",
        _build_tag_emitted_termination,
        _check_tag_emitted_termination,
    ),
    ConformanceCase(
        "tool_result_recording",
        "tools",
        "a successful tool call's name, args, and result are all recorded on one step",
        _build_tool_result_recording,
        _check_tool_result_recording,
    ),
    ConformanceCase(
        "tool_exception_handling",
        "tools",
        "a raising tool callback surfaces as an error result, not an uncaught exception",
        _build_tool_exception_is_surfaced_not_raised,
        _check_tool_exception_is_surfaced_not_raised,
    ),
    ConformanceCase(
        "missing_callback_handling",
        "tools",
        "a tool call with no registered callback surfaces as an error result",
        _build_missing_callback_reported_as_error_result,
        _check_missing_callback_reported_as_error_result,
    ),
    ConformanceCase(
        "trace_completeness",
        "trace",
        "the full step sequence (llm_call, tool_call, llm_call) is present and ordered",
        _build_well_formed_trace,
        _check_well_formed_trace,
    ),
]


# ---------------------------------------------------------------------------
# pydantic_ai-shaped cases: built by constructing a real pydantic_ai.Agent
# wired to a scripted FunctionModel and fake tool callables directly (bypassing
# PydanticAIAdapter.build(), same as the raw_api cases bypass RawAPIAdapter.build()
# — conformance tests run(), the execution contract, not file/client wiring).
#
# Several of these are *expected* to fail for PydanticAIAdapter today: its
# module docstring already documents that mid-run termination
# (explicit_tool/tag_emitted) isn't enforced, and pydantic-ai propagates a
# raising tool callback as an exception rather than an error result the model
# can see. That's the point — the matrix should show it, not hide it, and a
# clean `supports()` declaration is not proof either behavior actually holds.
# ---------------------------------------------------------------------------


def _pydantic_ai_agent(fn, tools: list, *, max_turns: int = 5):
    from pydantic_ai import Agent
    from pydantic_ai.models.function import FunctionModel

    spec = make_spec(max_turns=max_turns)
    agent = Agent(FunctionModel(fn), tools=tools)
    return RunnableAgent(spec=spec, handle=agent), "go"


def _pai_build_max_turns():
    from pydantic_ai import Tool
    from pydantic_ai.messages import ModelResponse, ToolCallPart

    def noop(**_kwargs) -> str:
        return "ok"

    def fn(_messages, _info):
        return ModelResponse(parts=[ToolCallPart(tool_name="noop", args={})])

    return _pydantic_ai_agent(fn, [Tool(function=noop, name="noop", description="")], max_turns=3)


def _pai_check_max_turns(t: Trajectory) -> None:
    assert t.stopped_reason == "max_turns", f"expected max_turns termination, got {t.stopped_reason!r}"


def _pai_build_explicit_tool_termination():
    from pydantic_ai import Tool
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

    calls = {"n": 0}

    def submit(**_kwargs) -> str:
        return "recorded"

    def fn(_messages, _info):
        calls["n"] += 1
        if calls["n"] == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name="submit", args={"answer": "42"})])
        return ModelResponse(parts=[TextPart(content="done")])

    return _pydantic_ai_agent(fn, [Tool(function=submit, name="submit", description="")])


def _pai_check_explicit_tool_termination(t: Trajectory) -> None:
    assert t.stopped_reason == "explicit_tool:submit", (
        f"expected explicit_tool termination, got {t.stopped_reason!r} "
        "(PydanticAIAdapter does not enforce mid-run explicit_tool termination — see its module docstring)"
    )


def _pai_build_tag_emitted_termination():
    from pydantic_ai.messages import ModelResponse, TextPart

    def fn(_messages, _info):
        return ModelResponse(parts=[TextPart(content="finishing up <done>result</done>")])

    return _pydantic_ai_agent(fn, [])


def _pai_check_tag_emitted_termination(t: Trajectory) -> None:
    assert t.stopped_reason == "tag_emitted:done", (
        f"expected tag_emitted termination, got {t.stopped_reason!r} "
        "(PydanticAIAdapter does not enforce mid-run tag_emitted termination — see its module docstring)"
    )


def _pai_build_tool_result_recording():
    from pydantic_ai import Tool
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

    calls = {"n": 0}

    def noop(x: int = 0) -> str:
        return f"ok:{x}"

    def fn(_messages, _info):
        calls["n"] += 1
        if calls["n"] == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name="noop", args={"x": 1})])
        return ModelResponse(parts=[TextPart(content="done")])

    return _pydantic_ai_agent(fn, [Tool(function=noop, name="noop", description="")])


def _pai_check_tool_result_recording(t: Trajectory) -> None:
    calls = t.tool_calls
    assert len(calls) == 1, f"expected exactly one tool_call step, got {len(calls)}"
    step = calls[0]
    assert step.tool_name == "noop"
    assert step.tool_result is not None and "ok:" in step.tool_result


def _pai_build_tool_exception_handling():
    from pydantic_ai import Tool
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

    calls = {"n": 0}

    def fn(_messages, _info):
        calls["n"] += 1
        if calls["n"] == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name="flaky", args={})])
        return ModelResponse(parts=[TextPart(content="recovered")])

    return _pydantic_ai_agent(fn, [Tool(function=raising_tool("kaboom"), name="flaky", description="")])


def _pai_check_tool_exception_handling(t: Trajectory) -> None:
    calls = t.tool_calls
    assert calls and calls[0].tool_result is not None and "kaboom" in calls[0].tool_result, (
        "a raising tool callback should surface as an error result on the trace, not crash the run "
        "(pydantic-ai currently propagates the exception out of run_sync instead)"
    )


def _pai_build_trace_completeness():
    from pydantic_ai import Tool
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

    calls = {"n": 0}

    def noop(x: int = 0) -> str:
        return f"ok:{x}"

    def fn(_messages, _info):
        calls["n"] += 1
        if calls["n"] == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name="noop", args={"a": 1})])
        return ModelResponse(parts=[TextPart(content="final answer")])

    return _pydantic_ai_agent(fn, [Tool(function=noop, name="noop", description="")])


def _pai_check_trace_completeness(t: Trajectory) -> None:
    assert t.final_output == "final answer"
    step_types = [s.step_type for s in t.steps]
    assert step_types == [StepType.llm_call, StepType.tool_call, StepType.llm_call], step_types


PYDANTIC_AI_CASES: list[ConformanceCase] = [
    ConformanceCase(
        "max_turn_termination",
        "loop.max_turns",
        "run() stops exactly at loop.max_turns and reports stopped_reason='max_turns'",
        _pai_build_max_turns,
        _pai_check_max_turns,
    ),
    ConformanceCase(
        "explicit_tool_termination",
        "termination.explicit_tool",
        "calling the declared explicit-tool ends the run without executing that tool",
        _pai_build_explicit_tool_termination,
        _pai_check_explicit_tool_termination,
    ),
    ConformanceCase(
        "tag_emitted_termination",
        "termination.tag_emitted",
        "emitting the declared stop tag in text ends the run",
        _pai_build_tag_emitted_termination,
        _pai_check_tag_emitted_termination,
    ),
    ConformanceCase(
        "tool_result_recording",
        "tools",
        "a successful tool call's name, args, and result are all recorded on one step",
        _pai_build_tool_result_recording,
        _pai_check_tool_result_recording,
    ),
    ConformanceCase(
        "tool_exception_handling",
        "tools",
        "a raising tool callback surfaces as an error result, not an uncaught exception",
        _pai_build_tool_exception_handling,
        _pai_check_tool_exception_handling,
    ),
    ConformanceCase(
        "trace_completeness",
        "trace",
        "the full step sequence (llm_call, tool_call, llm_call) is present and ordered",
        _pai_build_trace_completeness,
        _pai_check_trace_completeness,
    ),
]


def generate_matrix(results: list[ScenarioResult]) -> str:
    """Render conformance results as a markdown support matrix.

    A declaration in `AdapterCapabilities` says a feature *should* work; this
    table says whether the same scripted scenario actually passed. The two
    are not the same claim — do not treat a clean `supports()` output as
    proof of behavior.
    """
    case_ids = sorted({r.case_id for r in results})
    runtimes = sorted({r.runtime for r in results})
    by_key = {(r.case_id, r.runtime): r for r in results}

    symbol = {"pass": "✅", "fail": "❌", "error": "⚠️", "not_applicable": "—"}

    lines = ["| scenario | " + " | ".join(runtimes) + " |", "|---|" + "---|" * len(runtimes)]
    for case_id in case_ids:
        row = [case_id]
        for runtime in runtimes:
            result = by_key.get((case_id, runtime))
            row.append(symbol[result.status] if result else "—")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# gateway-shaped cases: the same scenarios as RAW_API_CASES, with each scripted
# Anthropic response translated into an OpenAI chat-completions body and served
# by FakeChatClient — plus gateway-specific cost/served-model/argument cases.
# ---------------------------------------------------------------------------


def _as_gateway(build: Callable[[], tuple[RunnableAgent, str]]) -> Callable[[], tuple[RunnableAgent, str]]:
    from harnesskit.adapters.gateway import to_openai_tool
    from harnesskit.testing.fakes import FakeChatClient, chat_body_from_fake_response

    def _build() -> tuple[RunnableAgent, str]:
        agent, input_text = build()
        script = agent.handle["client"].messages.responses
        client = FakeChatClient([chat_body_from_fake_response(r) for r in script])
        handle = {
            "client": client,
            "tool_schemas": [to_openai_tool(s) for s in agent.handle["tool_schemas"]],
            "callbacks": agent.handle["callbacks"],
        }
        return RunnableAgent(spec=agent.spec, handle=handle), input_text

    return _build


def _gateway_agent(spec: HarnessSpec, bodies: list[dict], callbacks: dict, headers: list[dict] | None = None) -> RunnableAgent:
    from harnesskit.adapters.gateway import to_openai_tool
    from harnesskit.testing.fakes import FakeChatClient

    tools = [to_openai_tool({"name": t.name, "input_schema": {"type": "object", "properties": {}}}) for t in spec.tools]
    return RunnableAgent(spec=spec, handle={"client": FakeChatClient(bodies, headers), "tool_schemas": tools, "callbacks": callbacks})


def _gw_build_observed_cost() -> tuple[RunnableAgent, str]:
    from harnesskit.testing.fakes import chat_body

    spec = make_spec()
    body = chat_body("answer", model="served/alias-target", usage={"prompt_tokens": 5, "completion_tokens": 7, "cost": 0.0012})
    return _gateway_agent(spec, [body], {}), "go"


def _gw_check_observed_cost(t: Trajectory) -> None:
    step = t.steps[0]
    assert step.cost_status is not None and step.cost_status.value == "observed", step.cost_status
    assert step.cost_usd == 0.0012
    assert step.note and "served_model=served/alias-target" in step.note, "the gateway's served model must be recorded"


def _gw_build_unknown_cost() -> tuple[RunnableAgent, str]:
    from harnesskit.testing.fakes import chat_body

    spec = make_spec()
    spec.model.model_id = "some-gateway/unpriced-model"
    return _gateway_agent(spec, [chat_body("answer")], {}), "go"


def _gw_check_unknown_cost(t: Trajectory) -> None:
    step = t.steps[0]
    assert step.cost_usd is None and step.cost_status is not None and step.cost_status.value == "unavailable", (
        "an unpriced gateway model must report unavailable cost, never $0"
    )


def _gw_build_bad_arguments() -> tuple[RunnableAgent, str]:
    from harnesskit.testing.fakes import chat_body

    spec = make_spec(tools=[_NOOP_TOOL])
    bad = chat_body(None, [("noop", {})])
    bad["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{not json"
    return _gateway_agent(spec, [bad, chat_body("done")], {"noop": echo_tool()}), "go"


def _gw_check_bad_arguments(t: Trajectory) -> None:
    calls = t.tool_calls
    assert len(calls) == 1 and calls[0].tool_result and "invalid tool arguments" in calls[0].tool_result
    assert t.final_output == "done"


def _gw_build_max_tool_calls() -> tuple[RunnableAgent, str]:
    from harnesskit.testing.fakes import chat_body

    spec = make_spec(max_turns=10, tools=[_NOOP_TOOL])
    spec.loop.max_tool_calls = 2
    bodies = [chat_body(None, [("noop", {"i": i})]) for i in range(5)]
    return _gateway_agent(spec, bodies, {"noop": echo_tool()}), "go"


def _gw_check_max_tool_calls(t: Trajectory) -> None:
    assert t.stopped_reason == "max_tool_calls", t.stopped_reason
    assert len(t.tool_calls) == 2


GATEWAY_CASES: list[ConformanceCase] = [
    ConformanceCase(c.id, c.capability, c.description, _as_gateway(c.build), c.check) for c in RAW_API_CASES
] + [
    ConformanceCase(
        "observed_cost_and_served_model",
        "cost",
        "a gateway-reported cost is recorded as observed, and the served model is recorded",
        _gw_build_observed_cost,
        _gw_check_observed_cost,
    ),
    ConformanceCase(
        "unpriced_model_cost_unavailable",
        "cost",
        "an unpriced model with no reported cost is 'unavailable', never $0",
        _gw_build_unknown_cost,
        _gw_check_unknown_cost,
    ),
    ConformanceCase(
        "invalid_tool_arguments",
        "tools",
        "malformed tool-call JSON surfaces as an error result the model can see",
        _gw_build_bad_arguments,
        _gw_check_bad_arguments,
    ),
    ConformanceCase(
        "max_tool_calls",
        "loop.max_tool_calls",
        "run() stops before exceeding loop.max_tool_calls",
        _gw_build_max_tool_calls,
        _gw_check_max_tool_calls,
    ),
]
