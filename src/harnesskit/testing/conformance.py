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
