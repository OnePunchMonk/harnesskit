"""PydanticAIAdapter conformance (issue #6): same scripted scenarios as
raw_api, run against a real pydantic_ai.Agent wired to a scripted
FunctionModel. Some scenarios are *expected* to fail here — the adapter's own
docstring documents that mid-run termination isn't enforced — the point of
this suite is to report that explicitly rather than assume declarations hold.
"""
from __future__ import annotations

import pytest

pydantic_ai = pytest.importorskip("pydantic_ai")

from harnesskit.adapters.pydantic_ai_adapter import PydanticAIAdapter  # noqa: E402
from harnesskit.testing.conformance import PYDANTIC_AI_CASES, run_case  # noqa: E402

EXPECTED_STATUS = {
    "max_turn_termination": "pass",
    "explicit_tool_termination": "fail",  # not enforced mid-run, by design
    "tag_emitted_termination": "fail",  # not enforced mid-run, by design
    "tool_result_recording": "pass",
    "tool_exception_handling": "error",  # pydantic-ai propagates the raise
    "trace_completeness": "pass",
}


@pytest.mark.parametrize("case", PYDANTIC_AI_CASES, ids=[c.id for c in PYDANTIC_AI_CASES])
def test_pydantic_ai_adapter_conformance(case):
    result = run_case(PydanticAIAdapter(), case, runtime="pydantic_ai")
    expected = EXPECTED_STATUS[case.id]
    assert result.status == expected, f"{case.id}: expected {expected}, got {result.status} ({result.detail})"
