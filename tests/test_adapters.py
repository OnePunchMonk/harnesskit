import os
from pathlib import Path

from harnesskit.adapters import PydanticAIAdapter, RawAPIAdapter, check_support
from harnesskit.parser import load_harness

EXAMPLE = Path(__file__).parent.parent / "examples" / "react-web-researcher"


def test_check_support_flags_unsupported_loop_type():
    result = load_harness(EXAMPLE)
    result.spec.loop.type = result.spec.loop.type.__class__("rewoo")
    warnings = check_support(result.spec, RawAPIAdapter())
    assert any("loop.type" in w for w in warnings)


def test_check_support_flags_unsupported_memory_backend():
    result = load_harness(EXAMPLE)
    result.spec.memory.session = "persistent"
    warnings = check_support(result.spec, PydanticAIAdapter())
    assert any("memory.session" in w for w in warnings)


def test_check_support_clean_for_default_example():
    result = load_harness(EXAMPLE)
    assert check_support(result.spec, RawAPIAdapter()) == []
    assert check_support(result.spec, PydanticAIAdapter()) == []


def test_pydantic_ai_adapter_builds_with_example_tools(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-for-build-only")
    result = load_harness(EXAMPLE)
    adapter = PydanticAIAdapter()
    agent = adapter.build(result.spec)
    assert agent.handle is not None


def test_raw_api_adapter_loads_tool_callbacks():
    result = load_harness(EXAMPLE)
    os.environ.setdefault("ANTHROPIC_API_KEY", "dummy-for-build-only")
    adapter = RawAPIAdapter()
    agent = adapter.build(result.spec)
    assert set(agent.handle["callbacks"]) == {"search", "verify_citation"}
    assert all(cb is not None for cb in agent.handle["callbacks"].values())
