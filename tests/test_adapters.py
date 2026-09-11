import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from harnesskit.adapters import PydanticAIAdapter, RawAPIAdapter, SupportStatus, check_support, inspect_support
from harnesskit.cli.main import app
from harnesskit.parser import load_harness

EXAMPLE = Path(__file__).parent.parent / "examples" / "react-web-researcher"


def test_check_support_flags_unsupported_loop_type():
    result = load_harness(EXAMPLE)
    result.spec.loop.type = result.spec.loop.type.__class__("rewoo")
    warnings = check_support(result.spec, RawAPIAdapter())
    assert any("loop.type" in w for w in warnings)
    finding = inspect_support(result.spec, RawAPIAdapter())[0]
    assert finding.field == "loop.type"
    assert finding.status is SupportStatus.unsupported
    assert finding.runtime == "raw_api"


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
    pytest.importorskip("pydantic_ai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-for-build-only")
    result = load_harness(EXAMPLE)
    adapter = PydanticAIAdapter()
    agent = adapter.build(result.spec)
    assert agent.handle is not None


def test_raw_api_adapter_loads_tool_callbacks(monkeypatch):
    pytest.importorskip("anthropic")
    result = load_harness(EXAMPLE)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-for-build-only")
    adapter = RawAPIAdapter()
    agent = adapter.build(result.spec)
    assert set(agent.handle["callbacks"]) == {"search", "verify_citation"}
    assert all(cb is not None for cb in agent.handle["callbacks"].values())


def test_inspect_adapter_json_is_offline_and_machine_readable():
    result = CliRunner().invoke(app, ["inspect", str(EXAMPLE), "--adapter", "raw_api", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == []


def test_strict_preflight_rejects_before_provider_setup(tmp_path):
    harness_dir = tmp_path / "harness"
    shutil.copytree(EXAMPLE, harness_dir)
    with (harness_dir / "harness.yaml").open("a") as config:
        config.write("\nmemory:\n  session: persistent\n")

    result = CliRunner().invoke(app, ["run", str(harness_dir), "--input", "hello", "--strict"])

    assert result.exit_code == 1
    assert "Strict preflight rejected" in result.output
    assert "memory.session='persistent'" in result.output
