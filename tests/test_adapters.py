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
    # The example harness declares a cost-ceiling guardrail, a dedup
    # guardrail, and observation_masking compaction — none of which either
    # adapter enforces at runtime (issue #3 item 1). It's no longer "clean"
    # by design: those three gaps must surface, not be silently hidden.
    raw_warnings = check_support(result.spec, RawAPIAdapter())
    assert any("guardrails.cost-ceiling" in w for w in raw_warnings)
    assert any("guardrails.dedup" in w for w in raw_warnings)
    assert any("context.compaction" in w for w in raw_warnings)

    pydantic_warnings = check_support(result.spec, PydanticAIAdapter())
    assert any("guardrails.cost-ceiling" in w for w in pydantic_warnings)
    assert any("guardrails.dedup" in w for w in pydantic_warnings)
    assert any("context.compaction" in w for w in pydantic_warnings)


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
    findings = json.loads(result.output)
    # The example declares a cost-ceiling guardrail, a dedup guardrail, and
    # observation_masking compaction, none of which raw_api enforces at
    # runtime (issue #3 item 1) — these three gaps must be visible here.
    fields = {f["field"] for f in findings}
    assert "guardrails.cost-ceiling" in fields
    assert "guardrails.dedup" in fields
    assert "context.compaction" in fields
    for f in findings:
        assert f["status"] == "unsupported"


def test_strict_preflight_rejects_before_provider_setup(tmp_path):
    harness_dir = tmp_path / "harness"
    shutil.copytree(EXAMPLE, harness_dir)
    with (harness_dir / "harness.yaml").open("a") as config:
        config.write("\nmemory:\n  session: persistent\n")

    result = CliRunner().invoke(app, ["run", str(harness_dir), "--input", "hello", "--strict"])

    assert result.exit_code == 1
    assert "Strict preflight rejected" in result.output
    assert "memory.session='persistent'" in result.output


def test_raw_api_capabilities_match_what_run_actually_enforces():
    """RawAPIAdapter.run() enforces stop tags, explicit-tool stop, and the
    `for turn in range(max_turns)` bound — nothing else. supports() must say
    exactly that, not claim more (issue #3 item 1)."""
    caps = RawAPIAdapter().supports()
    assert caps.enforced_termination_types == {"tag_emitted", "explicit_tool", "max_turns"}
    assert caps.enforced_guardrails == set()
    assert caps.enforced_compaction_strategies == set()
    assert caps.supports_routing is False
    assert caps.supports_max_tool_calls is False


def test_pydantic_ai_capabilities_match_what_run_actually_enforces():
    """PydanticAIAdapter.run() only enforces max_turns via UsageLimits;
    explicit_tool/tag_emitted are not enforced mid-run (see module
    docstring), and no guardrail/compaction/routing/max_tool_calls either."""
    caps = PydanticAIAdapter().supports()
    assert caps.enforced_termination_types == {"max_turns"}
    assert caps.enforced_guardrails == set()
    assert caps.enforced_compaction_strategies == set()
    assert caps.supports_routing is False
    assert caps.supports_max_tool_calls is False


def test_inspect_support_flags_each_declared_but_unenforced_gap():
    """Five small cases: a guardrail, a termination type, a compaction
    strategy, model.routing, and loop.max_tool_calls, each declared but not
    in RawAPIAdapter's enforced sets, each must produce a finding."""
    result = load_harness(EXAMPLE)

    # 1. guardrail
    spec = result.spec.model_copy(deep=True)
    findings = inspect_support(spec, RawAPIAdapter())
    assert any(f.field == "guardrails.cost-ceiling" for f in findings)

    # 2. termination type not enforced by raw_api
    spec = result.spec.model_copy(deep=True)
    spec.termination.append(spec.termination[0].__class__(type="idle_detection", value=None))
    findings = inspect_support(spec, RawAPIAdapter())
    assert any(f.field == "termination" and f.requested == "idle_detection" for f in findings)

    # 3. compaction strategy
    spec = result.spec.model_copy(deep=True)
    assert any(f.field == "context.compaction" for f in inspect_support(spec, RawAPIAdapter()))

    # 4. model.routing
    spec = result.spec.model_copy(deep=True)
    spec.model.routing = {"plan": "claude-opus-5"}
    findings = inspect_support(spec, RawAPIAdapter())
    assert any(f.field == "model.routing" for f in findings)

    # 5. loop.max_tool_calls
    spec = result.spec.model_copy(deep=True)
    spec.loop.max_tool_calls = 5
    findings = inspect_support(spec, RawAPIAdapter())
    assert any(f.field == "loop.max_tool_calls" and f.requested == "5" for f in findings)


def test_lint_flags_react_web_researcher_declared_but_unenforced_gaps():
    """Issue #3's own example: cost_threshold guardrail, dedup guardrail,
    and observation_masking compaction must all surface as lint findings,
    not silently pass ("No issues found")."""
    from harnesskit.linter import lint

    result = load_harness(EXAMPLE)
    findings = lint(result.spec)
    unenforced = [f for f in findings if f.rule == "declared-but-unenforced"]
    messages = " ".join(f.message for f in unenforced)
    assert "guardrails.cost-ceiling" in messages
    assert "guardrails.dedup" in messages
    assert "context.compaction" in messages
    assert "raw_api" in messages


def test_run_prints_enforcement_gap_warning_without_strict(monkeypatch):
    """The whole point is that this is visible by DEFAULT, not buried behind
    --strict."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")

    class _FakeAdapter:
        def supports(self):
            return RawAPIAdapter().supports()

        def build(self, spec):
            raise RuntimeError("build should not be reached before warnings print")

    import harnesskit.cli.main as main_mod

    monkeypatch.setitem(main_mod.ADAPTERS, "raw_api", lambda: _FakeAdapter())
    result = CliRunner().invoke(app, ["run", str(EXAMPLE), "--input", "hello"])

    assert "declared but not enforced at runtime by adapter 'raw_api'" in result.output


def test_run_strict_turns_enforcement_gap_into_hard_failure():
    result = CliRunner().invoke(app, ["run", str(EXAMPLE), "--input", "hello", "--strict"])

    assert result.exit_code == 1
    assert "Strict preflight rejected" in result.output
    assert "declared but not enforced at runtime by adapter 'raw_api'" in result.output


def test_eval_strict_turns_enforcement_gap_into_hard_failure():
    result = CliRunner().invoke(app, ["eval", str(EXAMPLE), "--strict"])

    assert result.exit_code == 1
    assert "Strict preflight rejected" in result.output
    assert "declared but not enforced at runtime by adapter 'raw_api'" in result.output
