from pathlib import Path

from harnesskit.linter import Severity
from harnesskit.scaffold import ScaffoldPlan, generate_harness, validate_and_fix
from harnesskit.scaffold.plan import PlannedTool


def test_generate_harness_for_research_agent_domain(tmp_path):
    plan = ScaffoldPlan(
        name="fact-finder",
        description="Answers factual questions using web search.",
        purpose="answer factual questions",
        domain="research_agent",
        guardrails_needed=["cost_ceiling", "dedup"],
        max_turns=8,
    )
    target = tmp_path / "fact-finder"
    generate_harness(plan, target)

    assert (target / "harness.yaml").exists()
    assert (target / "system_prompt.md").exists()
    assert (target / "tools" / "search.json").exists()
    assert (target / "tools" / "search.py").exists()
    assert (target / "tools" / "verify_citation.json").exists()  # from the research_agent template default


def test_generate_harness_refuses_nonempty_dir(tmp_path):
    plan = ScaffoldPlan(name="x", description="d", purpose="p", domain="generic")
    target = tmp_path / "x"
    target.mkdir()
    (target / "already-here.txt").write_text("hi")
    try:
        generate_harness(plan, target)
        assert False, "expected FileExistsError"
    except FileExistsError:
        pass


def test_generated_harness_lints_clean_after_self_validation(tmp_path):
    plan = ScaffoldPlan(
        name="fact-finder",
        description="Answers factual questions using web search.",
        purpose="answer factual questions",
        domain="research_agent",
        guardrails_needed=["cost_ceiling", "dedup"],
        max_turns=8,
    )
    target = tmp_path / "fact-finder"
    generate_harness(plan, target)

    result, findings = validate_and_fix(target)
    errors = [f for f in findings if f.severity == Severity.error]
    assert errors == [], errors


def test_self_validation_adds_verifier_when_domain_has_none(tmp_path):
    plan = ScaffoldPlan(
        name="lookup-bot",
        description="Looks up account info.",
        purpose="look up accounts",
        domain="customer_support",
        guardrails_needed=["cost_ceiling"],
        max_turns=6,
    )
    target = tmp_path / "lookup-bot"
    generate_harness(plan, target)

    result, findings = validate_and_fix(target)
    assert any(t.name == "self_check" for t in result.spec.tools)
    assert not any(f.rule == "verifier-gap" for f in findings)


def test_plan_extra_tools_extend_template_defaults(tmp_path):
    plan = ScaffoldPlan(
        name="fact-finder-plus",
        description="Research agent with a summarizer.",
        purpose="research and summarize",
        domain="research_agent",
        tools=[PlannedTool(name="summarize", description="Summarize a document.", properties={"text": "string"})],
        guardrails_needed=["cost_ceiling"],
        max_turns=10,
    )
    target = tmp_path / "fact-finder-plus"
    generate_harness(plan, target)
    result, _ = validate_and_fix(target)
    tool_names = {t.name for t in result.spec.tools}
    assert {"search", "verify_citation", "summarize"} <= tool_names
