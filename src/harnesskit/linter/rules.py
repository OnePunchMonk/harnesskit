"""Static analysis rules over a HarnessSpec.

Each rule is a plain function: (HarnessSpec) -> list[Finding]. New rules are
added by writing one of these and registering it in ALL_RULES. This keeps the
linter itself dumb — it just runs every rule and collects findings — so the
interesting logic lives in isolated, testable functions per known failure mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from harnesskit.format.spec import CompactionStrategy, HarnessSpec


class Severity(str, Enum):
    error = "error"
    warning = "warning"
    info = "info"


@dataclass
class Finding:
    rule: str
    severity: Severity
    message: str
    fix: str | None = None


def rule_missing_termination(spec: HarnessSpec) -> list[Finding]:
    """§4.2 — every harness needs at least one reliable termination condition."""
    if not spec.termination:
        return [
            Finding(
                rule="missing-termination",
                severity=Severity.error,
                message="No termination conditions defined.",
                fix="Add at least one of: explicit_tool, idle_detection, max_turns, timeout.",
            )
        ]
    types = {t.type for t in spec.termination}
    if types == {"timeout"}:
        return [
            Finding(
                rule="timeout-only-termination",
                severity=Severity.warning,
                message="timeout is the only termination condition — expensive failure mode if the agent hangs before then.",
                fix="Add explicit_tool or max_turns as a cheaper backstop.",
            )
        ]
    return []


def rule_unbounded_context(spec: HarnessSpec) -> list[Finding]:
    """§4.3 — context growth without a bound is a common runaway-cost failure."""
    findings = []
    if spec.context.compaction == CompactionStrategy.none and spec.context.budget_tokens is None:
        if spec.loop.max_turns > 10:
            findings.append(
                Finding(
                    rule="unbounded-context",
                    severity=Severity.warning,
                    message=f"compaction=none with no context_budget and max_turns={spec.loop.max_turns}.",
                    fix="Set context.budget_tokens or choose a compaction strategy (observation_masking recommended).",
                )
            )
    return findings


def rule_verifier_gap(spec: HarnessSpec) -> list[Finding]:
    """§4.5 — the agent does work but nothing checks the output."""
    has_verify_tool = any("verify" in t.name.lower() or "check" in t.name.lower() for t in spec.tools)
    has_verify_guardrail = any("output" in g.check.lower() or "verify" in g.check.lower() for g in spec.guardrails)
    has_verify_routing = "verify" in spec.model.routing
    if not (has_verify_tool or has_verify_guardrail or has_verify_routing):
        return [
            Finding(
                rule="verifier-gap",
                severity=Severity.warning,
                message="No tool, guardrail, or model-routing step verifies the agent's own output.",
                fix="Add a verification tool, an output-checking guardrail, or a verify step in model.routing.",
            )
        ]
    return []


def rule_guardrail_completeness(spec: HarnessSpec) -> list[Finding]:
    """§4.7 — check for guardrails covering the most common failure modes."""
    findings = []
    checks = [g.check.lower() for g in spec.guardrails]
    if not any("dedup" in c for c in checks) and spec.loop.max_turns > 5:
        findings.append(
            Finding(
                rule="no-dedup-guardrail",
                severity=Severity.info,
                message="No dedup-detection guardrail; a stuck loop can repeat the same tool call until max_turns.",
                fix="Add a guardrail with check='dedup_detection'.",
            )
        )
    if not any("cost" in c for c in checks):
        findings.append(
            Finding(
                rule="no-cost-ceiling",
                severity=Severity.warning,
                message="No cost-ceiling guardrail defined.",
                fix="Add a guardrail with check='cost_threshold:<usd>' and enforce='abort'.",
            )
        )
    has_fs_or_shell = any(t.source.value == "builtin" and t.ref in ("shell", "filesystem") for t in spec.tools)
    if has_fs_or_shell and not any(t.permissions for t in spec.tools):
        findings.append(
            Finding(
                rule="unscoped-fs-shell",
                severity=Severity.warning,
                message="Filesystem/shell tool has no permission scoping.",
                fix="Set tool.permissions (e.g. ['read'], ['write:./workdir']).",
            )
        )
    return findings


def rule_cost_ceiling_estimate(spec: HarnessSpec) -> list[Finding]:
    """§4.6 — rough worst-case cost check against any declared cost guardrail."""
    ceiling = None
    for g in spec.guardrails:
        if g.check.startswith("cost_threshold:"):
            try:
                ceiling = float(g.check.split(":", 1)[1])
            except ValueError:
                pass
    if ceiling is None:
        return []
    # crude estimate: ~1500 tokens/turn blended, ~$6/M tokens blended rate
    est_tokens = spec.loop.max_turns * 1500
    est_cost = est_tokens / 1_000_000 * 6.0
    if est_cost > ceiling:
        return [
            Finding(
                rule="cost-ceiling-exceeded",
                severity=Severity.warning,
                message=f"Worst-case estimated cost (~${est_cost:.2f} over {spec.loop.max_turns} turns) exceeds cost_threshold=${ceiling:.2f}.",
                fix="Lower loop.max_turns, tighten the cost_threshold, or add cheaper model routing for non-planning steps.",
            )
        ]
    return []


def rule_tool_coverage(spec: HarnessSpec) -> list[Finding]:
    """§4.4 — cheap heuristic: prompt mentions a capability with no matching tool."""
    prompt = spec.scaffold.system_prompt.lower()
    tool_refs = " ".join(t.name.lower() + " " + t.ref.lower() for t in spec.tools)
    findings = []
    checks = {
        "run tests": ["shell", "test", "exec"],
        "search the web": ["search", "browser", "web"],
    }
    for phrase, keywords in checks.items():
        if phrase in prompt and not any(k in tool_refs for k in keywords):
            findings.append(
                Finding(
                    rule="tool-coverage-gap",
                    severity=Severity.error,
                    message=f"System prompt mentions '{phrase}' but no matching tool is registered.",
                    fix=f"Add a tool whose name/ref relates to: {', '.join(keywords)}.",
                )
            )
    return findings


ALL_RULES = [
    rule_missing_termination,
    rule_unbounded_context,
    rule_verifier_gap,
    rule_guardrail_completeness,
    rule_cost_ceiling_estimate,
    rule_tool_coverage,
]
