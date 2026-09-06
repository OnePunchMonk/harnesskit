"""Spec inference (design doc §6.1): one LLM call that turns a natural-
language description into a ScaffoldPlan. This is the only place in the
scaffolder that talks to a model — everything after this (templates,
generation, validation) is deterministic.
"""
from __future__ import annotations

from harnesskit.scaffold.plan import ScaffoldPlan
from harnesskit.scaffold.templates import TEMPLATES

_SYSTEM = f"""You turn a one-sentence to one-paragraph description of an agent into a
structured harness plan by calling propose_harness_plan exactly once.

Available domains (pick the closest match, or "generic" if none fit well):
{", ".join(TEMPLATES.keys())}

Guidelines:
- `name` should be a short kebab-case identifier suitable as a directory name.
- `tools` should list only the tools genuinely needed beyond the domain's defaults —
  it's fine to return an empty list if the domain's default tools cover it.
- `guardrails_needed` should be chosen from: cost_ceiling, dedup, pii_redaction,
  permission_scoping, output_schema_validation. Always include cost_ceiling.
- `max_turns` should reflect realistic task complexity: 5-10 for simple lookups,
  15-25 for multi-step work like coding tasks.
- `stop_tag` is the XML tag the agent should emit to signal it's done, e.g. "answer"."""


def infer_plan(nl_spec: str, model_id: str = "claude-sonnet-5") -> ScaffoldPlan:
    try:
        import anthropic
    except ImportError as e:
        raise ImportError("Scaffolding from a natural-language spec requires: pip install anthropic") from e

    client = anthropic.Anthropic()
    schema = ScaffoldPlan.model_json_schema()
    schema.pop("description", None)

    response = client.messages.create(
        model=model_id,
        max_tokens=1024,
        system=_SYSTEM,
        messages=[{"role": "user", "content": nl_spec}],
        tools=[{"name": "propose_harness_plan", "description": "Propose a structured harness plan.", "input_schema": schema}],
        tool_choice={"type": "tool", "name": "propose_harness_plan"},
    )

    tool_use = next(b for b in response.content if b.type == "tool_use")
    plan = ScaffoldPlan.model_validate(tool_use.input)
    plan.description = plan.description or nl_spec
    return plan
