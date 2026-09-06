"""Materializes a ScaffoldPlan into a real harness directory (design doc §6):
harness.yaml, system_prompt.md, and one tools/<name>.json + tools/<name>.py
stub per tool. Purely deterministic — no model calls happen here.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from harnesskit.scaffold.plan import PlannedTool, ScaffoldPlan
from harnesskit.scaffold.templates import TEMPLATES, ToolStub

_GUARDRAIL_BUILDERS = {
    "cost_ceiling": lambda plan: {
        "name": "cost-ceiling",
        "trigger": "on_budget_check",
        "check": f"cost_threshold:{plan.cost_ceiling_usd:.2f}",
        "enforce": "abort",
    },
    "dedup": lambda plan: {
        "name": "dedup",
        "trigger": "pre_tool_call",
        "check": "dedup_detection",
        "enforce": "retry_with_reflection",
    },
    "pii_redaction": lambda plan: {
        "name": "pii-redaction",
        "trigger": "post_tool_call",
        "check": "pii_scan",
        "enforce": "cancel_tool",
        "message": "Sensitive data detected in tool output; redact before continuing.",
    },
    "output_schema_validation": lambda plan: {
        "name": "output-schema",
        "trigger": "post_turn",
        "check": "schema_validation",
        "enforce": "retry_with_reflection",
    },
}


def _merged_tools(plan: ScaffoldPlan) -> list[ToolStub | PlannedTool]:
    template = TEMPLATES[plan.resolved_domain()]
    by_name = {t.name: t for t in template.default_tools}
    for t in plan.tools:
        by_name[t.name] = t  # plan-specified tools override/extend the template defaults
    return list(by_name.values())


def generate_harness(plan: ScaffoldPlan, target_dir: Path) -> None:
    if target_dir.exists() and any(target_dir.iterdir()):
        raise FileExistsError(f"{target_dir} already exists and is not empty")

    template = TEMPLATES[plan.resolved_domain()]
    tools_dir = target_dir / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)

    tool_entries = []
    for tool in _merged_tools(plan):
        properties = getattr(tool, "properties", {}) or {"input": "string"}
        schema = {
            "name": tool.name,
            "description": tool.description,
            "input_schema": {
                "type": "object",
                "properties": {k: {"type": v} for k, v in properties.items()},
                "required": list(properties.keys()),
            },
        }
        (tools_dir / f"{tool.name}.json").write_text(json.dumps(schema, indent=2) + "\n")
        (tools_dir / f"{tool.name}.py").write_text(
            f'"""Callback for the `{tool.name}` tool — generated stub, wire up real logic."""\n\n\n'
            f"def run(**kwargs) -> str:\n"
            f'    raise NotImplementedError("wire up {tool.name} before running this harness for real")\n'
        )
        tool_entries.append({"name": tool.name, "source": "custom", "ref": f"tools/{tool.name}.json", "permissions": []})

    guardrails = [_GUARDRAIL_BUILDERS[label](plan) for label in plan.guardrails_needed if label in _GUARDRAIL_BUILDERS]
    if not any(g["check"].startswith("cost_threshold") for g in guardrails):
        guardrails.insert(0, _GUARDRAIL_BUILDERS["cost_ceiling"](plan))

    system_prompt_lines = [f"You are {plan.name}, {plan.purpose or plan.description}"]
    if template.system_prompt_hint:
        system_prompt_lines.append(template.system_prompt_hint)
    system_prompt_lines.append(f"When you have finished, emit your answer wrapped in <{plan.stop_tag}>...</{plan.stop_tag}> and stop.")
    system_prompt_lines.append("\nTask: {{task}}")
    (target_dir / "system_prompt.md").write_text("\n\n".join(system_prompt_lines) + "\n")

    manifest = {
        "schema_version": 1,
        "metadata": {
            "name": plan.name,
            "version": "0.1.0",
            "description": plan.description,
        },
        "model": {"provider": "anthropic", "model_id": "claude-sonnet-5"},
        "loop": {"type": template.loop_type, "max_turns": plan.max_turns},
        "tools": tool_entries,
        "guardrails": guardrails,
        "termination": [
            {"type": "tag_emitted", "value": plan.stop_tag},
            {"type": "max_turns", "value": plan.max_turns},
        ],
        "eval": {"cases": []},
        "scaffold": {
            "system_prompt": "system_prompt.md",
            "system_prompt_is_file": True,
            "template_vars": ["task"],
        },
    }
    (target_dir / "harness.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, width=100))
