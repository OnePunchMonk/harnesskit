"""Self-validation loop (design doc §6.3): after generation, lint the result
and make one deterministic corrective pass for the most common fixable
finding (a missing verifier), then re-lint. This is intentionally narrow —
a real auto-fix-everything loop is out of scope for v0; the goal is that
`harness init --spec ...` doesn't hand you a harness with an obvious,
mechanically-fixable gap.
"""
from __future__ import annotations

import json
from pathlib import Path

from harnesskit.linter import Finding, Severity, lint
from harnesskit.parser import LoadResult, load_harness


def _add_self_check_tool(target_dir: Path) -> None:
    tools_dir = target_dir / "tools"
    tools_dir.mkdir(exist_ok=True)
    schema = {
        "name": "self_check",
        "description": "Re-read your own draft output and confirm it satisfies the task before finalizing.",
        "input_schema": {
            "type": "object",
            "properties": {"draft_output": {"type": "string"}},
            "required": ["draft_output"],
        },
    }
    (tools_dir / "self_check.json").write_text(json.dumps(schema, indent=2) + "\n")
    (tools_dir / "self_check.py").write_text(
        '"""Callback for the generated `self_check` verifier tool."""\n\n\n'
        "def run(draft_output: str) -> str:\n"
        '    return f"self-check noted: {len(draft_output)} chars of draft output recorded"\n'
    )

    manifest_path = target_dir / "harness.yaml"
    import yaml

    raw = yaml.safe_load(manifest_path.read_text())
    raw.setdefault("tools", []).append(
        {"name": "self_check", "source": "custom", "ref": "tools/self_check.json", "permissions": []}
    )
    manifest_path.write_text(yaml.safe_dump(raw, sort_keys=False, width=100))


def validate_and_fix(target_dir: Path) -> tuple[LoadResult, list[Finding]]:
    result = load_harness(target_dir)
    findings = lint(result.spec)

    if any(f.rule == "verifier-gap" for f in findings):
        _add_self_check_tool(target_dir)
        result = load_harness(target_dir)
        findings = lint(result.spec)

    return result, findings
