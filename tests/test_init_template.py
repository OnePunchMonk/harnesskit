"""`harness init --template` and `harness templates` (issue #4 item 5):
the template path is offline and deterministic — no LLM call.
"""
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from harnesskit.cli.main import app
from harnesskit.linter import Severity
from harnesskit.parser import load_harness
from harnesskit.scaffold.templates import TEMPLATES


def test_templates_command_lists_all_six(monkeypatch):
    result = CliRunner().invoke(app, ["templates"])
    assert result.exit_code == 0, result.output
    for name in TEMPLATES:
        assert name in result.output


def test_init_with_template_is_offline_and_deterministic(tmp_path, monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("harness init --template must never make a network/LLM call")

    monkeypatch.setattr("harnesskit.scaffold.infer.infer_plan", _boom, raising=False)

    target = tmp_path / "my-agent"
    result = CliRunner().invoke(app, ["init", "my-agent", "--template", "coding_agent", "--directory", str(target)])

    assert result.exit_code == 0, result.output
    assert "no LLM call" in result.output
    assert (target / "harness.yaml").exists()
    assert (target / "tools" / "shell.json").exists()

    loaded = load_harness(target)
    from harnesskit.linter import lint

    findings = lint(loaded.spec)
    errors = [f for f in findings if f.severity == Severity.error]
    assert errors == [], errors


def test_init_with_unknown_template_lists_valid_choices(tmp_path):
    target = tmp_path / "nope"
    result = CliRunner().invoke(app, ["init", "nope", "--template", "not-a-template", "--directory", str(target)])
    assert result.exit_code == 1
    for name in TEMPLATES:
        assert name in result.output


def test_init_rejects_both_spec_and_template():
    result = CliRunner().invoke(app, ["init", "x", "--spec", "an agent", "--template", "generic"])
    assert result.exit_code == 1
    assert "either --spec or --template" in result.output
