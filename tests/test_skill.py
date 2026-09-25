"""The shipped harnesskit Agent Skill: valid frontmatter, commands that exist,
a harness.yaml reference that actually validates, and `harness skill install`."""
from __future__ import annotations

import re
from pathlib import Path

import yaml
from typer.testing import CliRunner

from harnesskit.cli.main import app
from harnesskit.format.spec import HarnessSpec

SKILL = Path(__file__).parent.parent / "src" / "harnesskit" / "skill"


def test_skill_frontmatter_follows_agent_skills_format():
    text = (SKILL / "SKILL.md").read_text()
    match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert match, "SKILL.md must start with YAML frontmatter"
    meta = yaml.safe_load(match.group(1))
    assert meta["name"] == "harnesskit"
    assert re.fullmatch(r"[a-z0-9-]+", meta["name"])
    assert 0 < len(meta["description"]) <= 1024


def test_every_harness_command_in_the_skill_exists():
    runner = CliRunner()
    text = "\n".join(p.read_text() for p in SKILL.rglob("*.md"))
    code = "\n".join(re.findall(r"```.*?```", text, re.S) + re.findall(r"`([^`\n]+)`", text))
    commands = set(re.findall(r"(?:^|\s)harness ([a-z]+)", code, re.M))
    assert {"init", "eval", "train", "params", "inspect", "lint", "conformance", "export"} <= commands
    for command in commands:
        r = runner.invoke(app, [command, "--help"])
        assert r.exit_code == 0, f"`harness {command}` referenced by the skill does not exist"


def test_format_reference_example_validates():
    text = (SKILL / "references" / "harness-format.md").read_text()
    block = re.search(r"```yaml\n(.*?)```", text, re.S).group(1)
    raw = yaml.safe_load(block)
    raw["scaffold"] = {"system_prompt": "inline", "system_prompt_is_file": False}
    spec = HarnessSpec.model_validate(raw)
    assert spec.eval.cases[0].split == "train" and len(spec.trainable) == 4


def test_skill_install(tmp_path):
    runner = CliRunner()
    r = runner.invoke(app, ["skill", "install", "--dest", str(tmp_path)])
    assert r.exit_code == 0, r.output
    installed = tmp_path / "harnesskit"
    assert (installed / "SKILL.md").is_file() and (installed / "references" / "evals.md").is_file()
    r = runner.invoke(app, ["skill", "install", "--dest", str(tmp_path)])
    assert r.exit_code == 1 and "--force" in r.output
    assert runner.invoke(app, ["skill", "install", "--dest", str(tmp_path), "--force"]).exit_code == 0
