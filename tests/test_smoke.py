from pathlib import Path

from harnesskit.linter import Severity, lint
from harnesskit.parser import load_harness

EXAMPLE = Path(__file__).parent.parent / "examples" / "react-web-researcher"


def test_load_example_harness():
    result = load_harness(EXAMPLE)
    assert result.spec.metadata.name == "react-web-researcher"
    assert len(result.spec.tools) == 2


def test_lint_example_harness_has_no_errors():
    result = load_harness(EXAMPLE)
    findings = lint(result.spec)
    errors = [f for f in findings if f.severity == Severity.error]
    assert errors == [], errors
