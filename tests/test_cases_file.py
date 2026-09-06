from pathlib import Path

from harnesskit.parser import load_harness

EXAMPLE = Path(__file__).parent.parent / "examples" / "react-web-researcher"


def test_cases_file_merges_with_inline_cases():
    result = load_harness(EXAMPLE)
    ids = {c.id for c in result.spec.eval.cases}
    assert "capital-lookup" in ids  # inline
    assert "elevation-lookup" in ids  # from eval/cases.jsonl


def test_cases_file_maps_target_to_ground_truth():
    result = load_harness(EXAMPLE)
    case = next(c for c in result.spec.eval.cases if c.id == "elevation-lookup")
    assert case.ground_truth == "Kilimanjaro"


def test_missing_cases_file_raises(tmp_path):
    from harnesskit.parser import HarnessLoadError

    harness_dir = tmp_path / "h"
    harness_dir.mkdir()
    (harness_dir / "system_prompt.md").write_text("You are a bot. Task: {{task}}")
    (harness_dir / "harness.yaml").write_text(
        """schema_version: 1
metadata:
  name: broken
eval:
  cases_file: eval/does_not_exist.jsonl
scaffold:
  system_prompt: system_prompt.md
"""
    )
    try:
        load_harness(harness_dir)
        assert False, "expected HarnessLoadError"
    except HarnessLoadError as e:
        assert "cases_file" in str(e)
