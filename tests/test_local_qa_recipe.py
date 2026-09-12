"""End-to-end test for the local-qa-recipe walkthrough (issue #9, P1 rest):
asserts each of the six steps produces the expected artifact/signal."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

RECIPE_DIR = Path(__file__).parent.parent / "examples" / "local_qa_recipe"


@pytest.fixture
def recipe_module(tmp_path, monkeypatch):
    """Run the recipe against a scratch copy so exported bundles/baselines
    written under .harness/ don't dirty the tracked example directory."""
    workdir = tmp_path / "local_qa_recipe"
    shutil.copytree(RECIPE_DIR, workdir)

    if str(workdir) not in sys.path:
        sys.path.insert(0, str(workdir))
    sys.modules.pop("run_recipe", None)
    sys.modules.pop("agent", None)
    import run_recipe as module

    yield module, workdir

    sys.modules.pop("run_recipe", None)
    sys.modules.pop("agent", None)
    if str(workdir) in sys.path:
        sys.path.remove(str(workdir))


def test_recipe_runs_all_six_steps(recipe_module):
    module, workdir = recipe_module
    artifacts = module.run_all(workdir)

    # 1. baseline run: exactly one intentional failure (nepal_capital).
    baseline_suite = artifacts["baseline_suite"]
    assert len(baseline_suite.results) == 4
    failing_ids = [r.case.id for r in baseline_suite.results if not r.passed]
    assert failing_ids == ["nepal_capital"]

    # 2. inspection surfaces the failure with its wrapped-agent lookup log.
    failure = artifacts["failure"]
    assert failure.case.id == "nepal_capital"
    assert failure.trajectory.final_output is not None
    assert "Kathmandu" not in failure.trajectory.final_output

    # 3. promoted to a named, persisted regression baseline.
    baseline_path = artifacts["baseline_path"]
    assert baseline_path.exists()
    assert baseline_path.name == "regression-nepal_capital.json"

    # 4. replay under a looser scorer changes the score, without any new
    # calls to the wrapped agent.
    replayed_suite = artifacts["replayed_suite"]
    replayed_nepal = next(r for r in replayed_suite.results if r.case.id == "nepal_capital")
    assert replayed_nepal.passed  # now passes: loosened to "mentions Nepal"
    original_nepal = next(r for r in baseline_suite.results if r.case.id == "nepal_capital")
    assert not original_nepal.passed  # original (stricter) result is unchanged
    assert artifacts["calls_after_replay"] == 4  # one call per case from step 1, none added

    # 5. changing the wrapped agent blocks a replay-only claim with an
    # explicit message, rather than silently reusing old trajectories.
    block_message = artifacts["block_message"]
    assert "fresh" in block_message
    assert "v1-keyword-overlap" in block_message
    assert "v2-with-fact-table" in block_message

    # 6. two fresh runs compare cleanly, and the winner exports/re-imports.
    v1_suite, v2_suite = artifacts["v1_suite"], artifacts["v2_suite"]
    assert v2_suite.pass_rate > v1_suite.pass_rate  # v2's fact table fixes nepal_capital
    bundle_path = artifacts["bundle_path"]
    assert bundle_path.exists()


def test_change_harness_and_block_raises_for_real_version_change(recipe_module):
    module, _workdir = recipe_module
    from agent import ImprovedQAAgent, NaiveQAAgent

    message = module.step5_change_harness_and_block(NaiveQAAgent().version, ImprovedQAAgent())
    assert "requires" not in message or "run" in message  # sanity: it's an explanatory message
    assert "v2-with-fact-table" in message


def test_change_harness_and_block_allows_unchanged_agent(recipe_module):
    module, _workdir = recipe_module
    from agent import NaiveQAAgent

    result = module.step5_change_harness_and_block(NaiveQAAgent().version, NaiveQAAgent())
    assert result == "no change detected; replay remains valid"
