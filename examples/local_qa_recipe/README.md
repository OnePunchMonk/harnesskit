# local-qa-recipe

A ~15-minute walkthrough of the local debugging workflow (issue #9, P1),
using a plain callback-based agent over a tiny fixture text corpus —
deliberately generic (local lookup/extraction/QA), not a specialized
research task.

`agent.py` is written as if it were someone else's existing code: a
`NaiveQAAgent` class whose only public surface is `answer_question(question)
-> str`. We never touch its internals — we wrap that one method with
`harnesskit.adapters.callback_adapter.CallbackAdapter`.

Run it:

```bash
python examples/local_qa_recipe/run_recipe.py
```

## The six steps

1. **Run a shipped offline recipe.** `step1_run_baseline` wraps
   `NaiveQAAgent.answer_question` with `CallbackAdapter` and runs it over the
   harness's four eval cases via `harnesskit.eval.engine.run_case` — no
   network call, no API key. One case (`nepal_capital`) fails on purpose:
   the corpus mentions Nepal but never its capital, so a keyword-overlap
   agent has no way to answer it correctly.

2. **Inspect the failure.** `CallbackAdapter` only observes the wrapped
   agent's final output and wall-clock duration (see its module docstring:
   it does *not* get tool/step-level tracing for an arbitrary external
   callback). `NaiveQAAgent` keeps its own `call_log` of which document and
   sentence it picked per question; `step2_inspect_failure` reads that log
   alongside the failed case's expected vs. actual output.

3. **Promote to a regression case.** The failing case already has an
   explicit `expected_output_contains: ["Kathmandu"]` — that's what makes it
   a reviewed regression, not just a fluke. `step3_promote_to_regression`
   saves its trajectory as a named baseline (`regression-nepal_capital`) so
   future changes are compared against this exact recorded failure.

4. **Change the scorer, replay, no new calls.** `step4_change_scorer_and_replay`
   loosens `nepal_capital`'s assertion from an exact fact match to "did the
   answer at least mention Nepal" and re-scores the *same* cached
   trajectories. The wrapped agent's `call_log` length is asserted
   unchanged — the score moved, the agent did not run again.

5. **Changing the agent blocks a replay-only claim.** `step5_change_harness_and_block`
   simulates swapping in `ImprovedQAAgent` (a different `version`) and shows
   that comparing its behavior requires a fresh run: attempting to treat old
   trajectories as evidence for the new agent raises
   `HarnessChangeRequiresFreshRunError` with an explicit message, instead of
   silently reusing stale data.

6. **Compare two fresh runs, export the winner.** `step6_compare_and_export`
   runs `NaiveQAAgent` (v1) and `ImprovedQAAgent` (v2, which adds a small
   fact-table fallback) fresh, End-to-end, and compares them with
   `harnesskit.eval.engine.compare`. It then saves v2's trajectories as a
   baseline, exports the harness as a `.harn` bundle
   (`harnesskit.packaging.bundle.export_bundle`), and re-imports it into a
   scratch directory to prove the exported artifact round-trips.

See `tests/test_local_qa_recipe.py` for an automated assertion of each step's
artifact/signal.
