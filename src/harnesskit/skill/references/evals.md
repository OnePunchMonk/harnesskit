# Designing an eval suite harnesskit can train against

**Start from real failures.** Collect real inputs and write notes on what a
good answer requires. Group the notes into failure modes (for example,
"misses the escalation rule" or "wrong field format") and make sure each
failure mode has cases. This is Hamel Husain and Shreya Shankar's
error-analysis method; generic benchmarks don't show whether your
application fails.

**Prefer mechanical assertions, in this order:**
1. Final state or tool usage: `expected_tools`, and the arguments of an
   `explicit_tool` submit call, which become the output.
2. `ground_truth` for exact, canonical outputs, such as extracted fields
   serialized as JSON.
3. `expected_output_contains` for required facts or decisions. Keep these
   specific: a check that passes on any long answer is a hole an optimizer
   will find.
4. Resource limits (`max_turns`, `max_cost_usd`) as secondary checks.

**Include negative and abstain cases.** Cover out-of-scope questions,
missing data, and requests that must be escalated. Without them, training
rewards confident guessing.

**Splits.** train gives the optimizer failure evidence, val selects among
candidates, and test is run once at the end. Keep each split representative
of all failure modes. Small splits (fewer than 10 test cases) usually give
`inconclusive`, which is expected.

**Human review.** Criteria drift: people refine what "good" means while
grading, so they can't fully define it upfront (Shankar et al., "Who
Validates the Validators?"). Put uncertain expectations in `NEEDS_REVIEW.md`
and have the user confirm them before training. Never make a failing model
output the expected answer.

**Before training, check:**
- every case has at least one assertion (`harness eval` reports unscored
  cases);
- a trivially bad harness fails most cases (the assertions have teeth);
- no test-case text appears in prompts, tools, or skills.
