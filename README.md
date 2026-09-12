# harnesskit

Infrastructure for building agent harnesses — what pytest is to testing, or
unsloth is to fine-tuning. Not another agent framework: a format, linter,
and eval loop for the harness (loop strategy, tools, guardrails, prompt)
you build on top of an existing framework or the raw API.

See [`harness_toolkit_design_doc.md`](./harness_toolkit_design_doc.md) for
the full design (architecture, all 7 components, milestones, open questions),
and [`AGENTS.md`](./AGENTS.md) for conventions when working on the codebase
itself. Two scope questions are tracked as issues rather than settled in
code: [#1 benchmarking](../../issues/1), [#2 deployment](../../issues/2).

Requires Python 3.10+.

## Status

All seven components from the design doc have a working first version:
format (M1), parser + linter (M2), trace collector + eval engine (M3), two
adapters — raw-API and Pydantic AI v2 (M4), the NL-spec scaffolder (M5), and
CLI polish — watch mode, `.harn` export/import (M6). Eval comparisons are
case-paired and reject mismatched suites, so baselines cannot silently compare
different benchmarks. Adapters report typed capability findings, and `--strict`
on `run`/`eval` rejects unsupported declared behavior before any provider
client or tool callback is initialized. The base suite runs offline;
provider-adapter build checks run only with their optional extras. `harness
replay` re-scores a saved baseline against the current eval suite without
constructing an adapter at all, so the example harness's fixture baseline is
a real credential-free demo, not just a config file. `harness init --template`
and `harness templates` expose the six scaffold templates directly, offline
and deterministically, with no LLM call. `harness show` and `harness eval
--failed-only` pretty-print a trajectory step by step (llm/tool calls,
tokens, cost, stop reason) so a failing case is debuggable without opening
JSON. `harness eval --parallel N` runs independent cases concurrently in a
bounded thread pool. `--cache` (on `run`/`eval`, raw_api adapter only) caches
model-call responses on disk under `.harness/cache/`, keyed on (model,
system prompt, messages, tools), so iterating on scorers/guardrails/eval
assertions costs nothing on a cache hit.

## Try it

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Base package: fully offline, no provider SDK or credentials required.
pytest -q
harness lint examples/react-web-researcher
harness inspect examples/react-web-researcher
harness inspect examples/react-web-researcher --adapter raw_api --json
harness conformance                                # adapter conformance suite
harness templates                                  # list the six built-in scaffold templates
harness init my-agent --template coding_agent      # generate from a template, offline, no LLM call

# Re-score a fixture baseline (see examples/react-web-researcher/eval/fixtures/):
# no adapter, no API key, no network call. Deliberately shows one pass and one
# understandable failure.
harness replay examples/react-web-researcher \
  --baseline examples/react-web-researcher/eval/fixtures/fixture-demo.baseline.json

# Optional provider adapters. Install only the adapter you intend to run.
pip install -e ".[anthropic]"   # add [pydantic-ai] for that adapter
export ANTHROPIC_API_KEY=...

harness init my-agent                              # minimal skeleton, no API call
harness init --spec "a support agent that can escalate to a human"  # scaffolder (LLM call)
harness lint examples/react-web-researcher         # static analysis
harness inspect examples/react-web-researcher
harness run examples/react-web-researcher --input "What is 2+2?"
harness run examples/react-web-researcher --input "..." --cache   # cache model calls under .harness/cache/
harness eval examples/react-web-researcher         # run the eval suite
harness eval examples/react-web-researcher --parallel 4           # run cases concurrently
harness eval examples/react-web-researcher --adapter pydantic_ai
harness eval examples/react-web-researcher --failed-only          # step-by-step trace for failing cases only
harness eval examples/react-web-researcher --save-baseline v1     # also records model/adapter/git-sha/timestamp
harness eval examples/react-web-researcher --compare v1   # regression gate; warns loudly on environment drift
harness show examples/react-web-researcher/.harness/runs/<run-id>.json   # pretty-print a saved trajectory
harness watch examples/react-web-researcher        # re-lint on every file change
harness export examples/react-web-researcher -o react.harn
harness import react.harn --directory ./react-copy # all offline
```

The base test suite skips optional-adapter build tests when their SDK is not
installed. CI runs those tests separately with the relevant extras, so a base
installation never needs provider packages, credentials, network access, or
model downloads.

## Compatibility policy

`AdapterCapabilities.supports()` is a declaration, not a proof — an adapter
can claim to support a loop strategy or hook point and still get it wrong.
`harness run` and `harness eval` warn about unsupported adapter features by
default for compatibility; add `--strict` to reject them before an adapter,
provider client, or tool callback is initialized, using `inspect_support()`
findings (`harness inspect --adapter <name> --json` for the machine-readable
form).

The conformance suite in `harnesskit.testing.conformance` closes the gap
between declaration and behavior: it runs a fixed set of deterministic
scenarios (max-turn termination, explicit-tool/tag-emitted termination, tool
result recording, tool exceptions, missing-callback handling, trace
completeness) against a real `HarnessAdapter.run()`, using a scripted client
(`harnesskit.testing.fakes.FakeAnthropicClient`) and fake tools — no API key,
no network call. `harness conformance` runs it against both adapters (skipping
pydantic_ai automatically when that extra isn't installed) and prints a
support matrix; third-party adapter authors can import `RAW_API_CASES` /
`PYDANTIC_AI_CASES` and `run_case` directly to conformance-test their own
adapter (see the module docstring for a minimal example). A scenario passing
means the adapter's contract holds for that behavior; it does not mean the
adapter's native transcript matches another adapter's byte-for-byte —
runtimes may legitimately differ in how they get there.

The two adapters are not identical on this matrix, and that's the point:
`PydanticAIAdapter` doesn't enforce mid-run `explicit_tool`/`tag_emitted`
termination (its own module docstring says so), and pydantic-ai currently
propagates a raising tool callback as an exception instead of an error
result the model can see — both show up as an honest ❌/⚠️ in
`harness conformance` rather than a silently-passing declaration.

## Evidence integrity and the public API

The things a comparison depends on being trustworthy are enforced, not just
documented:

- **Run storage** uses collision-resistant run ids (timestamp + random
  suffix) and atomic (write-temp-then-rename) writes, so two runs started in
  the same millisecond never overwrite each other and a crash mid-write
  never corrupts a stored run.
- **Named baselines** can't be silently replaced: `save_baseline(...)`
  raises `FileExistsError` unless called with `overwrite=True` (or `harness
  eval --save-baseline NAME --overwrite-baseline` on the CLI). Old
  (pre-schema_version) baseline and run files still load; a baseline saved by
  a newer, incompatible harnesskit raises a specific `IncompatibleArtifactError`
  instead of an opaque crash.
- **Baseline provenance**: `--save-baseline` also records the model id,
  adapter, harness name+version, git sha (best-effort, `None` outside a git
  repo), and timestamp it was captured with. `--compare` loads that record
  and prints a loud warning — not a quiet log line — when the current run's
  own model/adapter/harness-version/git-sha differ from it, so an
  environment change is never silently reported as a harness regression. A
  legacy (pre-provenance) baseline reports "no provenance recorded (legacy
  baseline)" instead of silently skipping the check.
- **Cost accounting** distinguishes observed, estimated, known-zero, and
  unavailable cost (`CostStatus`) — an unknown-model call is reported as
  unavailable, never as `$0`, and `SuiteResult.avg_cost_usd` excludes
  unavailable cases rather than averaging them in as zero
  (`unavailable_cost_case_ids` surfaces which cases those were).
- **Errored/unscored cases stay in the denominator**: `run_suite` catches a
  failing case and keeps it in `SuiteResult.results` with `status=errored`
  instead of dropping it, so `pass_rate` reflects every attempted case.
- **Bundle import** validates the archive's exact member set against its
  checksum manifest (no unlisted extra file, no missing entry, no
  duplicates), rejects unsafe paths (`../`, absolute) and symlinks, and
  checks the bundle format version — all before extracting a single file.
  Checksums prove a file matches the manifest, not that the manifest's
  publisher is trustworthy.

A small, deliberate public API re-exports these pieces (and load/replay/
compare) for direct `import harnesskit` use — see the module docstring in
`src/harnesskit/__init__.py` for the full list. Everything else in
`harnesskit.*` remains CLI-only/internal and may change without notice.

## Layout

```
src/harnesskit/
  format/     HarnessSpec — the typed harness.yaml model tree
  parser/     loads a harness dir into a validated HarnessSpec
  linter/     static analysis rules (infinite loops, missing termination, ...)
  trace/      trajectory capture, cost estimation, run + baseline storage
  eval/       scorers, run/replay/compare/regression-gate, MockAdapter for tests
  adapters/   translates HarnessSpec -> a runnable agent: raw_api, pydantic_ai, callback
  scaffold/   NL spec -> ScaffoldPlan -> generated harness + self-validation
  packaging/  .harn bundle export/import with checksums, secrets excluded
  cli/        `harness` command, wires everything together
examples/react-web-researcher/   a working example harness.yaml
examples/local_qa_recipe/        the local debugging walkthrough below
```

## Bring your own agent, and the local debugging walkthrough

`harnesskit.adapters.callback_adapter.CallbackAdapter` wraps an existing,
already-working agent — anything with a plain `(question) -> answer`
entrypoint — without rewriting its internal loop. It only *observes* that
agent's final output and wall-clock duration; it cannot enforce the wrapped
agent's own timeouts or spend limits, and `CallbackAdapter.feature_status()`
labels each capability explicitly as `controlled`, `observed`, or
`unavailable` rather than implying parity with a fully harnesskit-run
adapter (see `harnesskit.adapters.base.FeatureStatus`).

`examples/local_qa_recipe/` puts this to work end-to-end: a naive
keyword-lookup QA agent over a tiny fixture text corpus, deliberately
non-financial and generic (not a specialized research task). Run
`python examples/local_qa_recipe/run_recipe.py` (or read
`examples/local_qa_recipe/README.md`) for the ~15-minute walkthrough: run
the offline recipe, inspect an intentionally-failed case, promote it to a
reviewed regression baseline, change the scorer and replay (same
trajectories, different score, no new agent calls), attempt to change the
agent and get an explicit "this requires a fresh run" error instead of a
silently-reused old score, then compare two fresh runs and export/re-import
the winner as a `.harn` bundle. `tests/test_local_qa_recipe.py` asserts each
step's artifact/signal.
