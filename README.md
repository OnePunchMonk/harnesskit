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

# Any model behind an OpenAI-compatible AI gateway (LiteLLM, OpenRouter, Portkey,
# Vercel AI Gateway, vLLM, Ollama, ...). No provider SDK needed.
export HARNESSKIT_GATEWAY_BASE_URL=https://openrouter.ai/api/v1   # or http://localhost:4000 for LiteLLM
export HARNESSKIT_GATEWAY_API_KEY=...
harness eval examples/react-web-researcher --adapter gateway

# Let your coding assistant do the abstracting: install the harnesskit Agent Skill.
harness skill install --dest .claude/skills        # any skills-compatible assistant; see `harness skill show`

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

`inspect_support`/`check_support` also cover a second, distinct kind of gap:
features that are declared and load/lint fine but are silently not enforced
at runtime by a given adapter — guardrails, `context.compaction`,
`model.routing`, `loop.max_tool_calls`, and termination condition types not
in the adapter's declared `enforced_*` sets. These print as "declared but
not enforced at runtime by adapter '<runtime>'" rather than "unsupported",
and `harness lint` flags them too (`declared-but-unenforced`), naming which
adapter(s) won't honour them. This is visibility only — harnesskit does not
enforce any of these at runtime yet; see the design doc for the planned
shared runtime layer.

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

## Gateway models and the abstractor workflow

The intended workflow is:
1. A team picks a cheap, low-latency model behind an AI gateway.
2. They describe their business use case to a coding assistant.
3. The coding assistant, acting as the **abstractor** (not the doer), turns
   that description into a harness.
4. harnesskit does the rest.

Two pieces support this workflow:

- **`GatewayAdapter`** (`--adapter gateway`) runs a harness against any
  OpenAI-compatible Chat Completions endpoint. It uses only the standard
  library and needs no SDK.
  - Configuration comes from `HARNESSKIT_GATEWAY_BASE_URL` and
    `HARNESSKIT_GATEWAY_API_KEY`, falling back to `OPENAI_BASE_URL` and
    `OPENAI_API_KEY`. It is read at build time, so `harness inspect` never
    needs credentials.
  - It enforces `max_turns`, `max_tool_calls`, and `explicit_tool` and
    `tag_emitted` termination.
  - Each step records the model the gateway actually served.
  - Cost is **observed** when the gateway reports it (OpenRouter's
    `usage.cost`, LiteLLM's `x-litellm-response-cost` header). Otherwise it
    is estimated from the pricing table, or marked unavailable. It is never
    reported as `$0`.
  - A failed request stays on the trace, and the case counts as errored.
  - `harness conformance` runs every raw_api scenario against the gateway
    adapter. It also runs gateway-specific scenarios for cost, served model,
    malformed tool arguments, and `max_tool_calls`.
- **The harnesskit Agent Skill** (`harness skill install`, source in
  `src/harnesskit/skill/`) teaches a skills-compatible coding assistant the
  abstractor workflow:
  1. Pin down the use case and mechanical success criteria.
  2. Scaffold the harness from a template.
  3. Write bounded tools.
  4. Write an outcome-asserted eval suite with splits, and a human-review
     list for uncertain expectations.
  5. Declare `trainable:` parameters.
  6. Verify with `lint`, `inspect`, `params`, and `conformance`.
  7. Get an explicit budget, then run `eval` and `train` on the gateway.
  8. Report the verdict as is.

  The skill forbids hard-coding answers, editing eval cases to pass, and
  spending money without a budget.

## Training a harness (`requires_grad` for harness components) — experimental

A harness can declare which of its components an optimizer may edit, the
harness equivalent of `requires_grad=True`:

```yaml
trainable:
  - name: system_prompt
    target: scaffold.system_prompt        # the prompt file
  - name: style_skill
    target: file:skills/answer_style.md   # any file: a skill, prompt fragment, or code component
  - name: search_description
    target: json:tools/search.json#/description   # tool descriptions, never their schema
  - name: max_turns
    target: loop.max_turns
    kind: int
    min: 2
    max: 12
  - name: model
    target: model.model_id
    kind: choice
    choices: [claude-haiku-4-5-20251001, claude-sonnet-5]
```

Anything not declared is **frozen**. That includes tools and their
permissions, guardrails, termination, eval cases, `harness.yaml` as a file,
anything under `eval/`, and whole tool schemas. `HarnessModule.materialize()`
writes each candidate as a new harness directory, reloads it, and rejects it
if anything outside the declared targets changed. This means a proposer
cannot quietly change what a tool does or what the eval measures.

`harness train` (or `harnesskit.train.Trainer`) runs a bounded loop:

1. Build **evidence** from the current best harness's *train* cases: failing
   inputs, expected checks, actual outputs, failed scorers, and tool calls.
   This plays the role of a backward pass, but it is not a gradient.
2. A **proposer** turns that evidence into edits. `random` is equal-budget
   random search over bounded choice and numeric parameters (the baseline to
   beat). `scripted:<file.json>` is a list of hand-written candidates. `llm`
   and `harness:<dir>` use a *proposer harness* run through any adapter.
   Because the proposer is itself a harness, its own prompt can be declared
   trainable and optimized the same way, which is the recursive case.
3. Each candidate is evaluated on train, then on **val** only if train did
   not get worse. It is accepted only if it beats the current best on val.
4. The original and the selected harness are each run **once** on the
   untouched **test** split and compared case by case. The verdict is
   `improved` only when the test gain's paired 95% CI excludes zero. A
   positive but uncertain gain is `inconclusive`. Everything else is
   `no_supported_improvement`, which is a valid and common outcome.

Splits come from a per-case `split: train|val|test`, or from a seeded hash of
case ids when no case declares one. Every candidate (accepted, rejected,
invalid, or duplicate), its diff, and every evaluation and proposal cost go
into `train_report.json`. `--max-cost` is an admission check: it compares
spend so far plus the largest evaluation seen so far against the cap. It is
not a guaranteed bill cap, and any overrun is recorded. If some cost is
unavailable, training stops rather than pretend the cap is enforced, unless
`--allow-unmetered` is passed. Exporting a trained harness always includes
its trainable files.

Optimizer features, all offline-tested:

- **Failure attribution.** Each trainable parameter receives the train
  failures it plausibly affected:
  - loop limits get failures that stopped on that limit;
  - a tool description gets failures that called or expected that tool;
  - prompts and files get every failure.

  This is a heuristic, labeled as one in the proposer input.
- **Optimizer memory.** The proposer sees past candidates' edits, outcomes,
  and train scores, but never val scores. Every decision is also appended to
  `history.jsonl` in the work directory.
- **Incremental text edits.** A text parameter can be edited with
  `{"op": "append"|"prepend", "text": ...}` or
  `{"op": "replace", "old": ..., "new": ...}` instead of a full rewrite.
  This follows ACE, and avoids eroding a long prompt through repeated
  rewrites.
- **`--selection pareto`.** Samples the next parent from the per-case Pareto
  front of train outcomes, as GEPA does. A candidate that alone solves some
  case keeps being explored. The default is `greedy`.
- **`--screen-size N`.** Evaluates a candidate on N train cases first and
  finishes the train evaluation only if it isn't worse there. Screening
  costs are recorded like any other evaluation.

Proposers can run through the gateway too, for example `--proposer llm
--proposer-adapter gateway --proposer-model <model>`.

`examples/trainable_qa/` is an offline, credential-free demo in which random
search finds a better retrieval config (see its README for what that result
does and doesn't show). `harness params <dir>` lists a harness's parameters.

## Layout

```
src/harnesskit/
  format/     HarnessSpec — the typed harness.yaml model tree
  parser/     loads a harness dir into a validated HarnessSpec
  linter/     static analysis rules (infinite loops, missing termination, ...)
  trace/      trajectory capture, cost estimation, run + baseline storage
  eval/       scorers, run/replay/compare/regression-gate, MockAdapter for tests
  adapters/   translates HarnessSpec -> a runnable agent: raw_api, gateway (OpenAI-compatible), pydantic_ai, callback
  scaffold/   NL spec -> ScaffoldPlan -> generated harness + self-validation
  packaging/  .harn bundle export/import with checksums, secrets excluded
  train/      trainable parameters, evidence, proposers, bounded train/val/test optimizer
  skill/      the harnesskit Agent Skill (SKILL.md) for coding assistants
  cli/        `harness` command, wires everything together
examples/react-web-researcher/   a working example harness.yaml
examples/local_qa_recipe/        the local debugging walkthrough below
examples/trainable_qa/           offline `harness train` demo (trainable params, splits, verdicts)
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
