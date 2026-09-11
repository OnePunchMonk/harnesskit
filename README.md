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
a real credential-free demo, not just a config file.

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

# Re-score a fixture baseline (see examples/react-web-researcher/eval/fixtures/):
# no adapter, no API key, no network call. Deliberately shows one pass and one
# understandable failure.
harness replay examples/react-web-researcher \
  --baseline examples/react-web-researcher/eval/fixtures/fixture-demo.baseline.json

# Optional provider adapters. Install only the adapter you intend to run.
pip install -e ".[anthropic]"   # add [pydantic-ai] for that adapter
export ANTHROPIC_API_KEY=...

harness init my-agent                              # minimal skeleton, no API call
harness init --spec "a support agent that can escalate to a human"  # scaffolder
harness lint examples/react-web-researcher         # static analysis
harness inspect examples/react-web-researcher
harness run examples/react-web-researcher --input "What is 2+2?"
harness eval examples/react-web-researcher         # run the eval suite
harness eval examples/react-web-researcher --adapter pydantic_ai
harness eval examples/react-web-researcher --save-baseline v1
harness eval examples/react-web-researcher --compare v1   # regression gate
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

## Layout

```
src/harnesskit/
  format/     HarnessSpec — the typed harness.yaml model tree
  parser/     loads a harness dir into a validated HarnessSpec
  linter/     static analysis rules (infinite loops, missing termination, ...)
  trace/      trajectory capture, cost estimation, run + baseline storage
  eval/       scorers, run/replay/compare/regression-gate, MockAdapter for tests
  adapters/   translates HarnessSpec -> a runnable agent: raw_api, pydantic_ai
  scaffold/   NL spec -> ScaffoldPlan -> generated harness + self-validation
  packaging/  .harn bundle export/import with checksums, secrets excluded
  cli/        `harness` command, wires everything together
examples/react-web-researcher/   a working example harness.yaml
```
