# harnesskit

Infrastructure for building agent harnesses — what pytest is to testing, or
unsloth is to fine-tuning. Not another agent framework: a format, linter,
and eval loop for the harness (loop strategy, tools, guardrails, prompt)
you build on top of an existing framework or the raw API.

See [`harness_toolkit_design_doc.md`](./harness_toolkit_design_doc.md) for
the full design (architecture, all 7 components, milestones, open questions).
Two scope questions are tracked as issues rather than settled in code:
[#1 benchmarking](../../issues/1), [#2 deployment](../../issues/2).

## Status

All seven components from the design doc have a working first version:
format (M1), parser + linter (M2), trace collector + eval engine (M3), two
adapters — raw-API and Pydantic AI v2 (M4), the NL-spec scaffolder (M5), and
CLI polish — watch mode, `.harn` export/import (M6). The base suite runs
offline; provider-adapter build checks run only with their optional extras.

## Try it

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Base package: fully offline, no provider SDK or credentials required.
pytest -q
harness lint examples/react-web-researcher
harness inspect examples/react-web-researcher

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
