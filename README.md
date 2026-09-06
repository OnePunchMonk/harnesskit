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

M1 (format), M2 (parser + linter), and M3 (trace + eval engine) are working.
M4 has two adapters: a raw-API adapter (no framework) and a Pydantic AI v2
adapter. See the milestone map in the design doc for what's next (a second
framework adapter to stress-test the format further, then the scaffolder).

## Try it

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[anthropic]"   # add [pydantic-ai] too if you want that adapter
export ANTHROPIC_API_KEY=...

harness init my-agent                          # scaffold a new harness
harness lint examples/react-web-researcher     # static analysis
harness inspect examples/react-web-researcher
harness run examples/react-web-researcher --input "What is 2+2?"
harness eval examples/react-web-researcher     # run the eval suite
harness eval examples/react-web-researcher --adapter pydantic_ai
harness eval examples/react-web-researcher --save-baseline v1
harness eval examples/react-web-researcher --compare v1   # regression gate
pytest
```

## Layout

```
src/harnesskit/
  format/    HarnessSpec — the typed harness.yaml model tree
  parser/    loads a harness dir into a validated HarnessSpec
  linter/    static analysis rules (infinite loops, missing termination, ...)
  trace/     trajectory capture, cost estimation, run + baseline storage
  eval/      scorers, run/replay/compare/regression-gate, MockAdapter for tests
  adapters/  translates HarnessSpec -> a runnable agent: raw_api, pydantic_ai
  scaffold/  NL spec -> generated harness directory            [not yet built]
  cli/       `harness` command, wires everything together
examples/react-web-researcher/   a working example harness.yaml
```
