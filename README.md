# harnesskit

Infrastructure for building agent harnesses — what pytest is to testing, or
unsloth is to fine-tuning. Not another agent framework: a format, linter,
and eval loop for the harness (loop strategy, tools, guardrails, prompt)
you build on top of an existing framework or the raw API.

See [`harness_toolkit_design_doc.md`](./harness_toolkit_design_doc.md) for
the full design (architecture, all 7 components, milestones, open questions).

## Status

M1 (format) and M2 (parser + linter) are working. See the milestone map in
the design doc for what's next (trace/eval engine, adapters, scaffolder).

## Try it

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .

harness init my-agent          # scaffold a new harness
harness lint examples/react-web-researcher   # static analysis
harness inspect examples/react-web-researcher
pytest
```

## Layout

```
src/harnesskit/
  format/    HarnessSpec — the typed harness.yaml model tree
  parser/    loads a harness dir into a validated HarnessSpec
  linter/    static analysis rules (infinite loops, missing termination, ...)
  eval/      trajectory scoring + A/B + regression gates      [not yet built]
  adapters/  translates HarnessSpec -> a runnable agent per framework  [not yet built]
  trace/     trajectory capture + cost tracking                [not yet built]
  scaffold/  NL spec -> generated harness directory            [not yet built]
  cli/       `harness` command, wires everything together
examples/react-web-researcher/   a working example harness.yaml
```
