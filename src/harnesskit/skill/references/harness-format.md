# harness.yaml reference (schema_version 1)

```yaml
schema_version: 1
metadata: {name: ticket-triage, version: 0.1.0, description: "..."}

model:
  provider: gateway            # free-form label
  model_id: "openrouter/some-cheap-model"   # sent verbatim to the gateway
  temperature: 0.0

loop:
  type: react                  # the gateway and raw_api adapters run react
  max_turns: 8
  max_tool_calls: 12           # enforced by the gateway adapter

tools:
  - name: lookup_order
    source: custom
    ref: tools/lookup_order.json   # {name, description, input_schema}; sibling .py has run(**kwargs) -> str
    permissions: [read:orders]

termination:
  - {type: explicit_tool, value: submit}   # the run ends when the model calls `submit`; its args become the output
  - {type: tag_emitted, value: done}       # or ends when <done> appears in text
  - {type: max_turns}

guardrails: []                 # declared only; check `harness inspect` for enforcement
memory: {session: none}

eval:
  cases:
    - id: refund_over_limit
      split: train             # train | val | test (all cases or none)
      input: "Customer asks for a $900 refund on order 1234"
      expected_output_contains: ["escalate"]
      expected_tools: [lookup_order]
      max_turns: 6
  cases_file: eval/cases.jsonl # optional; one EvalCase per line (Inspect's `target` accepted)

trainable:                     # requires_grad; everything else is frozen
  - {name: system_prompt, target: scaffold.system_prompt}
  - {name: lookup_desc, target: "json:tools/lookup_order.json#/description"}
  - {name: turns, target: loop.max_turns, kind: int, min: 2, max: 12}
  - {name: model, target: model.model_id, kind: choice, choices: [cheap-a, cheap-b]}

scaffold:
  system_prompt: system_prompt.md   # path (default) or inline with system_prompt_is_file: false

packaging:
  include: [tools/, skills/, eval/, AGENTS.md]
```

Trainable targets:
- `scaffold.system_prompt`, `loop.max_turns`, `loop.max_tool_calls`,
  `context.budget_tokens`, `model.temperature`, `model.model_id`
- `file:<path>`: any text file inside the harness, except `harness.yaml`,
  the `eval/` directory, `.harness/`, and whole tool schemas
- `json:<path>#<pointer>`: one JSON value. For tool schemas, only
  `.../description` pointers are allowed.

Kinds are `text`, `int`, `float` (both take `min`/`max`), and `choice`
(which takes `choices`). `requires_grad: false` declares a parameter for
visibility but keeps it frozen.
