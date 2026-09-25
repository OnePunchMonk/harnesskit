---
name: harnesskit
description: Turn a business use case into a tested, trainable agent harness with harnesskit. Use when someone wants an AI agent for a specific job, wants to run it on a cheap model behind an AI gateway (LiteLLM, OpenRouter, Portkey, ...), or wants to evaluate, improve, or ship an existing harness. You are the abstractor. You write the harness, tools, and eval cases. harnesskit runs, measures, and optimizes it.
---

# harnesskit: from use case to shipped harness

Your role is the **abstractor**, not the doer. Don't solve the user's task
yourself or hand-write answers into prompts. Your job is to turn their use
case into a harness (`harness.yaml`, a prompt, tools, and eval cases) so that
a cheap production model can do the task reliably. harnesskit runs,
measures, and trains that harness.

Work through the steps in order. Don't skip the eval step. A harness without
outcome checks can't be improved or trusted.

## 1. Pin down the use case

Ask about, or confirm from the repo, the following. Write the answers into
`USECASE.md` in the harness directory:

- **Job**: one sentence of the form "given X, produce Y" (for example, "given
  a support ticket, draft a reply or escalate").
- **Success**: how to tell, mechanically, that an output is right: a field
  value, a state change, a test that passes, a phrase that must or must not
  appear, an escalation decision.
- **Inputs and data**: where inputs come from, and what data or APIs the
  agent may read or change.
- **Constraints**: latency target, budget per task, privacy and PII rules,
  and actions that need a human.
- **Failure costs**: which mistakes are expensive (a wrong refund is worse
  than a slow reply). This decides which eval cases matter most.

If success can't be checked mechanically for part of the job, say so. Those
cases need human grading (step 4), not a guess.

## 2. Pick the runtime and model

- Default to the **gateway** adapter, which works with any OpenAI-compatible
  endpoint:
  ```bash
  export HARNESSKIT_GATEWAY_BASE_URL=https://<gateway>/v1   # LiteLLM, OpenRouter, Portkey, vLLM, Ollama, ...
  export HARNESSKIT_GATEWAY_API_KEY=...                     # never write keys into files
  ```
- Set `model.model_id` to the gateway's model name. Start with the cheapest,
  lowest-latency model the user will accept. The harness exists so that a
  small model is enough.
- Use `raw_api` only for direct Anthropic access, and `callback` (Python API)
  to wrap an agent that already exists.

## 3. Scaffold and write the harness

```bash
harness templates                                  # pick the closest one
harness init <name> --template <template>          # offline, no LLM call
```

Then edit the files. See `references/harness-format.md` for every field.

- **Prompt** (`system_prompt.md`): state the job, the output format, when to
  use each tool, and when to stop or escalate. Keep facts and business rules
  in files or tools, not buried in the prompt.
- **Tools** (`tools/<name>.json` with an input schema, plus `tools/<name>.py`
  with `run(**kwargs) -> str`): each tool should do one clear thing, return
  bounded and structured output, and return actionable error strings instead
  of raising. List side effects in `permissions`.
- **Loop limits**: set `loop.max_turns`, `loop.max_tool_calls`, and a
  `termination` condition (`explicit_tool` for a submit or finish tool is
  best).
- **Guardrails** and routing are declared features. Check with `harness
  inspect --adapter gateway` which ones the adapter actually enforces, and
  tell the user about any gaps.

## 4. Write the eval suite (the most important step)

See `references/evals.md`. In short:

- Write 20 or more cases, taken from real examples the user gives you
  wherever possible. Include the expensive failure modes from step 1, plus
  cases where the right answer is "I don't know" or "escalate".
- Every case needs an **outcome assertion**: `ground_truth`,
  `expected_output_contains`, `expected_tools`, or a limit. A case without
  one counts as *unscored*, not passing.
- Never produce expected outputs by running a model and copying its answer.
  If you're unsure what the right answer is, add the case to
  `NEEDS_REVIEW.md` and ask the user to confirm it.
- Assign `split: train | val | test` to every case, roughly 50/25/25, or
  leave split off everywhere to get a seeded hash split. Never put test-case
  content into prompts, tools, or skills.

## 5. Declare what may be trained

List the components an optimizer may change under `trainable:`. This is the
harness equivalent of `requires_grad=True`. Everything else is frozen and
enforced frozen.

```yaml
trainable:
  - {name: system_prompt, target: scaffold.system_prompt, max_chars: 6000}
  - {name: lookup_description, target: "json:tools/lookup.json#/description"}
  - {name: playbook, target: "file:skills/playbook.md"}
  - {name: max_turns, target: loop.max_turns, kind: int, min: 2, max: 12}
```

Never make tools' behavior, permissions, guardrails, or eval files
trainable. harnesskit rejects those targets anyway.

## 6. Verify before spending anything

```bash
harness lint <dir>
harness inspect <dir> --adapter gateway --json     # what is actually enforced
harness params <dir>                               # the requires_grad set
harness conformance                                # adapter contract, offline
```

Fix lint errors. Report the "declared but not enforced" gaps to the user.

## 7. Baseline, then train (costs money, so ask first)

Get the user's explicit budget before any live run. Then:

```bash
harness eval <dir> --adapter gateway --save-baseline v0
harness train <dir> --adapter gateway \
    --proposer llm --proposer-adapter gateway --proposer-model <strong-model> \
    --max-cost <budget> --steps 3 --candidates 4 --out runs/train-1
```

Read `runs/train-1/train_report.json`:

- The verdict is `improved` only if the gain on the held-out test cases
  holds up statistically. `inconclusive` and `no_supported_improvement` are
  legitimate outcomes. Report them as they are. Never claim an improvement
  that val alone showed.
- Report the full spend, including proposal calls, and whether any cost was
  unavailable.
- Look at `best.diff` and read every changed line before recommending it.

## 8. Ship

```bash
harness export runs/train-1/best -o <name>.harn   # or the original dir
```

Hand over:
- the harness directory or bundle
- `USECASE.md`
- `NEEDS_REVIEW.md`
- the baseline and train reports
- known gaps (unenforced features, weak eval coverage)

## Rules

- Don't do the task inside the harness. No hard-coded answers and no copied
  eval outputs.
- Don't edit eval cases to make a harness pass. If a case is wrong, tell the
  user.
- Never write secrets, private customer data, or API keys into harness
  files.
- Don't launch paid evaluations or training without an explicit budget.
- A tool named `verify` doesn't prove anything was verified. Only outcome
  assertions count as evidence.
