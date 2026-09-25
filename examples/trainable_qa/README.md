# trainable-qa: `requires_grad` for a harness, offline

This is a deterministic keyword-lookup QA agent, with no model call and no
network access. Its retrieval and answer settings live in `agent_config.json`.
`harness.yaml` marks three of them trainable and freezes everything else:

```yaml
trainable:
  - name: answer_mode
    target: "json:agent_config.json#/answer_mode"
    kind: choice
    choices: [first_sentence, best_sentence]
  - name: min_overlap
    target: "json:agent_config.json#/min_overlap"
    kind: int
    min: 0
    max: 3
  ...
```

The eval cases carry `split: train | val | test`. Run the demo:

```bash
harness params examples/trainable_qa
harness train examples/trainable_qa \
    --adapter custom:examples/trainable_qa/agent.py:make_adapter \
    --proposer random --steps 4 --candidates 3 --out /tmp/qa-train
# or: python examples/trainable_qa/train_demo.py /tmp/qa-train
```

What happens:

1. The starting config fails every case: it always returns the first
   sentence and never abstains.
2. Random search mutates one parameter per candidate. Each candidate is a
   full harness directory under `/tmp/qa-train/candidates/`. A candidate is
   evaluated on train first and on val only if train did not get worse. It
   is accepted only if it beats the current best on val.
3. The selected harness and the original are each run once on the four test
   cases, and the two runs are compared case by case.
4. With seed 0, test pass rate goes from 0.00 to 0.75 and the verdict is
   `improved`. The best harness is in `/tmp/qa-train/best/`, the diff is in
   `best.diff`, and every candidate, including rejected and duplicate ones,
   is in `train_report.json`.

This result only shows that the loop works. It is not a benchmark. There are
four test cases, and selecting on val is itself imperfect here. The config
chosen on val scores 3/4 on test. `strip_stopwords: true` with
`min_overlap: 1` would score 4/4 on test, but it only scores 2/4 on val, so
the trainer correctly never selects it. The callback agent reports no cost, so the report says spend is
incomplete instead of claiming $0. Passing `--max-cost` without
`--allow-unmetered` stops training because the cap cannot be enforced.

To let a model propose edits, including edits to text parameters such as
prompts, skills, or tool descriptions, use `--proposer llm` (requires
`pip install -e ".[anthropic]"` and an API key). You can also pass
`--proposer harness:<dir>` to use your own proposer harness. The proposer
sees parameters and train evidence only, never val or test cases.
