# Fixture trajectories

`fixture-demo.baseline.json` is a hand-written baseline in the same format
`harness eval --save-baseline` produces — it was never run against a real
model. It exists so `harness replay` has something to re-score offline, with
no API key and no adapter construction:

```bash
harness replay examples/react-web-researcher \
  --baseline examples/react-web-researcher/eval/fixtures/fixture-demo.baseline.json
```

It is a demonstration of the scoring engine, not evidence of model
performance:

- `capital-lookup` calls `search`, then answers with `<answer>Brasília</answer>`
  — passes every configured scorer.
- `elevation-lookup` includes one failed `search` call (a timeout) followed by
  a successful retry — both a successful and a failing tool step — but its
  final answer paraphrases instead of matching `ground_truth` ("Kilimanjaro")
  exactly, so `exact_output` fails. That's the "at least one understandable
  failure" a clean install should be able to see without touching a provider.
