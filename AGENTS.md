# Harnesskit: coding-agent instructions and implementation backlog

## Mission

Build **Unsloth for harness engineering**: make building, debugging, evaluating,
and improving agent harnesses accessible, fast, and technically trustworthy.
Earn this through a useful developer workflow and reproducible quality/cost
improvements. Performance gains are hypotheses until measured.

The target workflow is: bring an agent or recipe → establish a baseline →
inspect a failure → change the harness → compare fairly → export a tested artifact.

This file is both repository guidance and an editable task board. It does not
claim that the planned features already exist.

## How to assign work

Give an agent this repository and an instruction such as:

> Read AGENTS.md. Implement HK-03 and its acceptance criteria. Verify prerequisites
> against the code, preserve unrelated changes, run relevant checks, and commit
> the result on a feature branch. Update the task status and evidence here.

Implement the task IDs the user assigns. If asked to work through this backlog,
choose the earliest ready task and proceed in dependency order. Reading this file
alone does not authorize implementing the entire backlog. Do not spawn additional
agents unless the user authorizes delegation.

Task states: `TODO`, `IN_PROGRESS`, `BLOCKED`, `DONE`. Claim a task by recording its
branch in the table. `DONE` means its acceptance criteria have evidence, not merely
that code was written. Distinguish implemented-on-branch from merged/integrated.
For `BLOCKED`, record the exact missing dependency and the next actionable step.

When multiple agents work concurrently, use separate branches/worktrees and
explicit task assignments. Avoid concurrent edits to shared modules. Each agent
updates only its task row and adds a task-specific completion note; reconcile
task-board conflicts during integration without overwriting others' evidence.

## Current baseline and source of truth

This plan was prepared against main commit `be0063a`. Recheck current code and
GitHub state before assuming a gap remains. Existing local branch
`fix/paired-eval-comparisons`, commit `7255ad9`, contains paired-comparison work;
inspect and reuse it through the normal integration workflow instead of duplicating
it. Do not assume that commit has landed in the branch you are using.

- [Detailed research and implementation plan](https://github.com/OnePunchMonk/harnesskit/issues/6#issuecomment-5574017325)
- [Review findings #3](https://github.com/OnePunchMonk/harnesskit/issues/3)
- [Broader roadmap #4](https://github.com/OnePunchMonk/harnesskit/issues/4)
- [Offline replay #5](https://github.com/OnePunchMonk/harnesskit/issues/5)
- [Adapter conformance #6](https://github.com/OnePunchMonk/harnesskit/issues/6)
- `harness_toolkit_design_doc.md`: design context, not proof of implementation.

Follow explicit user instructions first, then this file. Resolve conflicts between
the old design and the task's contract in a short implementation note. Do not add
an unsupported feature merely because the schema or design document names it.

## Repository map

| Path | Responsibility |
|---|---|
| `src/harnesskit/format/spec.py` | Typed harness configuration |
| `src/harnesskit/parser/` | Loading, reference resolution, validation |
| `src/harnesskit/adapters/` | Runtime translation and capability contracts |
| `src/harnesskit/trace/` | Events, usage, run and baseline persistence |
| `src/harnesskit/eval/` | Scoring, replay, comparisons, regression gates |
| `src/harnesskit/linter/` | Structural checks and explicitly heuristic advice |
| `src/harnesskit/scaffold/` | Templates and generated harnesses |
| `src/harnesskit/packaging/` | Portable artifact export/import |
| `src/harnesskit/cli/main.py` | CLI integration; keep reusable logic elsewhere |
| `examples/`, `tests/` | Runnable recipes and verification |

## Engineering rules

- Preserve Python >=3.10 compatibility as declared in `pyproject.toml`.
- Keep the base package usable without provider SDKs, credentials, network, or
  model downloads. Optional integrations belong in explicit dependency extras.
- Use typed contracts and small modules. Keep the CLI thin and share logic with
  the Python API. Add abstractions when required by a concrete supported use case.
- Never silently accept unsupported required runtime behavior. Distinguish
  declared, translated, enforced, observed, and unavailable capabilities.
- Separate hard validation from heuristic lint advice. A tool named `verify`
  does not prove that verification occurred or that its judgment is correct.
- Preserve existing files, uncommitted work, and active branches. Use a feature
  branch. Do not reset, force-push, or rewrite unrelated history.
- Commit scoped changes when requested. Push, open PRs, post comments, deploy, or
  spend money only within the user's authorization. Existing authorization persists;
  do not ask repeatedly for routine implementation steps.
- Do not launch paid model evaluations without an authorized scope and budget.
  Use deterministic fixtures for development and label synthetic results clearly.
- Keep secrets, private prompts, and sensitive tool results out of commits and
  public reports. Record redaction and its effect on reproducibility.
- Checksums establish integrity relative to a manifest, not publisher authenticity.
  A subprocess is not by itself a security sandbox.
- New configuration must have documented semantics, failure behavior, runtime
  support, and tests. Introduce versioned migrations for persisted schema changes.

## Evaluation and execution invariants

1. A missing, unscored, errored, or cancelled case cannot silently become a pass.
2. Compare compatible baseline/candidate cases by stable identity; explain missing
   coverage and keep paired trials paired. Do not silently compare different suites.
3. Record all attempts and costs, including failed calls, tools, judges, routing,
   retries, and optimizer proposals. Label estimates and pricing identities.
4. Reserve admitted inference work before dispatch where the runtime owns execution.
   Reconcile observed cost afterward, including overruns; do not claim a guaranteed
   cloud bill cap when metering or startup costs make it impossible.
5. Re-scoring a trace evaluates new scorers. It does not evaluate how a changed
   prompt, context policy, tool, or guardrail would have changed the trajectory.
6. Cached responses are not independent stochastic trials. Reuse requires complete
   dependency identity; mutable/effectful tools need explicit replay semantics.
7. Keep tuning, candidate selection, and final evaluation separate. Generated tests
   and a candidate's own judge are not sufficient independent acceptance evidence.
8. Pin relevant code, model, adapter, tool, data, scorer, and environment identities.
   Identical seeds do not promise identical behavior across providers/hardware.

## Task board

Dependencies are implementation dependencies. A dependency completed on another
branch must be integrated or explicitly stacked before dependent work starts.

| ID | Task | Depends on | Status | Branch / evidence |
|---|---|---|---|---|
| HK-01 | Baseline verification and offline CI | — | DONE | `feat/hk-01-baseline-ci`: Python 3.12 clean base install: 26 passed, 2 optional-adapter tests skipped; full local environment: 28 passed |
| HK-02 | Scoring correctness and paired comparisons | HK-01 | TODO | Inspect `7255ad9` first |
| HK-03 | Structured capabilities and strict preflight | HK-01 | TODO | Issue #6 |
| HK-04 | Immutable run manifests and storage | HK-02 | TODO | |
| HK-05 | Offline replay CLI and fixture demo | HK-02, HK-04 | TODO | Issue #5 |
| HK-06 | Scripted adapter conformance suite | HK-03 | TODO | Issue #6 |
| HK-07 | Trace inspection and regression-case export | HK-04, HK-05 | TODO | |
| HK-08 | Stable SDK and existing-agent integration | HK-03, HK-04, HK-06 | TODO | |
| HK-09 | Usage accounting and budget admission | HK-03, HK-04 | TODO | |
| HK-10 | First complete recipe and benchmark manifest | HK-05, HK-07, HK-09 | TODO | |
| HK-11 | Observation-masking policy and ablation | HK-06, HK-09, HK-10 | TODO | |
| HK-12 | Resumable bounded parallel evaluation | HK-04, HK-09, HK-10 | TODO | |
| HK-13 | Dependency-aware caching and watch mode | HK-05, HK-11, HK-12 | TODO | |
| HK-14 | Reproducible bundles and import validation | HK-04, HK-08 | TODO | |
| HK-15 | Tool-interface optimization experiment | HK-10, HK-11 | TODO | |
| HK-16 | Conditional verification experiment | HK-09, HK-10 | TODO | |
| HK-17 | Routing and escalation experiment | HK-09, HK-10, HK-16 | TODO | |
| HK-18 | Second-runtime qualification and replication | HK-06, HK-08, HK-10, HK-14 | TODO | |
| HK-19 | Bounded harness optimizer | HK-11, HK-15, HK-16, HK-17, HK-18 | TODO | |
| HK-20 | Release documentation and external pilot | HK-07, HK-08, HK-10, HK-14 | TODO | |

### HK-01 — Baseline verification and offline CI

Inventory supported fields, available tests, and optional dependencies. Establish
an isolated development environment and a CI matrix including Python 3.10 and a
newer supported release. Make optional SDK tests explicit rather than requiring
provider packages for base installation. Document development dependencies.

Accept when base installation/import and offline tests work without keys/network;
optional integration jobs are separately identified; existing failures and test
commands are recorded. Do not mask failures with broad exception handling/skips.

Completion evidence: base `.[dev]` install on Python 3.12 passed without the
Anthropic or Pydantic AI SDKs (`26 passed, 2 skipped`); the installed optional
adapter environment passed all 28 tests. `.github/workflows/ci.yml` runs the
base suite on Python 3.10 and 3.12 and optional adapter builds separately.

### HK-02 — Scoring correctness and paired comparisons

Inspect the existing paired-comparison patch first. Define applicable scorers,
explicit task-outcome assertions, case identity, missing coverage, and threshold
semantics. Implement supported scoring modes or reject unsupported ones clearly.

Accept when assertion-free cases are unscored, wrong task outputs fail, reordered
paired cases compare correctly, missing/duplicate/incompatible cases are handled
explicitly, and bootstrap resampling preserves pairing. Test CLI and library gates.

### HK-03 — Structured capabilities and strict preflight

Return typed support findings with field, requested value, status, reason, and
runtime identity. Add adapter-aware inspect output including machine-readable JSON
and strict run/eval preflight. Document compatibility with existing warning mode.

Accept when unsupported required behavior fails before provider/tool initialization;
inspect output identifies actual limitations; supported configurations still run.

### HK-04 — Immutable run manifests and storage

Introduce versioned run/experiment manifests and collision-resistant IDs. Record
resolved configuration, case identities, relevant revisions, and execution mode.
Preserve run history and baseline provenance; migrate/read legacy artifacts explicitly.

Accept when repeated evaluations do not overwrite each other, incompatible baselines
are diagnosed, legacy behavior is documented/tested, and a run can be reconstructed
from its manifest to the extent its provider/environment permits.

### HK-05 — Offline replay CLI and fixture demo

Expose baseline replay without constructing an adapter. Report expected/found/missing
coverage, with incomplete results failing by default. Ship labeled fixture traces
that include both useful successes and understandable failures.

Accept when an offline user changes an assertion and sees the verdict change;
provider construction is prohibited by the test; replay preserves original artifacts;
the tutorial explains when a fresh run is required. Follow issue #5's criteria.

### HK-06 — Scripted adapter conformance suite

Inject clients/tools and run shared cases against the current adapters. Cover turn
limits, final-tool termination, errors, tool results, trace completeness, and
unsupported features. Generate a support matrix from tested capabilities.

Accept when a deliberately broken adapter fails, unsupported cases are explicit,
and a third-party adapter can reuse the documented helper without real API calls.
Native runtime differences must remain visible rather than being normalized away.

### HK-07 — Trace inspection and regression-case export

Add a terminal view of failed cases, observable failure events, tool arguments/results,
retries, context changes, and costs. Add reviewed regression-case export with editable
expectations and sanitized fixtures. Keep inferred diagnoses labeled as hypotheses.

Accept when a known failure can be located and converted into a replayable test;
sensitive fields follow retention/redaction settings; exported expected outputs are
not automatically inferred as truth from failed model responses.

### HK-08 — Stable SDK and existing-agent integration

Expose a small public API for load, validate, run, replay, and compare. Support an
existing-agent callback/protocol without forcing a YAML rewrite. Distinguish runtime
control from observational integration and explicitly report unavailable events.

Accept when a documented standalone agent is evaluated through the public API,
the CLI reuses that API, and missing internal instrumentation does not imply enforcement.

### HK-09 — Usage accounting and budget admission

Separate input/output/cache token accounting and estimated/measured cost. Version
pricing identities. Add atomic reservations and reconciliation for owned execution,
including failed/retried calls. Define cancellation and partially known usage behavior.

Accept when insufficient budget prevents dispatch, concurrent calls cannot reserve
the same balance, incurred overruns remain recorded, and unknown pricing is explicit.
Do not meter external agents as controlled calls unless integration supports it.

### HK-10 — First complete recipe and benchmark manifest

Build an extraction/local-lookup recipe with deterministic field and final-state
checks. Include fixtures, a live configuration, development/validation/final splits,
stable example IDs, repeated-trial reporting, and an explicit experiment budget.

Accept when the offline walkthrough works and a live report can be reproduced from
its manifest. If live execution lacks authorization, report that criterion pending;
never substitute fixture scores. Report success, total cost, cost per successful
task, failures, and latency with appropriate sample-size limitations.

### HK-11 — Observation masking and ablation

Implement a conservative context-policy interface and masking policy, preserving
provider-valid messages, pending tool pairs, and declared essential information.
Compare against unmanaged context and a simple window on long-dependency examples.

Accept when transformation invariants pass and a budgeted live experiment reports
both quality and cost, including negative results. A claim of improvement must survive
held-out evaluation. A negative result can complete the experiment without enabling
the policy as the default. Summarization can follow as another measured policy.

### HK-12 — Resumable bounded parallel evaluation

Add concurrency limits, provider rate-limit handling, per-case persistence, cancellation,
and experiment resume. Define retry identities and uncertain external completion.

Accept when completed cases are not rerun inadvertently, interrupted cases are visible,
budgets remain correct under concurrency, and wall-time gains are reported separately
from spend. Do not promise exactly-once effects for arbitrary external tools.

### HK-13 — Dependency-aware caching and watch mode

Cache only requests with complete matching identities. Distinguish scorer-only
replay, fixture execution, and fresh execution. Classify edits conservatively.

Accept when model/prompt/tool/schema/settings changes invalidate affected entries;
mutable tools default to non-reuse unless explicitly supported; reports disclose
cache hits; cached outcomes never inflate independent-trial counts.

### HK-14 — Reproducible bundles and import validation

Version the bundle manifest and preserve tested configuration/provenance without
including private traces by default. Validate archive membership against checksums,
names, duplicate entries, supported versions, and resolved extraction boundaries.
Validate fully before making an imported artifact available for execution.

Accept when valid round trips work, unlisted/invalid members and path escapes fail,
partial failures do not leave a runnable partial import, and required dependencies
are explicit. Treat existing public findings in issue #3 as prerequisites to release.

### HK-15 — Tool-interface experiment

Compare generic and structured/bounded tool observations, descriptions, and error
messages while holding tool semantics fixed. Change one factor at a time initially.

Accept when reports include final-state correctness, argument errors, retries, tokens,
and cost across held-out cases. Reject apparent gains caused by changing task meaning.
Record where a second model fails to reproduce the gain.

### HK-16 — Conditional verification experiment

Compare never-verify, always-verify, and deterministic failure-triggered verification.
Begin with schema checks, known expected state, and tests. Measure false accepts and
false rejects against independent outcomes, including wrong-verifier examples.

Accept when verification overhead is counted and selective policies are evaluated at
matched budgets. A policy that saves money by missing required failures is not a win.

### HK-17 — Routing and escalation experiment

Compare strong-model-only, cheap-model-only, and a static routed policy with one
interpretable escalation rule. Account for routing, fallback, and duplicated work.

Accept when results include hard cases and abandoned attempts; success/cost tradeoffs
are explicit; routing savings use whole-request costs rather than per-call prices.

### HK-18 — Second-runtime qualification and replication

Qualify the supported recipe against another runtime using conformance, then have
an independent developer reproduce a report. Prefer already-supported adapters before
adding new framework dependencies. Record version support and observed differences.

Accept when integration instructions and artifacts reproduce supported behavior.
If independent feedback is unavailable, record the pending criterion; an agent's own
rerun is not independent developer validation.

### HK-19 — Bounded harness optimizer

Search a small typed space of prompt/context/retry/verification/routing changes with
an explicit cost cap. Compare original, manual, random-search, and prompt-only
baselines. Persist rejected candidates and produce reviewable diffs and reports.

Accept when selection never tunes on the final holdout, proposal/evaluation expense
is included, semantics/permissions cannot be silently changed, and the optimizer
can return “no supported improvement.” Prefer simple search if richer search loses.

### HK-20 — Release documentation and external pilot

Publish tested setup, SDK examples, schema/capability reference, migrations, and
complete recipes. Observe unfamiliar developers completing the failure-to-fix loop.

Accept when documented commands pass applicable checks, known limitations are clear,
and pilot feedback records setup friction and useful outcomes. The ten-minute offline
onboarding goal is a target to test, not an established claim. External coordination
requires user authorization; prepare the materials before requesting it.

## Reading tied to implementation

Use the linked deep dive for annotated exercises, falsification criteria, and the
full bibliography. Read the relevant sources for your task, not every paper first.
Historical paper results are not guaranteed benefits on current models/workloads.

| Tasks | Reading | Question to answer |
|---|---|---|
| HK-02, HK-10 | [AI Agents That Matter](https://arxiv.org/abs/2407.01502) | Is improvement due to better policy or more compute/selection? |
| HK-02, HK-16 | [τ-bench](https://arxiv.org/abs/2406.12045) | Does the environment reach the right state consistently? |
| HK-03, HK-06 | [Pydantic AI testing](https://pydantic.dev/docs/ai/guides/testing/) | How can runtime behavior be tested without live inference? |
| HK-08, HK-10 | [Inspect agents](https://inspect.aisi.org.uk/agents.html) | Which evaluation infrastructure can be reused? |
| HK-11 | [JetBrains context-management research](https://blog.jetbrains.com/research/2025/12/efficient-context-management/) | Where does masking lose required evidence? |
| HK-12 | [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence) | Which state persists and which external effects can repeat? |
| HK-15 | [SWE-agent](https://arxiv.org/abs/2405.15793) | Does interface design change behavior with the same model? |
| HK-15 | [MCP tools, version 2025-06-18](https://modelcontextprotocol.io/specification/2025-06-18/server/tools) | What are schema, error, and annotation trust semantics? |
| HK-16 | [Agentless](https://arxiv.org/abs/2407.01489) | Does a simple fixed workflow beat the more adaptive policy? |
| HK-17 | [ReWOO](https://arxiv.org/abs/2305.18323) | Which steps can be planned before observations arrive? |
| HK-19 | [DSPy optimizers](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/learn/optimization/optimizers.md), [GEPA](https://arxiv.org/abs/2507.19457) | Does policy search add value over prompt-only/equal-budget random search? |

## Validation and handoff

Use the repository's configured interpreter/environment. Baseline commands are:

```sh
python -m pytest -q
python -m compileall -q src
git diff --check
```

Run targeted tests during implementation and the applicable full offline suite before
handoff. Install dependencies in an isolated environment when authorized; do not
modify global Python. Document any unavailable dependency or pre-existing failure.
For docs-only changes, inspect content/links as relevant and run `git diff --check`;
do not launch provider calls or a full model benchmark.

For each completed task, append a short note below with: task ID, branch, implementation
commit if already known, changed behavior, commands/results, remaining limitations,
and integration status. Commit the final status/evidence update with the work or in
a small follow-up; do not invent a commit hash or repeatedly amend to self-reference.

Before committing, inspect the staged diff and stage only intended task files. Final
handoff should say what changed, what passed, what remains, and the commit/branch.

## Completion notes

No implementation tasks have been claimed or completed through this task board yet.
Existing functionality and the paired-comparison branch must be audited before work.

## User-added tasks

Copy this template, assign the next unused ID, and add a row to the task board:

```text
### HK-XX — Task title
Status: TODO
Priority:
Depends on:
Scope / expected behavior:
Likely files:
Acceptance criteria:
Validation commands / evidence:
Out of scope:
Branch / integration status:
```
