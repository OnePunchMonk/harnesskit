# Harness Toolkit: Design Document

**Codename**: `harnesskit` (placeholder)
**Author**: @avaya
**Date**: September 2026
**Status**: Draft v0.1

---

## 0. Problem Statement

Building a custom agent harness today requires hand-rolling the loop, hand-tuning the prompt, hand-debugging infinite loops and context blowup, and discovering only at the end whether the harness works. This pain is repeatable and generalizable, not domain-specific. General-purpose harnesses (Claude Code, Codex CLI) are too expensive for most business workflows and too generic for specialized tasks. The NLAH paper showed that task-specific harness specs achieve 47.2% success vs 30.4% for generic code harnesses, with LLM calls dropping from 1,200 to 34, confirming that custom, well-specified harnesses are both cheaper and more effective. [33][48]

**This toolkit makes writing, debugging, and iterating on agent harnesses fast and hard-to-get-wrong.** It is to harness development what pytest is to testing or what unsloth is to fine-tuning: the infrastructure layer that turns a slow, artisanal process into a fast, repeatable, quality-controlled one.

---

## 1. Architecture Overview

The toolkit has **seven components** arranged in three tiers:

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│                                                                 │
│  CLI (`harness` command)          Python SDK (programmatic)     │
│  init · lint · eval · ab · watch · export · import · run        │
└─────────────┬───────────────────────────────┬───────────────────┘
              │                               │
┌─────────────▼───────────────────────────────▼───────────────────┐
│                        CORE ENGINE                              │
│                                                                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│  │  Scaffolder   │ │   Linter     │ │     Eval Engine          │ │
│  │  (init)       │ │   (lint)     │ │  (eval / ab / watch)     │ │
│  └──────┬───────┘ └──────┬───────┘ └──────────┬───────────────┘ │
│         │                │                     │                 │
│  ┌──────▼────────────────▼─────────────────────▼───────────────┐ │
│  │              Harness Format Parser + Validator               │ │
│  │              (loads, validates, resolves harness.yaml)        │ │
│  └──────────────────────────┬──────────────────────────────────┘ │
└─────────────────────────────┼───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                     RUNTIME LAYER                               │
│                                                                 │
│  ┌──────────────────────┐  ┌──────────────────────────────────┐ │
│  │  Adapter Registry     │  │  Trace Collector + Cost Tracker  │ │
│  │  (pydantic-ai,        │  │  (captures trajectory, tokens,   │ │
│  │   langgraph,          │  │   cost, timing per step)         │ │
│  │   smolagents,         │  │                                  │ │
│  │   raw-api)            │  │                                  │ │
│  └──────────────────────┘  └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

Each component is a distinct sub-problem. The rest of this document defines them.

---

## 2. Component 1: The Harness Format

### What it is
A declarative specification for an agent harness: a `harness.yaml` file plus a directory structure containing skills, tools, guardrails, eval cases, hooks, and adapters. The format is the foundation; every other component reads it.

### Sub-problems to solve

#### 2.1 Schema Design
Define a YAML schema that can express at minimum four genuinely different harness types (coding agent, document review, data extraction, customer support) without hacks. The schema must cover:

- **Metadata** (name, version, author, license, description)
- **Model configuration** (provider, model ID, temperature, per-step model routing)
- **Loop strategy** (react, plan-then-execute, ReWOO, code-action, custom)
- **Tool registry** (built-in tools, MCP servers, custom tool definitions)
- **Skills** (SKILL.md bundles with progressive-disclosure activation)
- **Guardrails** (runtime enforcement rules, budget limits, dedup detection)
- **Memory configuration** (session memory strategy, persistent memory)
- **Context management** (budget, compaction strategy, preservation rules)
- **Termination conditions** (explicit stop tool, idle detection, timeout, budget exhaustion)
- **Eval criteria** (metrics, scorers, thresholds, regression gates)
- **Hooks** (lifecycle interception points: pre/post tool, pre/post turn, on-error)
- **Scaffold** (system prompt, instance prompt template with variable substitution)
- **Packaging** (what to include/exclude in an exported bundle)

#### 2.2 Loop Strategy Abstraction
The loop is the heart of the harness. The format must represent multiple loop strategies declaratively, because the choice of loop strategy fundamentally changes the harness's behavior, cost, and failure modes. ReAct loops interleave reasoning and action; Plan-then-Execute separates planning from execution; ReWOO decouples reasoning from observation entirely (2 LLM calls regardless of tool count vs N+1 for ReAct); code-action loops (smolagents-style) write Python instead of making JSON tool calls. [1][4][5][48]

Each strategy has different:
- Cost profiles (ReWOO is cheapest for multi-tool tasks)
- Failure modes (ReAct loops infinitely; Plan-then-Execute fails on re-planning)
- Context growth patterns (ReAct grows linearly; code-action can be more compact)
- Appropriate use cases (ReWOO for independent tools; ReAct for dependent observations)

The format needs to express the chosen strategy plus its parameters (max turns, max tool calls, etc.) and the termination conditions specific to that strategy.

**Open question**: Should `custom` strategy allow an arbitrary Python function, or should we define a small DSL for loop composition (e.g., "plan, then for each step: execute + verify + retry")?

#### 2.3 Tool Definition Format
Tools come from three sources:
1. **Built-in tools** (filesystem, shell, browser) with permission scoping
2. **MCP servers** (referenced by URL, with capability filtering)
3. **Custom tools** (defined as YAML schema + Python callback, or as a code file)

The format must represent all three uniformly enough for the linter and eval engine to reason about them (e.g., "this harness has no tool that can verify its output, which is a known failure mode"). MCP is the de facto standard for tool integration with 6,400+ servers in the registry, so the format must compose with it natively, not reinvent it. [7][8]

#### 2.4 Guardrail Rule Format
Inspired by AgentSpec (ICSE 2026), guardrail rules need:
- **Trigger**: when the rule fires (pre_tool_call, post_tool_call, pre_turn, post_turn, on_budget_check)
- **Check**: what condition to evaluate (dedup detection, cost threshold, PII scan, custom predicate)
- **Enforce**: what action to take (cancel_tool with message, retry_with_reflection, abort, ask_user)

AgentSpec demonstrated that this trigger/check/enforce pattern eliminates 90%+ unsafe executions with millisecond overhead. The key design insight is that enforcement happens at the runtime scope where feedback is created, not as optional local parameters. [12][13][14]

#### 2.5 Eval Case Format
Each eval case must specify:
- **Input**: the task to give the agent
- **Expected behavior**: which tools should be called, what the output should contain, max turns, max cost
- **Scoring mode**: exact match, semantic match, trajectory match (exact order, in-order, any-order), LLM-as-judge
- **Ground truth** (optional): for tasks where a deterministic correct answer exists

This should be Inspect-compatible where possible, since Inspect AI (UK AISI) is the most credible eval framework and is used by METR, Anthropic, DeepMind, and xAI. [9][38][39]

#### 2.6 Composability with Existing Standards
The format must reference, not replace:
- **AGENTS.md** for repo-local agent instructions (60k+ repos already use it) [6]
- **SKILL.md** for skill bundles (283k+ skills tracked on SkillsMP) [6][7]
- **MCP** for tool integration [7][8]
- **OpenTelemetry** for trace export [10]

#### 2.7 Validation and Schema Enforcement
A JSON Schema or Pydantic model that validates `harness.yaml` at load time, catching:
- Missing required fields
- Invalid strategy/compaction/termination combinations
- Tool references that don't resolve
- Eval thresholds without corresponding scorers
- Guardrail rules with invalid trigger/check/enforce combinations

---

## 3. Component 2: The Format Parser + Validator

### What it is
The internal library that loads a harness directory, validates it against the schema, resolves all file references, and produces a typed in-memory `HarnessSpec` object that every other component consumes.

### Sub-problems to solve

#### 3.1 File Resolution
Walk the harness directory, resolve all `path:` references in `harness.yaml` to actual files, and report clear errors when files are missing or malformed.

#### 3.2 Typed Internal Representation
A Pydantic model tree (`HarnessSpec`) that represents the fully resolved harness. Every downstream component (linter, eval engine, scaffolder, adapter) works with this typed object, not raw YAML.

#### 3.3 Schema Versioning
The format will evolve. The parser must handle multiple schema versions with clear migration paths. Include a `format_version` field in `harness.yaml` from day one.

#### 3.4 Partial Harness Support
During `harness init`, the scaffolder generates a harness incrementally. The parser must handle partially complete harnesses (e.g., no eval suite yet) with warnings, not errors.

---

## 4. Component 3: The Linter

### What it is
Static analysis of a harness spec to detect known failure modes before the harness ever runs. This is the "hard-to-get-wrong" differentiator. The linter answers: "Is this harness structurally sound, or will it blow up at runtime?"

### Sub-problems to solve

#### 4.1 Infinite Loop Detection
The IAL-Scan paper (arXiv:2607.01641) demonstrated that infinite agentic loops are a structural failure detectable by static analysis. They built an Agent IR and Agentic Loop Dependence Graph (ALDG) to find feedback paths that can repeatedly reach costly operations without effective bounds, scanning 6,549 repos and finding 74 real issues. [14][15]

Port a simplified version of this approach:
- Build a graph of the harness's feedback paths (tool calls that can trigger re-entry, retry logic, sub-agent delegation)
- Check that every cycle in the graph has at least one effective bound (max_turns, max_tool_calls, timeout, dedup detection)
- Flag paths where bounds exist but are unreachable (e.g., a dedup window that's larger than max_turns)

#### 4.2 Missing Termination Conditions
Check that the harness has at least one reliable termination condition and that it's reachable from all states. Common failures:
- No `final_answer` tool defined and no idle detection
- Timeout is the only termination condition (expensive failure mode)
- Termination depends entirely on the LLM deciding to stop (unreliable)

#### 4.3 Unbounded Context Growth
Analyze the loop strategy and context management configuration to detect scenarios where context grows without bound:
- ReAct loop with `compaction: none` and no `context_budget`
- Tools that return unbounded output without truncation
- Skills that are `always_loaded` and collectively exceed the context budget

Context management is a critical sub-problem; approximately 65% of enterprise AI failures trace back to context drift and memory loss during multi-step reasoning. [16][17]

#### 4.4 Tool Coverage Gaps
Check whether the harness has tools that match its stated purpose:
- System prompt mentions "run tests" but no shell tool is available
- System prompt mentions "search the web" but no browser/search tool is configured
- Eval cases expect specific tool calls that aren't registered

#### 4.5 Verifier Gap Detection
A known failure mode: the agent does work but never verifies its output. The linter should check:
- Does the loop strategy include a verification step?
- Is there a tool that can check the agent's own output?
- Do guardrails include any output validation?

#### 4.6 Cost Ceiling Analysis
Estimate the worst-case cost of a single harness run based on:
- max_turns * estimated tokens per turn (from model config)
- max_tool_calls * estimated cost per tool call
- Flag if worst-case cost exceeds the guardrail `cost_ceiling`

#### 4.7 Guardrail Completeness
Check that the harness has guardrails for the most common failure modes:
- Infinite loops (dedup detection or hard turn limit)
- Cost runaway (cost ceiling)
- Sensitive data leakage (PII redaction if tools access external data)
- Unauthorized actions (permission scoping on filesystem/shell tools)

#### 4.8 Output Format
Produce a structured lint report with severity levels (error, warning, info) and actionable fix suggestions, similar to ESLint or ruff output.

---

## 5. Component 4: The Eval Engine

### What it is
The runtime that executes a harness against its eval suite and produces scores. This is the "training loss curve": fast, local, clear pass/fail feedback on every harness change.

### Sub-problems to solve

#### 5.1 Trajectory Capture
Every harness run during eval must produce a complete trajectory: every LLM call (input/output), every tool call (name, args, result), every state transition, timing, token count, and cost. This trajectory is what scorers evaluate. Trajectory evaluation is now standard practice: it scores the path the agent takes, not just the final answer, covering tool correctness, argument correctness, step efficiency, plan adherence, reasoning quality, and safety. [20][21][22][23]

#### 5.2 Scorer Framework
Scorers take a trajectory + expected behavior and produce a normalized score (0.0-1.0). Two types:

**Deterministic scorers** (fast, cheap, preferred):
- Tool call presence/absence (did the agent call the expected tools?)
- Tool call ordering (exact match, in-order match, any-order match)
- Output containment (does the output contain expected strings/patterns?)
- Step count (how many turns vs max allowed?)
- Cost (actual vs budget)
- Latency (wall-clock time)
- Loop health (any duplicate tool calls?)

**LLM-as-judge scorers** (slower, more expensive, for semantic evaluation):
- Task completion (did the agent achieve the goal?)
- Reasoning quality (is the reasoning chain sound?)
- Output quality (is the final output good?)

Each scorer produces a `Score` object with value (0.0-1.0), threshold (pass/fail), and explanation. The Princeton reliability framework defines 12 metrics across 4 dimensions (consistency, robustness, predictability, safety) that should inform the default metric set. [11][22]

#### 5.3 Cheap Eval Modes
The eval loop must be cheap enough to use during development, not just in CI. This requires:

**Trajectory replay**: Re-score a cached trajectory without re-calling the LLM. Change a scorer, see new scores instantly. This is the cheapest operation.

**Mocked tools**: Replace real tool calls with canned responses. Tests loop logic and prompt behavior without real API calls or side effects.

**Sampled runs**: Run 3 of 30 cases for fast iteration; full suite for CI.

**Cached LLM responses**: If the prompt hasn't changed, reuse the previous LLM response. Invalidate on prompt/tool/guardrail changes.

#### 5.4 A/B Comparison
`harness ab` runs two harness versions against the same eval cases and produces a side-by-side comparison: which version is better on which metrics, with confidence intervals. This requires:
- Paired evaluation (same cases, same order, same random seeds where applicable)
- Statistical significance testing (bootstrap CI or similar, since N is often small)
- Clear visual output (table with deltas, colored pass/fail)

#### 5.5 Regression Detection
`harness eval --compare baseline` compares the current run against a stored baseline and fails if any metric regresses beyond the configured tolerance. This is the CI gate. Requires:
- Storing eval run results as versioned artifacts (JSON, one per run)
- Diffing two runs and computing per-metric deltas
- Configurable tolerance per metric (e.g., 5% regression allowed on cost, 0% on safety)

#### 5.6 Watch Mode
`harness watch` re-runs evals on every file change in the harness directory. Like pytest-watch. Must be smart about what to re-run: if only a scorer changed, replay cached trajectories; if the prompt changed, re-run LLM calls; if only a guardrail changed, re-run affected cases.

#### 5.7 Cost Tracking as a First-Class Metric
Every eval run reports cost alongside quality. The "training loss curve" is actually two curves: quality and cost. Businesses optimize for the Pareto frontier of both. Track:
- Total tokens (input + output, per model)
- Total cost (USD, computed from provider pricing)
- Cost per task (for comparison across harness versions)
- Cost breakdown by component (LLM calls vs tool calls)

---

## 6. Component 5: The Scaffolder

### What it is
`harness init` takes a natural-language spec (one sentence to one paragraph describing the agent's purpose) and generates a complete, runnable harness directory. This is the "magic moment" for adoption.

### Sub-problems to solve

#### 6.1 Spec Parsing
Interpret the user's natural-language description to extract:
- What the agent does (purpose, domain)
- What tools it needs (file access, web search, database, APIs)
- What loop strategy fits (simple Q&A = single turn; multi-step research = ReAct; parallel tool use = ReWOO)
- What guardrails are needed (sensitive data? external actions? cost sensitivity?)
- What eval criteria make sense (task completion, tool accuracy, cost)

This is an LLM call. The quality of the scaffold depends on the quality of this extraction.

#### 6.2 Template Library
A set of harness templates (coding agent, document review, data extraction, customer support, research agent, etc.) that the scaffolder selects from and customizes. Templates encode best practices:
- Coding agent: shell + filesystem tools, test-before-approve guardrail, compaction for long sessions
- Document review: file read tools, PII redaction guardrail, semantic scoring
- Data extraction: structured output, schema validation guardrail, exact-match scoring
- Customer support: memory for conversation context, escalation tool, sentiment scoring

#### 6.3 Self-Validation Loop
After generating the harness, the scaffolder runs `harness lint` and `harness eval` (against a minimal smoke-test suite auto-generated with the harness) to validate it. If lint fails, it loops back and fixes the issues. This makes the scaffolder self-correcting.

#### 6.4 Framework-Specific Scaffold
The scaffolder generates an adapter for the chosen framework (e.g., `--framework pydantic-ai`) that wires the harness spec to the framework's API. This means the scaffolder needs to know the target framework's interface.

#### 6.5 Interactive Mode
For complex harnesses, offer an interactive mode where the scaffolder asks clarifying questions:
- "Your agent accesses external APIs. Should tool calls require user approval?"
- "You mentioned cost sensitivity. What's the max cost per task?"
- "Should the agent retry on tool errors, or abort?"

Keep this minimal; default to best-practice answers if the user skips.

---

## 7. Component 6: The Adapter Registry

### What it is
The translation layer between the framework-agnostic `harness.yaml` spec and a specific agent framework's runtime API. The adapter takes a `HarnessSpec` and produces a runnable agent in the target framework.

### Sub-problems to solve

#### 7.1 Adapter Interface
Define a clean, minimal interface that every adapter must implement:

```python
class HarnessAdapter(Protocol):
    def build(self, spec: HarnessSpec) -> RunnableAgent:
        """Convert a harness spec into a runnable agent."""
        ...
    
    def run(self, agent: RunnableAgent, input: str) -> Trajectory:
        """Execute the agent on an input, returning a full trajectory."""
        ...
    
    def supports(self) -> AdapterCapabilities:
        """Declare which harness features this adapter supports."""
        ...
```

#### 7.2 Feature Mapping
Not every framework supports every harness feature. The adapter must declare its capabilities (loop strategies, tool types, hook points, memory backends) and the parser/linter must warn when a harness uses features the target adapter doesn't support.

For example:
- Pydantic AI v2 supports `tool_calls_limit`, typed tools, dependency injection, and structured outputs natively. [3][5]
- LangGraph supports explicit state graphs, checkpointing, interrupts for human-in-the-loop, and parallel node execution. [1]
- smolagents supports code-action loops natively (agent writes Python, not JSON tool calls) but has minimal built-in guardrails. [4]

#### 7.3 First Adapter: Pydantic AI v2
Build this first. Pydantic AI v2 is harness-first, typed, growing, not provider-locked, and its "capabilities as composable units" model maps well to the harness format. [3][5]

Map:
- `loop.strategy: react` to Pydantic AI's default agent loop
- `loop.max_tool_calls` to `tool_calls_limit`
- `tools` to Pydantic AI tool definitions with typed schemas
- `guardrails` to pre/post tool hooks
- `memory` to Pydantic AI's context management
- `scaffold.system_prompt` to the agent's system prompt
- `eval` to test harness using the eval engine

#### 7.4 Second Adapter: LangGraph
Build this second. LangGraph's explicit state graph will force you to discover what's truly framework-agnostic in the format and what you accidentally assumed.

#### 7.5 Raw API Adapter
A fallback adapter that uses raw OpenAI/Anthropic API calls without any framework. This is the simplest adapter and proves the format isn't framework-dependent.

#### 7.6 Adapter Stability
Frameworks ship breaking changes. Pin adapter versions to framework versions and test adapters in CI against framework releases. This is a maintenance burden that must be budgeted for.

---

## 8. Component 7: The Trace Collector + Cost Tracker

### What it is
A lightweight instrumentation layer that wraps every LLM call and tool call during a harness run, capturing the full trajectory with timing, tokens, and cost. This feeds the eval engine and the CLI's reporting.

### Sub-problems to solve

#### 8.1 Trace Schema
Define a trace format that captures:
- **Run metadata**: harness name, version, timestamp, model(s) used, adapter
- **Steps**: ordered list of (step_type, input, output, tokens_in, tokens_out, cost_usd, duration_ms, tool_name, tool_args, tool_result)
- **Aggregates**: total turns, total tool calls, total tokens, total cost, total duration
- **Events**: guardrail firings, compaction events, error/retry events

#### 8.2 Provider-Specific Cost Calculation
Map token counts to USD using provider-specific pricing. Maintain a pricing table (or fetch from a known source) for major providers (OpenAI, Anthropic, Google, Mistral, local models = $0).

#### 8.3 OpenTelemetry Export
Export traces in OTel format for integration with existing observability stacks (Langfuse, Braintrust, Arize, etc.). [10]

#### 8.4 Trace Storage
Store traces as JSON files in the harness directory (e.g., `.harness/runs/`) for replay, comparison, and debugging. Keep this simple; local files, not a database.

#### 8.5 Trace Viewer (Stretch Goal)
A simple terminal-based or browser-based trace viewer that shows the trajectory step-by-step, highlighting:
- Which tools were called and what they returned
- Where guardrails fired
- Where the agent looped or retried
- Token/cost breakdown per step

---

## 9. Component 8: The CLI

### What it is
The user-facing interface that ties all components together. All commands operate on a harness directory.

### Commands

| Command | What it does | Components used |
|---|---|---|
| `harness init <spec>` | Scaffold a new harness from a natural-language spec | Scaffolder, Parser, Linter |
| `harness lint <dir>` | Static analysis for known failure modes | Parser, Linter |
| `harness eval <dir>` | Run the harness against its eval suite | Parser, Adapter, Trace Collector, Eval Engine |
| `harness ab <dir1> <dir2>` | A/B test two harness versions | Parser, Adapter, Trace Collector, Eval Engine |
| `harness watch <dir>` | Re-run evals on file changes | Parser, Adapter, Trace Collector, Eval Engine |
| `harness run <dir> --input <task>` | Execute the harness on a single input | Parser, Adapter, Trace Collector |
| `harness export <dir>` | Package the harness as a portable bundle | Parser |
| `harness import <bundle>` | Unpack a harness bundle | Parser |
| `harness inspect <dir>` | Show harness summary (tools, skills, guardrails, eval metrics) | Parser |

### Sub-problems

#### 9.1 CLI Framework
Use Click or Typer. Produce rich terminal output (tables, colored pass/fail, progress bars for eval runs).

#### 9.2 Configuration
Global config (`~/.harnesskit/config.yaml`) for:
- Default framework (pydantic-ai, langgraph, etc.)
- Default model provider + API key references (never store keys in harness.yaml)
- Default eval settings (sample size, parallelism)

#### 9.3 Error Reporting
Clear, actionable error messages. When something fails, tell the user exactly what went wrong and what to do about it. No stack traces unless `--verbose`.

---

## 10. Cross-Cutting Concerns

### 10.1 Context Management Strategies

Context management is a sub-problem that cuts across the format, linter, and adapter. The format must represent the strategy; the linter must check it's sane; the adapter must implement it.

The landscape of strategies is well-studied and converges on a few patterns. [16][17][18][19]

| Strategy | Token Reduction | Best For | Trade-off |
|---|---|---|---|
| Sliding window | 40-70% | Multi-turn chat | Loses old context entirely |
| Turn summarization | 30-60% | Long conversations | Expensive (requires LLM call); JetBrains found it actually worse than masking |
| Observation masking | 30-50% | Agent workflows | 2.6% higher solve rates, 52% cheaper than summarization |
| RAG retrieval | 50-80% | Document-heavy tasks | Retrieval blind spots |
| Priority-based eviction | 20-50% | Mixed workflows | Complex to tune |
| Prompt caching | 50-90% (effective cost) | Repeated system prompts | Provider-dependent |

Key finding from JetBrains Research: observation masking (replace older environment observations with placeholders while preserving reasoning) achieved 2.6% higher task solve rates while being 52% cheaper than LLM-based summarization. Summarization inadvertently extended agent trajectories by 13-15% by smoothing over failure signals. [16][17]

**Design decision**: The format should support at least `sliding_window`, `observation_masking`, `auto_summarize`, and `none`. The linter should warn when `none` is combined with a high `max_turns` and no `context_budget`.

### 10.2 Model Routing

Many business harnesses don't need a frontier model for every step. The format should support per-step model selection:
- Planning step: strong model (high reasoning)
- Tool selection: cheap model (pattern matching)
- Verification: medium model (checking, not generating)
- Summarization/compaction: cheap model

This directly reduces cost. ReWOO already demonstrates this principle: 2 LLM calls regardless of tool count vs N+1 for ReAct. [1][4]

### 10.3 Packaging / Export Format

A `.harn` bundle is a ZIP file containing:
- `harness.yaml` (the spec)
- All referenced files (skills/, tools/, guardrails/, eval/, hooks/)
- `AGENTS.md` (if present)
- A `manifest.json` with:
  - Bundle format version
  - Required adapter(s)
  - Required MCP servers (URLs, not included)
  - Required model providers (names, not keys)
  - SHA-256 checksums for all files

The bundle does NOT include:
- `.env` or API keys
- Adapter code (framework-specific, not portable)
- Trace data / run history
- `node_modules` / `__pycache__` / other runtime artifacts

### 10.4 Security Model

Harness bundles from untrusted sources can contain:
- Malicious hook code (arbitrary Python in `hooks/`)
- Malicious tool definitions (arbitrary Python in `tools/`)
- Prompt injection in `scaffold.system_prompt`

Mitigations:
- `harness import` shows a diff of all code files before executing anything
- Hooks and custom tools run in a subprocess sandbox by default
- `harness lint` scans for known dangerous patterns in hook/tool code

---

## 11. Dependency Graph / Build Order

```
                    ┌─────────────────┐
                    │  Harness Format  │  ← Define this FIRST
                    │  (Schema + Spec) │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Parser +        │  ← Build second
                    │  Validator       │
                    └───┬────┬────┬───┘
                        │    │    │
              ┌─────────▼┐ ┌▼────▼────────┐
              │  Linter   │ │ Trace        │  ← Build third (parallel)
              │           │ │ Collector    │
              └─────────┬┘ └──────┬───────┘
                        │         │
                        │  ┌──────▼───────┐
                        │  │ Eval Engine   │  ← Build fourth
                        │  │              │     (needs trace collector)
                        │  └──────┬───────┘
                        │         │
              ┌─────────▼─────────▼───────┐
              │  Adapter Registry          │  ← Build fifth
              │  (Pydantic AI first)       │    (needs parser + tracer)
              └─────────┬─────────────────┘
                        │
              ┌─────────▼─────────────────┐
              │  Scaffolder                │  ← Build sixth
              │  (needs lint + eval        │    (needs everything)
              │   for self-validation)     │
              └─────────┬─────────────────┘
                        │
              ┌─────────▼─────────────────┐
              │  CLI                       │  ← Build last
              │  (wraps all components)    │    (wires components together)
              └───────────────────────────┘
```

### Milestone Map

| Milestone | Deliverables | What You Can Demo |
|---|---|---|
| M1: Format | `harness.yaml` schema, 4 hand-written example harnesses, Pydantic model tree | "Here's what a harness spec looks like" |
| M2: Parser + Linter | Load/validate harness dirs, 8+ lint rules with structured output | `harness lint ./my-harness/` catches real problems |
| M3: Trace + Eval | Trajectory capture, 6+ built-in scorers, comparison, regression gates | `harness eval` gives a training-loss-curve experience |
| M4: First Adapter | Pydantic AI v2 adapter, `harness run` works end-to-end | Run a real harness from a spec file |
| M5: Scaffolder | `harness init` from NL spec with lint + smoke-test validation loop | The "magic moment": spec to working harness in 2 minutes |
| M6: CLI + Polish | All commands, watch mode, export/import, error UX | Ship it |

---

## 12. Open Questions

1. **YAML vs TOML vs code-first?** YAML is the most common choice for declarative configs and what SKILL.md uses for frontmatter. TOML is stricter. A pure-Python DSL (like Pydantic models) is more type-safe but harder to edit by hand. Current leaning: YAML with strong validation, because the target user is editing this file manually and YAML is the lingua franca of config files in this ecosystem.

2. **How much should the format encode the prompt?** The `scaffold.system_prompt` field puts the prompt inside the spec, which is good for portability but means prompt changes require spec changes. Alternative: the scaffold section references an external `.md` file, which is more natural for long prompts and aligns with the AGENTS.md pattern.

3. **Should guardrails be runtime-only or also compile-time?** AgentSpec is runtime enforcement. IAL-Scan is static analysis. The linter does static analysis; guardrails do runtime enforcement. These are complementary, and both should be supported, but the user should understand which is which.

4. **How do you handle multi-agent harnesses?** The current format assumes a single agent. Multi-agent harnesses (e.g., researcher + writer + reviewer) need a way to define agent topology, handoff conditions, and shared state. This could be a `topology` section in `harness.yaml` or a separate concern entirely. LangGraph and CrewAI have different approaches. Defer to v0.2 but design the format so it doesn't preclude multi-agent later.

5. **Eval cost budget**: How much should a full eval suite run cost? The target should be < $1 for the full suite during development (using mocked tools + sampled runs + trajectory replay). CI runs can be more expensive. The eval engine needs to surface cost estimates before running.

6. **What about hosted/SaaS eval?** The eval engine should be local-first (no account required, no data sent anywhere) but support export to hosted platforms (Braintrust, LangSmith, Langfuse) for teams that want dashboards and collaboration.

---

## 13. What Would Kill This

- **The format is wrong.** If the format can't express real harnesses cleanly, everything built on it is wasted. Validate against 4+ diverse use cases before writing any tooling code.
- **The eval loop is too slow or too expensive.** If `harness eval` takes 5 minutes or costs $10, nobody uses it during development. Cheap eval modes (replay, mocks, sampling) are essential.
- **Adapter maintenance burden.** If adapters break on every framework release, you spend all your time on maintenance. Pin framework versions, test in CI, and keep the adapter interface minimal.
- **"Just a format, not a tool."** The format alone is not enough. The linter, eval engine, and scaffolder are what make people use the format. Ship tools, not specs.
- **Scope creep into framework territory.** This is a toolkit for building harnesses, not a harness itself. If you start implementing loop strategies, tool dispatch, or memory backends, you're competing with LangGraph/Pydantic AI instead of composing with them.
