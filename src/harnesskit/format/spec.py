"""Typed representation of a harness.yaml — the HarnessSpec model tree.

Every other component (linter, eval engine, scaffolder, adapter) consumes
this typed object, never raw YAML. See format/loader.py for how a harness
directory becomes a HarnessSpec.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class Metadata(BaseModel):
    name: str
    version: str = "0.1.0"
    author: str | None = None
    license: str | None = None
    description: str | None = None


class ModelStep(str, Enum):
    plan = "plan"
    act = "act"
    verify = "verify"
    summarize = "summarize"


class ModelConfig(BaseModel):
    provider: str = "anthropic"
    model_id: str = "claude-sonnet-5"
    temperature: float = 0.0
    min_context: int | None = None
    requires_tool_use: bool = True
    # per-step overrides, e.g. {"plan": "claude-opus-5", "act": "claude-haiku-4-5-20251001"}
    routing: dict[ModelStep, str] = Field(default_factory=dict)


class LoopStrategyType(str, Enum):
    react = "react"
    plan_execute = "plan_execute"
    rewoo = "rewoo"
    code_action = "code_action"
    custom = "custom"


class LoopConfig(BaseModel):
    type: LoopStrategyType = LoopStrategyType.react
    max_turns: int = 15
    max_tool_calls: int | None = None
    custom_ref: str | None = None  # "custom:strategy.py:MyStrategy" when type == custom


class ToolSource(str, Enum):
    builtin = "builtin"
    mcp = "mcp"
    custom = "custom"


class ToolDef(BaseModel):
    name: str
    source: ToolSource
    # builtin: tool name from the built-in registry (filesystem, shell, browser)
    # mcp: server URL + optional capability filter
    # custom: path to a JSON schema file, optionally paired with a Python callback
    ref: str
    permissions: list[str] = Field(default_factory=list)


class GuardrailTrigger(str, Enum):
    pre_tool_call = "pre_tool_call"
    post_tool_call = "post_tool_call"
    pre_turn = "pre_turn"
    post_turn = "post_turn"
    on_budget_check = "on_budget_check"


class GuardrailEnforce(str, Enum):
    cancel_tool = "cancel_tool"
    retry_with_reflection = "retry_with_reflection"
    abort = "abort"
    ask_user = "ask_user"


class GuardrailRule(BaseModel):
    name: str
    trigger: GuardrailTrigger
    check: str  # e.g. "dedup_detection", "cost_threshold:2.00", "pii_scan", "custom:guardrails.py:my_check"
    enforce: GuardrailEnforce
    message: str | None = None


class CompactionStrategy(str, Enum):
    none = "none"
    sliding_window = "sliding_window"
    observation_masking = "observation_masking"
    auto_summarize = "auto_summarize"


class ContextConfig(BaseModel):
    budget_tokens: int | None = None
    compaction: CompactionStrategy = CompactionStrategy.none


class MemoryConfig(BaseModel):
    session: Literal["none", "in_memory", "persistent"] = "in_memory"
    persistent_backend: str | None = None


class TerminationCondition(BaseModel):
    type: Literal["explicit_tool", "idle_detection", "timeout", "budget_exhaustion", "tag_emitted", "max_turns"]
    value: str | int | None = None


class ScoringMode(str, Enum):
    exact_match = "exact_match"
    semantic_match = "semantic_match"
    trajectory_exact = "trajectory_exact"
    trajectory_in_order = "trajectory_in_order"
    trajectory_any_order = "trajectory_any_order"
    llm_judge = "llm_judge"


class EvalCase(BaseModel):
    id: str
    input: str
    expected_tools: list[str] = Field(default_factory=list)
    expected_output_contains: list[str] = Field(default_factory=list)
    max_turns: int | None = None
    max_cost_usd: float | None = None
    scoring_mode: ScoringMode = ScoringMode.exact_match
    ground_truth: str | None = None
    # Which partition this case belongs to when the harness is trained
    # (`harness train`): "train" cases feed failure evidence to the proposer,
    # "val" cases select among candidates, "test" cases are evaluated once at
    # the end. Ignored by `harness eval`, which always runs every case.
    split: Literal["train", "val", "test"] | None = None


class EvalConfig(BaseModel):
    cases: list[EvalCase] = Field(default_factory=list)
    cases_file: str | None = None  # path to a jsonl file of EvalCase records
    thresholds: dict[str, float] = Field(default_factory=dict)


class ParameterKind(str, Enum):
    text = "text"
    int = "int"
    float = "float"
    choice = "choice"


class TrainableParam(BaseModel):
    """One harness component the optimizer may edit — the harness analogue of
    a tensor with ``requires_grad=True``. Everything not declared here is
    frozen; see ``harnesskit.train.params`` for the supported ``target``
    forms and the components that can never be made trainable."""

    name: str
    target: str  # "scaffold.system_prompt", "loop.max_turns", "file:skills/x.md", "json:agent.json#/mode", ...
    kind: ParameterKind = ParameterKind.text
    requires_grad: bool = True
    description: str | None = None  # what the parameter means, shown to proposers
    choices: list[bool | int | float | str] | None = None  # bool first: keeps YAML true/false from becoming 1/0
    min: float | None = None
    max: float | None = None
    max_chars: int | None = None


class HookPoint(str, Enum):
    pre_tool = "pre_tool"
    post_tool = "post_tool"
    pre_turn = "pre_turn"
    post_turn = "post_turn"
    on_error = "on_error"


class Hook(BaseModel):
    point: HookPoint
    ref: str  # "custom:hooks.py:my_hook"


class ScaffoldConfig(BaseModel):
    system_prompt: str  # inline text OR a path reference resolved by the loader
    system_prompt_is_file: bool = True
    template_vars: list[str] = Field(default_factory=list)


class PackagingConfig(BaseModel):
    include: list[str] = Field(default_factory=lambda: ["tools/", "skills/", "guardrails/", "eval/", "hooks/", "AGENTS.md"])
    exclude: list[str] = Field(default_factory=lambda: [".harness/runs/", "__pycache__/", ".env"])


class HarnessSpec(BaseModel):
    schema_version: int = 1
    metadata: Metadata
    model: ModelConfig = Field(default_factory=ModelConfig)
    loop: LoopConfig = Field(default_factory=LoopConfig)
    tools: list[ToolDef] = Field(default_factory=list)
    guardrails: list[GuardrailRule] = Field(default_factory=list)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    termination: list[TerminationCondition] = Field(default_factory=list)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    hooks: list[Hook] = Field(default_factory=list)
    scaffold: ScaffoldConfig
    packaging: PackagingConfig = Field(default_factory=PackagingConfig)
    trainable: list[TrainableParam] = Field(default_factory=list)

    # populated by the loader, not part of the YAML itself
    source_dir: Path | None = Field(default=None, exclude=True)

    model_config = {"arbitrary_types_allowed": True}
