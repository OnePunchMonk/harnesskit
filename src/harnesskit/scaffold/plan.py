"""The intermediate representation between an NL spec and a generated harness
directory (design doc §6.1). infer.py produces one from a natural-language
description; generate.py turns one into files on disk. Kept as a separate,
inspectable object so a plan can be reviewed or hand-edited before it's
materialized — you're never forced to trust the LLM call blindly.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from harnesskit.scaffold.templates import TEMPLATES

Domain = str  # validated against TEMPLATES keys at generation time, not via Literal, so new templates don't need a code change here


class PlannedTool(BaseModel):
    name: str
    description: str
    properties: dict[str, str] = Field(default_factory=dict)  # property_name -> JSON-schema type


class ScaffoldPlan(BaseModel):
    name: str
    description: str
    domain: Domain
    purpose: str
    tools: list[PlannedTool] = Field(default_factory=list)
    guardrails_needed: list[str] = Field(default_factory=list)
    max_turns: int = 10
    cost_ceiling_usd: float = 0.50
    stop_tag: str = "answer"

    def resolved_domain(self) -> str:
        return self.domain if self.domain in TEMPLATES else "generic"
