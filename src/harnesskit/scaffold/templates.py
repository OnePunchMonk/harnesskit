"""Template library (design doc §6.2) — encodes best practices per domain so
the scaffolder isn't inventing a harness shape from nothing every time. The
NL-spec inference step (infer.py) picks one of these; generate.py fills it in.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolStub:
    name: str
    description: str
    properties: dict[str, str] = field(default_factory=lambda: {"input": "string"})


@dataclass
class Template:
    domain: str
    loop_type: str
    default_tools: list[ToolStub]
    guardrails: list[str]  # labels resolved to real GuardrailRule objects in generate.py
    max_turns: int
    cost_ceiling_usd: float
    system_prompt_hint: str


TEMPLATES: dict[str, Template] = {
    "coding_agent": Template(
        domain="coding_agent",
        loop_type="react",
        default_tools=[
            ToolStub("shell", "Run a shell command and return its output.", {"command": "string"}),
            ToolStub("read_file", "Read a file's contents.", {"path": "string"}),
            ToolStub("write_file", "Write content to a file.", {"path": "string", "content": "string"}),
            ToolStub("run_tests", "Run the project's test suite and return pass/fail output.", {}),
        ],
        guardrails=["cost_ceiling", "dedup", "permission_scoping"],
        max_turns=25,
        cost_ceiling_usd=2.00,
        system_prompt_hint="Always run the test suite before declaring the task complete.",
    ),
    "document_review": Template(
        domain="document_review",
        loop_type="react",
        default_tools=[
            ToolStub("read_document", "Read a document's contents by path or ID.", {"doc_id": "string"}),
            ToolStub("flag_issue", "Record an issue found in the document.", {"location": "string", "issue": "string"}),
        ],
        guardrails=["cost_ceiling", "pii_redaction"],
        max_turns=15,
        cost_ceiling_usd=0.75,
        system_prompt_hint="Never quote sensitive personal information verbatim in your output; redact it.",
    ),
    "data_extraction": Template(
        domain="data_extraction",
        loop_type="react",
        default_tools=[
            ToolStub("read_source", "Read the raw source data to extract from.", {"source_id": "string"}),
            ToolStub("validate_schema", "Validate extracted data against the target schema.", {"data": "string"}),
        ],
        guardrails=["cost_ceiling", "output_schema_validation"],
        max_turns=10,
        cost_ceiling_usd=0.50,
        system_prompt_hint="Emit only structured data matching the target schema — no prose commentary.",
    ),
    "customer_support": Template(
        domain="customer_support",
        loop_type="react",
        default_tools=[
            ToolStub("lookup_order", "Look up an order or account by ID.", {"account_id": "string"}),
            ToolStub("escalate", "Escalate the conversation to a human agent.", {"reason": "string"}),
        ],
        guardrails=["cost_ceiling", "dedup"],
        max_turns=12,
        cost_ceiling_usd=0.30,
        system_prompt_hint="Escalate to a human rather than guessing when you are not confident or the user is upset.",
    ),
    "research_agent": Template(
        domain="research_agent",
        loop_type="react",
        default_tools=[
            ToolStub("search", "Search the web and return top results.", {"query": "string"}),
            ToolStub("verify_citation", "Confirm a claim is supported by a fetched source.", {"claim": "string", "source_url": "string"}),
        ],
        guardrails=["cost_ceiling", "dedup"],
        max_turns=12,
        cost_ceiling_usd=0.50,
        system_prompt_hint="Verify a claim with verify_citation before stating it as fact.",
    ),
    "generic": Template(
        domain="generic",
        loop_type="react",
        default_tools=[],
        guardrails=["cost_ceiling"],
        max_turns=10,
        cost_ceiling_usd=0.50,
        system_prompt_hint="",
    ),
}
