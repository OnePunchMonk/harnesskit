"""Scripted fakes for adapter conformance testing (design doc §7; issue #6).

These stand in for a real provider client so the conformance suite in
`harnesskit.testing.conformance` needs no API key and makes no network call.
`FakeAnthropicClient` mimics the shape of `anthropic.Anthropic().messages`
closely enough for `RawAPIAdapter.run()` to consume it unmodified — nothing
in the adapter has to know it's talking to a fake.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class FakeUsage:
    input_tokens: int = 10
    output_tokens: int = 10


@dataclass
class FakeBlock:
    type: str
    text: str | None = None
    name: str | None = None
    input: dict | None = None
    id: str | None = None


@dataclass
class FakeResponse:
    content: list[FakeBlock]
    usage: FakeUsage = field(default_factory=FakeUsage)


def text_block(text: str) -> FakeBlock:
    return FakeBlock(type="text", text=text)


def tool_use_block(name: str, args: dict | None = None, *, id: str = "toolu_1") -> FakeBlock:
    return FakeBlock(type="tool_use", name=name, input=args or {}, id=id)


@dataclass
class _FakeMessages:
    responses: list[FakeResponse]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError(
                "FakeAnthropicClient script exhausted: the adapter made more "
                "messages.create() calls than the scenario scripted"
            )
        return self.responses.pop(0)


class FakeAnthropicClient:
    """Drop-in replacement for `anthropic.Anthropic()`, driven by a fixed
    script of responses given in call order. Never touches the network."""

    def __init__(self, script: list[FakeResponse]) -> None:
        self.messages = _FakeMessages(list(script))

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.messages.calls


def raising_tool(message: str = "boom") -> Callable[..., str]:
    """A fake tool callback that always raises, for exception-handling scenarios."""

    def _run(**_kwargs: Any) -> str:
        raise RuntimeError(message)

    return _run


def echo_tool() -> Callable[..., str]:
    """A fake tool callback that echoes its kwargs back, for trace-completeness checks."""

    def _run(**kwargs: Any) -> str:
        return f"ok:{kwargs}"

    return _run
