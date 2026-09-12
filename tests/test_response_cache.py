"""Response caching (issue #4 item 9): a cache hit on (model, system,
messages, tools) must skip the actual provider call entirely.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from harnesskit.adapters.base import RunnableAgent
from harnesskit.adapters.raw_api import RawAPIAdapter
from harnesskit.parser import load_harness

EXAMPLE = Path(__file__).parent.parent / "examples" / "react-web-researcher"


class _FakeUsage:
    def __init__(self, tokens_in=10, tokens_out=5):
        self.input_tokens = tokens_in
        self.output_tokens = tokens_out


class _FakeBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, text="<answer>done</answer>"):
        self.content = [_FakeBlock(text)]
        self.usage = _FakeUsage()


class _CountingClient:
    def __init__(self):
        self.call_count = 0
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.call_count += 1
        return _FakeResponse()


def _agent(spec, client):
    return RunnableAgent(spec=spec, handle={"client": client, "tool_schemas": [], "callbacks": {}})


def test_identical_calls_hit_cache_once(tmp_path):
    result = load_harness(EXAMPLE)
    result.spec.source_dir = tmp_path

    adapter = RawAPIAdapter(cache=True)
    client = _CountingClient()

    adapter.run(_agent(result.spec, client), "what is the capital of Brazil?")
    adapter.run(_agent(result.spec, client), "what is the capital of Brazil?")

    assert client.call_count == 1


def test_cache_disabled_calls_provider_every_time(tmp_path):
    result = load_harness(EXAMPLE)
    result.spec.source_dir = tmp_path

    adapter = RawAPIAdapter(cache=False)
    client = _CountingClient()

    adapter.run(_agent(result.spec, client), "what is the capital of Brazil?")
    adapter.run(_agent(result.spec, client), "what is the capital of Brazil?")

    assert client.call_count == 2


def test_different_input_is_a_cache_miss(tmp_path):
    result = load_harness(EXAMPLE)
    result.spec.source_dir = tmp_path

    adapter = RawAPIAdapter(cache=True)
    client = _CountingClient()

    adapter.run(_agent(result.spec, client), "what is the capital of Brazil?")
    adapter.run(_agent(result.spec, client), "what is the capital of France?")

    assert client.call_count == 2
