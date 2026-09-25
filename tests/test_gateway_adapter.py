"""GatewayAdapter: OpenAI-compatible gateways, offline (scripted client and a
localhost HTTP server; no provider SDK, key, or external network)."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from typer.testing import CliRunner

from harnesskit.adapters import ADAPTERS, GatewayAdapter
from harnesskit.adapters.gateway import GatewayConfigError, GatewayRequestError, HTTPChatClient
from harnesskit.cli.main import app
from harnesskit.eval import run_suite
from harnesskit.parser import load_harness
from harnesskit.testing.conformance import GATEWAY_CASES, run_case
from harnesskit.testing.fakes import FakeChatClient, chat_body


@pytest.mark.parametrize("case", GATEWAY_CASES, ids=lambda c: c.id)
def test_gateway_conformance(case):
    result = run_case(GatewayAdapter(), case, runtime="gateway")
    assert result.status == "pass", result.detail


def _harness(tmp_path: Path) -> Path:
    d = tmp_path / "h"
    (d / "tools").mkdir(parents=True, exist_ok=True)
    (d / "tools" / "lookup.json").write_text(json.dumps({
        "name": "lookup", "description": "Look up a fact.",
        "input_schema": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
    }))
    (d / "tools" / "lookup.py").write_text("def run(key):\n    return {'capital': 'Paris'}.get(key, 'unknown')\n")
    (d / "prompt.md").write_text("Answer using the lookup tool.\n")
    (d / "harness.yaml").write_text(
        "metadata: {name: gw}\n"
        "model: {provider: gateway, model_id: 'openrouter/cheap-fast-model'}\n"
        "loop: {type: react, max_turns: 4}\n"
        "tools: [{name: lookup, source: custom, ref: tools/lookup.json}]\n"
        "eval:\n  cases:\n    - {id: capital, input: 'Capital of France?', expected_output_contains: [Paris]}\n"
        "scaffold: {system_prompt: prompt.md}\n"
    )
    return d


def test_build_converts_tools_and_runs_suite_with_scripted_client(tmp_path):
    client = FakeChatClient([chat_body(None, [("lookup", {"key": "capital"})]), chat_body("It is Paris.")])
    spec = load_harness(_harness(tmp_path)).spec
    suite = run_suite(spec, GatewayAdapter(client=client))
    assert suite.pass_rate == 1.0
    first = client.calls[0]
    assert first["model"] == "openrouter/cheap-fast-model"
    assert first["messages"][0] == {"role": "system", "content": "Answer using the lookup tool.\n"}
    assert first["tools"][0]["function"]["name"] == "lookup"
    assert first["tools"][0]["function"]["parameters"]["required"] == ["key"]
    second = client.calls[1]["messages"]
    assert second[-1] == {"role": "tool", "tool_call_id": "call_0", "content": "Paris"}
    traj = suite.results[0].trajectory
    assert [s.tool_result for s in traj.tool_calls] == ["Paris"]


def test_missing_base_url_fails_at_build_not_inspect(tmp_path, monkeypatch):
    for var in ("HARNESSKIT_GATEWAY_BASE_URL", "OPENAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    adapter = ADAPTERS["gateway"]()
    assert adapter.supports().runtime == "gateway"  # capability inspection needs no config
    spec = load_harness(_harness(tmp_path)).spec
    with pytest.raises(GatewayConfigError, match="HARNESSKIT_GATEWAY_BASE_URL"):
        adapter.build(spec)
    monkeypatch.setenv("HARNESSKIT_GATEWAY_BASE_URL", "ftp://nope")
    with pytest.raises(GatewayConfigError, match="http"):
        adapter.build(spec)


class _Gateway(BaseHTTPRequestHandler):
    requests: list[dict] = []
    replies: list[tuple[int, dict, dict]] = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        status, payload, headers = type(self).replies.pop(0)
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def local_gateway():
    _Gateway.requests, _Gateway.replies = [], []
    server = HTTPServer(("127.0.0.1", 0), _Gateway)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1", _Gateway
    server.shutdown()


def test_http_client_end_to_end_with_litellm_cost_header(tmp_path, monkeypatch, local_gateway):
    url, handler = local_gateway
    handler.replies = [
        (200, chat_body(None, [("lookup", {"key": "capital"})], model="served-a"), {"x-litellm-response-cost": "0.0003"}),
        (200, chat_body("Paris."), {}),
    ]
    monkeypatch.setenv("HARNESSKIT_GATEWAY_BASE_URL", url)
    monkeypatch.setenv("HARNESSKIT_GATEWAY_API_KEY", "sk-test")
    spec = load_harness(_harness(tmp_path)).spec
    suite = run_suite(spec, GatewayAdapter())
    assert suite.pass_rate == 1.0
    assert handler.requests[0]["path"] == "/v1/chat/completions"
    assert handler.requests[0]["auth"] == "Bearer sk-test"
    steps = [s for s in suite.results[0].trajectory.steps if s.step_type.value == "llm_call"]
    assert steps[0].cost_status.value == "observed" and steps[0].cost_usd == 0.0003
    assert "served_model=served-a" in steps[0].note
    assert steps[1].cost_status.value == "unavailable"  # unpriced model, no reported cost
    assert suite.avg_cost_usd is None and suite.unavailable_cost_case_ids == ["capital"]


def test_http_error_is_a_visible_errored_case(tmp_path, monkeypatch, local_gateway):
    url, handler = local_gateway
    handler.replies = [(429, {"error": {"message": "rate limited"}}, {})]
    with pytest.raises(GatewayRequestError, match="HTTP 429"):
        HTTPChatClient(url).create({"model": "m", "messages": []})
    handler.replies = [(500, {"error": "boom"}, {})]
    monkeypatch.setenv("HARNESSKIT_GATEWAY_BASE_URL", url)
    suite = run_suite(load_harness(_harness(tmp_path)).spec, GatewayAdapter())
    assert suite.errored_case_ids == ["capital"] and suite.pass_rate == 0.0
    assert "HTTP 500" in suite.results[0].error


def test_cli_eval_with_gateway_adapter(tmp_path, monkeypatch, local_gateway):
    url, handler = local_gateway
    handler.replies = [(200, chat_body("Paris is the capital."), {})]
    monkeypatch.setenv("HARNESSKIT_GATEWAY_BASE_URL", url)
    r = CliRunner().invoke(app, ["eval", str(_harness(tmp_path)), "--adapter", "gateway"])
    assert r.exit_code == 0, r.output
    assert handler.requests and handler.requests[0]["body"]["model"] == "openrouter/cheap-fast-model"
