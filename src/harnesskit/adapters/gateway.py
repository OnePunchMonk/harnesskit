"""OpenAI-compatible gateway adapter: run a harness against any model behind an
AI gateway (LiteLLM, OpenRouter, Portkey, Vercel AI Gateway, vLLM, Ollama, ...)
that speaks the Chat Completions API.

No provider SDK is required: requests go over the standard library's HTTP
client. Configuration comes from the environment at ``build()`` time, so
``harness inspect --adapter gateway`` never needs credentials:

- ``HARNESSKIT_GATEWAY_BASE_URL`` (or ``OPENAI_BASE_URL``): e.g.
  ``https://openrouter.ai/api/v1`` or ``http://localhost:4000``.
- ``HARNESSKIT_GATEWAY_API_KEY`` (or ``OPENAI_API_KEY``): optional for local
  gateways that don't need one.

``spec.model.model_id`` is sent verbatim as the gateway's model name. The
gateway may serve a different model (aliases, fallbacks), so each llm_call
step records the ``served_model`` the response names.

Cost is ``observed`` only when the response carries a cost
(``usage.cost``, as OpenRouter returns, or LiteLLM's
``x-litellm-response-cost`` header). Otherwise it is estimated from the
local pricing table when the model is listed there, and ``unavailable`` when
it isn't. It is never reported as $0.

Tool convention matches ``RawAPIAdapter``: a custom tool at
``tools/<name>.json`` (``name``/``description``/``input_schema``) with an
optional sibling ``tools/<name>.py`` exposing ``run(**kwargs) -> str``.
Enforced at runtime: ``loop.max_turns``, ``termination`` types
``max_turns``/``explicit_tool``/``tag_emitted``, and ``loop.max_tool_calls``.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from harnesskit.adapters._tool_loading import load_tool_callback
from harnesskit.adapters.base import AdapterCapabilities, RunnableAgent
from harnesskit.format.spec import HarnessSpec
from harnesskit.trace.schema import AttemptStatus, CostStatus, Step, StepType, Trajectory, estimate_cost_usd


class GatewayConfigError(Exception):
    """The gateway base URL is missing or invalid."""


class GatewayRequestError(Exception):
    """The gateway returned an HTTP error or an unusable response."""


@dataclass
class ChatResponse:
    body: dict[str, Any]
    headers: dict[str, str]


class ChatClient(Protocol):
    def create(self, payload: dict[str, Any]) -> ChatResponse: ...


@dataclass
class HTTPChatClient:
    base_url: str
    api_key: str | None = None
    timeout_s: float = 120.0
    extra_headers: dict[str, str] | None = None

    def create(self, payload: dict[str, Any]) -> ChatResponse:
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json", **(self.extra_headers or {})}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:  # noqa: S310 — URL is user-configured
                return ChatResponse(json.loads(resp.read().decode()), {k.lower(): v for k, v in resp.headers.items()})
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            raise GatewayRequestError(f"gateway returned HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise GatewayRequestError(f"could not reach gateway at {url}: {e.reason}") from e


def to_openai_tool(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a harnesskit tool schema (Anthropic-style) to Chat Completions form."""
    return {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema.get("description", ""),
            "parameters": schema.get("input_schema") or {"type": "object", "properties": {}},
        },
    }


def _observed_cost(response: ChatResponse) -> float | None:
    usage = response.body.get("usage") or {}
    cost = usage.get("cost")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
        return float(cost)
    header = response.headers.get("x-litellm-response-cost")
    try:
        return float(header) if header is not None else None
    except ValueError:
        return None


class GatewayAdapter:
    """Implements HarnessAdapter for a ReAct loop over an OpenAI-compatible endpoint."""

    runtime = "gateway"

    def __init__(self, client: ChatClient | None = None, max_tokens: int = 4096):
        self._client = client
        self.max_tokens = max_tokens

    def supports(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            loop_strategies={"react"},
            tool_sources={"custom"},
            hook_points=set(),
            memory_backends={"none", "in_memory"},
            runtime="gateway",
            enforced_termination_types={"max_turns", "explicit_tool", "tag_emitted"},
            supports_max_tool_calls=True,
        )

    def _make_client(self) -> ChatClient:
        if self._client is not None:
            return self._client
        base_url = os.environ.get("HARNESSKIT_GATEWAY_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        if not base_url:
            raise GatewayConfigError(
                "set HARNESSKIT_GATEWAY_BASE_URL (or OPENAI_BASE_URL) to your gateway's OpenAI-compatible base URL"
            )
        if not base_url.startswith(("http://", "https://")):
            raise GatewayConfigError(f"gateway base URL must start with http:// or https://, got {base_url!r}")
        api_key = os.environ.get("HARNESSKIT_GATEWAY_API_KEY") or os.environ.get("OPENAI_API_KEY")
        return HTTPChatClient(base_url, api_key)

    def build(self, spec: HarnessSpec) -> RunnableAgent:
        client = self._make_client()
        tool_schemas, callbacks = [], {}
        for tool in spec.tools:
            if tool.source.value != "custom":
                continue
            if spec.source_dir is None:
                raise GatewayConfigError("spec must be loaded via harnesskit.parser.load_harness to resolve tools")
            tool_schemas.append(to_openai_tool(json.loads((spec.source_dir / tool.ref).read_text())))
            callbacks[tool.name] = load_tool_callback(spec.source_dir, tool.ref)
        return RunnableAgent(spec=spec, handle={"client": client, "tool_schemas": tool_schemas, "callbacks": callbacks})

    def run(self, agent: RunnableAgent, input: str) -> Trajectory:
        spec = agent.spec
        client: ChatClient = agent.handle["client"]
        tool_schemas = agent.handle["tool_schemas"]
        callbacks = agent.handle["callbacks"]

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": spec.scaffold.system_prompt.replace("{{task}}", input)},
            {"role": "user", "content": input},
        ]
        trajectory = Trajectory(
            harness_name=spec.metadata.name,
            harness_version=spec.metadata.version,
            model_id=spec.model.model_id,
            adapter="gateway",
            input=input,
        )
        stop_tags = [str(t.value) for t in spec.termination if t.type == "tag_emitted"]
        explicit_tools = [str(t.value) for t in spec.termination if t.type == "explicit_tool"]
        max_tool_calls = spec.loop.max_tool_calls
        last_text = ""

        for _ in range(spec.loop.max_turns):
            payload: dict[str, Any] = {
                "model": spec.model.model_id,
                "messages": list(messages),  # snapshot: the loop keeps appending
                "temperature": spec.model.temperature,
                "max_tokens": self.max_tokens,
            }
            if tool_schemas:
                payload["tools"] = tool_schemas
            t0 = time.time()
            try:
                response = client.create(payload)
                choice = response.body["choices"][0]
                message = choice["message"]
            except (GatewayRequestError, KeyError, IndexError, TypeError) as e:
                # The failed attempt stays on the trace (it may still have cost money).
                trajectory.steps.append(
                    Step(
                        step_type=StepType.llm_call,
                        input=json.dumps(messages[-1], default=str),
                        cost_usd=None,
                        cost_status=CostStatus.unavailable,
                        attempt_status=AttemptStatus.failed,
                        duration_ms=int((time.time() - t0) * 1000),
                        note=f"gateway request failed: {e}",
                    )
                )
                trajectory.stopped_reason = "error"
                trajectory.final_output = None
                raise GatewayRequestError(str(e)) from e
            duration_ms = int((time.time() - t0) * 1000)

            usage = response.body.get("usage") or {}
            tokens_in, tokens_out = usage.get("prompt_tokens"), usage.get("completion_tokens")
            observed = _observed_cost(response)
            if observed is not None:
                cost, cost_status = observed, CostStatus.observed
            elif tokens_in is not None and tokens_out is not None:
                cost, cost_status = estimate_cost_usd(spec.model.model_id, tokens_in, tokens_out)
            else:
                cost, cost_status = None, CostStatus.unavailable

            text = message.get("content") or ""
            if isinstance(text, list):  # some gateways return content parts
                text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
            last_text = text
            tool_calls = message.get("tool_calls") or []
            served = response.body.get("model")
            trajectory.steps.append(
                Step(
                    step_type=StepType.llm_call,
                    input=json.dumps(messages[-1], default=str),
                    output=text,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    tokens_observed=tokens_in is not None,
                    cost_usd=cost,
                    cost_status=cost_status,
                    duration_ms=duration_ms,
                    note=f"served_model={served}; finish_reason={choice.get('finish_reason')}",
                )
            )
            assistant: dict[str, Any] = {"role": "assistant", "content": text or None}
            if tool_calls:
                assistant["tool_calls"] = tool_calls
            messages.append(assistant)

            for tag in stop_tags:
                if f"<{tag}>" in text:
                    trajectory.final_output = text
                    trajectory.stopped_reason = f"tag_emitted:{tag}"
                    return trajectory

            if not tool_calls:
                trajectory.final_output = text
                trajectory.stopped_reason = "no_tool_use" if not explicit_tools else "idle"
                return trajectory

            for call in tool_calls:
                fn = call.get("function") or {}
                name = fn.get("name") or ""
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("arguments must be a JSON object")
                    arg_error = None
                except ValueError as e:
                    args, arg_error = {}, f"error: invalid tool arguments: {e}"

                if name in explicit_tools and arg_error is None:
                    trajectory.final_output = json.dumps(args)
                    trajectory.stopped_reason = f"explicit_tool:{name}"
                    return trajectory

                if max_tool_calls is not None and len(trajectory.tool_calls) >= max_tool_calls:
                    trajectory.final_output = last_text or None
                    trajectory.stopped_reason = "max_tool_calls"
                    return trajectory

                t_tool = time.time()
                callback = callbacks.get(name)
                if arg_error is not None:
                    result_text = arg_error
                elif callback is None:
                    result_text = f"error: tool '{name}' has no callback wired up"
                else:
                    try:
                        result_text = str(callback(**args))
                    except Exception as e:  # noqa: BLE001 — surfaced to the model, not raised
                        result_text = f"error: {e}"
                trajectory.steps.append(
                    Step(
                        step_type=StepType.tool_call,
                        tool_name=name,
                        tool_args=args,
                        tool_result=result_text,
                        duration_ms=int((time.time() - t_tool) * 1000),
                    )
                )
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": result_text})

        trajectory.stopped_reason = "max_turns"
        trajectory.final_output = last_text or None
        return trajectory
