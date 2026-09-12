"""Response cache (issue #4 item 9): key a single model-call boundary on a
deterministic hash of (model, system prompt, message history, tool schema
list) and store the raw provider response under
``<harness_dir>/.harness/cache/<hash>.json``.

Opt-in only (``--cache`` on the CLI, or an adapter constructed with
``cache=True``) — caching silently reusing a stale response would be a
correctness surprise for anyone who wants a fresh call every time. Purely a
filesystem cache: no eviction policy beyond "delete the directory."
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from harnesskit.trace.store import _atomic_write_text


def cache_dir(harness_dir: Path) -> Path:
    d = harness_dir / ".harness" / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_key(model_id: str, system: str, messages: list, tools: list | None) -> str:
    """Deterministic hash over the exact inputs to a model call. `messages`
    and `tools` must already be JSON-serializable (plain dicts/lists) —
    callers pass the same structures they're about to send to the provider."""
    payload = {
        "model": model_id,
        "system": system,
        "messages": messages,
        "tools": tools or [],
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_cached_response(harness_dir: Path, key: str) -> dict | None:
    path = cache_dir(harness_dir) / f"{key}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_cached_response(harness_dir: Path, key: str, response_payload: dict) -> Path:
    path = cache_dir(harness_dir) / f"{key}.json"
    _atomic_write_text(path, json.dumps(response_payload, indent=2))
    return path
