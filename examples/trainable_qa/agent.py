"""A deterministic keyword-lookup QA agent whose behavior is driven by
``agent_config.json`` — the file whose values this example declares
trainable. No model call, no network: the point is to exercise the
harnesskit.train loop end to end, offline.

``make_adapter(spec)`` is the adapter factory the trainer calls for every
candidate: it reads the *candidate's* ``agent_config.json`` from
``spec.source_dir`` so each candidate harness actually runs with its own
parameter values.
"""
from __future__ import annotations

import json
from pathlib import Path

from harnesskit.adapters.callback_adapter import CallbackAdapter, wrap_simple_callback

CORPUS_DIR = Path(__file__).parent / "corpus"
STOPWORDS = {
    "a", "an", "and", "the", "is", "are", "was", "were", "of", "in", "on", "at", "to", "for",
    "what", "which", "who", "whom", "how", "when", "where", "does", "did", "do", "its", "it",
    "tell", "me", "about", "by",
}


def _words(text: str, strip_stopwords: bool) -> set[str]:
    words = {w.strip(".,?!'\"").lower() for w in text.split()}
    words.discard("")
    return words - STOPWORDS if strip_stopwords else words


class ConfigurableQAAgent:
    def __init__(self, config: dict):
        self.config = config
        self.corpus = {p.stem: p.read_text().strip() for p in sorted(CORPUS_DIR.glob("*.txt"))}

    def answer(self, question: str) -> str:
        strip = bool(self.config.get("strip_stopwords", False))
        q = _words(question, strip)
        doc = max(self.corpus.values(), key=lambda text: len(q & _words(text, strip)))
        sentences = [s.strip() for s in doc.split(". ") if s.strip()]
        best = max(sentences, key=lambda s: len(q & _words(s, strip)))
        if len(q & _words(best, strip)) < int(self.config.get("min_overlap", 0)):
            return "I don't know."
        if self.config.get("answer_mode") == "best_sentence":
            return best
        return sentences[0]


def make_adapter(spec) -> CallbackAdapter:
    config = json.loads((Path(spec.source_dir) / "agent_config.json").read_text())
    agent = ConfigurableQAAgent(config)
    return CallbackAdapter(callback=wrap_simple_callback(agent.answer), model_id="local:keyword-lookup", runtime_name="callback")
