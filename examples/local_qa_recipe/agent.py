"""A plain, pre-existing callback-based agent — pretend this is someone
else's code that harnesskit had no hand in writing. It has its own internal
"loop" (a keyword-overlap lookup over a small corpus) that we are not
rewriting; we are only wrapping its entrypoint function.

`answer_question(question) -> str` is the entire public surface. Everything
below it (corpus loading, scoring, sentence selection) is the wrapped
agent's own implementation detail.
"""
from __future__ import annotations

from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_corpus() -> dict[str, str]:
    return {p.stem: p.read_text() for p in sorted(FIXTURES_DIR.glob("*.txt"))}


def _words(text: str) -> set[str]:
    return {w.strip(".,?!").lower() for w in text.split() if w.strip(".,?!")}


class NaiveQAAgent:
    """Keyword-overlap lookup/extraction over the fixture corpus. Deliberately
    naive: it cannot answer a question whose answer isn't literally present
    in the corpus (e.g. "what is the capital of Nepal?" — the corpus
    mentions Nepal but never its capital). That's a real, honest limitation
    of this kind of agent, not a bug in the harness — the whole point of the
    recipe is to catch exactly this with an eval case.

    `version` and `call_log` exist only so the recipe script can (a) tell
    when the wrapped agent's own logic changed [step 5] and (b) show what
    document/score it picked for each question [step 2's "tool/result"
    inspection] — neither is something `CallbackAdapter` itself provides,
    since from harnesskit's point of view this whole class is one opaque
    callback.
    """

    version = "v1-keyword-overlap"

    def __init__(self) -> None:
        self.corpus = load_corpus()
        self.call_log: list[dict] = []

    def answer_question(self, question: str) -> str:
        q_words = _words(question)
        best_doc_name, best_doc_text, best_score = None, "", -1
        for name, text in self.corpus.items():
            score = len(q_words & _words(text))
            if score > best_score:
                best_doc_name, best_doc_text, best_score = name, text, score

        sentences = [s.strip() for s in best_doc_text.split(". ") if s.strip()]
        best_sentence = max(
            sentences, key=lambda s: len(q_words & _words(s)), default=""
        )
        self.call_log.append(
            {
                "question": question,
                "picked_doc": best_doc_name,
                "overlap_score": best_score,
                "picked_sentence": best_sentence,
            }
        )
        return best_sentence


class ImprovedQAAgent(NaiveQAAgent):
    """A "fresh execution" candidate for step 6: adds a tiny fact table for
    questions whose answer isn't literally in the corpus text. This is a
    behavioral change to the wrapped agent, not just a rescoring — comparing
    it against v1 requires running both fresh (see run_recipe.step6)."""

    version = "v2-with-fact-table"

    _FACTS = {"capital of nepal": "Kathmandu is the capital of Nepal."}

    def answer_question(self, question: str) -> str:
        key = question.strip("?").lower()
        for fact_key, fact_answer in self._FACTS.items():
            if fact_key in key:
                self.call_log.append(
                    {"question": question, "picked_doc": "fact_table", "overlap_score": None, "picked_sentence": fact_answer}
                )
                return fact_answer
        return super().answer_question(question)
