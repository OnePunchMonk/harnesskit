"""Train the trainable-qa harness offline with random search.

    python examples/trainable_qa/train_demo.py [work_dir]

Equivalent CLI:

    harness train examples/trainable_qa \
        --adapter custom:examples/trainable_qa/agent.py:make_adapter \
        --proposer random --steps 4 --candidates 3 --out <work_dir>
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from agent import make_adapter  # noqa: E402

from harnesskit.train import HarnessModule, RandomSearchProposer, TrainBudget, Trainer  # noqa: E402


def main(work_dir: Path) -> None:
    module = HarnessModule(HERE)
    for name, param in module.named_parameters():
        print(f"requires_grad  {name:16s} = {param.value!r}")
    trainer = Trainer(
        module,
        adapter_factory=make_adapter,
        proposer=RandomSearchProposer(),
        work_dir=work_dir,
        budget=TrainBudget(max_steps=4, candidates_per_step=3),
        seed=0,
    )
    result = trainer.fit()
    for c in result.candidates:
        val = f"val={c.val.pass_rate:.2f}" if c.val else ""
        print(f"{c.id} step{c.step} {c.status:9s} {c.edits} {val} — {c.reason}")
    print(f"\nverdict: {result.verdict} — {result.verdict_reason}")
    print(f"best state: {result.best_state}")
    print(f"report: {work_dir / 'train_report.json'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="harness-train-")))
