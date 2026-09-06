from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from harnesskit.adapters import RawAPIAdapter
from harnesskit.eval import SuiteResult, check_regression, compare, run_suite
from harnesskit.eval.engine import CaseResult
from harnesskit.eval.scorers import score_trajectory
from harnesskit.linter import Severity, lint
from harnesskit.parser import HarnessLoadError, load_harness
from harnesskit.trace import list_baselines, load_baseline, save_trajectory
from harnesskit.trace import save_baseline as save_baseline_fn

app = typer.Typer(no_args_is_help=True, help="harnesskit: infrastructure for building agent harnesses")
console = Console()


@app.command()
def lint_cmd(directory: Path = typer.Argument(Path("."), help="Harness directory")) -> None:
    """Static analysis for known failure modes."""
    try:
        result = load_harness(directory)
    except HarnessLoadError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)

    for w in result.warnings:
        console.print(f"[yellow]warning[/yellow]: {w}")

    findings = lint(result.spec)
    if not findings:
        console.print("[green]✓[/green] No issues found.")
        raise typer.Exit(0)

    table = Table(show_header=True, header_style="bold")
    table.add_column("severity")
    table.add_column("rule")
    table.add_column("message")
    table.add_column("fix")
    color = {Severity.error: "red", Severity.warning: "yellow", Severity.info: "cyan"}
    for f in findings:
        table.add_row(f"[{color[f.severity]}]{f.severity.value}[/{color[f.severity]}]", f.rule, f.message, f.fix or "")
    console.print(table)

    if any(f.severity == Severity.error for f in findings):
        raise typer.Exit(1)


app.command(name="lint")(lint_cmd)


@app.command()
def inspect(directory: Path = typer.Argument(Path("."), help="Harness directory")) -> None:
    """Show a harness summary: tools, guardrails, loop strategy, eval cases."""
    try:
        result = load_harness(directory)
    except HarnessLoadError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)
    spec = result.spec

    console.print(f"[bold]{spec.metadata.name}[/bold] v{spec.metadata.version}")
    if spec.metadata.description:
        console.print(spec.metadata.description)
    console.print(f"\nmodel: {spec.model.provider}/{spec.model.model_id}")
    console.print(f"loop: {spec.loop.type.value} (max_turns={spec.loop.max_turns})")
    console.print(f"tools: {', '.join(t.name for t in spec.tools) or '(none)'}")
    console.print(f"guardrails: {', '.join(g.name for g in spec.guardrails) or '(none)'}")
    console.print(f"termination: {', '.join(t.type for t in spec.termination) or '(none)'}")
    console.print(f"eval cases: {len(spec.eval.cases)}")


@app.command()
def init(
    name: str = typer.Argument(..., help="Harness name"),
    directory: Path = typer.Option(None, help="Where to create it (default: ./<name>)"),
) -> None:
    """Scaffold a minimal, runnable harness directory (full NL-spec scaffolder: see scaffold/ module)."""
    target = directory or Path(name)
    if target.exists() and any(target.iterdir()):
        console.print(f"[red]✗[/red] {target} already exists and is not empty.")
        raise typer.Exit(1)

    (target / "tools").mkdir(parents=True, exist_ok=True)
    (target / "system_prompt.md").write_text(f"You are {name}, an agent that ...\n")
    (target / "harness.yaml").write_text(
        f"""schema_version: 1
metadata:
  name: {name}
  version: 0.1.0
  description: "TODO: describe what this harness does"

model:
  provider: anthropic
  model_id: claude-sonnet-5

loop:
  type: react
  max_turns: 10

tools: []

guardrails:
  - name: cost-ceiling
    trigger: on_budget_check
    check: "cost_threshold:1.00"
    enforce: abort

termination:
  - type: explicit_tool
    value: final_answer
  - type: max_turns
    value: 10

eval:
  cases: []

scaffold:
  system_prompt: system_prompt.md
  system_prompt_is_file: true
"""
    )
    console.print(f"[green]✓[/green] Created harness at {target}/")
    console.print("  Next: edit tools, add eval cases, then run `harness lint .`")


def _fail(message: str, verbose: bool, exc: Exception | None = None) -> None:
    console.print(f"[red]✗[/red] {message}")
    if verbose and exc is not None:
        console.print_exception()
    raise typer.Exit(1)


@app.command()
def run(
    directory: Path = typer.Argument(Path(".")),
    input: str = typer.Option(..., "--input", help="Task to run the harness on"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    """Execute the harness on a single input via the raw-API adapter."""
    try:
        result = load_harness(directory)
    except HarnessLoadError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)

    adapter = RawAPIAdapter()
    try:
        agent = adapter.build(result.spec)
        trajectory = adapter.run(agent, input)
    except ImportError as e:
        _fail(str(e), verbose, e)
    except Exception as e:  # noqa: BLE001 — provider/auth errors surfaced concisely by default
        _fail(f"Adapter run failed: {e}", verbose, e)
    path = save_trajectory(directory, trajectory)

    console.print(f"[bold]stopped:[/bold] {trajectory.stopped_reason}  "
                  f"[bold]turns:[/bold] {trajectory.turns}  "
                  f"[bold]cost:[/bold] ${trajectory.total_cost_usd:.4f}")
    console.print(f"\n{trajectory.final_output or '(no final output)'}")
    console.print(f"\n[dim]trace saved to {path}[/dim]")


@app.command()
def eval_cmd(
    directory: Path = typer.Argument(Path(".")),
    sample: int = typer.Option(None, "--sample", help="Run only the first N cases"),
    save_baseline: str = typer.Option(None, "--save-baseline", help="Snapshot this run under a name for future --compare"),
    compare_baseline: str = typer.Option(None, "--compare", help="Diff this run against a saved baseline; fail on regression"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    """Run the harness against its eval suite via the raw-API adapter."""
    try:
        result = load_harness(directory)
    except HarnessLoadError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)
    if not result.spec.eval.cases:
        console.print("[yellow]No eval cases defined in this harness.[/yellow]")
        raise typer.Exit(1)

    try:
        suite = run_suite(result.spec, RawAPIAdapter(), sample=sample)
    except ImportError as e:
        _fail(str(e), verbose, e)
    except Exception as e:  # noqa: BLE001 — provider/auth errors surfaced concisely by default
        _fail(f"Adapter run failed: {e}", verbose, e)
    for r in suite.results:
        save_trajectory(directory, r.trajectory, run_id=r.case.id)

    table = Table(show_header=True, header_style="bold")
    table.add_column("case")
    table.add_column("pass")
    table.add_column("turns")
    table.add_column("cost")
    table.add_column("scores")
    for r in suite.results:
        mark = "[green]✓[/green]" if r.passed else "[red]✗[/red]"
        score_str = ", ".join(f"{s.name}={s.value:.2f}" for s in r.scores if not s.passed) or "all pass"
        table.add_row(r.case.id, mark, str(r.trajectory.turns), f"${r.trajectory.total_cost_usd:.4f}", score_str)
    console.print(table)
    console.print(
        f"\npass_rate={suite.pass_rate:.2f}  avg_turns={suite.avg_turns:.1f}  "
        f"avg_cost=${suite.avg_cost_usd:.4f}  dup_tool_calls={suite.total_duplicate_tool_calls}"
    )

    if save_baseline:
        path = save_baseline_fn(directory, save_baseline, {r.case.id: r.trajectory for r in suite.results})
        console.print(f"[dim]baseline '{save_baseline}' saved to {path}[/dim]")

    if compare_baseline:
        try:
            baseline_trajectories = load_baseline(directory, compare_baseline)
        except FileNotFoundError as e:
            available = list_baselines(directory)
            console.print(f"[red]✗[/red] {e}" + (f" (available: {', '.join(available)})" if available else ""))
            raise typer.Exit(1)

        baseline_results = [
            CaseResult(case=c, trajectory=baseline_trajectories[c.id], scores=score_trajectory(baseline_trajectories[c.id], c))
            for c in result.spec.eval.cases
            if c.id in baseline_trajectories
        ]
        baseline_suite = SuiteResult(harness_name=f"{result.spec.metadata.name}@{compare_baseline}", results=baseline_results)

        reg = check_regression(baseline_suite, suite, result.spec.eval.thresholds)
        ab_table = Table(show_header=True, header_style="bold", title=f"vs baseline '{compare_baseline}'")
        ab_table.add_column("metric")
        ab_table.add_column("baseline")
        ab_table.add_column("current")
        ab_table.add_column("Δ")
        ab = compare(baseline_suite, suite)
        for d in ab.deltas:
            sign = "+" if d.delta >= 0 else ""
            ab_table.add_row(d.metric, f"{d.a:.3f}", f"{d.b:.3f}", f"{sign}{d.delta:.3f}")
        console.print(ab_table)

        if not reg.ok:
            console.print("[red]✗ regression detected:[/red]")
            for d in reg.regressions:
                console.print(f"  - {d.metric}: {d.a:.3f} -> {d.b:.3f} (Δ {d.delta:+.3f})")
            raise typer.Exit(1)
        console.print("[green]✓[/green] no regression beyond configured tolerances")

    if suite.pass_rate < 1.0:
        raise typer.Exit(1)


app.command(name="eval")(eval_cmd)


@app.command()
def ab(dir1: Path, dir2: Path, sample: int = typer.Option(None, "--sample"), verbose: bool = typer.Option(False, "--verbose")) -> None:
    """A/B two harness versions on the same eval suite."""
    try:
        r1, r2 = load_harness(dir1), load_harness(dir2)
    except HarnessLoadError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)

    adapter = RawAPIAdapter()
    try:
        suite_a = run_suite(r1.spec, adapter, sample=sample)
        suite_b = run_suite(r2.spec, adapter, sample=sample)
    except ImportError as e:
        _fail(str(e), verbose, e)
    except Exception as e:  # noqa: BLE001 — provider/auth errors surfaced concisely by default
        _fail(f"Adapter run failed: {e}", verbose, e)
    result = compare(suite_a, suite_b)

    table = Table(show_header=True, header_style="bold")
    table.add_column("metric")
    table.add_column(result.name_a)
    table.add_column(result.name_b)
    table.add_column("Δ")
    for d in result.deltas:
        sign = "+" if d.delta >= 0 else ""
        table.add_row(d.metric, f"{d.a:.3f}", f"{d.b:.3f}", f"{sign}{d.delta:.3f}")
    console.print(table)
    lo, hi = result.pass_rate_ci
    console.print(f"\n95% CI on pass_rate delta: [{lo:+.2f}, {hi:+.2f}]  (n={len(suite_a.results)}, bootstrap)")


@app.command()
def watch(directory: Path = typer.Argument(Path("."))) -> None:
    """Re-run evals on file changes."""
    console.print("[yellow]not yet implemented[/yellow]")
    raise typer.Exit(2)


@app.command(name="export")
def export_cmd(directory: Path = typer.Argument(Path("."))) -> None:
    """Package the harness as a portable .harn bundle."""
    console.print("[yellow]not yet implemented[/yellow]")
    raise typer.Exit(2)


@app.command(name="import")
def import_cmd(bundle: Path) -> None:
    """Unpack a .harn bundle into a harness directory."""
    console.print("[yellow]not yet implemented[/yellow]")
    raise typer.Exit(2)


if __name__ == "__main__":
    app()
