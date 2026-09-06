from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from harnesskit.linter import Severity, lint
from harnesskit.parser import HarnessLoadError, load_harness

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


@app.command()
def run(
    directory: Path = typer.Argument(Path(".")),
    input: str = typer.Option(..., "--input", help="Task to run the harness on"),
) -> None:
    """Execute the harness on a single input (requires an adapter — not yet wired up)."""
    console.print("[yellow]not yet implemented[/yellow]: requires an Adapter (see adapters/ module).")
    raise typer.Exit(2)


@app.command()
def eval_cmd(directory: Path = typer.Argument(Path("."))) -> None:
    """Run the harness against its eval suite (requires an adapter — not yet wired up)."""
    console.print("[yellow]not yet implemented[/yellow]: requires the Eval Engine + an Adapter.")
    raise typer.Exit(2)


app.command(name="eval")(eval_cmd)


@app.command()
def ab(dir1: Path, dir2: Path) -> None:
    """A/B two harness versions on the same eval suite."""
    console.print("[yellow]not yet implemented[/yellow]")
    raise typer.Exit(2)


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
