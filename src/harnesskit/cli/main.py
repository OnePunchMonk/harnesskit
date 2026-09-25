from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from harnesskit.adapters import ADAPTERS, SupportStatus, check_support, inspect_support
from harnesskit.eval import SuiteComparisonError, SuiteResult, check_regression, compare, replay_suite, run_suite
from harnesskit.eval.engine import CaseResult
from harnesskit.eval.scorers import score_trajectory
from harnesskit.linter import Severity, lint
from harnesskit.packaging import export_bundle, import_bundle, preview_bundle
from harnesskit.parser import HarnessLoadError, load_harness
from harnesskit.trace import Trajectory, current_provenance, list_baselines, load_baseline, load_baseline_provenance, save_trajectory
from harnesskit.trace import save_baseline as save_baseline_fn
from harnesskit.trace.store import _git_sha as _git_sha_for_warning

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
def inspect(
    directory: Path = typer.Argument(Path("."), help="Harness directory"),
    adapter_name: str | None = typer.Option(None, "--adapter", help="Inspect support for this adapter"),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable support findings"),
) -> None:
    """Show a harness summary: tools, guardrails, loop strategy, eval cases."""
    try:
        result = load_harness(directory)
    except HarnessLoadError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)
    spec = result.spec

    if as_json:
        if adapter_name is None:
            console.print("[red]✗[/red] --json requires --adapter")
            raise typer.Exit(1)
        adapter_cls = ADAPTERS.get(adapter_name)
        if adapter_cls is None:
            console.print(f"[red]✗[/red] Unknown adapter '{adapter_name}'. Available: {', '.join(ADAPTERS)}")
            raise typer.Exit(1)
        console.print_json(json.dumps([finding.to_dict() for finding in inspect_support(spec, adapter_cls())]))
        return

    console.print(f"[bold]{spec.metadata.name}[/bold] v{spec.metadata.version}")
    if spec.metadata.description:
        console.print(spec.metadata.description)
    console.print(f"\nmodel: {spec.model.provider}/{spec.model.model_id}")
    console.print(f"loop: {spec.loop.type.value} (max_turns={spec.loop.max_turns})")
    console.print(f"tools: {', '.join(t.name for t in spec.tools) or '(none)'}")
    console.print(f"guardrails: {', '.join(g.name for g in spec.guardrails) or '(none)'}")
    console.print(f"termination: {', '.join(t.type for t in spec.termination) or '(none)'}")
    console.print(f"eval cases: {len(spec.eval.cases)}")
    if adapter_name is not None:
        adapter_cls = ADAPTERS.get(adapter_name)
        if adapter_cls is None:
            console.print(f"[red]✗[/red] Unknown adapter '{adapter_name}'. Available: {', '.join(ADAPTERS)}")
            raise typer.Exit(1)
        findings = inspect_support(spec, adapter_cls())
        if not findings:
            console.print(f"[green]✓[/green] All declared features are supported by {adapter_name}.")
        else:
            for finding in findings:
                console.print(
                    f"[yellow]{finding.status.value}[/yellow] {finding.field}={finding.requested}: {finding.reason}"
                )


@app.command()
def conformance(
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable scenario results"),
) -> None:
    """Run the scripted adapter conformance suite (no harness dir, no API key).

    Tests that adapters actually behave the way their declared capabilities
    claim, using scripted clients and fake tools — see `harnesskit.testing`.
    """
    from harnesskit.testing.conformance import RAW_API_CASES, generate_matrix, run_case

    adapter = ADAPTERS["raw_api"]()
    results = [run_case(adapter, case, runtime="raw_api") for case in RAW_API_CASES]

    try:
        from harnesskit.testing.conformance import PYDANTIC_AI_CASES

        pai_adapter = ADAPTERS["pydantic_ai"]()
        results += [run_case(pai_adapter, case, runtime="pydantic_ai") for case in PYDANTIC_AI_CASES]
    except ImportError:
        pass  # pydantic-ai not installed; report raw_api only

    if as_json:
        console.print_json(json.dumps([{"case_id": r.case_id, "runtime": r.runtime, "status": r.status, "detail": r.detail} for r in results]))
        raise typer.Exit(0 if all(r.status == "pass" for r in results) else 1)

    console.print(generate_matrix(results))
    for r in results:
        if r.status != "pass":
            console.print(f"[red]✗[/red] {r.case_id} ({r.runtime}): {r.detail}")
    raise typer.Exit(0 if all(r.status == "pass" for r in results) else 1)


@app.command()
def templates() -> None:
    """List the built-in scaffold templates usable with `harness init --template`."""
    from harnesskit.scaffold.templates import TEMPLATES

    table = Table(show_header=True, header_style="bold")
    table.add_column("template")
    table.add_column("description")
    for name, template in TEMPLATES.items():
        table.add_row(name, template.description or "(no description)")
    console.print(table)


@app.command()
def init(
    name: str = typer.Argument(None, help="Harness name (omit when using --spec; the plan names itself)"),
    spec: str = typer.Option(None, "--spec", help="Natural-language description — scaffolds via an LLM call + self-validation"),
    template: str = typer.Option(
        None, "--template", help="Generate directly from a named template — offline, deterministic, no LLM call. See `harness templates`."
    ),
    directory: Path = typer.Option(None, help="Where to create it (default: ./<name>)"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    """Scaffold a harness directory: a minimal skeleton by default, a
    filled-in one from a natural-language description via --spec, or a
    filled-in one from a named template via --template (offline, no LLM call)."""
    if spec and template:
        console.print("[red]✗[/red] Pass either --spec or --template, not both.")
        raise typer.Exit(1)

    if template:
        from harnesskit.scaffold import generate_harness
        from harnesskit.scaffold.plan import ScaffoldPlan
        from harnesskit.scaffold.templates import TEMPLATES

        if template not in TEMPLATES:
            console.print(f"[red]✗[/red] Unknown template '{template}'. Valid choices: {', '.join(TEMPLATES)}")
            raise typer.Exit(1)
        if not name:
            console.print("[red]✗[/red] Provide a harness name.")
            raise typer.Exit(1)

        t = TEMPLATES[template]
        plan = ScaffoldPlan(
            name=name,
            description=t.description or f"A {template} harness.",
            purpose=t.description or template,
            domain=template,
            guardrails_needed=list(t.guardrails),
            max_turns=t.max_turns,
            cost_ceiling_usd=t.cost_ceiling_usd,
        )
        target = directory or Path(name)
        try:
            generate_harness(plan, target)
        except FileExistsError as e:
            console.print(f"[red]✗[/red] {e}")
            raise typer.Exit(1)
        console.print(f"[green]✓[/green] Generated {template} harness at {target}/ (offline, no LLM call)")
        console.print("  Wire up the tools/*.py stubs, add eval cases, then `harness eval .`")
        return

    if spec:
        from harnesskit.scaffold import generate_harness, infer_plan, validate_and_fix

        try:
            plan = infer_plan(spec)
        except ImportError as e:
            _fail(str(e), verbose, e)
        except Exception as e:  # noqa: BLE001 — provider/auth errors surfaced concisely by default
            _fail(f"Spec inference failed: {e}", verbose, e)

        target = directory or Path(name or plan.name)
        try:
            generate_harness(plan, target)
        except FileExistsError as e:
            console.print(f"[red]✗[/red] {e}")
            raise typer.Exit(1)

        console.print(f"[green]✓[/green] Generated {plan.resolved_domain()} harness at {target}/ from spec")

        result, findings = validate_and_fix(target)
        errors = [f for f in findings if f.severity == Severity.error]
        warnings = [f for f in findings if f.severity != Severity.error]
        for f in warnings:
            console.print(f"  [yellow]{f.severity.value}[/yellow]: {f.message}")
        if errors:
            for f in errors:
                console.print(f"  [red]error[/red]: {f.message}  ({f.fix})")
            console.print(f"\n[yellow]Generated with {len(errors)} unresolved lint error(s)[/yellow] — see above.")
        else:
            console.print("\n[green]✓[/green] Self-validation: harness lint clean.")
        console.print("  Wire up the tools/*.py stubs, add eval cases, then `harness eval .`")
        return

    if not name:
        console.print("[red]✗[/red] Provide a harness name, or use --spec for a natural-language description.")
        raise typer.Exit(1)

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


def _resolve_adapter(name: str, spec, *, strict: bool = False, cache: bool = False) -> object:
    adapter_cls = ADAPTERS.get(name)
    if adapter_cls is None:
        console.print(f"[red]✗[/red] Unknown adapter '{name}'. Available: {', '.join(ADAPTERS)}")
        raise typer.Exit(1)
    if cache and name == "raw_api":
        adapter = adapter_cls(cache=True)
    else:
        if cache:
            console.print(f"[yellow]warning[/yellow]: --cache is only supported by the raw_api adapter; ignored for '{name}'")
        adapter = adapter_cls()
    warnings = check_support(spec, adapter)
    if strict and warnings:
        console.print("[red]✗[/red] Strict preflight rejected unsupported required behavior:")
        for warning in warnings:
            console.print(f"  - {warning}")
        raise typer.Exit(1)
    for warning in warnings:
        console.print(f"[yellow]warning[/yellow]: {warning}")
    return adapter


def _render_suite_table(suite: SuiteResult) -> Table:
    table = Table(show_header=True, header_style="bold")
    table.add_column("case")
    table.add_column("pass")
    table.add_column("turns")
    table.add_column("cost")
    table.add_column("scores")
    for r in suite.results:
        mark = "[green]✓[/green]" if r.passed else "[red]✗[/red]"
        if r.trajectory is None:
            score_str = f"{r.status.value}: {r.error or 'no trajectory'}"
            table.add_row(r.case.id, mark, "-", "-", score_str)
            continue
        if not r.is_scored:
            score_str = "unscored: add an outcome, trajectory, or budget assertion"
        else:
            score_str = ", ".join(f"{s.name}={s.value:.2f}" for s in r.scores if not s.passed) or "all pass"
        cost_str = "unavailable" if r.trajectory.has_unavailable_cost else f"${r.trajectory.total_cost_usd:.4f}"
        table.add_row(r.case.id, mark, str(r.trajectory.turns), cost_str, score_str)
    return table


def _truncate(text: str | None, limit: int = 400) -> str:
    if not text:
        return "(none)"
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [dim]({len(text) - limit} more chars)[/dim]"


def render_trajectory(trajectory: Trajectory, console: Console, label: str | None = None) -> None:
    """Pretty-print a single Trajectory step by step — shared by `harness
    show` and `harness eval --failed-only` so a failing case is debuggable
    inline without opening JSON."""
    from harnesskit.trace.schema import StepType

    if label:
        console.print(f"\n[bold]== {label} ==[/bold]")
    console.print(f"[dim]input:[/dim] {_truncate(trajectory.input, 200)}")
    for i, step in enumerate(trajectory.steps, start=1):
        if step.step_type == StepType.llm_call:
            console.print(f"[bold cyan]{i}. llm_call[/bold cyan]  tokens_in={step.tokens_in} tokens_out={step.tokens_out}")
            console.print(f"   [dim]input:[/dim]  {_truncate(step.input)}")
            console.print(f"   [dim]output:[/dim] {_truncate(step.output)}")
        elif step.step_type == StepType.tool_call:
            console.print(f"[bold magenta]{i}. tool_call[/bold magenta]  tool={step.tool_name}")
            console.print(f"   [dim]args:[/dim]   {step.tool_args}")
            console.print(f"   [dim]result:[/dim] {_truncate(step.tool_result)}")
        else:
            console.print(f"[bold yellow]{i}. {step.step_type.value}[/bold yellow]  {step.note or ''}")
    cost_str = "unavailable" if trajectory.has_unavailable_cost else f"${trajectory.total_cost_usd:.4f}"
    console.print(
        f"[bold]stopped_reason:[/bold] {trajectory.stopped_reason}  "
        f"[bold]total_tokens:[/bold] {trajectory.total_tokens}  "
        f"[bold]total_cost:[/bold] {cost_str}"
    )
    console.print(f"[bold]final_output:[/bold] {_truncate(trajectory.final_output)}")


@app.command()
def show(
    path: Path = typer.Argument(..., help="A saved run JSON, or a baseline-style {case_id: trajectory} JSON"),
    case: str = typer.Option(None, "--case", help="Case id to show, when `path` is a baseline-style file with multiple trajectories"),
) -> None:
    """Pretty-print a saved trajectory step by step: llm/tool calls, tokens,
    costs, and where it stopped — the debugging view for a failing eval case."""
    raw = json.loads(path.read_text())

    def _as_trajectory(data: dict) -> Trajectory:
        if isinstance(data, dict) and "schema_version" in data and "trajectory" in data:
            return Trajectory.model_validate(data["trajectory"])
        return Trajectory.model_validate(data)

    # A single run artifact: {"schema_version":..., "trajectory": {...}} or a raw Trajectory dict.
    if isinstance(raw, dict) and ("steps" in raw or "trajectory" in raw):
        render_trajectory(_as_trajectory(raw), console)
        return

    # Baseline-style {case_id: trajectory}, optionally wrapped as {"baselines": {...}}.
    cases = raw.get("baselines", raw) if isinstance(raw, dict) else raw
    if case:
        if case not in cases:
            console.print(f"[red]✗[/red] No case '{case}' in {path}. Available: {', '.join(cases)}")
            raise typer.Exit(1)
        render_trajectory(Trajectory.model_validate(cases[case]), console, label=case)
        return

    if len(cases) == 1:
        ((only_id, only_traj),) = cases.items()
        render_trajectory(Trajectory.model_validate(only_traj), console, label=only_id)
        return

    console.print(f"Multiple cases in {path} — pass --case to pick one: {', '.join(cases)}")
    raise typer.Exit(1)


def _suite_summary_line(suite: SuiteResult) -> str:
    avg_cost = "unavailable" if suite.avg_cost_usd is None else f"${suite.avg_cost_usd:.4f}"
    line = (
        f"\npass_rate={suite.pass_rate:.2f}  avg_turns={suite.avg_turns:.1f}  "
        f"avg_cost={avg_cost}  dup_tool_calls={suite.total_duplicate_tool_calls}"
    )
    if suite.errored_case_ids:
        line += f"\n[red]errored:[/red] {', '.join(suite.errored_case_ids)}"
    if suite.unavailable_cost_case_ids:
        line += f"\n[yellow]unavailable cost for:[/yellow] {', '.join(suite.unavailable_cost_case_ids)}"
    return line


@app.command()
def run(
    directory: Path = typer.Argument(Path(".")),
    input: str = typer.Option(..., "--input", help="Task to run the harness on"),
    adapter_name: str = typer.Option("raw_api", "--adapter", help=f"One of: {', '.join(ADAPTERS)}"),
    strict: bool = typer.Option(False, "--strict", help="Reject unsupported declared behavior before adapter setup"),
    cache: bool = typer.Option(False, "--cache", help="Cache model-call responses under .harness/cache/ (raw_api adapter only)"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    """Execute the harness on a single input."""
    try:
        result = load_harness(directory)
    except HarnessLoadError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)

    adapter = _resolve_adapter(adapter_name, result.spec, strict=strict, cache=cache)
    try:
        agent = adapter.build(result.spec)
        trajectory = adapter.run(agent, input)
    except ImportError as e:
        _fail(str(e), verbose, e)
    except Exception as e:  # noqa: BLE001 — provider/auth errors surfaced concisely by default
        _fail(f"Adapter run failed: {e}", verbose, e)
    path = save_trajectory(directory, trajectory)

    cost_str = "unavailable" if trajectory.has_unavailable_cost else f"${trajectory.total_cost_usd:.4f}"
    console.print(f"[bold]stopped:[/bold] {trajectory.stopped_reason}  "
                  f"[bold]turns:[/bold] {trajectory.turns}  "
                  f"[bold]cost:[/bold] {cost_str}")
    console.print(f"\n{trajectory.final_output or '(no final output)'}")
    console.print(f"\n[dim]trace saved to {path}[/dim]")


@app.command()
def eval_cmd(
    directory: Path = typer.Argument(Path(".")),
    sample: int = typer.Option(None, "--sample", help="Run only the first N cases"),
    adapter_name: str = typer.Option("raw_api", "--adapter", help=f"One of: {', '.join(ADAPTERS)}"),
    strict: bool = typer.Option(False, "--strict", help="Reject unsupported declared behavior before adapter setup"),
    save_baseline: str = typer.Option(None, "--save-baseline", help="Snapshot this run under a name for future --compare"),
    overwrite_baseline: bool = typer.Option(
        False, "--overwrite-baseline", help="Allow --save-baseline to replace an existing baseline of the same name"
    ),
    compare_baseline: str = typer.Option(None, "--compare", help="Diff this run against a saved baseline; fail on regression"),
    failed_only: bool = typer.Option(False, "--failed-only", help="Print the full step-by-step trace for every failing case"),
    parallel: int = typer.Option(1, "--parallel", help="Run up to N eval cases concurrently (default: serial)"),
    cache: bool = typer.Option(False, "--cache", help="Cache model-call responses under .harness/cache/ (raw_api adapter only)"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    """Run the harness against its eval suite."""
    try:
        result = load_harness(directory)
    except HarnessLoadError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)
    if not result.spec.eval.cases:
        console.print("[yellow]No eval cases defined in this harness.[/yellow]")
        raise typer.Exit(1)

    adapter = _resolve_adapter(adapter_name, result.spec, strict=strict, cache=cache)
    try:
        suite = run_suite(result.spec, adapter, sample=sample, max_workers=max(1, parallel))
    except ImportError as e:
        _fail(str(e), verbose, e)
    except Exception as e:  # noqa: BLE001 — provider/auth errors surfaced concisely by default
        _fail(f"Adapter run failed: {e}", verbose, e)
    for r in suite.results:
        if r.trajectory is not None:
            save_trajectory(directory, r.trajectory, run_id=r.case.id)

    console.print(_render_suite_table(suite))
    console.print(_suite_summary_line(suite))

    if failed_only:
        failing = [r for r in suite.results if not r.passed]
        if not failing:
            console.print("\n[green]✓[/green] no failing cases to show")
        for r in failing:
            if r.trajectory is None:
                console.print(f"\n[bold]== {r.case.id} ==[/bold]  [red]{r.status.value}[/red]: {r.error or 'no trajectory'}")
                continue
            render_trajectory(r.trajectory, console, label=r.case.id)

    if save_baseline:
        scored_trajectories = {r.case.id: r.trajectory for r in suite.results if r.trajectory is not None}
        provenance = current_provenance(
            directory,
            model_id=result.spec.model.model_id,
            adapter=adapter_name,
            harness_name=result.spec.metadata.name,
            harness_version=result.spec.metadata.version,
        )
        try:
            path = save_baseline_fn(directory, save_baseline, scored_trajectories, overwrite=overwrite_baseline, provenance=provenance)
        except FileExistsError as e:
            console.print(f"[red]✗[/red] {e}")
            raise typer.Exit(1)
        console.print(f"[dim]baseline '{save_baseline}' saved to {path}[/dim]")

    if compare_baseline:
        try:
            baseline_trajectories = load_baseline(directory, compare_baseline)
        except FileNotFoundError as e:
            available = list_baselines(directory)
            console.print(f"[red]✗[/red] {e}" + (f" (available: {', '.join(available)})" if available else ""))
            raise typer.Exit(1)

        baseline_prov = load_baseline_provenance(directory, compare_baseline)
        if baseline_prov.is_legacy:
            console.print(f"[yellow]![/yellow] baseline '{compare_baseline}': no provenance recorded (legacy baseline)")
        else:
            current_model = result.spec.model.model_id
            current_git_sha = _git_sha_for_warning(directory)
            mismatches = []
            if baseline_prov.model_id != current_model:
                mismatches.append(f"model={baseline_prov.model_id!r} vs {current_model!r}")
            if baseline_prov.adapter != adapter_name:
                mismatches.append(f"adapter={baseline_prov.adapter!r} vs {adapter_name!r}")
            if baseline_prov.harness_version != result.spec.metadata.version:
                mismatches.append(f"harness_version={baseline_prov.harness_version!r} vs {result.spec.metadata.version!r}")
            if baseline_prov.git_sha != current_git_sha:
                mismatches.append(f"git={baseline_prov.git_sha!r} vs {current_git_sha!r}")
            if mismatches:
                console.print(
                    f"[bold red]⚠ baseline '{compare_baseline}' was captured under a different environment:[/bold red] "
                    + "; ".join(mismatches)
                    + " — this comparison may reflect environment drift, not a real regression."
                )

        current_cases = [case_result.case for case_result in suite.results]
        missing_case_ids = [case.id for case in current_cases if case.id not in baseline_trajectories]
        if missing_case_ids:
            console.print(
                f"[red]✗[/red] Baseline '{compare_baseline}' is missing eval case(s): {', '.join(missing_case_ids)}. "
                "Save a new baseline before comparing."
            )
            raise typer.Exit(1)
        baseline_results = [
            CaseResult(case=case, trajectory=baseline_trajectories[case.id], scores=score_trajectory(baseline_trajectories[case.id], case))
            for case in current_cases
        ]
        baseline_suite = SuiteResult(harness_name=f"{result.spec.metadata.name}@{compare_baseline}", results=baseline_results)

        try:
            reg = check_regression(baseline_suite, suite, result.spec.eval.thresholds)
            ab = compare(baseline_suite, suite)
        except SuiteComparisonError as e:
            console.print(f"[red]✗[/red] Cannot compare baseline: {e}")
            raise typer.Exit(1)
        ab_table = Table(show_header=True, header_style="bold", title=f"vs baseline '{compare_baseline}'")
        ab_table.add_column("metric")
        ab_table.add_column("baseline")
        ab_table.add_column("current")
        ab_table.add_column("Δ")
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
def replay(
    directory: Path = typer.Argument(Path(".")),
    baseline: str = typer.Option(
        ..., "--baseline", help="A baseline name saved via `harness eval --save-baseline`, or a path to a baseline JSON file"
    ),
    partial: bool = typer.Option(False, "--partial", help="Score the cases that have a trajectory even if some are missing"),
) -> None:
    """Re-score a saved baseline against the current eval suite — no adapter,
    no provider client, no API key, no network call."""
    try:
        result = load_harness(directory)
    except HarnessLoadError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)
    if not result.spec.eval.cases:
        console.print("[yellow]No eval cases defined in this harness.[/yellow]")
        raise typer.Exit(1)

    baseline_path = Path(baseline)
    if baseline_path.is_file():
        trajectories = {
            case_id: Trajectory.model_validate(data) for case_id, data in json.loads(baseline_path.read_text()).items()
        }
        baseline_label = str(baseline_path)
    else:
        try:
            trajectories = load_baseline(directory, baseline)
        except FileNotFoundError as e:
            available = list_baselines(directory)
            console.print(f"[red]✗[/red] {e}" + (f" (available: {', '.join(available)})" if available else ""))
            raise typer.Exit(1)
        baseline_label = baseline

    suite = replay_suite(result.spec, trajectories)
    expected = len(result.spec.eval.cases)
    found = len(suite.results)
    console.print(f"replaying '{baseline_label}': expected={expected}  found={found}  missing={len(suite.missing_case_ids)}")

    if suite.missing_case_ids:
        console.print(f"[yellow]missing trajectories for:[/yellow] {', '.join(suite.missing_case_ids)}")
        if not partial:
            console.print(
                "[red]✗[/red] refusing to report a complete result while cases are missing; pass --partial to score what's available"
            )
            raise typer.Exit(1)

    console.print(_render_suite_table(suite))
    console.print(_suite_summary_line(suite))

    if suite.missing_case_ids or suite.pass_rate < 1.0:
        raise typer.Exit(1)


@app.command()
def ab(
    dir1: Path,
    dir2: Path,
    sample: int = typer.Option(None, "--sample"),
    adapter_name: str = typer.Option("raw_api", "--adapter", help=f"One of: {', '.join(ADAPTERS)}"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    """A/B two harness versions on the same eval suite. Both must use the same adapter."""
    try:
        r1, r2 = load_harness(dir1), load_harness(dir2)
    except HarnessLoadError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)

    adapter = _resolve_adapter(adapter_name, r1.spec)
    try:
        suite_a = run_suite(r1.spec, adapter, sample=sample)
        suite_b = run_suite(r2.spec, adapter, sample=sample)
    except ImportError as e:
        _fail(str(e), verbose, e)
    except Exception as e:  # noqa: BLE001 — provider/auth errors surfaced concisely by default
        _fail(f"Adapter run failed: {e}", verbose, e)
    try:
        result = compare(suite_a, suite_b)
    except SuiteComparisonError as e:
        console.print(f"[red]✗[/red] Cannot run A/B comparison: {e}")
        raise typer.Exit(1)

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
def watch(
    directory: Path = typer.Argument(Path(".")),
    sample: int = typer.Option(None, "--sample", help="Also re-run this many eval cases on change (costs API calls)"),
    adapter_name: str = typer.Option("raw_api", "--adapter"),
) -> None:
    """Re-lint (and optionally re-eval a sample) on every file change under the harness directory."""
    import time

    console.print(f"[dim]watching {directory} — Ctrl+C to stop[/dim]")

    def _fingerprint() -> dict[str, float]:
        return {
            str(p): p.stat().st_mtime
            for p in directory.rglob("*")
            if p.is_file() and ".harness" not in p.parts and "__pycache__" not in p.parts
        }

    def _check_once() -> None:
        try:
            result = load_harness(directory)
        except HarnessLoadError as e:
            console.print(f"[red]✗[/red] {e}")
            return
        findings = lint(result.spec)
        errors = [f for f in findings if f.severity == Severity.error]
        warnings = [f for f in findings if f.severity != Severity.error]
        status = "[green]✓ lint clean[/green]" if not findings else f"[yellow]{len(warnings)} warning(s)[/yellow], [red]{len(errors)} error(s)[/red]"
        console.print(f"[dim]{time.strftime('%H:%M:%S')}[/dim] {status}")
        for f in findings:
            color = {Severity.error: "red", Severity.warning: "yellow", Severity.info: "cyan"}[f.severity]
            console.print(f"  [{color}]{f.severity.value}[/{color}] {f.rule}: {f.message}")

        if sample and result.spec.eval.cases and not errors:
            try:
                adapter = ADAPTERS[adapter_name]()
                suite = run_suite(result.spec, adapter, sample=sample)
                avg_cost = "unavailable" if suite.avg_cost_usd is None else f"${suite.avg_cost_usd:.4f}"
                console.print(f"  eval (sample={sample}): pass_rate={suite.pass_rate:.2f} avg_cost={avg_cost}")
            except Exception as e:  # noqa: BLE001 — keep the watch loop alive on a bad run
                console.print(f"  [red]eval failed:[/red] {e}")

    last = _fingerprint()
    _check_once()
    try:
        while True:
            time.sleep(1.0)
            current = _fingerprint()
            if current != last:
                last = current
                _check_once()
    except KeyboardInterrupt:
        console.print("\n[dim]stopped watching[/dim]")


@app.command(name="export")
def export_cmd(
    directory: Path = typer.Argument(Path(".")),
    output: Path = typer.Option(None, "--output", "-o", help="Bundle path (default: <harness-name>.harn)"),
    include_eval_summary: str = typer.Option(
        None, "--include-eval-summary", help="Embed a self-reported summary of this saved baseline (see: harness eval --save-baseline)"
    ),
) -> None:
    """Package the harness as a portable .harn bundle."""
    try:
        result = load_harness(directory)
    except HarnessLoadError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)

    out = output or Path(f"{result.spec.metadata.name}.harn")
    try:
        bundle = export_bundle(directory, out, include_eval_summary=include_eval_summary)
    except FileNotFoundError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)
    console.print(f"[green]✓[/green] Wrote {bundle.path} ({bundle.file_count} files)")
    if bundle.required_mcp_servers:
        console.print(f"  requires MCP servers: {', '.join(bundle.required_mcp_servers)}")
    if bundle.eval_summary:
        s = bundle.eval_summary
        avg_cost = "unavailable" if s["avg_cost_usd"] is None else f"${s['avg_cost_usd']:.4f}"
        console.print(
            f"  [dim]eval_summary (self-reported, unverified):[/dim] pass_rate={s['pass_rate']:.2f} "
            f"avg_cost={avg_cost} over {s['case_count']} case(s) from baseline '{s['baseline_name']}'"
        )


@app.command(name="import")
def import_cmd(
    bundle: Path,
    directory: Path = typer.Option(None, help="Where to unpack it (default: ./<harness-name>)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the code-review confirmation prompt"),
) -> None:
    """Unpack a .harn bundle into a harness directory."""
    try:
        preview = preview_bundle(bundle)
    except (KeyError, FileNotFoundError) as e:
        console.print(f"[red]✗[/red] Not a valid .harn bundle: {e}")
        raise typer.Exit(1)

    console.print(f"Bundle: {preview.manifest['harness_name']} v{preview.manifest['harness_version']}")
    console.print(f"  {len(preview.files)} files")
    summary = preview.manifest.get("eval_summary")
    if summary:
        console.print(
            f"  [dim]author-reported eval:[/dim] pass_rate={summary['pass_rate']:.2f} "
            f"over {summary['case_count']} case(s) — [yellow]self-reported by the author, not independently verified[/yellow]"
        )
    if preview.code_files:
        console.print(f"  [yellow]{len(preview.code_files)} Python file(s) will be added — review before running this harness:[/yellow]")
        for f in preview.code_files:
            console.print(f"    - {f}")
    if not yes:
        confirmed = typer.confirm("Unpack this bundle?", default=False)
        if not confirmed:
            console.print("Aborted.")
            raise typer.Exit(1)

    target = directory or Path(preview.manifest["harness_name"])
    try:
        import_bundle(bundle, target)
    except (FileExistsError, ValueError) as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)
    console.print(f"[green]✓[/green] Unpacked to {target}/ — checksums verified")


@app.command()
def params(
    directory: Path = typer.Argument(Path(".")),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
) -> None:
    """List a harness's declared trainable parameters (its `requires_grad` set)."""
    from harnesskit.train import HarnessModule, ParameterError

    try:
        module = HarnessModule(directory)
    except (HarnessLoadError, ParameterError) as e:
        _fail(str(e), False)
    described = [p.describe() for p in module.parameters()]
    if as_json:
        console.print_json(json.dumps(described))
        return
    if not described:
        console.print("[yellow]No trainable parameters declared.[/yellow] Add a `trainable:` section to harness.yaml.")
        return
    table = Table(show_header=True, header_style="bold")
    for col in ("name", "requires_grad", "kind", "target", "constraints", "value"):
        table.add_column(col)
    for d in described:
        constraints = ", ".join(f"{k}={d[k]}" for k in ("choices", "min", "max", "max_chars") if k in d)
        table.add_row(d["name"], "yes" if d["requires_grad"] else "[dim]frozen[/dim]", d["kind"], d["target"], constraints, _truncate(json.dumps(d["value"]), 60))
    console.print(table)
    console.print("[dim]Everything not listed as requires_grad is frozen; candidates that change it are rejected.[/dim]")


@app.command()
def train(
    directory: Path = typer.Argument(Path(".")),
    out: Path = typer.Option(..., "--out", help="Empty work directory for candidates, best harness, and train_report.json"),
    adapter_name: str = typer.Option("raw_api", "--adapter", help=f"One of: {', '.join(ADAPTERS)}, or custom:<file.py>:<factory>"),
    proposer_name: str = typer.Option("random", "--proposer", help="random | scripted:<file.json> | llm | harness:<dir>"),
    proposer_adapter: str = typer.Option("raw_api", "--proposer-adapter", help="Adapter that runs an llm/harness proposer"),
    proposer_model: str = typer.Option("claude-sonnet-5", "--proposer-model", help="Model id for --proposer llm"),
    steps: int = typer.Option(3, "--steps", min=1),
    candidates: int = typer.Option(4, "--candidates", min=1, help="Candidates proposed per step"),
    max_candidates: int = typer.Option(None, "--max-candidates", min=1),
    max_cost: float = typer.Option(None, "--max-cost", help="Admission cap in USD across evaluation and proposal spend"),
    allow_unmetered: bool = typer.Option(False, "--allow-unmetered", help="Continue under --max-cost even when some cost is unavailable"),
    min_improvement: float = typer.Option(0.0, "--min-improvement", help="Required val pass-rate gain to accept a candidate"),
    seed: int = typer.Option(0, "--seed"),
    parallel: int = typer.Option(1, "--parallel", min=1),
    as_json: bool = typer.Option(False, "--json", help="Print the full report as JSON"),
) -> None:
    """Optimize a harness's trainable parameters: propose on train evidence,
    select on val, report once on held-out test."""
    from harnesskit.train import HarnessModule, ParameterError, TrainBudget, Trainer, TrainError
    from harnesskit.train.loading import resolve_adapter_factory, resolve_proposer

    try:
        module = HarnessModule(directory)
        factory = resolve_adapter_factory(adapter_name)
        proposer = resolve_proposer(proposer_name, adapter=proposer_adapter, model_id=proposer_model)
        budget = TrainBudget(steps, candidates, max_candidates, max_cost, allow_unmetered)
        trainer = Trainer(module, factory, proposer, out, budget=budget, seed=seed, min_improvement=min_improvement, max_workers=parallel)
        result = trainer.fit()
    except (HarnessLoadError, ParameterError, TrainError, ValueError, OSError) as e:
        _fail(str(e), False)
    if as_json:
        console.print_json(json.dumps(result.to_dict(), default=str))
        return
    s = result.splits
    console.print(f"splits ({s.source}): train={len(s.train)} val={len(s.val)} test={len(s.test)}")
    table = Table(show_header=True, header_style="bold")
    for col in ("id", "step", "status", "edits", "train", "val", "reason"):
        table.add_column(col)
    color = {"accepted": "green", "rejected": "yellow", "invalid": "red", "duplicate": "dim"}
    for c in result.candidates:
        table.add_row(
            c.id, str(c.step), f"[{color[c.status]}]{c.status}[/{color[c.status]}]", _truncate(json.dumps(c.edits, default=str), 60),
            f"{c.train.pass_rate:.2f}" if c.train else "-", f"{c.val.pass_rate:.2f}" if c.val else "-", c.reason,
        )
    console.print(table)
    console.print(f"val pass rate: {result.initial_val.pass_rate:.2f} -> {result.best_val.pass_rate:.2f} (selection split; optimistic)")
    spend = result.spend.to_dict()
    cost = f"${spend['known_total_usd']:.4f} known" + ("" if spend["complete"] else f", {spend['entries_with_unavailable_cost']} entries with unavailable cost")
    console.print(f"spend: {cost}")
    console.print(f"stopped: {result.stop_reason}")
    for note in result.notes:
        console.print(f"[dim]note:[/dim] {note}")
    if result.best_diff:
        console.print("[bold]best diff:[/bold]")
        console.print(result.best_diff, markup=False, highlight=False)
    verdict_color = {"improved": "green", "inconclusive": "yellow"}.get(result.verdict, "red")
    console.print(f"[{verdict_color}]verdict: {result.verdict}[/{verdict_color}] — {result.verdict_reason}")
    console.print(f"report: {out / 'train_report.json'}")


if __name__ == "__main__":
    app()
