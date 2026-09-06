from harnesskit.format.spec import HarnessSpec
from harnesskit.linter.rules import ALL_RULES, Finding, Severity


def lint(spec: HarnessSpec) -> list[Finding]:
    findings: list[Finding] = []
    for rule in ALL_RULES:
        findings.extend(rule(spec))
    order = {Severity.error: 0, Severity.warning: 1, Severity.info: 2}
    return sorted(findings, key=lambda f: order[f.severity])


__all__ = ["lint", "Finding", "Severity"]
