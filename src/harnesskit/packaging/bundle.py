"""Packaging / export format (design doc §10.3): a `.harn` bundle is a ZIP
containing harness.yaml, every file it references, and a manifest.json with
checksums. Explicitly excludes secrets, run/baseline history, and anything
adapter-specific — a bundle is meant to be portable across adapters, not a
snapshot of one particular run.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from harnesskit.format.spec import PackagingConfig
from harnesskit.parser import load_harness
from harnesskit.trace import load_baseline

BUNDLE_FORMAT_VERSION = 1

# bundle_format_version values this build knows how to import. Bump when the
# manifest/checksum shape changes in a way older code can't safely read.
SUPPORTED_BUNDLE_FORMAT_VERSIONS = {1}


class BundleValidationError(ValueError):
    """Raised when a bundle fails structural or security validation, before
    any member has been extracted. Checksums prove a file matches what the
    manifest says it should be — they do not prove the manifest itself, or
    the bundle's publisher, is trustworthy."""

_ALWAYS_EXCLUDE = [
    ".env",
    ".env.*",
    "__pycache__/*",
    "*.pyc",
    ".harness/runs/*",
    ".harness/baselines/*",
    ".git/*",
]


def _is_excluded(rel_path: str, exclude_patterns: list[str]) -> bool:
    patterns = exclude_patterns + _ALWAYS_EXCLUDE
    return any(fnmatch.fnmatch(rel_path, pat.rstrip("/") + "/*") or fnmatch.fnmatch(rel_path, pat) for pat in patterns)


def _forced_includes(harness_dir: Path) -> set[str]:
    """Files always bundled regardless of packaging.include: the manifest
    itself, plus whatever it points scaffold.system_prompt at — the loader
    resolves that reference into inline text before HarnessSpec validation,
    so by the time we have a HarnessSpec the original path is gone and we
    have to re-read the raw manifest to find it. Files named by `trainable:`
    file/json targets are forced too, so an exported trained harness never
    silently loses its trained values."""
    import yaml

    forced = {"harness.yaml"}
    raw = yaml.safe_load((harness_dir / "harness.yaml").read_text()) or {}
    scaffold = raw.get("scaffold", {})
    if scaffold.get("system_prompt_is_file", True) and "system_prompt" in scaffold:
        forced.add(scaffold["system_prompt"])
    for param in raw.get("trainable") or []:
        target = str(param.get("target", ""))
        if target.startswith("file:"):
            forced.add(Path(target[len("file:"):]).as_posix())
        elif target.startswith("json:"):
            forced.add(Path(target[len("json:"):].partition("#")[0]).as_posix())
    return forced


def _iter_included_files(harness_dir: Path, packaging: PackagingConfig):
    forced = _forced_includes(harness_dir)
    for path in sorted(harness_dir.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(harness_dir).as_posix()
        if _is_excluded(rel, packaging.exclude):
            continue
        if rel in forced or any(
            rel == inc.rstrip("/") or rel.startswith(inc.rstrip("/") + "/") for inc in packaging.include
        ):
            yield path, rel


@dataclass
class BundleResult:
    path: Path
    file_count: int
    required_mcp_servers: list[str]
    required_adapters: list[str]
    eval_summary: dict | None = None


def _self_reported_eval_summary(harness_dir: Path, spec, baseline_name: str) -> dict:
    """A harness's own eval results, embedded on export so a consumer can see
    something before adopting it (issue #1) — explicitly labeled as
    self-reported and unverified. This is NOT a benchmark result: it's
    whatever the harness author's own eval.cases happened to score against
    whatever adapter/model they ran locally. harnesskit has no mechanism to
    verify it and never will (design doc §13 — no gatekept task suite)."""
    from harnesskit.eval import replay_suite

    trajectories = load_baseline(harness_dir, baseline_name)
    suite = replay_suite(spec, trajectories)
    return {
        "self_reported": True,
        "verified_by": None,
        "note": "Scored by the harness author's own eval.cases, replayed from a locally saved baseline. "
                "Not independently verified — treat as a claim, not a benchmark result.",
        "baseline_name": baseline_name,
        "case_count": len(suite.results),
        "pass_rate": suite.pass_rate,
        "avg_turns": suite.avg_turns,
        "avg_cost_usd": suite.avg_cost_usd,
    }


def export_bundle(harness_dir: Path, output_path: Path, include_eval_summary: str | None = None) -> BundleResult:
    result = load_harness(harness_dir)
    spec = result.spec

    files = list(_iter_included_files(harness_dir, spec.packaging))
    checksums = {}
    for path, rel in files:
        checksums[rel] = hashlib.sha256(path.read_bytes()).hexdigest()

    required_mcp = [t.ref for t in spec.tools if t.source.value == "mcp"]
    manifest = {
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "harness_name": spec.metadata.name,
        "harness_version": spec.metadata.version,
        "required_adapters": [],  # adapter code is never bundled — the format is adapter-agnostic
        "required_mcp_servers": required_mcp,
        "required_model_providers": [spec.model.provider],
        "checksums": checksums,
    }
    if include_eval_summary:
        manifest["eval_summary"] = _self_reported_eval_summary(harness_dir, spec, include_eval_summary)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, rel in files:
            zf.write(path, rel)
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

    return BundleResult(
        path=output_path,
        file_count=len(files),
        required_mcp_servers=required_mcp,
        required_adapters=manifest["required_adapters"],
        eval_summary=manifest.get("eval_summary"),
    )


@dataclass
class ImportPreview:
    files: list[str]
    code_files: list[str]  # .py files — reviewed before anything executes (design doc §10.4)
    manifest: dict


def preview_bundle(bundle_path: Path) -> ImportPreview:
    with zipfile.ZipFile(bundle_path) as zf:
        names = [n for n in zf.namelist() if n != "manifest.json"]
        manifest = json.loads(zf.read("manifest.json"))
    code_files = [n for n in names if n.endswith(".py")]
    return ImportPreview(files=names, code_files=code_files, manifest=manifest)


def _is_safe_member_path(name: str) -> bool:
    """Reject anything that could escape the extraction directory: absolute
    paths and any `..` path segment."""
    if name in ("", "."):
        return True
    p = PurePosixPath(name)
    if p.is_absolute():
        return False
    return ".." not in p.parts


def _member_is_symlink(info: zipfile.ZipInfo) -> bool:
    # Unix file mode is packed into the top 16 bits of external_attr when the
    # archive was written on a Unix system; S_IFLNK marks a symlink entry,
    # which could otherwise point outside the extraction directory.
    mode = info.external_attr >> 16
    return stat.S_ISLNK(mode)


def _validate_bundle_members(zf: zipfile.ZipFile) -> dict:
    """Validate the archive's member set against its own manifest, and
    reject anything unsafe, before a single byte is extracted:

    - the exact file SET must match (nothing in the archive that isn't in
      the checksum manifest, and nothing in the manifest missing from the
      archive) — an unlisted extra file is exactly how an attacker could
      smuggle something past a checksum-only check.
    - no duplicate member names.
    - no unsafe member paths (absolute, or containing `..`) and no symlinks.
    - a `bundle_format_version` this build understands.
    """
    infos = zf.infolist()
    names = [i.filename for i in infos]

    if "manifest.json" not in names:
        raise BundleValidationError("Bundle is missing manifest.json")

    if len(names) != len(set(names)):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise BundleValidationError(f"Bundle contains duplicate member(s): {', '.join(dupes)}")

    manifest = json.loads(zf.read("manifest.json"))
    version = manifest.get("bundle_format_version")
    if version not in SUPPORTED_BUNDLE_FORMAT_VERSIONS:
        raise BundleValidationError(
            f"Unsupported bundle_format_version={version!r}; this build of harnesskit "
            f"understands {sorted(SUPPORTED_BUNDLE_FORMAT_VERSIONS)}."
        )

    checksums = manifest.get("checksums", {})
    archive_files = {n for n in names if n != "manifest.json"}
    manifest_files = set(checksums.keys())

    extra = archive_files - manifest_files
    missing = manifest_files - archive_files
    if extra or missing:
        details = []
        if extra:
            details.append(f"present in bundle but not listed in manifest: {', '.join(sorted(extra))}")
        if missing:
            details.append(f"listed in manifest but missing from bundle: {', '.join(sorted(missing))}")
        raise BundleValidationError(
            "Bundle member set does not match its checksum manifest exactly (" + "; ".join(details) + ")"
        )

    for info in infos:
        if not _is_safe_member_path(info.filename):
            raise BundleValidationError(f"Unsafe member path in bundle: {info.filename!r}")
        if _member_is_symlink(info):
            raise BundleValidationError(f"Bundle contains a symlink member, which is not allowed: {info.filename!r}")

    return manifest


def import_bundle(bundle_path: Path, target_dir: Path) -> Path:
    if target_dir.exists() and any(target_dir.iterdir()):
        raise FileExistsError(f"{target_dir} already exists and is not empty")

    with zipfile.ZipFile(bundle_path) as zf:
        # Validate the full member set, paths, and schema version FIRST —
        # nothing is written to target_dir until every check below passes.
        manifest = _validate_bundle_members(zf)

        for rel, expected_sha in manifest["checksums"].items():
            actual = hashlib.sha256(zf.read(rel)).hexdigest()
            if actual != expected_sha:
                raise ValueError(f"Checksum mismatch for {rel} — bundle may be corrupted or tampered with")

        target_dir.mkdir(parents=True, exist_ok=True)
        target_resolved = target_dir.resolve()
        for name in zf.namelist():
            if name == "manifest.json":
                continue
            dest = (target_dir / name).resolve()
            if dest != target_resolved and target_resolved not in dest.parents:
                raise BundleValidationError(f"Bundle member would extract outside target directory: {name!r}")
            zf.extract(name, target_dir)
    return target_dir
