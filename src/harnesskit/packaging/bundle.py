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
import zipfile
from dataclasses import dataclass
from pathlib import Path

from harnesskit.format.spec import PackagingConfig
from harnesskit.parser import load_harness

BUNDLE_FORMAT_VERSION = 1

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
    have to re-read the raw manifest to find it."""
    import yaml

    forced = {"harness.yaml"}
    raw = yaml.safe_load((harness_dir / "harness.yaml").read_text()) or {}
    scaffold = raw.get("scaffold", {})
    if scaffold.get("system_prompt_is_file", True) and "system_prompt" in scaffold:
        forced.add(scaffold["system_prompt"])
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


def export_bundle(harness_dir: Path, output_path: Path) -> BundleResult:
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


def import_bundle(bundle_path: Path, target_dir: Path) -> Path:
    if target_dir.exists() and any(target_dir.iterdir()):
        raise FileExistsError(f"{target_dir} already exists and is not empty")
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        for rel, expected_sha in manifest["checksums"].items():
            actual = hashlib.sha256(zf.read(rel)).hexdigest()
            if actual != expected_sha:
                raise ValueError(f"Checksum mismatch for {rel} — bundle may be corrupted or tampered with")
        for name in zf.namelist():
            if name == "manifest.json":
                continue
            zf.extract(name, target_dir)
    return target_dir
