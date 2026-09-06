import zipfile
from pathlib import Path

import pytest

from harnesskit.packaging import export_bundle, import_bundle, preview_bundle
from harnesskit.parser import load_harness

EXAMPLE = Path(__file__).parent.parent / "examples" / "react-web-researcher"


def test_export_bundle_includes_expected_files(tmp_path):
    bundle_path = tmp_path / "react-web-researcher.harn"
    result = export_bundle(EXAMPLE, bundle_path)

    assert result.path.exists()
    with zipfile.ZipFile(bundle_path) as zf:
        names = set(zf.namelist())
    assert "harness.yaml" in names
    assert "system_prompt.md" in names
    assert "tools/search.json" in names
    assert "manifest.json" in names


def test_export_bundle_excludes_runs_and_env(tmp_path):
    # simulate a harness dir with local run history and a stray .env, as if
    # someone ran `harness eval` before exporting
    harness_dir = tmp_path / "h"
    for name in EXAMPLE.rglob("*"):
        if name.is_file():
            rel = name.relative_to(EXAMPLE)
            dest = harness_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(name.read_bytes())
    (harness_dir / ".harness" / "runs").mkdir(parents=True)
    (harness_dir / ".harness" / "runs" / "abc.json").write_text("{}")
    (harness_dir / ".env").write_text("ANTHROPIC_API_KEY=secret")

    bundle_path = tmp_path / "out.harn"
    export_bundle(harness_dir, bundle_path)
    with zipfile.ZipFile(bundle_path) as zf:
        names = zf.namelist()
    assert not any(n.startswith(".harness/runs") for n in names)
    assert ".env" not in names


def test_import_bundle_round_trip(tmp_path):
    bundle_path = tmp_path / "out.harn"
    export_bundle(EXAMPLE, bundle_path)

    target = tmp_path / "imported"
    import_bundle(bundle_path, target)

    result = load_harness(target)
    assert result.spec.metadata.name == "react-web-researcher"
    assert len(result.spec.tools) == 2


def test_import_bundle_detects_tampering(tmp_path):
    bundle_path = tmp_path / "out.harn"
    export_bundle(EXAMPLE, bundle_path)

    tampered_path = tmp_path / "tampered.harn"
    with zipfile.ZipFile(bundle_path) as src, zipfile.ZipFile(tampered_path, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "harness.yaml":
                data = data + b"\n# tampered"
            dst.writestr(item, data)

    with pytest.raises(ValueError, match="Checksum mismatch"):
        import_bundle(tampered_path, tmp_path / "imported-tampered")


def test_preview_bundle_lists_python_files(tmp_path):
    bundle_path = tmp_path / "out.harn"
    export_bundle(EXAMPLE, bundle_path)
    preview = preview_bundle(bundle_path)
    assert "tools/search.py" in preview.code_files
    assert preview.manifest["harness_name"] == "react-web-researcher"
