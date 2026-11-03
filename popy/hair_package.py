"""Package only the actually validated hair-repair artifacts, keeping assets local."""

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

from .compile_ir import ROOT, jsonwrite


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(source, destination, prefix=""):
    files = sorted(p for p in source.rglob("*") if p.is_file())
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as z:
        for path in files:
            z.write(
                path, str(Path(prefix) / path.relative_to(source)).replace("\\", "/")
            )
    with zipfile.ZipFile(destination) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(set(z.namelist()))


def package(include_editor=True):
    report = ROOT / "artifacts/hair_repair_report"
    metrics = json.loads((report / "metrics.json").read_text())
    model = ROOT / "output/POPY_hair_clearance"
    assert metrics["provenance"]["moc_sha256"] == sha(model / "POPY.moc3")
    blend = ROOT / "artifacts/POPY_hair_clearance.blend"
    assert metrics["provenance"]["corrected_blend_sha256"] == sha(blend)
    assert metrics["provenance"]["source_unchanged"]
    delivery = ROOT / "delivery"
    delivery.mkdir(exist_ok=True)
    directories = [model] + (
        [ROOT / "output/POPY_hair_clearance_balanced"] if include_editor else []
    )
    for directory in directories:
        if not directory.exists():
            continue
        jsonwrite(
            directory / "SHA256.json",
            {
                p.relative_to(directory).as_posix(): sha(p)
                for p in sorted(directory.rglob("*"))
                if p.is_file() and p.name != "SHA256.json"
            },
        )
    archive(model, delivery / "POPY_hair_fixed.zip", "POPY")
    editable = ROOT / "output/POPY_hair_clearance_balanced"
    if include_editor and (editable / "POPY.cmo3").exists():
        check = json.loads((editable / "cmo_roundtrip.json").read_text())
        assert check["drawables"] == metrics["model"]["artmesh_count"]
        assert check["official_editor_verified"] is False
        assert metrics["editor"]["moc_sha256"] == sha(editable / "POPY.moc3")
        assert metrics["editor"]["cmo_sha256"] == sha(editable / "POPY.cmo3")
        archive(editable, delivery / "POPY_hair_fixed_editable.zip", "POPY")
    shutil.copy2(blend, delivery / "POPY_hair_fixed.blend")
    archive(report, delivery / "POPY_hair_repair_report.zip")
    # Only repository-tracked source is bundled; ignored character assets,
    # credentials, failed experiments and generated binaries cannot enter it.
    tracked = (
        subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
        )
        .stdout.decode("utf8")
        .split("\0")
    )
    source_zip = delivery / "converter_source.zip"
    with zipfile.ZipFile(
        source_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as z:
        for name in tracked:
            if not name:
                continue
            path = ROOT / name
            if not path.resolve().is_relative_to(ROOT.resolve()) or path.is_symlink():
                raise ValueError("Source bundle path escaped repository: " + name)
            z.write(path, name)
    with zipfile.ZipFile(source_zip) as z:
        assert z.testzip() is None
    legacy_hashes = delivery / "SHA256.json"
    if legacy_hashes.exists():
        old = json.loads(legacy_hashes.read_text())
        old[source_zip.name] = sha(source_zip)
        jsonwrite(legacy_hashes, old)
    files = [
        delivery / name
        for name in [
            "POPY_hair_fixed.zip",
            "POPY_hair_fixed_editable.zip",
            "POPY_hair_fixed.blend",
            "POPY_hair_repair_report.zip",
            "converter_source.zip",
        ]
        if (delivery / name).exists()
        and (include_editor or name != "POPY_hair_fixed_editable.zip")
    ]
    jsonwrite(delivery / "SHA256_hair_repair.json", {p.name: sha(p) for p in files})
    print(json.dumps({p.name: p.stat().st_size for p in files}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-editor", action="store_true")
    package(not parser.parse_args().no_editor)
