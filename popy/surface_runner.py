"""End-to-end anatomical visibility regression, using real Blender and Ayagami."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from .cli import run
from .compile_ir import ROOT, jsonwrite
from .surface_regression import assess, cases, diagnostic_rig, labels
from .surface_report import generate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=ROOT / "output/POPY/POPY.model3.json")
    p.add_argument("--rig", type=Path, default=ROOT / "artifacts/rig/POPY.rig.json")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/surfaces")
    p.add_argument("--source", type=Path, default=ROOT / "artifacts/POPY_source.blend")
    p.add_argument(
        "--blender",
        default=r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
    )
    p.add_argument("--reuse-reference", action="store_true")
    a = p.parse_args()
    a.out = a.out.resolve()
    a.model = a.model.resolve()
    a.rig = a.rig.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    case_file = ROOT / "artifacts/surface_cases.json"
    label_file = ROOT / "artifacts/surface_labels.json"
    jsonwrite(case_file, cases())
    jsonwrite(label_file, {n: v.tolist() for n, v in labels().items()})
    reference = ROOT / "artifacts/surface_reference"
    source_hash = hashlib.sha256(a.source.read_bytes()).hexdigest()
    if not a.reuse_reference:
        run(
            "surface_blender",
            [
                a.blender,
                "-b",
                a.source,
                "-t",
                "6",
                "--python",
                "blender/sample.py",
                "--",
                "--references-only",
                "--case-file",
                case_file,
                "--case-output",
                ROOT / "artifacts/surface_reference_cases.json",
                "--reference-dir",
                reference,
                "--surface-labels",
                label_file,
            ],
        )
    objects = json.loads((ROOT / "artifacts/samples/manifest.json").read_text())[
        "objects"
    ]
    for c in cases():
        for suffix in [".png", "_surfaces.png"] + [
            f"__{o['name']}.npy" for o in objects
        ]:
            path = reference / (c["name"] + suffix)
            if not path.exists():
                raise FileNotFoundError(path)
    env = os.environ.copy()
    env.setdefault("JAVA_HOME", r"C:\lib\java")
    env.setdefault("JAVA_OPTS", "-Xmx7g -XX:+UseG1GC -XX:+ExitOnOutOfMemoryError")
    diagnostic = diagnostic_rig(a.rig, a.out / "diagnostic_rig")
    exporter = (
        ROOT
        / "exporter/build/install/popy-cubism-exporter/bin/popy-cubism-exporter.bat"
    )
    run(
        "surface_diagnostic_export",
        [exporter, diagnostic, a.out / "diagnostic_model", "--no-cmo"],
        env,
    )
    renderer = ROOT / "rust/target/release/popy-validate.exe"
    run("surface_native", [renderer, a.model, case_file, a.out / "runtime"])
    run(
        "surface_diagnostic",
        [
            renderer,
            a.out / "diagnostic_model/POPY.model3.json",
            case_file,
            a.out / "surfaces",
            "--images-only",
        ],
    )
    result = assess(a.out / "surfaces", a.out / "assessment")
    from .native_glue_check import check

    check(a.rig, a.out / "runtime", a.out / "native_glue.json")
    baseline = ROOT / "artifacts/feature_before"
    if (baseline / "surfaces/runtime_report.json").exists():
        assess(baseline / "surfaces", baseline / "assessment")
    manifest = json.loads(a.model.read_text())
    moc = a.model.parent / manifest["FileReferences"]["Moc"]
    assert source_hash == hashlib.sha256(a.source.read_bytes()).hexdigest()
    jsonwrite(
        a.out / "provenance.json",
        {
            "moc_sha256": hashlib.sha256(moc.read_bytes()).hexdigest(),
            "rig_sha256": hashlib.sha256(a.rig.read_bytes()).hexdigest(),
            "source_sha256": source_hash,
            "cases_sha256": hashlib.sha256(case_file.read_bytes()).hexdigest(),
            "labels_sha256": hashlib.sha256(label_file.read_bytes()).hexdigest(),
            "reference_reused": a.reuse_reference,
            "source_unchanged": True,
            "summary": result["summary"],
        },
    )
    generate(a.out)


if __name__ == "__main__":
    main()
