"""Reproducible POPY hair repair on the audited source and existing verified rigs."""

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

from .cli import run
from .compile_ir import ROOT, jsonwrite
from .hair_clearance_ir import convert
from .hair_regression import assess, cases
from .surface_regression import diagnostic_rig


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "stage",
        choices=[
            "all",
            "blender",
            "rig",
            "export",
            "validate",
            "editor",
            "report",
            "test",
            "package",
        ],
    )
    p.add_argument("--source", type=Path, default=ROOT / "artifacts/POPY_source.blend")
    p.add_argument(
        "--blender",
        default=r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
    )
    p.add_argument("--no-editor", action="store_true")
    a = p.parse_args()
    os.chdir(ROOT)
    stages = (
        ["blender", "rig", "export", "validate", "editor", "report", "test", "package"]
        if a.stage == "all"
        else [a.stage]
    )
    quality = ROOT / "artifacts/hair_clearance_rig"
    balanced = ROOT / "artifacts/hair_clearance_balanced_rig"
    model = ROOT / "output/POPY_hair_clearance"
    editable = ROOT / "output/POPY_hair_clearance_balanced"
    case_file = ROOT / "artifacts/hair_cases.json"
    reference = ROOT / "artifacts/hair_clearance_reference"
    exporter = (
        ROOT
        / "exporter/build/install/popy-cubism-exporter/bin/popy-cubism-exporter.bat"
    )
    renderer = ROOT / "rust/target/release/popy-validate.exe"
    env = os.environ.copy()
    env.setdefault("JAVA_HOME", r"C:\lib\java")
    env.setdefault("JAVA_OPTS", "-Xmx7g -XX:+UseG1GC -XX:+ExitOnOutOfMemoryError")
    for stage in stages:
        if stage == "blender":
            source_hash = hashlib.sha256(a.source.read_bytes()).hexdigest()
            jsonwrite(case_file, cases())
            run(
                "hair_install",
                [
                    a.blender,
                    "-b",
                    a.source.resolve(),
                    "-t",
                    "6",
                    "--python-exit-code",
                    "1",
                    "--python",
                    "blender/hair_collision.py",
                    "--",
                    ROOT / "artifacts/POPY_hair_clearance.blend",
                ],
            )
            samples = ROOT / "artifacts/hair_collision_samples"
            samples.mkdir(exist_ok=True)
            # References-only still records the evaluated source inventory separately.
            shutil.copy2(
                ROOT / "artifacts/samples/manifest.json", samples / "manifest.json"
            )
            original_samples = ROOT / "artifacts/hair_baseline_samples"
            original_samples.mkdir(exist_ok=True)
            shutil.copy2(
                ROOT / "artifacts/samples/manifest.json",
                original_samples / "manifest.json",
            )
            run(
                "hair_original_reference",
                [
                    a.blender,
                    "-b",
                    a.source.resolve(),
                    "-t",
                    "6",
                    "--python-exit-code",
                    "1",
                    "--python",
                    "blender/sample.py",
                    "--",
                    "--out",
                    original_samples,
                    "--references-only",
                    "--case-file",
                    case_file,
                    "--case-output",
                    ROOT / "artifacts/hair_original_cases.json",
                    "--reference-dir",
                    ROOT / "artifacts/hair_reference",
                    "--surface-labels",
                    ROOT / "artifacts/surface_labels.json",
                ],
            )
            run(
                "hair_blender_reference",
                [
                    a.blender,
                    "-b",
                    ROOT / "artifacts/POPY_hair_clearance.blend",
                    "-t",
                    "6",
                    "--python-exit-code",
                    "1",
                    "--python",
                    "blender/sample.py",
                    "--",
                    "--out",
                    samples,
                    "--references-only",
                    "--case-file",
                    case_file,
                    "--case-output",
                    ROOT / "artifacts/hair_clearance_cases.json",
                    "--reference-dir",
                    reference,
                    "--surface-labels",
                    ROOT / "artifacts/surface_labels.json",
                    "--collision-audit",
                ],
            )
            for case in cases():
                if not (reference / (case["name"] + "_collision.json")).exists():
                    raise RuntimeError(
                        "Blender did not produce the requested collision audit"
                    )
            after_hash = hashlib.sha256(a.source.read_bytes()).hexdigest()
            assert source_hash == after_hash, "Source .blend changed"
            jsonwrite(
                ROOT / "artifacts/hair_input.json",
                {
                    "source": str(a.source.resolve()),
                    "sha256_before": source_hash,
                    "sha256_after": after_hash,
                },
            )
        elif stage == "rig":
            convert(ROOT / "artifacts/rig/POPY.rig.json", quality)
            if not a.no_editor:
                convert(ROOT / "artifacts/rig_balanced/POPY.rig.json", balanced)
        elif stage == "export":
            run(
                "hair_quality_export",
                [exporter, quality / "POPY.rig.json", model, "--no-cmo"],
                env,
            )
        elif stage == "validate":
            diagnostic = diagnostic_rig(
                quality / "POPY.rig.json",
                ROOT / "artifacts/hair_clearance_diagnostic_rig",
            )
            run(
                "hair_diagnostic_export",
                [
                    exporter,
                    diagnostic,
                    ROOT / "artifacts/hair_clearance_diagnostic_model",
                    "--no-cmo",
                ],
                env,
            )
            for suffix, source_cases in [
                ("", case_file),
                ("_expression", ROOT / "artifacts/surface_cases.json"),
            ]:
                run(
                    "hair_runtime" + suffix,
                    [
                        renderer,
                        model / "POPY.model3.json",
                        source_cases,
                        ROOT / ("artifacts/hair_clearance" + suffix + "_runtime"),
                    ],
                )
                run(
                    "hair_surface" + suffix,
                    [
                        renderer,
                        ROOT
                        / "artifacts/hair_clearance_diagnostic_model/POPY.model3.json",
                        source_cases,
                        ROOT / ("artifacts/hair_clearance" + suffix + "_surfaces"),
                        "--images-only",
                    ],
                )
            assess(
                case_file,
                reference,
                ROOT / "artifacts/hair_clearance_surfaces",
                ROOT / "artifacts/hair_clearance_assessment/after",
            )
            baseline = ROOT / "artifacts/hair_before/surfaces"
            old_diagnostic = diagnostic_rig(
                ROOT / "artifacts/rig/POPY.rig.json",
                ROOT / "artifacts/hair_baseline_diagnostic_rig",
            )
            run(
                "hair_baseline_diagnostic_export",
                [
                    exporter,
                    old_diagnostic,
                    ROOT / "artifacts/hair_baseline_diagnostic_model",
                    "--no-cmo",
                ],
                env,
            )
            run(
                "hair_baseline_runtime",
                [
                    renderer,
                    ROOT / "output/POPY/POPY.model3.json",
                    case_file,
                    ROOT / "artifacts/hair_before/runtime",
                ],
            )
            run(
                "hair_baseline_diagnostic",
                [
                    renderer,
                    ROOT / "artifacts/hair_baseline_diagnostic_model/POPY.model3.json",
                    case_file,
                    baseline,
                    "--images-only",
                ],
            )
            assess(
                case_file,
                reference,
                baseline,
                ROOT / "artifacts/hair_clearance_assessment/before",
            )
            from .native_glue_check import check

            check(
                quality / "POPY.rig.json",
                ROOT / "artifacts/hair_clearance_expression_runtime",
                ROOT / "artifacts/hair_clearance_native_glue.json",
            )
            from .surface_regression import assess as expression_assess

            expression_assess(
                ROOT / "artifacts/hair_clearance_expression_surfaces",
                ROOT / "artifacts/hair_clearance_expression_assessment",
            )
            from .feature_regression import assess as feature_assess
            from .feature_regression import feature_labels

            feature_classes = feature_labels(ROOT / "artifacts/rig/POPY.rig.json")
            label_file = ROOT / "artifacts/hair_feature_labels.json"
            jsonwrite(label_file, {k: v.tolist() for k, v in feature_classes.items()})
            feature_reference = ROOT / "artifacts/hair_feature_reference"
            run(
                "hair_feature_blender",
                [
                    a.blender,
                    "-b",
                    ROOT / "artifacts/POPY_hair_clearance.blend",
                    "-t",
                    "6",
                    "--python-exit-code",
                    "1",
                    "--python",
                    "blender/sample.py",
                    "--",
                    "--out",
                    ROOT / "artifacts/hair_collision_samples",
                    "--references-only",
                    "--case-file",
                    case_file,
                    "--case-output",
                    ROOT / "artifacts/hair_feature_cases.json",
                    "--reference-dir",
                    feature_reference,
                    "--surface-labels",
                    label_file,
                ],
            )
            for name, rig in [
                ("before", ROOT / "artifacts/rig/POPY.rig.json"),
                ("after", quality / "POPY.rig.json"),
            ]:
                diagnostic = diagnostic_rig(
                    rig, ROOT / f"artifacts/hair_features_{name}_rig", feature_classes
                )
                run(
                    f"hair_feature_{name}_export",
                    [
                        exporter,
                        diagnostic,
                        ROOT / f"artifacts/hair_features_{name}_model",
                        "--no-cmo",
                    ],
                    env,
                )
                run(
                    f"hair_feature_{name}_runtime",
                    [
                        renderer,
                        ROOT / f"artifacts/hair_features_{name}_model/POPY.model3.json",
                        case_file,
                        ROOT / f"artifacts/hair_features_{name}_runtime",
                        "--images-only",
                    ],
                )
                feature_assess(
                    feature_reference,
                    ROOT / f"artifacts/hair_features_{name}_runtime",
                    ROOT / f"artifacts/hair_feature_assessment/{name}",
                )
        elif stage == "editor" and not a.no_editor:
            writer = env.copy()
            writer["JAVA_OPTS"] = os.environ.get(
                "POPY_CMO_WRITE_JAVA_OPTS",
                "-Xmx16g -XX:+UseG1GC -XX:+ExitOnOutOfMemoryError",
            )
            run(
                "hair_editor_export",
                [exporter, balanced / "POPY.rig.json", editable],
                writer,
            )
            writer["JAVA_OPTS"] = os.environ.get(
                "POPY_CMO_VERIFY_JAVA_OPTS",
                "-Xmx24g -XX:+UseG1GC -XX:+ExitOnOutOfMemoryError",
            )
            run(
                "hair_editor_readback",
                [exporter, "--verify-cmo", editable / "POPY.cmo3"],
                writer,
            )
            run(
                "hair_editor_runtime",
                [
                    renderer,
                    editable / "POPY.model3.json",
                    case_file,
                    ROOT / "artifacts/hair_clearance_balanced_runtime",
                ],
            )
        elif stage == "report":
            run("hair_report", [sys.executable, "-m", "popy.hair_report"])
        elif stage == "test":
            run(
                "hair_python_tests",
                [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            )
            run(
                "hair_rust_tests",
                [
                    "cargo",
                    "test",
                    "--manifest-path",
                    "rust/Cargo.toml",
                    "-p",
                    "popy-geometry",
                    "--locked",
                ],
            )
            run("hair_report_after_tests", [sys.executable, "-m", "popy.hair_report"])
        elif stage == "package":
            run(
                "hair_package",
                [sys.executable, "-m", "popy.hair_package"]
                + (["--no-editor"] if a.no_editor else []),
            )


if __name__ == "__main__":
    main()
