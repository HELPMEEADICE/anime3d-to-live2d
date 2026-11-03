"""Staged, repeatable neck/collar repair on the verified POPY hair baseline."""

import argparse
import hashlib
import html
import json
import os
import shutil
import subprocess
import sys
import zipfile

from PIL import Image, ImageDraw

from .cli import run
from .collar_clearance import convert
from .collar_regression import assess, cases, labels
from .compile_ir import ROOT, jsonwrite
from .feature_regression import assess as assess_features
from .feature_regression import feature_labels
from .hair_package import archive, sha
from .native_glue_check import check
from .surface_regression import diagnostic_rig

BASE = ROOT / "artifacts/neck_repair"
RIG = BASE / "rig/POPY.rig.json"
BASELINE = ROOT / "artifacts/hair_clearance_rig/POPY.rig.json"
MODEL = ROOT / "output/POPY_neck_fixed"


def invariants():
    before, after = [json.loads(p.read_text()) for p in [BASELINE, RIG]]
    original = {m["id"]: m for m in before["meshes"]}
    targets = {d["target"] for d in after["surface_visibility"]["collar_clearance"]}
    masks_equal, tracks_equal = True, True
    for mesh in after["meshes"]:
        if mesh["id"] not in original:
            continue
        old = original[mesh["id"]]
        if mesh["id"] not in targets:
            masks_equal &= all(
                old.get(k) == mesh.get(k) for k in ["masked_by", "invert_mask"]
            )
        tracks_equal &= all(
            old.get(k) == mesh.get(k)
            for k in old
            if k not in ["masked_by", "invert_mask"]
        )
    # Compaction retains the previous binary prefix, including every visible
    # keyform and original helper. Check actual floats as well as metadata.
    old_binary = BASELINE.parent / before["keyform_data"]
    new_binary = RIG.parent / after["keyform_data"]
    with new_binary.open("rb") as stream:
        tracks_equal &= hashlib.sha256(
            stream.read(old_binary.stat().st_size)
        ).hexdigest() == sha(old_binary)
    assert tracks_equal and masks_equal
    return {
        "visible_geometry_and_order_equal": bool(tracks_equal),
        "non_neck_masks_equal": bool(masks_equal),
        "neck_targets": len(targets),
    }


def report():
    destination = BASE / "report"
    destination.mkdir(exist_ok=True)
    before, after = [
        json.loads((BASE / ("assessment_" + s) / "metrics.json").read_text())
        for s in ["before", "after"]
    ]
    rig = json.loads(RIG.read_text())
    legacy = [
        json.loads((BASE / p / "runtime_report.json").read_text())
        for p in ["legacy_runtime", "legacy_hair_runtime"]
    ]
    # Each native report stores one entry per evaluated case.
    legacy_count = sum(len(r["cases"]) for r in legacy)
    result = {
        "before": before,
        "after": after,
        "invariants": invariants(),
        "native_glue": json.loads((BASE / "native_glue.json").read_text()),
        "legacy_runtime_poses": legacy_count,
        "moc_sha256": sha(MODEL / "POPY.moc3"),
        "baseline_moc_sha256": sha(ROOT / "output/POPY_hair_clearance/POPY.moc3"),
        "model": {
            k: rig["diagnostics"][k]
            for k in [
                "artmesh_count",
                "vertices",
                "triangles",
                "glues",
                "animated_glues",
                "masked_drawables",
            ]
        },
        "collar_clearance": rig["diagnostics"]["collar_clearance"],
        "source_blend_sha256": sha(ROOT / "artifacts/POPY_source.blend"),
        "reference_blend_sha256": sha(ROOT / "artifacts/POPY_hair_clearance.blend"),
        "limitations": [
            "Approximate curved neck patches with support planes and interpolate their 105 sampled clip fields; extreme compound pose residuals remain.",
            "This update exports the quality MOC only. Prior CMO3 packages do not contain this collar fix.",
            "Blender source geometry is unchanged: the reported defect was reproduced in native sorting, absent in the Blender reference.",
        ],
    }
    result["features"] = {
        "before": json.loads(
            (ROOT / "artifacts/hair_feature_assessment/after/metrics.json").read_text()
        ),
        "after": json.loads((BASE / "features_assessment/metrics.json").read_text()),
    }
    for side in ["before", "after"]:
        timing = BASE / ("timing_" + side) / "runtime_report.json"
        if timing.exists():
            values = sorted(
                c["pose_to_gpu_complete_ms"]
                for c in json.loads(timing.read_text())["cases"]
            )
            result.setdefault("timing", {})[side] = {
                "poses": len(values),
                "median_pose_to_gpu_complete_ms": values[len(values) // 2],
            }
    jsonwrite(destination / "metrics.json", result)
    preview = Image.new("RGB", (1200, 260), "#222222")
    draw = ImageDraw.Draw(preview)
    for column, (folder, title) in enumerate(
        [("reference", "Blender reference"), ("before", "Before"), ("after", "After")]
    ):
        for case in cases():
            source = BASE / folder / (case["name"] + ".png")
            target = destination / folder / source.name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(source, target)
        image = Image.open(BASE / folder / "reported_neck.png").convert("RGB")
        preview.paste(
            image.crop((320, 435, 440, 505)).resize((400, 233)), (column * 400, 27)
        )
        draw.text((column * 400 + 10, 5), title, fill="white")
    preview.save(destination / "comparison.jpg", quality=95)
    for side in ["before", "after"]:
        directory = destination / ("error_" + side)
        directory.mkdir(exist_ok=True)
        for case in cases():
            shutil.copy2(
                BASE / ("assessment_" + side) / (case["name"] + "_errors.png"),
                directory / (case["name"] + ".png"),
            )
    options = "".join(f"<option>{html.escape(c['name'])}</option>" for c in cases())
    page = """<!doctype html><meta charset="utf-8"><title>POPY neck/collar repair</title>
<style>body{background:#222;color:#eee;font:16px sans-serif;margin:24px}img{max-width:100%}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.grid img{width:100%}pre{white-space:pre-wrap}</style>
<h1>POPY neck/collar repair</h1><p>Blender source depth, preceding hair model, repaired native MOC. All 79 cases use the same canvas and pose. Metrics use original neck bounds and 1px boundary tolerance.</p>
<img src="comparison.jpg"><p><select id="pose">OPTIONS</select></p>
<div class="grid"><div>Blender reference<img id="reference"></div><div>Before<img id="before"></div><div>After<img id="after"></div></div>
<p>Red: excess neck exposure. Blue: missing neck. 1px tolerance.</p><div class="grid"><div>Before errors<img id="error_before"></div><div>After errors<img id="error_after"></div></div>
<pre>SUMMARY</pre><p><a href="metrics.json">Complete measured results and provenance</a></p>
<script>const pose=document.getElementById('pose');function update(){for(const id of ['reference','before','after','error_before','error_after'])document.getElementById(id).src=id+'/'+encodeURIComponent(pose.value)+'.png'}pose.onchange=update;update();</script>"""
    summary = {k: result[k] for k in ["model", "invariants", "limitations"]}
    summary.update(
        before=before["summary"],
        after=after["summary"],
        temporal_before=before["temporal_false_change"],
        temporal_after=after["temporal_false_change"],
    )
    (destination / "index.html").write_text(
        page.replace("OPTIONS", options).replace(
            "SUMMARY", html.escape(json.dumps(summary, indent=2))
        ),
        encoding="utf8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return result


def package():
    metrics = json.loads((BASE / "report/metrics.json").read_text())
    assert metrics["moc_sha256"] == sha(MODEL / "POPY.moc3")
    for name in ["collar_python_tests", "collar_rust_tests"]:
        log = ROOT / "artifacts/logs" / (name + ".log")
        shutil.copy2(log, BASE / "report" / log.name)
    jsonwrite(
        MODEL / "SHA256.json",
        {
            p.name: sha(p)
            for p in MODEL.iterdir()
            if p.is_file() and p.name != "SHA256.json"
        },
    )
    delivery = ROOT / "delivery"
    delivery.mkdir(exist_ok=True)
    archive(MODEL, delivery / "POPY_neck_fixed.zip", "POPY")
    archive(BASE / "report", delivery / "POPY_neck_repair_report.zip")
    tracked = (
        subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
        )
        .stdout.decode("utf8")
        .split("\0")
    )
    source_zip = delivery / "neck_converter_source.zip"
    with zipfile.ZipFile(source_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in filter(None, tracked):
            path = ROOT / name
            if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
                raise ValueError("Source bundle path escapes repository")
            z.write(path, name)
    with zipfile.ZipFile(source_zip) as z:
        assert z.testzip() is None
    jsonwrite(
        delivery / "SHA256_neck_repair.json",
        {
            name: sha(delivery / name)
            for name in [
                "POPY_neck_fixed.zip",
                "POPY_neck_repair_report.zip",
                "neck_converter_source.zip",
            ]
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "all",
            "prepare",
            "rig",
            "export",
            "reference",
            "validate",
            "report",
            "test",
            "package",
        ],
    )
    parser.add_argument(
        "--blender",
        default=r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
    )
    args = parser.parse_args()
    os.chdir(ROOT)
    BASE.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.setdefault("JAVA_HOME", r"C:\lib\java")
    env.setdefault("JAVA_OPTS", "-Xmx7g -XX:+UseG1GC -XX:+ExitOnOutOfMemoryError")
    exporter = (
        ROOT
        / "exporter/build/install/popy-cubism-exporter/bin/popy-cubism-exporter.bat"
    )
    renderer = ROOT / "rust/target/release/popy-validate.exe"
    stages = (
        [
            "prepare",
            "rig",
            "export",
            "reference",
            "validate",
            "report",
            "test",
            "package",
        ]
        if args.stage == "all"
        else [args.stage]
    )
    for stage in stages:
        if stage == "prepare":
            jsonwrite(BASE / "cases.json", cases())
            jsonwrite(
                BASE / "labels.json", {k: v.tolist() for k, v in labels().items()}
            )
            (BASE / "samples").mkdir(exist_ok=True)
            shutil.copy2(
                ROOT / "artifacts/samples/manifest.json", BASE / "samples/manifest.json"
            )
        elif stage == "rig":
            convert(BASELINE, RIG.parent)
        elif stage == "export":
            run("collar_export", [exporter, RIG, MODEL, "--no-cmo"], env)
        elif stage == "reference":
            source = ROOT / "artifacts/POPY_hair_clearance.blend"
            before = sha(source)
            run(
                "collar_reference",
                [
                    args.blender,
                    "-b",
                    source,
                    "-t",
                    "6",
                    "--python-exit-code",
                    "1",
                    "--python",
                    "blender/sample.py",
                    "--",
                    "--out",
                    BASE / "samples",
                    "--references-only",
                    "--case-file",
                    BASE / "cases.json",
                    "--case-output",
                    BASE / "reference_cases.json",
                    "--reference-dir",
                    BASE / "reference",
                    "--surface-labels",
                    BASE / "labels.json",
                ],
            )
            assert before == sha(source)
        elif stage == "validate":
            for side, rig, model in [
                ("before", BASELINE, ROOT / "output/POPY_hair_clearance"),
                ("after", RIG, MODEL),
            ]:
                diagnostic = diagnostic_rig(
                    rig, BASE / ("diagnostic_" + side), labels()
                )
                run(
                    "collar_diagnostic_export_" + side,
                    [exporter, diagnostic, BASE / ("model_" + side), "--no-cmo"],
                    env,
                )
                run(
                    "collar_diagnostic_runtime_" + side,
                    [
                        renderer,
                        BASE / ("model_" + side) / "POPY.model3.json",
                        BASE / "cases.json",
                        BASE / ("surfaces_" + side),
                        "--images-only",
                    ],
                )
                run(
                    "collar_runtime_" + side,
                    [
                        renderer,
                        model / "POPY.model3.json",
                        BASE / "cases.json",
                        BASE / side,
                        "--images-only",
                    ],
                )
                assess(
                    BASE / "reference",
                    BASE / ("surfaces_" + side),
                    BASE / ("assessment_" + side),
                    BASE / "cases.json",
                )
            for file, folder in [
                ("surface_cases.json", "legacy_runtime"),
                ("hair_cases.json", "legacy_hair_runtime"),
            ]:
                run(
                    "collar_" + folder,
                    [
                        renderer,
                        MODEL / "POPY.model3.json",
                        ROOT / "artifacts" / file,
                        BASE / folder,
                    ],
                )
            check(RIG, BASE / "legacy_runtime", BASE / "native_glue.json")
            features = diagnostic_rig(
                RIG, BASE / "features_rig", feature_labels(BASELINE)
            )
            run(
                "collar_feature_export",
                [exporter, features, BASE / "features_model", "--no-cmo"],
                env,
            )
            run(
                "collar_feature_runtime",
                [
                    renderer,
                    BASE / "features_model/POPY.model3.json",
                    ROOT / "artifacts/hair_cases.json",
                    BASE / "features_runtime",
                    "--images-only",
                ],
            )
            assess_features(
                ROOT / "artifacts/hair_feature_reference",
                BASE / "features_runtime",
                BASE / "features_assessment",
            )
            for side, model in [
                ("before", ROOT / "output/POPY_hair_clearance"),
                ("after", MODEL),
            ]:
                run(
                    "collar_timing_" + side,
                    [
                        renderer,
                        model / "POPY.model3.json",
                        BASE / "cases.json",
                        BASE / ("timing_" + side),
                        "--images-only",
                    ],
                )
        elif stage == "report":
            report()
        elif stage == "test":
            run(
                "collar_python_tests",
                [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            )
            run(
                "collar_rust_tests",
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
        elif stage == "package":
            package()


if __name__ == "__main__":
    main()
