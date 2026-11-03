"""Repeatable source-depth facial repair, real MOC regression and local delivery."""

import argparse
import html
import json
import os
import shutil
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw

from .cli import run
from .collar_regression import assess as assess_collar
from .collar_regression import labels as collar_labels
from .compile_ir import ROOT, jsonwrite
from .face_attribution import screen
from .face_regression import (
    assess,
    assess_oral_ids,
    assess_teeth,
    cases,
    labels,
    oral_labels,
    reported_spots,
)
from .feature_regression import assess as assess_features
from .feature_regression import feature_labels
from .hair_package import archive, sha
from .native_glue_check import check
from .surface_regression import diagnostic_rig

BASE = ROOT / "artifacts/face_repair"
BASELINE = ROOT / "artifacts/neck_repair/rig/POPY.rig.json"
RIG = BASE / "final_rig/POPY.rig.json"
MODEL = ROOT / "output/POPY_face_fixed"
OLD_MODEL = ROOT / "output/POPY_neck_fixed"
EXPORTER = (
    ROOT / "exporter/build/install/popy-cubism-exporter/bin/popy-cubism-exporter.bat"
)
RENDERER = ROOT / "rust/target/release/popy-validate.exe"


def export(rig, model, label):
    env = os.environ.copy()
    env.setdefault("JAVA_HOME", r"C:\lib\java")
    env.setdefault("JAVA_OPTS", "-Xmx12g -XX:+UseG1GC -XX:+ExitOnOutOfMemoryError")
    run(label, [EXPORTER, rig, model, "--no-cmo"], env)


def render(model, case_file, destination, label, geometry=False):
    cmd = [RENDERER, model / "POPY.model3.json", case_file, destination]
    if not geometry:
        cmd.append("--images-only")
    run(label, cmd)


def primary():
    for side, rig, model in [("before", BASELINE, OLD_MODEL), ("after", RIG, MODEL)]:
        diagnostic = diagnostic_rig(rig, BASE / ("diagnostic_" + side), labels())
        export(diagnostic, BASE / ("model_" + side), "face_id_export_" + side)
        render(model, BASE / "cases.json", BASE / side, "face_rgb_" + side)
        render(
            BASE / ("model_" + side),
            BASE / "cases.json",
            BASE / ("surfaces_" + side),
            "face_id_" + side,
        )
        assess(
            BASE / "reference",
            BASE / ("surfaces_" + side),
            BASE / ("assessment_" + side),
            BASE / "cases.json",
        )


def regress():
    probes = {"front", "all_extreme", "eye_45_-30_open", "continuous_blink_-45_013"}
    legacy_cases = json.loads((ROOT / "artifacts/surface_cases.json").read_text())
    probe_cases = [c for c in legacy_cases if c["name"] in probes]
    probe_cases.extend(
        c
        for c in cases()
        if c["name"]
        in ["expression_-45_-30_1", "expression_45_30_1", "compound_-45_-30_1"]
    )
    probe_cases.append(
        {
            "name": "proxy_smile_combination",
            "parameters": {
                "ParamAngleX": 37.5,
                "ParamAngleY": -22.5,
                "ParamAngleZ": 12,
                "ParamEyeLOpen": 0,
                "ParamEyeROpen": 0,
                "ParamEyeLSmile": 0.6,
                "ParamEyeRSmile": 0.4,
                "ParamMouthOpenY": 0.7,
                "ParamBodyAngleX": 7.5,
                "ParamBodyAngleY": -7.5,
            },
        }
    )
    jsonwrite(BASE / "probes.json", probe_cases)
    render(MODEL, BASE / "probes.json", BASE / "probes", "face_native_probes", True)
    check(RIG, BASE / "probes", BASE / "native_glue.json")
    render(
        MODEL, ROOT / "artifacts/surface_cases.json", BASE / "legacy", "face_legacy_264"
    )
    for name, class_map, case_file, reference, assessor in [
        (
            "features",
            feature_labels(BASELINE),
            ROOT / "artifacts/hair_cases.json",
            ROOT / "artifacts/hair_feature_reference",
            assess_features,
        ),
        (
            "collar",
            collar_labels(),
            ROOT / "artifacts/neck_repair/cases.json",
            ROOT / "artifacts/neck_repair/reference",
            assess_collar,
        ),
    ]:
        diagnostic = diagnostic_rig(RIG, BASE / (name + "_rig"), class_map)
        export(diagnostic, BASE / (name + "_model"), "face_" + name + "_export")
        render(
            BASE / (name + "_model"),
            case_file,
            BASE / (name + "_runtime"),
            "face_" + name + "_runtime",
        )
        assessor(
            reference,
            BASE / (name + "_runtime"),
            BASE / (name + "_assessment"),
            case_file,
        )
    export(BASELINE, BASE / "exporter_baseline", "face_exporter_baseline")
    jsonwrite(
        BASE / "exporter_baseline_check.json",
        {
            "byte_identical": sha(BASE / "exporter_baseline/POPY.moc3")
            == sha(OLD_MODEL / "POPY.moc3"),
            "sha256": sha(BASE / "exporter_baseline/POPY.moc3"),
        },
    )


def oral_regression():
    for side, rig in [("before", BASELINE), ("after", RIG)]:
        diagnostic = diagnostic_rig(rig, BASE / ("oral_diag_" + side), oral_labels())
        export(diagnostic, BASE / ("oral_model_" + side), "face_oral_export_" + side)
        render(
            BASE / ("oral_model_" + side),
            BASE / "oral_cases.json",
            BASE / ("oral_id_" + side),
            "face_oral_id_" + side,
        )
        jsonwrite(
            BASE / ("oral_assessment_" + side + ".json"),
            assess_oral_ids(
                BASE / "oral_reference",
                BASE / ("oral_id_" + side),
                BASE / "oral_cases.json",
            ),
        )


def invariants():
    old, new = [json.loads(p.read_text()) for p in [BASELINE, RIG]]
    old_forms = np.memmap(BASELINE.parent / old["keyform_data"], dtype="<f4", mode="r")
    new_forms = np.memmap(RIG.parent / new["keyform_data"], dtype="<f4", mode="r")
    checked, relocations = set(), 0

    def same_form(a, b, size):
        nonlocal relocations
        if a is None or b is None:
            assert a is None and b is None
            return
        key = (a, b, size)
        if key not in checked:
            # Compaction may relocate byte-identical referenced forms. Storage
            # offsets are not geometry; compare the actual original float data.
            assert np.array_equal(old_forms[a : a + size], new_forms[b : b + size])
            checked.add(key)
            relocations += a != b

    current = {m["id"]: m for m in new["meshes"]}
    config = new["diagnostics"]["face_clearance"]["config"]
    noses = set(new["diagnostics"]["face_clearance"]["nose"])
    for m in old["meshes"]:
        n = current[m["id"]]
        for key in ["positions", "uvs", "indices", "source", "axes"]:
            assert m.get(key) == n.get(key), (m["id"], key)
        assert m.get("masked_by") == n.get("masked_by")
        assert m.get("invert_mask") == n.get("invert_mask")
        assert len(m["cells"]) == len(n["cells"])
        size = len(m["positions"])
        for a, b in zip(m["cells"], n["cells"]):
            assert {k: v for k, v in a.items() if k not in ["order", "offset"]} == {
                k: v for k, v in b.items() if k not in ["order", "offset"]
            }
            same_form(a["offset"], b["offset"], size)
            assert (
                abs(
                    b["order"]
                    - a["order"]
                    - (config["nose_order_bias"] if m["id"] in noses else 0)
                )
                < 1e-8
            )
        assert len(m["blend_shapes"]) == len(n["blend_shapes"])
        for a, b in zip(m["blend_shapes"], n["blend_shapes"]):
            assert {k: v for k, v in a.items() if k != "offsets"} == {
                k: v for k, v in b.items() if k != "offsets"
            }
            assert len(a["offsets"]) == len(b["offsets"])
            for aa, bb in zip(a["offsets"], b["offsets"]):
                same_form(aa, bb, size)
    assert old["glues"] == new["glues"][: len(old["glues"])]
    prior = json.loads((ROOT / "artifacts/neck_repair/report/metrics.json").read_text())
    assert sha(ROOT / "artifacts/POPY_source.blend") == prior["source_blend_sha256"]
    assert (
        sha(ROOT / "artifacts/POPY_hair_clearance.blend")
        == prior["reference_blend_sha256"]
    )
    return {
        "source_geometry_and_glue_preserved": True,
        "original_masks_preserved": True,
        "original_blend_unchanged": True,
        "original_unique_forms_checked": len(checked),
        "relocated_original_forms": relocations,
        "original_order_changes": {m: config["nose_order_bias"] for m in noses},
        "hidden_original_masters": sum(
            bool(m.get("face_partition_master")) for m in new["meshes"]
        ),
    }


def proxy_error(ir):
    error = 0.0
    for case in json.loads((BASE / "probes.json").read_text()):
        raw = {
            m["id"]: m
            for m in json.loads(
                (BASE / "probes" / (case["name"] + ".geometry.json")).read_text()
            )
        }
        states = {k: np.array(m["vertices"]) for k, m in raw.items()}
        for glue in ir["glues"]:
            if not glue["id"].startswith("G_FaceProxy_"):
                continue
            child = next(m for m in ir["meshes"] if m["id"] == glue["mesh_b"])
            assert raw[glue["mesh_b"]]["depth"] == child.get(
                "face_order_override", raw[glue["mesh_a"]]["depth"]
            ), (case["name"], glue["mesh_b"], "proxy order changed")
            pairs = np.array(glue["pairs"])
            delta = (
                states[glue["mesh_a"]][pairs[:, 0].astype(int)]
                - states[glue["mesh_b"]][pairs[:, 1].astype(int)]
            )
            error = max(
                error, float(np.linalg.norm(delta, axis=1).max() * ir["canvas"][0] / 2)
            )
    return error


def report():
    destination = BASE / "report"
    destination.mkdir(exist_ok=True)
    ir = json.loads(RIG.read_text())
    before, after = [
        json.loads((BASE / ("assessment_" + s) / "metrics.json").read_text())
        for s in ["before", "after"]
    ]
    screening = json.loads((BASE / "screen/conflicts.json").read_text())
    result = {
        "before": before,
        "after": after,
        "invariants": invariants(),
        "native_glue": json.loads((BASE / "native_glue.json").read_text()),
        "proxy_error_px": proxy_error(ir),
        "moc_sha256": sha(MODEL / "POPY.moc3"),
        "baseline_moc_sha256": sha(OLD_MODEL / "POPY.moc3"),
        "source_blend_sha256": sha(ROOT / "artifacts/POPY_source.blend"),
        "reference_blend_sha256": sha(ROOT / "artifacts/POPY_hair_clearance.blend"),
        "screening": {
            "poses": len(screening["cases"]),
            "conflict_triangles": len(screening["triangles"]),
            "source_ir_sha256": screening["source_ir_sha256"],
        },
        "model": {
            k: ir["diagnostics"][k]
            for k in [
                "artmesh_count",
                "vertices",
                "triangles",
                "glues",
                "animated_glues",
                "masked_drawables",
                "keyform_bytes",
            ]
        },
        "face_clearance": ir["diagnostics"]["face_clearance"],
        "scalar_grid_compression": ir["diagnostics"]["scalar_grid_compression"],
        "rig_ir_sha256": sha(RIG),
        "keyforms_sha256": sha(RIG.parent / ir["keyform_data"]),
        "legacy_rendered_poses": len(
            json.loads((BASE / "legacy/runtime_report.json").read_text())["cases"]
        ),
        "features": {
            "before": json.loads(
                (
                    ROOT / "artifacts/neck_repair/features_assessment/metrics.json"
                ).read_text()
            ),
            "after": json.loads(
                (BASE / "features_assessment/metrics.json").read_text()
            ),
        },
        "collar": {
            "before": json.loads(
                (
                    ROOT / "artifacts/neck_repair/assessment_after/metrics.json"
                ).read_text()
            ),
            "after": json.loads((BASE / "collar_assessment/metrics.json").read_text()),
        },
        "limitations": [
            "Finite source sampling and local support-plane approximations cannot certify every real-valued parameter combination.",
            "Image metrics also contain source silhouette, projection, interpolation and the prior Blender hair-clearance approximation errors.",
            "This release updates the quality MOC only. Prior CMO3 projects do not contain this repair; official Cubism Editor compatibility is not claimed.",
        ],
    }
    result["model"]["moc_bytes"] = (MODEL / "POPY.moc3").stat().st_size
    result["model"].update(
        parameters=len(ir["parameters"]),
        deformers=3 + len(ir.get("body_warps", [None])),
        deformer_depth=3,
        texture_pages=len(ir["textures"]),
        authored_geometry_keyforms=sum(len(m["cells"]) for m in ir["meshes"]),
        max_artmesh_vertices=max(len(m["positions"]) // 2 for m in ir["meshes"]),
    )
    result["model"]["textures"] = []
    for name in ir["textures"]:
        with Image.open(RIG.parent / name) as texture:
            result["model"]["textures"].append(
                {"file": name, "size": list(texture.size)}
            )
    result["teeth"] = {
        side: assess_teeth(BASE / "reference", BASE / side, BASE / "cases.json")
        for side in ["before", "after"]
    }
    result["reported_spots"] = reported_spots(BASE)
    if (BASE / "oral_assessment_after.json").exists():
        result["oral_ids"] = {
            side: json.loads((BASE / ("oral_assessment_" + side + ".json")).read_text())
            for side in ["before", "after"]
        }
    result["exporter_baseline"] = json.loads(
        (BASE / "exporter_baseline_check.json").read_text()
    )
    if (BASE / "reproduction.json").exists():
        result["source_reproduction"] = json.loads(
            (BASE / "reproduction.json").read_text()
        )
    if (BASE / "packing_comparison.json").exists():
        result["packing_experiment"] = json.loads(
            (BASE / "packing_comparison.json").read_text()
        )
    if (BASE / "source_control").exists():
        result["original_geometry_control"] = assess(
            BASE / "source_control",
            BASE / "surfaces_after",
            BASE / "source_control_assessment",
            BASE / "source_control_cases.json",
        )
    for side in ["before", "after"]:
        values = sorted(
            c["pose_to_gpu_complete_ms"]
            for c in json.loads((BASE / side / "runtime_report.json").read_text())[
                "cases"
            ]
        )
        result.setdefault("timing", {})[side] = {
            "poses": len(values),
            "median_pose_to_gpu_complete_ms": float(np.median(values)),
            "max_ms": max(values),
        }
    jsonwrite(destination / "metrics.json", result)
    for folder, source, suffix in [
        ("reference", "reference", ".png"),
        ("before", "before", ".png"),
        ("after", "after", ".png"),
        ("error_before", "assessment_before", "_errors.png"),
        ("error_after", "assessment_after", "_errors.png"),
    ]:
        (destination / folder).mkdir(exist_ok=True)
        for case in cases():
            shutil.copy2(
                BASE / source / (case["name"] + suffix),
                destination / folder / (case["name"] + ".png"),
            )
    preview = Image.new("RGB", (1500, 660), "#222")
    draw = ImageDraw.Draw(preview)
    for row, name in enumerate(["reported_right", "reported_front"]):
        for col, folder in enumerate(["reference", "before", "after"]):
            image = Image.open(BASE / folder / (name + ".png")).convert("RGB")
            preview.paste(
                image.crop((275, 275, 525, 435)).resize((500, 320)),
                (col * 500, row * 330 + 10),
            )
            draw.text((col * 500 + 5, row * 330 + 10), folder + ": " + name, fill="red")
    preview.save(destination / "comparison.jpg", quality=96)
    summary = {
        k: result[k]
        for k in ["model", "screening", "invariants", "timing", "limitations"]
    }
    summary.update(
        before=before["summary"],
        after=after["summary"],
        temporal_before=before["temporal_false_change"],
        temporal_after=after["temporal_false_change"],
    )
    summary["teeth"] = {k: result["teeth"][k]["summary"] for k in ["before", "after"]}
    summary["features"] = {
        k: result["features"][k]["summary"] for k in ["before", "after"]
    }
    summary["collar"] = {k: result["collar"][k]["summary"] for k in ["before", "after"]}
    summary["native_glue_max_px"] = result["native_glue"]["max_px"]
    summary["proxy_error_px"] = result["proxy_error_px"]
    summary["reported_spots"] = result["reported_spots"]
    if "oral_ids" in result:
        summary["oral_ids"] = {
            side: result["oral_ids"][side]["summary"] for side in ["before", "after"]
        }
    options = "".join(f"<option>{html.escape(c['name'])}</option>" for c in cases())
    page = """<!doctype html><meta charset="utf-8"><title>POPY facial surface repair</title>
<style>body{background:#222;color:#eee;font:16px sans-serif;margin:24px}img{max-width:100%}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.grid img{width:100%}pre{white-space:pre-wrap}</style>
<h1>POPY 面部穿透遮盖回归</h1><p>独立 Blender / 修复前原生 MOC / 修复后原生 MOC，218 姿态。红色为错误露出，蓝色为缺失；指标含投影误差，有限测试不能证明全部连续参数无缺陷。</p>
<img src="comparison.jpg"><p><select id="pose">OPTIONS</select></p>
<div class="grid"><div>Blender<img id="reference"></div><div>Before<img id="before"></div><div>After<img id="after"></div></div>
<div class="grid"><div>Before errors<img id="error_before"></div><div>After errors<img id="error_after"></div></div>
<pre>SUMMARY</pre><p><a href="metrics.json">完整数值、来源哈希及回归记录</a></p>
<script>const pose=document.getElementById('pose');function update(){for(const id of ['reference','before','after','error_before','error_after'])document.getElementById(id).src=id+'/'+encodeURIComponent(pose.value)+'.png'}pose.onchange=update;update();</script>"""
    (destination / "index.html").write_text(
        page.replace("OPTIONS", options).replace(
            "SUMMARY", html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
        ),
        encoding="utf8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return result


def package():
    metrics = json.loads((BASE / "report/metrics.json").read_text())
    assert sha(MODEL / "POPY.moc3") == metrics["moc_sha256"]
    logs = BASE / "report/test_logs"
    logs.mkdir(exist_ok=True)
    for name in ["face_python_tests", "face_rust_tests"]:
        shutil.copy2(ROOT / "artifacts/logs" / (name + ".log"), logs)
    for name, source in [
        ("oral_reference", "oral_reference"),
        ("oral_before", "oral_id_before"),
        ("oral_after", "oral_id_after"),
    ]:
        folder = BASE / "report" / name
        folder.mkdir(exist_ok=True)
        for case in json.loads((BASE / "oral_cases.json").read_text()):
            suffix = "_surfaces.png" if name == "oral_reference" else ".png"
            shutil.copy2(
                BASE / source / (case["name"] + suffix),
                folder / (case["name"] + ".png"),
            )
    jsonwrite(
        MODEL / "SHA256.json",
        {
            p.name: sha(p)
            for p in MODEL.iterdir()
            if p.is_file() and p.name != "SHA256.json"
        },
    )
    delivery = ROOT / "delivery"
    archive(MODEL, delivery / "POPY_face_fixed.zip", "POPY")
    archive(BASE / "report", delivery / "POPY_face_repair_report.zip")
    tracked = (
        subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
        )
        .stdout.decode("utf8")
        .split("\0")
    )
    with zipfile.ZipFile(
        delivery / "face_converter_source.zip",
        "w",
        zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as z:
        for name in filter(None, tracked):
            path = ROOT / name
            if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
                raise ValueError("Source bundle escapes repository")
            z.write(path, name)
    filenames = [
        "POPY_face_fixed.zip",
        "POPY_face_repair_report.zip",
        "face_converter_source.zip",
    ]
    for name in filenames:
        with zipfile.ZipFile(delivery / name) as z:
            assert z.testzip() is None
    jsonwrite(
        delivery / "SHA256_face_repair.json",
        {name: sha(delivery / name) for name in filenames},
    )


def main():
    parser = argparse.ArgumentParser()
    stages = [
        "prepare",
        "rig",
        "export",
        "reference",
        "validate",
        "regress",
        "report",
        "test",
        "package",
    ]
    parser.add_argument("stage", choices=["all"] + stages)
    parser.add_argument(
        "--blender",
        default=r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
    )
    args = parser.parse_args()
    os.chdir(ROOT)
    BASE.mkdir(exist_ok=True)
    for stage in stages if args.stage == "all" else [args.stage]:
        if stage == "prepare":
            env = os.environ.copy()
            env.setdefault("JAVA_HOME", r"C:\lib\java")
            run(
                "face_exporter_build",
                [
                    ROOT / "vendor/psd2live/gradlew.bat",
                    "-p",
                    ROOT / "exporter",
                    "installDist",
                ],
                env,
            )
            run(
                "face_rust_build",
                [
                    "cargo",
                    "build",
                    "--release",
                    "--manifest-path",
                    "rust/Cargo.toml",
                    "-p",
                    "popy-geometry",
                    "--bin",
                    "face_conflicts",
                    "--locked",
                ],
            )
            screen(BASELINE, BASE / "screen")
            jsonwrite(BASE / "cases.json", cases())
            jsonwrite(
                BASE / "labels.json", {k: v.tolist() for k, v in labels().items()}
            )
            (BASE / "samples").mkdir(exist_ok=True)
            shutil.copy2(
                ROOT / "artifacts/samples/manifest.json", BASE / "samples/manifest.json"
            )
            jsonwrite(
                BASE / "oral_labels.json",
                {k: v.tolist() for k, v in oral_labels().items()},
            )
            jsonwrite(
                BASE / "oral_cases.json",
                [
                    c
                    for c in cases()
                    if c["name"].startswith(("reported_", "expression_", "compound_"))
                ],
            )
        elif stage == "rig":
            run(
                "face_rig",
                [sys.executable, "-m", "popy.face_clearance", "--out", RIG.parent],
            )
        elif stage == "export":
            export(RIG, MODEL, "face_export")
        elif stage == "reference":
            source = ROOT / "artifacts/POPY_hair_clearance.blend"
            before = sha(source)
            run(
                "face_reference",
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
            control = [
                c for c in cases() if c["name"] in ["grid_45_30", "compound_-45_-30_1"]
            ]
            jsonwrite(BASE / "source_control_cases.json", control)
            original = ROOT / "artifacts/POPY_source.blend"
            original_hash = sha(original)
            run(
                "face_source_control",
                [
                    args.blender,
                    "-b",
                    original,
                    "-t",
                    "4",
                    "--python-exit-code",
                    "1",
                    "--python",
                    "blender/sample.py",
                    "--",
                    "--out",
                    BASE / "samples",
                    "--references-only",
                    "--case-file",
                    BASE / "source_control_cases.json",
                    "--case-output",
                    BASE / "source_control_metadata.json",
                    "--reference-dir",
                    BASE / "source_control",
                    "--surface-labels",
                    BASE / "labels.json",
                ],
            )
            assert sha(original) == original_hash
            run(
                "face_oral_reference",
                [
                    args.blender,
                    "-b",
                    source,
                    "-t",
                    "4",
                    "--python-exit-code",
                    "1",
                    "--python",
                    "blender/sample.py",
                    "--",
                    "--out",
                    BASE / "samples",
                    "--references-only",
                    "--case-file",
                    BASE / "oral_cases.json",
                    "--case-output",
                    BASE / "oral_reference_cases.json",
                    "--reference-dir",
                    BASE / "oral_reference",
                    "--surface-labels",
                    BASE / "oral_labels.json",
                ],
            )
            assert sha(source) == before
        elif stage == "validate":
            primary()
        elif stage == "regress":
            regress()
            oral_regression()
        elif stage == "report":
            report()
        elif stage == "test":
            run(
                "face_python_tests",
                [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            )
            run(
                "face_rust_tests",
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
