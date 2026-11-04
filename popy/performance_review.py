"""Real-MOC performance/quality comparison; original fullbody assets are read-only."""

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .animation import evaluate
from .cli import run
from .compare import composite, ssim
from .compile_ir import ROOT, jsonwrite, project
from .face_repair import EXPORTER, RENDERER
from .performance_opt import optimize

BASE = ROOT / "artifacts/perf_opt"
ORIGINAL = ROOT / "output/POPY_fullbody"
EXACT = ROOT / "output/POPY_optimized_exact"
MODEL = ROOT / "output/POPY_fullbody_optimized"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build():
    env = os.environ.copy()
    env.setdefault("JAVA_HOME", r"C:\lib\java")
    run("performance_exporter_setup", [sys.executable, "-m", "popy.setup_exporter"])
    run(
        "performance_exporter_build",
        [
            ROOT / "vendor/psd2live/gradlew.bat",
            "-p",
            ROOT / "exporter",
            "installDist",
            "--console=plain",
        ],
        env,
    )
    run(
        "performance_rust_build",
        [
            "cargo",
            "build",
            "--release",
            "--locked",
            "--manifest-path",
            ROOT / "rust/Cargo.toml",
            "-p",
            "popy-validate",
            "--bins",
        ],
    )


def rig():
    for mode in ["exact", "precision"]:
        optimize(
            ROOT / "artifacts/fullbody/rig/POPY.rig.json", BASE / (mode + "_rig"), mode
        )


def export():
    env = os.environ.copy()
    env.setdefault("JAVA_HOME", r"C:\lib\java")
    env.setdefault("JAVA_OPTS", "-Xmx20g -XX:+UseG1GC -XX:+ExitOnOutOfMemoryError")
    for mode, target in [("exact", EXACT), ("precision", MODEL)]:
        run(
            "performance_export_" + mode,
            [
                EXPORTER,
                BASE / (mode + "_rig/POPY.rig.json"),
                target,
                "--no-cmo",
                "--deduplicate-positions",
            ],
            env,
        )


def test():
    run(
        "performance_all_tests",
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
            "-v",
        ],
    )
    run(
        "performance_rust_tests",
        [
            "cargo",
            "test",
            "--manifest-path",
            ROOT / "rust/Cargo.toml",
            "-p",
            "popy-geometry",
            "--locked",
        ],
    )
    record_tests()
    if (BASE / "report/metrics.json").exists():
        timing_report()


def record_tests():
    py = ROOT / "artifacts/logs/performance_all_tests.log"
    rs = ROOT / "artifacts/logs/performance_rust_tests.log"
    if not rs.exists():
        rs = ROOT / "artifacts/logs/perf_opt_rust_tests.log"
    py_text = py.read_text()
    rs_text = rs.read_text()
    count = re.search(r"Ran (\d+) tests", py_text)
    if count is None or not py_text.rstrip().endswith("OK") or "FAILED" in rs_text:
        raise AssertionError("Passing test logs required")
    result = {
        "python_tests": int(count[1]),
        "rust_tests": sum(
            map(int, re.findall(r"test result: ok\. (\d+) passed;", rs_text))
        ),
        "passed": True,
        "python_log_sha256": sha(py),
        "rust_log_sha256": sha(rs),
        "baseline_moc_sha256": sha(ORIGINAL / "POPY.moc3"),
        "precision_moc_sha256": sha(MODEL / "POPY.moc3"),
        "exact_moc_sha256": sha(EXACT / "POPY.moc3"),
    }
    jsonwrite(BASE / "tests.json", result)
    return result


def cases():
    parameters = json.loads((ROOT / "artifacts/fullbody/parameters.json").read_text())
    result = json.loads((ROOT / "artifacts/fullbody/cases.json").read_text())
    for p in parameters:
        for i, value in enumerate(np.linspace(p["min"], p["max"], 17)):
            result.append(
                {"name": f"sweep_{p['id']}_{i}", "parameters": {p["id"]: float(value)}}
            )
    rng = np.random.default_rng(20261104)
    for i in range(256):
        result.append(
            {
                "name": f"random_{i:03}",
                "parameters": {
                    p["id"]: float(rng.uniform(p["min"], p["max"])) for p in parameters
                },
            }
        )
    for path in sorted((ORIGINAL / "motions").glob("*.motion3.json")):
        m = json.loads(path.read_text())
        for i, t in enumerate(np.linspace(0, m["Meta"]["Duration"], 9)):
            result.append(
                {
                    "name": f"continuous_{path.stem}_{i}",
                    "parameters": evaluate(parameters, m, float(t)),
                }
            )
    jsonwrite(BASE / "equivalence_cases.json", result)
    sequence = []
    for name in ["wave_L", "emote_laugh_C", "walk_in_place"]:
        motion = json.loads(
            (ORIGINAL / "motions" / (name + ".motion3.json")).read_text()
        )
        for i, t in enumerate(np.linspace(0.01, motion["Meta"]["Duration"] - 0.01, 20)):
            sequence.append(
                {
                    "name": f"{name}_{i:03}",
                    "parameters": evaluate(parameters, motion, float(t)),
                }
            )
    sequence = [
        {"name": f"warmup_{i}", "parameters": c["parameters"]}
        for i, c in enumerate(sequence[:5])
    ] + sequence
    jsonwrite(BASE / "timing_cases.json", sequence)
    for target in [EXACT, MODEL]:
        if not (target / "POPY.moc3").exists():
            continue
        for path in ORIGINAL.iterdir():
            if path.name in ["POPY.moc3", "export_report.txt"]:
                continue
            if path.is_dir():
                shutil.copytree(path, target / path.name, dirs_exist_ok=True)
            else:
                shutil.copy2(path, target / path.name)
    return result


def validate(mode="precision"):
    target = EXACT if mode == "exact" else MODEL
    command = [
        ROOT / "rust/target/release/equivalence.exe",
        ORIGINAL / "POPY.model3.json",
        target / "POPY.model3.json",
        BASE / "equivalence_cases.json",
        BASE / (mode + "_equivalence.json"),
    ]
    if mode == "precision":
        command.append("--allow-difference")
    run("performance_" + mode + "_equivalence", command)
    path = BASE / (mode + "_equivalence.json")
    result = json.loads(path.read_text())
    result.update(
        baseline_moc_sha256=sha(ORIGINAL / "POPY.moc3"),
        candidate_moc_sha256=sha(target / "POPY.moc3"),
        cases_sha256=sha(BASE / "equivalence_cases.json"),
    )
    jsonwrite(path, result)


def images():
    for label, target in [("before", ORIGINAL), ("precision", MODEL), ("exact", EXACT)]:
        run(
            "performance_images_" + label,
            [
                RENDERER,
                target / "POPY.model3.json",
                ROOT / "artifacts/fullbody/cases.json",
                BASE / ("images_" + label),
                "--images-only",
            ],
        )
    run(
        "performance_geometry_precision",
        [
            RENDERER,
            MODEL / "POPY.model3.json",
            ROOT / "artifacts/fullbody/probes.json",
            BASE / "geometry_precision",
        ],
    )


def benchmark():
    rows = []
    # ABBA then reverse-order exact control; warmup excluded separately per run.
    for i, (label, target) in enumerate(
        [
            ("before", ORIGINAL),
            ("precision", MODEL),
            ("precision", MODEL),
            ("before", ORIGINAL),
            ("exact", EXACT),
            ("before", ORIGINAL),
            ("before", ORIGINAL),
            ("exact", EXACT),
        ]
    ):
        out = BASE / f"timing_{i}_{label}"
        run(
            f"performance_timing_{i}_{label}",
            [
                RENDERER,
                target / "POPY.model3.json",
                BASE / "timing_cases.json",
                out,
                "--timing-only",
            ],
        )
        report = json.loads((out / "runtime_report.json").read_text())
        times = [r["pose_to_gpu_complete_ms"] for r in report["cases"][5:]]
        rows.append(
            {
                "label": label,
                "run": i,
                "median_ms": float(np.median(times)),
                "p95_ms": float(np.quantile(times, 0.95)),
                "frames": len(times),
                "moc_parse_ms": report["moc_parse_ms"],
                "gpu": report["gpu"],
                "backend": report["backend"],
                "times_ms": times,
                "phase_medians_ms": {
                    k: float(np.median([r[k] for r in report["cases"][5:]]))
                    for k in [
                        "parameter_ms",
                        "prepare_ms",
                        "encode_ms",
                        "gpu_submit_and_wait_ms",
                    ]
                    if k in report["cases"][0]
                },
            }
        )
    cpu = []
    for label, target in [("precision", MODEL), ("exact", EXACT)]:
        for reverse in [False, True]:
            before, after = (target, ORIGINAL) if reverse else (ORIGINAL, target)
            path = BASE / (f"cpu_{label}_{int(reverse)}.json")
            run(
                f"performance_cpu_{label}_{int(reverse)}",
                [
                    ROOT / "rust/target/release/equivalence.exe",
                    before / "POPY.model3.json",
                    after / "POPY.model3.json",
                    BASE / "timing_cases.json",
                    path,
                    "--allow-difference",
                ],
            )
            values = json.loads(path.read_text())["cases"][5:]
            cpu.append(
                {
                    "mode": label,
                    "reverse": reverse,
                    "baseline_ms": [
                        r["after_driver_ms" if reverse else "before_driver_ms"]
                        for r in values
                    ],
                    "candidate_ms": [
                        r["before_driver_ms" if reverse else "after_driver_ms"]
                        for r in values
                    ],
                }
            )
    jsonwrite(
        BASE / "timing.json",
        {
            "runs": rows,
            "cpu_runs": cpu,
            "model_sha256": {
                label: sha(folder / "POPY.moc3")
                for label, folder in [
                    ("baseline", ORIGINAL),
                    ("exact", EXACT),
                    ("precision", MODEL),
                ]
            },
            "note": "Identical 60-pose wave/laugh/walk sequence, five warmup frames per process. ABBA ordering. Native parameter evaluation through GPU completion at 768x1024, full-size masks, reused cleared output target; no image readback in timing. Separate paired CPU-only drivers with input order reversed. Desktop not isolated. Not a real-time FPS claim.",
        },
    )


def report():
    ir = json.loads((ROOT / "artifacts/fullbody/rig/POPY.rig.json").read_text())
    source = ROOT / "artifacts/fullbody"
    camera = json.loads((source / "reference/camera.json").read_text())
    old = json.loads((source / "report/metrics.json").read_text())
    if old["moc_sha256"] != sha(ORIGINAL / "POPY.moc3"):
        raise ValueError("Original report does not match baseline MOC")
    previous = {c["name"]: c for c in old["cases"]}
    out = BASE / "report"
    out.mkdir(exist_ok=True)
    rows = []
    for case in json.loads((source / "cases.json").read_text()):
        name = case["name"]
        a = np.array(
            Image.open(BASE / "images_before" / (name + ".png")).convert("RGBA")
        )
        b = np.array(
            Image.open(BASE / "images_precision" / (name + ".png")).convert("RGBA")
        )
        exact = np.array(
            Image.open(BASE / "images_exact" / (name + ".png")).convert("RGBA")
        )
        if not np.array_equal(a, exact):
            raise AssertionError("Exact mode GPU pixel mismatch: " + name)
        ref = (
            np.array(
                Image.open(source / "reference" / (name + ".png")).convert("RGBA"),
                float,
            )
            / 255
        )
        af, bf = a.astype(float) / 255, b.astype(float) / 255
        ac, bc, rc = composite(af), composite(bf), composite(ref)
        ma, mb, mr = af[..., 3] > 0.5, bf[..., 3] > 0.5, ref[..., 3] > 0.5

        def metrics(image, mask, mr=mr, rc=rc):
            union = mr | mask
            return {
                "mae_foreground": float(np.abs(rc - image)[union].mean()),
                "silhouette_iou": float((mr & mask).sum() / max(union.sum(), 1)),
                "ssim": ssim(rc, image, union),
            }

        before, after = metrics(ac, ma), metrics(bc, mb)
        row = {
            "name": name,
            "before": before,
            "after": after,
            "changed_rgba_channels": int(np.count_nonzero(a != b)),
            "exact_changed_rgba_channels": 0,
        }
        geometry = BASE / "geometry_precision" / (name + ".geometry.json")
        if geometry.exists():
            native = {
                m["id"]: np.array(m["vertices"])
                for m in json.loads(geometry.read_text())
            }
            cache, distances = {}, []
            for mesh in ir["meshes"]:
                if not mesh.get("articulated") or mesh.get("mask_only"):
                    continue
                obj = mesh["source"]["object"]
                if obj not in cache:
                    cache[obj] = (
                        project(
                            np.load(
                                source / "reference" / (name + "__" + obj + ".npy")
                            ),
                            camera,
                        )
                        * ir["canvas"]
                    )
                ref_points = cache[obj][mesh["source"]["vertices"]]
                points = (
                    native[mesh["id"]] * ir["canvas"][0] / 2
                    + np.array(ir["canvas"]) / 2
                )
                distances.extend(np.linalg.norm(points - ref_points, axis=1))
            d = np.array(distances)
            row.update(
                before_rmse_px=previous[name]["articulated_rmse_px"],
                before_max_px=previous[name]["articulated_max_px"],
                after_rmse_px=float(np.sqrt(np.mean(d * d))),
                after_max_px=float(d.max()),
            )
        rows.append(row)
        if name in [
            "natural",
            "motion_wave_L_middle",
            "motion_emote_laugh_C_middle",
            "stress_all",
            "heldout_leg_L",
        ]:
            panel = Image.new("RGB", (1536, 1060), "#242424")
            panel.paste(Image.fromarray(np.uint8(ac * 255)), (0, 36))
            panel.paste(Image.fromarray(np.uint8(bc * 255)), (768, 36))
            ImageDraw.Draw(panel).text(
                (10, 10),
                name + " | original / optimized (same textures/topology)",
                fill="white",
            )
            panel.save(out / (name + ".jpg"), quality=92)
    geometric = [r for r in rows if "after_max_px" in r]
    evidence = {
        "baseline_moc_sha256": sha(ORIGINAL / "POPY.moc3"),
        "exact_moc_sha256": sha(EXACT / "POPY.moc3"),
        "precision_moc_sha256": sha(MODEL / "POPY.moc3"),
        "image_poses": len(rows),
        "geometry_poses": len(geometric),
        "before_geometry_max_px": max(r["before_max_px"] for r in geometric),
        "after_geometry_max_px": max(r["after_max_px"] for r in geometric),
        "all_geometry_rmse_improved": all(
            r["after_rmse_px"] <= r["before_rmse_px"] for r in geometric
        ),
        "all_geometry_max_improved": all(
            r["after_max_px"] <= r["before_max_px"] for r in geometric
        ),
        "mean_foreground_mae_before": float(
            np.mean([r["before"]["mae_foreground"] for r in rows])
        ),
        "mean_foreground_mae_after": float(
            np.mean([r["after"]["mae_foreground"] for r in rows])
        ),
        "worst_mae_delta": max(
            r["after"]["mae_foreground"] - r["before"]["mae_foreground"] for r in rows
        ),
        "cases": rows,
        "same_assets_sha256": {
            str(p.relative_to(ORIGINAL)): sha(p)
            for p in ORIGINAL.rglob("*")
            if p.is_file() and p.suffix in [".png", ".json"]
        },
        "compaction": {
            mode: json.loads((BASE / (mode + "_rig/optimization.json")).read_text())
            for mode in ["exact", "precision"]
        },
    }
    for relative, digest in evidence["same_assets_sha256"].items():
        if sha(MODEL / relative) != digest or sha(EXACT / relative) != digest:
            raise AssertionError(
                "Texture/motion/expression/manifest changed: " + relative
            )
    for mode in ["exact", "precision"]:
        path = BASE / (mode + "_equivalence.json")
        if path.exists():
            eq = json.loads(path.read_text())
            if (
                eq.get("candidate_moc_sha256") != evidence[mode + "_moc_sha256"]
                or eq.get("baseline_moc_sha256") != evidence["baseline_moc_sha256"]
            ):
                raise AssertionError("Stale equivalence proof: " + mode)
            evidence[mode + "_evaluation"] = {
                k: v for k, v in eq.items() if k != "cases"
            }
            evidence[mode + "_visible_delta_max_px"] = max(
                r["visible_max_px"] for r in eq["cases"]
            )
            if mode == "precision":
                evidence["face_hair_headwear_delta_max_px"] = max(
                    r["face_hair_headwear_max_px"] for r in eq["cases"]
                )
    return save_report(evidence)


def timing_report():
    evidence = json.loads((BASE / "report/metrics.json").read_text())
    for label, folder in [
        ("baseline", ORIGINAL),
        ("exact", EXACT),
        ("precision", MODEL),
    ]:
        if sha(folder / "POPY.moc3") != evidence[label + "_moc_sha256"]:
            raise ValueError("Cached quality proof does not match model")
    for relative, digest in evidence["same_assets_sha256"].items():
        if any(sha(folder / relative) != digest for folder in [ORIGINAL, EXACT, MODEL]):
            raise ValueError("Cached quality proof does not match assets: " + relative)
    return save_report(evidence)


def save_report(evidence):
    if (BASE / "tests.json").exists():
        evidence["test_results"] = json.loads((BASE / "tests.json").read_text())
    inventory = json.loads((BASE / "images_before/model_inventory.json").read_text())
    mask_sets = {tuple(m["clips"]) for m in inventory if m["clips"]}
    evidence["mask_statistics"] = {
        "unique_mask_sets": len(mask_sets),
        "masked_meshes": sum(bool(m["clips"]) for m in inventory),
        "largest_source_list": max(map(len, mask_sets), default=0),
    }
    for key in ["silhouette_iou", "ssim"]:
        evidence["mean_" + key + "_before"] = float(
            np.mean([r["before"][key] for r in evidence["cases"]])
        )
        evidence["mean_" + key + "_after"] = float(
            np.mean([r["after"][key] for r in evidence["cases"]])
        )
        evidence["worst_" + key + "_delta"] = min(
            r["after"][key] - r["before"][key] for r in evidence["cases"]
        )
    evidence["changed_image_poses"] = sum(
        r["changed_rgba_channels"] > 0 for r in evidence["cases"]
    )
    baseline_report = json.loads(
        (ROOT / "artifacts/fullbody/report/metrics.json").read_text()
    )
    evidence["retained_assets"] = {
        k: baseline_report["statistics"][k]
        for k in ["meshes", "vertices", "triangles", "authored_keyforms"]
    }
    evidence["retained_assets"].update(
        parameters=39, motions=58, expressions=24, textures=6, texture_size=[2048, 2048]
    )
    evidence["source_reference"] = (
        "Reused independent Blender source geometry/images from the unchanged fullbody baseline, same parameter cases and camera; no new Blender mutations"
    )
    evidence["official_cubism_core_editor_verified"] = False
    if (BASE / "timing.json").exists():
        evidence["timing"] = json.loads((BASE / "timing.json").read_text())
        for mode in ["baseline", "exact", "precision"]:
            if (
                evidence["timing"].get("model_sha256", {}).get(mode)
                != evidence[mode + "_moc_sha256"]
            ):
                raise ValueError("Timing model fingerprint mismatch: " + mode)
        runs = evidence["timing"]["runs"]
        evidence["matched_timing"] = {}
        for label, baseline_runs, candidate_runs in [
            ("precision", [0, 3], [1, 2]),
            ("exact", [5, 6], [4, 7]),
        ]:
            baseline = np.array(
                [t for r in runs if r["run"] in baseline_runs for t in r["times_ms"]]
            )
            candidate = np.array(
                [t for r in runs if r["run"] in candidate_runs for t in r["times_ms"]]
            )
            evidence["matched_timing"][label] = {
                "before_median_ms": float(np.median(baseline)),
                "after_median_ms": float(np.median(candidate)),
                "before_mean_ms": float(np.mean(baseline)),
                "after_mean_ms": float(np.mean(candidate)),
                "before_p95_ms": float(np.quantile(baseline, 0.95)),
                "after_p95_ms": float(np.quantile(candidate, 0.95)),
                "median_reduction_percent": float(
                    (1 - np.median(candidate) / np.median(baseline)) * 100
                ),
                "frames_each": len(baseline),
            }
        evidence["matched_cpu_timing"] = {}
        for label in ["precision", "exact"]:
            runs = evidence["timing"].get("cpu_runs", [])
            a = np.array(
                [t for r in runs if r["mode"] == label for t in r["baseline_ms"]]
            )
            b = np.array(
                [t for r in runs if r["mode"] == label for t in r["candidate_ms"]]
            )
            if len(a):
                evidence["matched_cpu_timing"][label] = {
                    "before_median_ms": float(np.median(a)),
                    "after_median_ms": float(np.median(b)),
                    "before_p95_ms": float(np.quantile(a, 0.95)),
                    "after_p95_ms": float(np.quantile(b, 0.95)),
                    "median_reduction_percent": float(
                        (1 - np.median(b) / np.median(a)) * 100
                    ),
                    "frames_each": len(a),
                }
    evidence["moc_bytes"] = {
        label: (path / "POPY.moc3").stat().st_size
        for label, path in [
            ("before", ORIGINAL),
            ("exact", EXACT),
            ("precision", MODEL),
        ]
    }
    out = BASE / "report"
    jsonwrite(out / "metrics.json", evidence)
    jsonwrite(ROOT / "benchmarks/popy/fullbody/performance_optimization.json", evidence)
    summary = {k: v for k, v in evidence.items() if k not in ["cases", "timing"]}
    timing_table = "<table><tr><th>Mode</th><th>CPU median before / after (ms)</th><th>Full frame median</th><th>Full frame mean</th><th>Full frame P95</th></tr>"
    for mode in ["exact", "precision"]:
        cpu = evidence.get("matched_cpu_timing", {}).get(mode, {})
        gpu = evidence.get("matched_timing", {}).get(mode, {})
        if not cpu or not gpu:
            continue
        timing_table += (
            "<tr><td>"
            + mode
            + "</td>"
            + "".join(
                "<td>" + f"{record[a]:.2f} / {record[b]:.2f}" + "</td>"
                for record, a, b in [
                    (cpu, "before_median_ms", "after_median_ms"),
                    (gpu, "before_median_ms", "after_median_ms"),
                    (gpu, "before_mean_ms", "after_mean_ms"),
                    (gpu, "before_p95_ms", "after_p95_ms"),
                ]
            )
            + "</tr>"
        )
    timing_table += "</table>"
    (out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>POPY performance</title><style>body{background:#222;color:#eee;font:16px system-ui;max-width:1550px;margin:auto;padding:24px}img{max-width:100%}pre{white-space:pre-wrap}table{border-collapse:collapse}th,td{border:1px solid #777;padding:10px}</style><h1>POPY: exact and precision-preserving compilation</h1><p>All source topology, textures, parameters and motions retained. Exact mode preserves every tested vertex and pixel. Precision mode improves sampled Blender skeleton reprojection; it changes floating-point accumulation.</p><p>GPU long frames remain. Precision-mode mean and P95 increased in this run; median improvement does not prove stable end-to-end acceleration. Desktop not isolated.</p>'
        + timing_table
        + "<details><summary>Full quality / timing / provenance evidence</summary><pre>"
        + html.escape(json.dumps(summary, indent=2))
        + "</pre></details>"
        + "".join('<img src="' + p.name + '">' for p in out.glob("*.jpg")),
        encoding="utf8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return evidence


def package():
    metrics = json.loads((BASE / "report/metrics.json").read_text())
    results = metrics.get("test_results", {})
    if not results.get("passed") or any(
        results.get(label + "_moc_sha256") != metrics[label + "_moc_sha256"]
        for label in ["baseline", "precision", "exact"]
    ):
        raise AssertionError("Matching passing test evidence is required")
    if (
        not metrics["all_geometry_rmse_improved"]
        or not metrics["all_geometry_max_improved"]
        or metrics["after_geometry_max_px"] >= 0.01
        or metrics["mean_foreground_mae_after"] > metrics["mean_foreground_mae_before"]
        or metrics["worst_mae_delta"] > 0.00001
        or metrics.get("face_hair_headwear_delta_max_px") != 0
        or metrics.get("exact_evaluation", {}).get("different_vertices") != 0
        or metrics.get("precision_evaluation", {}).get("poses", 0) < 1803
    ):
        raise AssertionError("Quality validation is incomplete or precision regressed")
    for label, target in [("optimized", MODEL), ("bitexact", EXACT)]:
        expected = metrics[
            ("precision" if label == "optimized" else "exact") + "_moc_sha256"
        ]
        if sha(target / "POPY.moc3") != expected:
            raise ValueError("Stale model proof")
        (target / "README.txt").write_text(
            "POPY fullbody "
            + label
            + "\n39 parameters / 58 motions / 24 expressions.\n"
            + (
                "Precision mode: mathematically equivalent native skeleton constant-lerp factorization; float rounding differs. All 81 sampled skeleton errors improve against Blender. GPU long frames remain; whole-frame mean/P95 did not improve in the measured run.\n"
                if label == "optimized"
                else "Exact mode: all vertices match bit-for-bit across 1803 tested poses; 362 rendered poses match pixel-for-pixel. CPU median reduction about 4% on the measured sequence.\n"
            )
            + "Textures and source topology unchanged. Orthographic, pinned Ayagami validation only. No new CMO3.\nDocs: https://github.com/HELPMEEADICE/anime3d-to-live2d/blob/main/docs/PERFORMANCE_OPTIMIZATION.md\nLocal report: artifacts/perf_opt/report/index.html\n",
            encoding="utf8",
        )
        archive = ROOT / "delivery" / ("POPY_fullbody_" + label + ".zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for path in target.rglob("*"):
                if path.is_file():
                    z.write(path, path.relative_to(target))
        metrics[label + "_zip_sha256"] = sha(archive)
    jsonwrite(
        ROOT / "delivery/SHA256_optimized.json",
        {k: v for k, v in metrics.items() if "sha256" in k},
    )
    report_zip = ROOT / "delivery/POPY_performance_report.zip"
    with zipfile.ZipFile(report_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in (BASE / "report").rglob("*"):
            if path.is_file():
                z.write(path, path.relative_to(BASE / "report"))
    manifest = ROOT / "delivery/SHA256_optimized.json"
    values = json.loads(manifest.read_text())
    values["performance_report_zip_sha256"] = sha(report_zip)
    jsonwrite(manifest, values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "all",
            "build",
            "rig",
            "export",
            "test",
            "cases",
            "validate",
            "images",
            "benchmark",
            "report",
            "timing_report",
            "package",
        ],
    )
    parser.add_argument("--mode", choices=["exact", "precision"], default="precision")
    args = parser.parse_args()
    for stage in (
        [
            "build",
            "rig",
            "export",
            "cases",
            "validate_exact",
            "validate",
            "images",
            "benchmark",
            "report",
            "test",
            "package",
        ]
        if args.stage == "all"
        else [args.stage]
    ):
        if stage == "validate":
            validate(args.mode)
        elif stage == "validate_exact":
            validate("exact")
        else:
            globals()[stage]()


if __name__ == "__main__":
    main()
