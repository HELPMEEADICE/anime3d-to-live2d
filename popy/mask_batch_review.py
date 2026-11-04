"""End-to-end lossless mask batching, real-MOC regression and paired GPU timing."""

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import sys
import zipfile

import numpy as np
from PIL import Image

from .cli import run
from .compile_ir import ROOT, jsonwrite
from .face_repair import EXPORTER, RENDERER
from .mask_batch import compile_plan, plan
from .performance_review import sha

CONFIG_PATH = ROOT / "config/mask_batch.json"
CONFIG = json.loads(CONFIG_PATH.read_text())
BASE = ROOT / CONFIG["artifact_directory"]
if os.environ.get("POPY_WGPU_BACKEND") == "dx12":
    BASE = BASE / "dx12"
MODEL = ROOT / CONFIG["output_model"]
EXACT = ROOT / CONFIG["baseline_model"]
INVENTORY = ROOT / CONFIG["inventory"]
CASES = ROOT / "artifacts/fullbody/cases.json"
NATIVE_CASES = ROOT / "artifacts/perf_opt/equivalence_cases.json"
TIMING_CASES = ROOT / "artifacts/perf_opt/timing_cases.json"
SOURCE = ROOT / CONFIG["source_ir"]


def build():
    run(
        "mask_batch_build",
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
    inventory = json.loads(INVENTORY.read_text())
    design = plan(
        inventory,
        CONFIG["vertex_budget"],
        CONFIG["minimum_uses"],
        CONFIG["helper_penalty"],
    )
    jsonwrite(BASE / "plan.json", design)
    compile_plan(SOURCE, BASE / "rig", design)


def export():
    env = os.environ.copy()
    env.setdefault("JAVA_HOME", r"C:\lib\java")
    env.setdefault("JAVA_OPTS", "-Xmx20g -XX:+UseG1GC -XX:+ExitOnOutOfMemoryError")
    run(
        "mask_batch_export_final",
        [
            EXPORTER,
            BASE / "rig/POPY.rig.json",
            MODEL,
            "--no-cmo",
            "--deduplicate-positions",
        ],
        env,
    )
    for p in EXACT.iterdir():
        if p.name in {"POPY.moc3", "export_report.txt"}:
            continue
        if p.is_dir():
            shutil.copytree(p, MODEL / p.name, dirs_exist_ok=True)
        else:
            shutil.copy2(p, MODEL / p.name)


def fingerprints():
    return {
        "baseline_moc_sha256": sha(EXACT / "POPY.moc3"),
        "candidate_moc_sha256": sha(MODEL / "POPY.moc3"),
        "native_cases_sha256": sha(NATIVE_CASES),
        "image_cases_sha256": sha(CASES),
        "plan_sha256": sha(BASE / "plan.json"),
        "configuration_sha256": sha(CONFIG_PATH),
    }


def inventory():
    data = {"inputs": fingerprints()}
    for label, model in [("baseline", EXACT), ("candidate", MODEL)]:
        path = BASE / ("inventory_" + label + ".json")
        run(
            "mask_batch_inventory_" + label,
            [
                ROOT / "rust/target/release/model_stats.exe",
                model / "POPY.model3.json",
                path,
            ],
        )
        data[label] = json.loads(path.read_text())
    jsonwrite(BASE / "inventory.json", data)


def validate():
    before = fingerprints()
    run(
        "mask_batch_native",
        [
            ROOT / "rust/target/release/mask_equivalence.exe",
            EXACT / "POPY.model3.json",
            MODEL / "POPY.model3.json",
            BASE / "plan.json",
            NATIVE_CASES,
            BASE / "native.json",
        ],
    )
    if fingerprints() != before:
        raise AssertionError("Inputs changed during native proof")
    jsonwrite(BASE / "native_inputs.json", before)


def images():
    before = fingerprints()
    # Re-render both actual MOCs with the same current executable.
    for label, model in [("baseline", EXACT), ("images", MODEL)]:
        run(
            "mask_batch_" + label,
            [
                RENDERER,
                model / "POPY.model3.json",
                CASES,
                BASE / label,
                "--images-only",
            ],
        )
    if fingerprints() != before:
        raise AssertionError("Inputs changed during GPU comparison")
    compare_images()


def compare_images():
    renderers = [
        json.loads((BASE / label / "runtime_report.json").read_text())
        for label in ["baseline", "images"]
    ]
    if any(renderers[0][k] != renderers[1][k] for k in ["gpu", "backend", "canvas"]):
        raise AssertionError(
            "Image comparison requires identical GPU, backend and canvas"
        )
    rows, total, maximum = [], 0, 0
    for c in json.loads(CASES.read_text()):
        name = c["name"]
        a = np.array(Image.open(BASE / "baseline" / (name + ".png"))).astype(np.int16)
        b = np.array(Image.open(BASE / "images" / (name + ".png"))).astype(np.int16)
        if a.shape != b.shape:
            raise AssertionError("Canvas changed")
        diff = np.abs(a - b)
        count, error = int(np.count_nonzero(diff)), int(diff.max())
        total += count
        maximum = max(maximum, error)
        rows.append(
            {"name": name, "different_channels": count, "max_channel_error": error}
        )
    evidence = {
        "poses": len(rows),
        "different_channels": total,
        "max_channel_error": maximum,
        "canvas": [768, 1024],
        "backend": renderers[0]["backend"],
        "gpu": renderers[0]["gpu"],
        "cases": rows,
        "inputs": fingerprints(),
    }
    jsonwrite(BASE / "pixels.json", evidence)
    if total:
        raise AssertionError("Lossless mask batching changed pixels")


def stats(values):
    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def benchmark():
    rows = []
    # Two ABBA blocks, independent processes; no image/native validation in parallel.
    for i, label in enumerate(["baseline", "candidate", "candidate", "baseline"] * 2):
        model = EXACT if label == "baseline" else MODEL
        out = BASE / f"timing_{i}_{label}"
        run(
            f"mask_batch_timing_{i}_{label}",
            [RENDERER, model / "POPY.model3.json", TIMING_CASES, out, "--timing-only"],
        )
        raw = json.loads((out / "runtime_report.json").read_text())
        samples = raw["cases"][5:]
        rows.append(
            {
                "label": label,
                "index": i,
                "gpu": raw["gpu"],
                "backend": raw["backend"],
                "cases": samples,
                "summary": {
                    k: stats([c[k] for c in samples])
                    for k in [
                        "pose_to_gpu_complete_ms",
                        "prepare_ms",
                        "encode_ms",
                        "gpu_submit_and_wait_ms",
                    ]
                },
            }
        )
    jsonwrite(
        BASE / "timing.json",
        {
            "runs": rows,
            "inputs": fingerprints(),
            "timing_cases_sha256": sha(TIMING_CASES),
            "warmup_per_process": 5,
            "note": "Two ABBA blocks; identical 60-pose wave/laugh/walk sequences, unmodified Ayagami, full 768x1024 masks, output target reused, GPU completion included, no readback. Desktop not isolated; native backend timing is not BrowserWebGPU.",
        },
    )


def report():
    (BASE / "report").mkdir(parents=True, exist_ok=True)
    inputs = fingerprints()
    native = json.loads((BASE / "native.json").read_text())
    if json.loads((BASE / "native_inputs.json").read_text()) != inputs:
        raise AssertionError("Native proof inputs changed; rerun validate")
    pixels = json.loads((BASE / "pixels.json").read_text())
    timing = json.loads((BASE / "timing.json").read_text())
    if pixels["inputs"] != inputs or timing["inputs"] != inputs:
        raise AssertionError("GPU evidence inputs changed; rerun images/benchmark")
    if timing["timing_cases_sha256"] != sha(TIMING_CASES):
        raise AssertionError("Timing sequence changed")
    if {r["backend"] for r in timing["runs"]} != {pixels["backend"]}:
        raise AssertionError("Image and timing backends differ; do not mix evidence")
    if (
        native["different_original_vertices"]
        or native["max_helper_copy_error_px"]
        or pixels["different_channels"]
    ):
        raise AssertionError("Lossless proof failed")
    design = json.loads((BASE / "plan.json").read_text())
    summary = {}
    for label in ["baseline", "candidate"]:
        samples = [c for r in timing["runs"] if r["label"] == label for c in r["cases"]]
        summary[label] = {
            k: stats([c[k] for c in samples])
            for k in [
                "pose_to_gpu_complete_ms",
                "prepare_ms",
                "encode_ms",
                "gpu_submit_and_wait_ms",
            ]
        }
    gain = {
        key: 1
        - summary["candidate"]["pose_to_gpu_complete_ms"][key]
        / summary["baseline"]["pose_to_gpu_complete_ms"][key]
        for key in ["median", "mean", "p95"]
    }
    manifest = json.loads((MODEL / "POPY.model3.json").read_text())
    refs = manifest["FileReferences"]
    for p in refs["Textures"]:
        if sha(MODEL / p) != sha(EXACT / p):
            raise AssertionError("Texture changed")
    if sha(MODEL / "POPY.model3.json") != sha(EXACT / "POPY.model3.json"):
        raise AssertionError("Runtime resources changed")
    resource_count = 0
    for p in EXACT.rglob("*"):
        if p.is_file() and p.name not in {"POPY.moc3", "export_report.txt"}:
            if sha(p) != sha(MODEL / p.relative_to(EXACT)):
                raise AssertionError("Resource changed: " + str(p))
            resource_count += 1
    inventory = json.loads((BASE / "images/model_inventory.json").read_text())
    result = {
        "method": design["method"],
        "inputs": inputs,
        "mask_batch": {
            k: v for k, v in design.items() if k not in ["unions", "groups"]
        },
        "native": {k: v for k, v in native.items() if k != "cases"},
        "pixels": {k: v for k, v in pixels.items() if k not in ["cases", "inputs"]},
        "timing_summary_ms": summary,
        "whole_frame_reduction": gain,
        "timing": timing,
        "candidate_moc_bytes": (MODEL / "POPY.moc3").stat().st_size,
        "baseline_moc_bytes": (EXACT / "POPY.moc3").stat().st_size,
        "textures_bit_identical": True,
        "runtime_manifest_bit_identical": True,
        "identical_resource_files": resource_count,
        "candidate_geometry": {
            "includes_hidden_carriers": True,
            "artmeshes": len(inventory),
            "vertices": sum(m["vertices"] for m in inventory),
            "triangles": sum(len(m["indices"]) // 3 for m in inventory),
        },
        "source_blend_sha256": sha(ROOT / "artifacts/POPY_source.blend"),
        "ayagami_commit": "0d1d7aa3efe57b1b363b72e672e353c52a9c15d6",
        "limitations": [
            "Native "
            + pixels["backend"]
            + " measured; BrowserWebGPU speed not measured",
            "No official Cubism Core or Editor compatibility verification",
            "Opaque texture alpha and invariant mask-source visibility required",
            "Original approximation/occlusion limitations remain; not a real-time guarantee",
        ],
    }
    if (BASE / "inventory.json").exists():
        model_stats = json.loads((BASE / "inventory.json").read_text())
        if model_stats["inputs"] != inputs:
            raise AssertionError("Model inventory inputs changed")
        result["model_stats"] = {k: v for k, v in model_stats.items() if k != "inputs"}
    if (BASE / "tests.json").exists():
        tests = json.loads((BASE / "tests.json").read_text())
        if tests.get("inputs") == inputs:
            result["tests"] = tests
    other = ROOT / CONFIG["artifact_directory"] / "report/metrics.json"
    if BASE.name == "dx12" and other.exists():
        previous = json.loads(other.read_text())
        if previous["inputs"] == inputs:
            result["other_backend"] = {
                k: previous[k]
                for k in [
                    "inputs",
                    "timing_summary_ms",
                    "whole_frame_reduction",
                    "limitations",
                ]
            }
            result["other_backend"]["full_record"] = (
                "benchmarks/popy/fullbody/mask_batch_vulkan.json"
            )
    jsonwrite(BASE / "report/metrics.json", result)
    jsonwrite(ROOT / "benchmarks/popy/fullbody/mask_batch.json", result)
    if pixels["backend"] == "Vulkan":
        jsonwrite(ROOT / "benchmarks/popy/fullbody/mask_batch_vulkan.json", result)
    table = "<table><tr><th>Model</th><th>Median ms</th><th>Mean ms</th><th>P95 ms</th></tr>"
    for label in summary:
        s = summary[label]["pose_to_gpu_complete_ms"]
        table += f"<tr><td>{label}</td><td>{s['median']:.2f}</td><td>{s['mean']:.2f}</td><td>{s['p95']:.2f}</td></tr>"
    table += "</table>"
    body = [
        "<!doctype html><meta charset=utf-8><title>POPY lossless mask batching</title>",
        "<style>body{font:16px system-ui;max-width:1300px;margin:auto;background:#18202a;color:#eee}table{border-collapse:collapse}td,th{padding:9px;border:1px solid #899}img{width:30%;background:#333}pre{white-space:pre-wrap}a{color:#9ce}</style>",
        "<h1>POPY lossless mask batching</h1><p>Original geometry, textures, parameters and full-resolution raster grid retained. Native proof and RGBA equality are required. Timing includes GPU completion, excludes PNG readback. Native timings do not establish browser FPS.</p>",
        table,
        "<details><summary>Evidence</summary><pre>"
        + html.escape(
            json.dumps({k: v for k, v in result.items() if k != "timing"}, indent=2)
        )
        + "</pre></details>",
    ]
    for c in json.loads(CASES.read_text()):
        name = c["name"]
        reference = os.path.relpath(
            ROOT / "artifacts/fullbody/reference" / (name + ".png"), BASE / "report"
        ).replace("\\", "/")
        body.append(
            f"<h3>{html.escape(name)} — zero RGBA difference</h3>"
            + f'<img loading="lazy" src="{reference}" alt="Blender reference">'
            + f'<img loading="lazy" src="../baseline/{name}.png" alt="Before">'
            + f'<img loading="lazy" src="../images/{name}.png" alt="After">'
        )
    (BASE / "report/index.html").write_text("\n".join(body), encoding="utf8")
    print(json.dumps({"timing": summary, "reduction": gain}, indent=2), flush=True)


def test():
    run(
        "mask_batch_all_tests",
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
    )
    run(
        "mask_batch_rust_tests",
        [
            "cargo",
            "test",
            "--locked",
            "--manifest-path",
            ROOT / "rust/Cargo.toml",
            "-p",
            "popy-geometry",
        ],
    )
    python_log = ROOT / "artifacts/logs/mask_batch_all_tests.log"
    rust_log = ROOT / "artifacts/logs/mask_batch_rust_tests.log"
    count = re.search(r"Ran (\d+) tests", python_log.read_text())
    if count is None:
        raise AssertionError("Missing Python test count")
    jsonwrite(
        BASE / "tests.json",
        {
            "passed": True,
            "python_tests": int(count[1]),
            "rust_tests": sum(
                map(
                    int,
                    re.findall(
                        r"test result: ok\. (\d+) passed;", rust_log.read_text()
                    ),
                )
            ),
            "inputs": fingerprints(),
            "python_log_sha256": sha(python_log),
            "rust_log_sha256": sha(rust_log),
        },
    )
    report()


def package():
    report()
    result = json.loads((BASE / "report/metrics.json").read_text())
    if not result.get("tests", {}).get("passed"):
        raise AssertionError("Run full tests before packaging")
    if "model_stats" not in result:
        raise AssertionError("Run actual model inventory before packaging")
    if any(result["whole_frame_reduction"][k] <= 0 for k in ["median", "mean", "p95"]):
        raise AssertionError(
            "Keep as experiment: measured median, mean and P95 must all improve before recommended packaging"
        )
    target = ROOT / "delivery/POPY_fullbody_maskbatched.zip"
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(MODEL.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(MODEL).as_posix())
        z.writestr("mask_batch_validation.json", json.dumps(result, indent=2))
        z.writestr(
            "README.txt",
            "Load POPY.model3.json in Ayagami. Lossless opaque mask batching, original 39 parameters / 58 motions / 24 expressions. No new CMO3 or official Core/Editor verification. See mask_batch_validation.json for actual native Vulkan timings and limitations.\n",
        )
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None:
            raise AssertionError("ZIP CRC failure")
        with z.open("POPY.moc3") as stream:
            if stream.read(4) != b"MOC3":
                raise AssertionError("Invalid MOC magic")
        with z.open("POPY.moc3") as stream:
            if (
                hashlib.file_digest(stream, "sha256").hexdigest()
                != result["inputs"]["candidate_moc_sha256"]
            ):
                raise AssertionError("Archive MOC differs from tested model")
        refs = json.loads(z.read("POPY.model3.json"))["FileReferences"]
        files = [refs["Moc"], *refs["Textures"]]
        files += [
            refs[k] for k in ["DisplayInfo", "Physics", "Pose", "UserData"] if k in refs
        ]
        for entries in refs.get("Motions", {}).values():
            for item in entries:
                files.append(item["File"])
                if item.get("Sound"):
                    files.append(item["Sound"])
        files += [item["File"] for item in refs.get("Expressions", [])]
        if any(name not in z.namelist() for name in files):
            raise AssertionError("Missing packaged runtime resource")
    jsonwrite(
        BASE / "package.json",
        {
            "path": str(target),
            "sha256": sha(target),
            "bytes": target.stat().st_size,
            "moc_sha256": result["inputs"]["candidate_moc_sha256"],
            "validated_manifest_references": len(files),
            "crc_verified": True,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    stages = [
        "build",
        "rig",
        "export",
        "inventory",
        "validate",
        "images",
        "benchmark",
        "report",
        "test",
        "package",
    ]
    parser.add_argument("stage", choices=["all"] + stages)
    args = parser.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    for stage in stages if args.stage == "all" else [args.stage]:
        globals()[stage]()


if __name__ == "__main__":
    main()
