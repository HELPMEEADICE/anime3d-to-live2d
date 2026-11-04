"""Rebuild, verify and package the exact channel/storage optimization."""

import argparse
import hashlib
import html
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from .cli import run
from .compile_ir import ROOT, jsonwrite
from .moc_pack import pack, runtime_extents
from .order_factor import compile_rig

CONFIG_PATH = ROOT / os.environ.get(
    "POPY_REPRESENTATION_CONFIG", "config/representation.json"
)
CONFIG = json.loads(CONFIG_PATH.read_text())
BASE = ROOT / CONFIG["artifact_directory"]
BEFORE = ROOT / CONFIG["baseline_model"]
UNPACKED = ROOT / CONFIG["unpacked_model"]
MODEL = ROOT / CONFIG["output_model"]
CASES = ROOT / "artifacts/fullbody/cases.json"
NATIVE = ROOT / "artifacts/perf_opt/equivalence_cases.json"
TIMING = ROOT / "artifacts/perf_opt/timing_cases.json"
RENDERER = ROOT / "rust/target/release/popy-validate.exe"


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def resource_sha(base):
    """Bind image/motion evidence to manifests and all referenced assets."""
    paths = {"POPY.model3.json"}

    def visit(value):
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str) and Path(value).suffix in {".json", ".png"}:
            paths.add(value)

    visit(json.loads((base / "POPY.model3.json").read_text())["FileReferences"])
    h = hashlib.sha256()
    for name in sorted(paths):
        h.update(name.encode() + b"\0" + bytes.fromhex(sha(base / name)))
    return h.hexdigest()


def fingerprints():
    result = {
        name: sha(p)
        for name, p in {
            "baseline_moc_sha256": BEFORE / "POPY.moc3",
            "candidate_moc_sha256": MODEL / "POPY.moc3",
            "plan_sha256": BASE / "rig/plan.json",
            "image_cases_sha256": CASES,
            "native_cases_sha256": NATIVE,
            "timing_cases_sha256": TIMING,
            "configuration_sha256": CONFIG_PATH,
        }.items()
    }
    result["baseline_runtime_assets_sha256"] = resource_sha(BEFORE)
    result["candidate_runtime_assets_sha256"] = resource_sha(MODEL)
    return result


def copy_assets(source, target):
    target.mkdir(parents=True, exist_ok=True)
    for p in source.iterdir():
        if p.name in {"POPY.moc3", "export_report.txt", "POPY.extents.json"}:
            continue
        if p.is_dir():
            shutil.copytree(p, target / p.name, dirs_exist_ok=True)
        else:
            shutil.copy2(p, target / p.name)


def build():
    env = os.environ.copy()
    env.setdefault("JAVA_HOME", r"C:\lib\java")
    run(
        "representation_build_exporter",
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
        "representation_build_rust",
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
    compile_rig(
        ROOT / CONFIG["source_ir"], BASE / "rig", CONFIG.get("rotation_carriers", False)
    )


def export():
    env = os.environ.copy()
    env.setdefault("JAVA_HOME", r"C:\lib\java")
    env.setdefault("JAVA_OPTS", "-Xmx20g -XX:+UseG1GC -XX:+ExitOnOutOfMemoryError")
    run(
        "representation_export",
        [
            ROOT
            / "exporter/build/install/popy-cubism-exporter/bin/popy-cubism-exporter.bat",
            BASE / "rig/POPY.rig.json",
            UNPACKED,
            "--no-cmo",
            "--deduplicate-positions",
        ],
        env,
    )
    copy_assets(BEFORE, UNPACKED)


def packing():
    pooled = BASE / "pooled"
    first = pack(
        UNPACKED / "POPY.moc3",
        pooled / "POPY.moc3",
    )
    copy_assets(BEFORE, pooled)
    second = pack(
        pooled / "POPY.moc3",
        MODEL / "POPY.moc3",
        extents=runtime_extents(pooled / "POPY.moc3"),
    )
    jsonwrite(
        BASE / "packing.json",
        {
            "input_bytes": first["input_bytes"],
            "output_bytes": second["output_bytes"],
            "pooling": first,
            "trimming": second,
        },
    )
    copy_assets(BEFORE, MODEL)


def validate():
    snapshot = fingerprints()
    run(
        "representation_native",
        [
            ROOT / "rust/target/release/hierarchy_equivalence.exe",
            BEFORE / "POPY.model3.json",
            MODEL / "POPY.model3.json",
            BASE / "rig/plan.json",
            NATIVE,
            BASE / "native.json",
        ],
    )
    if fingerprints() != snapshot:
        raise AssertionError("Inputs changed during native validation")
    jsonwrite(BASE / "native_inputs.json", snapshot)
    for label, p in [("baseline", BEFORE), ("candidate", MODEL)]:
        run(
            "representation_inventory_" + label,
            [
                ROOT / "rust/target/release/model_stats.exe",
                p / "POPY.model3.json",
                BASE / ("inventory_" + label + ".json"),
            ],
        )


def images():
    snapshot = fingerprints()
    env = os.environ.copy()
    env["POPY_WGPU_BACKEND"] = "dx12"
    for label, p in [("baseline", BEFORE), ("candidate", MODEL)]:
        run(
            "representation_images_" + label,
            [RENDERER, p / "POPY.model3.json", CASES, BASE / label, "--images-only"],
            env,
        )
    if fingerprints() != snapshot:
        raise AssertionError("Inputs changed during rendering")
    pixel_review()


def pixel_review():
    rows = []
    for c in json.loads(CASES.read_text()):
        n = c["name"]
        a, b = [
            np.asarray(
                Image.open(BASE / label / (n + ".png")).convert("RGBA"), dtype=np.int16
            )
            for label in ["baseline", "candidate"]
        ]
        if a.shape != b.shape:
            raise AssertionError("Canvas changed")
        d = np.abs(a - b)
        rows.append(
            {
                "name": n,
                "different_channels": int(np.count_nonzero(d)),
                "maximum": int(d.max()),
            }
        )
    renderers = [
        json.loads((BASE / label / "runtime_report.json").read_text())
        for label in ["baseline", "candidate"]
    ]
    if any(r["backend"] != "Dx12" for r in renderers):
        raise AssertionError("Renderer backends differ")
    result = {
        "poses": len(rows),
        "different_channels": sum(c["different_channels"] for c in rows),
        "maximum": max(c["maximum"] for c in rows),
        "cases": rows,
        "inputs": fingerprints(),
    }
    jsonwrite(BASE / "pixels.json", result)
    if result["different_channels"]:
        raise AssertionError("Lossless optimization changed pixels")


def stats(values):
    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def benchmark():
    snapshot = fingerprints()
    env = os.environ.copy()
    env["POPY_WGPU_BACKEND"] = "dx12"
    rows = []
    for i, label in enumerate(["baseline", "candidate", "candidate", "baseline"] * 2):
        model = BEFORE if label == "baseline" else MODEL
        label_run = f"representation_timing_{i}_{label}"
        path = BASE / f"timing_{i}_{label}"
        run(
            label_run,
            [RENDERER, model / "POPY.model3.json", TIMING, path, "--timing-only"],
            env,
        )
        raw = json.loads((path / "runtime_report.json").read_text())
        process = next(
            r
            for r in reversed(
                json.loads((ROOT / "artifacts/performance.json").read_text())
            )
            if r["stage"] == label_run
        )
        rows.append(
            {
                "label": label,
                "index": i,
                "gpu": raw["gpu"],
                "backend": raw["backend"],
                "peak_process_tree_rss_bytes": process["peak_process_tree_rss_bytes"],
                "cases": raw["cases"][5:],
            }
        )
    summary = {}
    for label in ["baseline", "candidate"]:
        subset = [r for r in rows if r["label"] == label]
        summary[label] = {
            k: stats([c[k] for r in subset for c in r["cases"]])
            for k in [
                "pose_to_gpu_complete_ms",
                "prepare_ms",
                "encode_ms",
                "gpu_submit_and_wait_ms",
            ]
        }
        summary[label]["peak_process_tree_rss_bytes"] = stats(
            [r["peak_process_tree_rss_bytes"] for r in subset]
        )
    if fingerprints() != snapshot:
        raise AssertionError("Inputs changed during benchmark")
    jsonwrite(
        BASE / "timing.json",
        {
            "inputs": fingerprints(),
            "warmup_per_process": 5,
            "runs": rows,
            "summary": summary,
            "note": "Two ABBA blocks, 60 timed poses each, full 768x1024 mask rendering, GPU completion included; no readback or concurrent validation. Native DX12 is not browser WebGPU. RSS includes process tree and is not GPU VRAM.",
        },
    )


def paired():
    """Complement independent-process timings; never replace their results."""
    snapshot = fingerprints()
    models = BASE / "paired_models.json"
    jsonwrite(
        models,
        [
            {
                "label": label,
                "manifest": str(path / "POPY.model3.json"),
                "moc_sha256": sha(path / "POPY.moc3"),
            }
            for label, path in [("baseline", BEFORE), ("candidate", MODEL)]
        ],
    )
    run(
        "representation_paired_cpu",
        [
            ROOT / "rust/target/release/drive_benchmark.exe",
            models,
            TIMING,
            BASE / "paired_cpu.json",
        ],
    )
    run(
        "representation_paired_gpu",
        [
            ROOT / "rust/target/release/render_benchmark.exe",
            BEFORE / "POPY.model3.json",
            MODEL / "POPY.model3.json",
            TIMING,
            BASE / "paired_gpu.json",
        ],
    )
    if fingerprints() != snapshot:
        raise AssertionError("Inputs changed during paired tests")
    for name in ["paired_cpu", "paired_gpu"]:
        path = BASE / (name + ".json")
        raw = json.loads(path.read_text())
        raw["inputs"] = snapshot
        jsonwrite(path, raw)


def report():
    native = json.loads((BASE / "native.json").read_text())
    pixels = json.loads((BASE / "pixels.json").read_text())
    timing = json.loads((BASE / "timing.json").read_text())
    inputs = fingerprints()
    if native["poses"] != len(json.loads(NATIVE.read_text())) or pixels["poses"] != len(
        json.loads(CASES.read_text())
    ):
        raise AssertionError("Incomplete quality cohort")
    if len(timing["runs"]) != 8 or any(len(r["cases"]) != 60 for r in timing["runs"]):
        raise AssertionError("Incomplete timing cohort")
    if any(
        x != inputs
        for x in [
            json.loads((BASE / "native_inputs.json").read_text()),
            pixels["inputs"],
            timing["inputs"],
        ]
    ):
        raise AssertionError("Stale evidence")
    for key in [
        "different_vertices",
        "different_visual_values",
        "different_effective_depths",
        "different_render_sequences",
    ]:
        if native[key]:
            raise AssertionError(key)
    if pixels["different_channels"]:
        raise AssertionError("Pixels changed")
    result = {
        "inputs": inputs,
        "native": native,
        "pixels": pixels,
        "timing": timing,
        "packing": json.loads((BASE / "packing.json").read_text()),
        "channel_factorization": json.loads((BASE / "rig/plan.json").read_text()),
        "inventory": {
            label: json.loads((BASE / ("inventory_" + label + ".json")).read_text())
            for label in ["baseline", "candidate"]
        },
    }
    for name in ["paired_cpu", "paired_gpu"]:
        path = BASE / (name + ".json")
        if path.exists():
            raw = json.loads(path.read_text())
            if raw.get("inputs") != inputs:
                raise AssertionError("Stale paired timing")
            metric = "driver_ms" if name == "paired_cpu" else "pose_to_gpu_complete_ms"
            raw["summary"] = {
                label: stats([r[metric] for r in raw["samples"] if r["label"] == label])
                for label in dict.fromkeys(r["label"] for r in raw["samples"])
            }
            result[name] = raw
    jsonwrite(BASE / "summary.json", result)
    public = (
        ROOT
        / "benchmarks/popy/fullbody"
        / (
            "representation_rotation.json"
            if CONFIG.get("rotation_carriers", False)
            else "representation.json"
        )
    )
    jsonwrite(public, result)
    (BASE / "report").mkdir(exist_ok=True)
    gallery = []
    for case in pixels["cases"]:
        name = case["name"]
        pictures = []
        reference = ROOT / "artifacts/fullbody/reference" / (name + ".png")
        if reference.exists():
            relative = Path(os.path.relpath(reference, BASE / "report")).as_posix()
            pictures.append(("Blender reference", relative))
        pictures.extend(
            (label, f"../{label}/{name}.png") for label in ["baseline", "candidate"]
        )
        gallery.append(
            "<details><summary>"
            + html.escape(name)
            + " — RGBA differences: "
            + str(case["different_channels"])
            + '</summary><div class="row">'
            + "".join(
                '<figure><img loading="lazy" src="'
                + html.escape(path, quote=True)
                + '"><figcaption>'
                + html.escape(label)
                + "</figcaption></figure>"
                for label, path in pictures
            )
            + "</div></details>"
        )
    (BASE / "report/index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>POPY lossless channel/storage optimization</title><h1>POPY 无损通道与存储优化</h1><p>All tested vertices, appearance values, effective depths, draw sequences and RGBA pixels are unchanged. This stage reduces file size/RAM; inspect timing independently.</p><pre>'
        + html.escape(
            json.dumps(
                {
                    "inputs": inputs,
                    "inventory": result["inventory"],
                    "timing": timing["summary"],
                    "native_poses": native["poses"],
                    "image_poses": pixels["poses"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        + "</pre><style>body{font:16px sans-serif;margin:24px;background:#202024;color:#eee}pre{overflow:auto}.row{display:flex;flex-wrap:wrap}figure{margin:10px}img{width:288px;background:repeating-conic-gradient(#333 0% 25%,#444 0% 50%) 0/20px 20px}details{border-top:1px solid #555;padding:8px}summary{cursor:pointer}</style><h2>Visual regression</h2>"
        + "".join(gallery),
        encoding="utf8",
    )


def package():
    data = json.loads((BASE / "summary.json").read_text())
    if data["inputs"] != fingerprints():
        raise AssertionError("Stale package evidence")
    manifest = json.loads((MODEL / "POPY.model3.json").read_text())
    refs = []

    def visit(v):
        if isinstance(v, dict):
            for x in v.values():
                visit(x)
        elif isinstance(v, list):
            for x in v:
                visit(x)
        elif isinstance(v, str) and Path(v).suffix in {".json", ".moc3", ".png"}:
            refs.append(v)

    visit(manifest["FileReferences"])
    if any(not (MODEL / p).is_file() for p in refs):
        raise AssertionError("Missing runtime resource")
    (MODEL / "README.txt").write_text(
        "POPY — verified channel/deformer/storage optimization\n"
        + (
            "ROTATION CANDIDATE: CPU and paired-GPU timings improved; independent-process frame timings regressed. Not the default package.\n"
            if CONFIG.get("rotation_carriers", False)
            else "DEFAULT STORAGE VERSION: exact file/RAM reduction; Glue frame-time bottlenecks remain.\n"
        )
        + "Load POPY.model3.json with the adjacent MOC3 and textures.\n"
        "39 parameters, 58 motions, 24 expressions, original topology/UV/textures retained.\n"
        "1,803 native poses and 362 DX12 RGBA renders match the baseline exactly.\n"
        "See verification.json inside the ZIP and docs/REPRESENTATION_OPTIMIZATION.md in the source repository for measurements and remaining limitations.\n"
        "This package contains no new CMO3. Official Cubism Core/Editor and browser performance have not been verified.\n",
        encoding="utf8",
    )
    target = ROOT / CONFIG["delivery"]
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(MODEL.rglob("*")):
            if p.is_file():
                z.write(p, "POPY/" + p.relative_to(MODEL).as_posix())
        z.write(BASE / "summary.json", "verification.json")
    with zipfile.ZipFile(target) as z:
        if z.testzip():
            raise AssertionError("ZIP CRC")
        if (
            hashlib.sha256(z.read("POPY/POPY.moc3")).hexdigest()
            != data["inputs"]["candidate_moc_sha256"]
        ):
            raise AssertionError("ZIP MOC differs")
    jsonwrite(
        BASE / "package.json",
        {
            "path": CONFIG["delivery"],
            "bytes": target.stat().st_size,
            "sha256": sha(target),
            "runtime_references_verified": len(refs),
            "crc_verified": True,
        },
    )


def test():
    run(
        "representation_python_tests",
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
    )
    run(
        "representation_rust_tests",
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


def main():
    stages = {
        "build": build,
        "rig": rig,
        "export": export,
        "pack": packing,
        "validate": validate,
        "images": images,
        "benchmark": benchmark,
        "paired": paired,
        "test": test,
        "report": report,
        "package": package,
    }
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["all", *stages])
    a = p.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    if a.stage == "all":
        # Isolate the large IR/JVM allocations between stages and record their
        # actual duration/peak process-tree RSS through the existing monitor.
        for name in stages:
            run(
                "representation_stage_" + name,
                [sys.executable, "-m", "popy.representation_review", name],
                os.environ.copy(),
            )
    else:
        stages[a.stage]()


if __name__ == "__main__":
    main()
