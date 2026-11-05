"""Exact storage reuse and adjacent draw batching of the delivered VTS rig.

Only adjacent, scalar-identical non-mask-source meshes are concatenated.
Geometry values and each vertex's interpolation program remain unchanged.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import zipfile

import numpy as np

from .compile_ir import ROOT, jsonwrite
from .moc_pack import pack, runtime_extents
from .representation_review import copy_assets, resource_sha, sha

BASE = ROOT / "artifacts/vts_exact"
SOURCE = ROOT / "output/POPY_VTS_Performance"
MODEL = ROOT / "output/POPY_VTS_Exact"
PREFIX = "vts_exact"


def signature(mesh):
    # A batch has identical scalar functions, interpolation axes, parent and
    # clipping. Being adjacent in the authored stable order ensures no third
    # drawable can sort between the members when their depth is tied.
    value = {
        k: mesh.get(k)
        for k in [
            "axes",
            "texture",
            "deformer",
            "projection_warp",
            "opacity",
            "masked_by",
            "invert_mask",
            "culling",
        ]
    }
    value["cells"] = [
        {k: v for k, v in cell.items() if k != "offset"} for cell in mesh["cells"]
    ]
    value["blends"] = [
        {k: v for k, v in blend.items() if k != "offsets"}
        | {"nulls": [v is None for v in blend["offsets"]]}
        for blend in mesh["blend_shapes"]
    ]
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).digest()


def groups(meshes, vertex_limit=60000):
    # Never merge a source: that could expand its coverage for a different
    # context, or change the original order of mask alpha accumulation.
    sources = {name for mesh in meshes for name in mesh["masked_by"]}
    result = []
    previous = None
    vertices = 0
    for mesh in meshes:
        n = len(mesh["positions"]) // 2
        key = signature(mesh) if mesh["id"] not in sources else None
        if key is not None and key == previous and vertices + n <= vertex_limit:
            result[-1].append(mesh)
            vertices += n
        else:
            result.append([mesh])
            vertices = n
        previous = key
    return result


def rig():
    source = ROOT / "artifacts/vts_fast/rig"
    out = BASE / "rig"
    ir = json.loads((source / "POPY.rig.json").read_text())
    return emit_rig(ir, source, out, groups(ir["meshes"]))


def emit_rig(ir, source, out, mesh_groups):
    """Concatenate unchanged vertex programs for a separately proven grouping."""
    out.mkdir(parents=True, exist_ok=True)
    original_count = len(ir["meshes"])
    data = np.memmap(source / ir["keyform_data"], "<f4", mode="r")
    merged = []
    mapping = []
    cursor = 0
    with (out / "keyforms.f32").open("wb") as stream:

        def emit(group, offsets):
            nonlocal cursor
            begin = cursor
            for mesh, offset in zip(group, offsets):
                payload = data[offset : offset + len(mesh["positions"])]
                stream.write(payload.tobytes())
                cursor += len(payload)
            return begin

        for group in mesh_groups:
            first = group[0]
            mesh = {
                **first,
                "positions": [],
                "uvs": [],
                "indices": [],
                "cells": [],
                "blend_shapes": [],
            }
            vertex = index = 0
            for member in group:
                mapping.append(
                    {
                        "source": member["id"],
                        "target": first["id"],
                        "vertex_start": vertex,
                        "vertex_count": len(member["positions"]) // 2,
                        "index_start": index,
                        "index_count": len(member["indices"]),
                    }
                )
                mesh["positions"].extend(member["positions"])
                mesh["uvs"].extend(member["uvs"])
                mesh["indices"].extend(i + vertex for i in member["indices"])
                vertex += len(member["positions"]) // 2
                index += len(member["indices"])
            for ci, cell in enumerate(first["cells"]):
                mesh["cells"].append(
                    {
                        **cell,
                        "offset": emit(
                            group, [m["cells"][ci]["offset"] for m in group]
                        ),
                    }
                )
            for bi, blend in enumerate(first["blend_shapes"]):
                offsets = [
                    None
                    if offset is None
                    else emit(
                        group, [m["blend_shapes"][bi]["offsets"][ki] for m in group]
                    )
                    for ki, offset in enumerate(blend["offsets"])
                ]
                mesh["blend_shapes"].append({**blend, "offsets": offsets})
            mesh["source"] = {
                "exact_adjacent_batch": [m["id"] for m in group],
                "geometry_arithmetic_changed": False,
            }
            merged.append(mesh)
    ir["meshes"] = merged
    ir["parts"] = [{"id": "P_Distilled", "mesh_ids": [m["id"] for m in merged]}]
    ir["diagnostics"]["exact_batching"] = {
        "before_meshes": original_count,
        "after_meshes": len(merged),
    }
    for texture in ir["textures"]:
        shutil.copy2(source / texture, out / texture)
    jsonwrite(out / "POPY.rig.json", ir)
    jsonwrite(BASE / "mapping.json", mapping)
    print(ir["diagnostics"]["exact_batching"], flush=True)
    return mapping


def export():
    env = os.environ.copy()
    env["JAVA_HOME"] = r"C:\lib\java"
    env["JAVA_OPTS"] = "-Xmx8g -XX:+UseG1GC -XX:+ExitOnOutOfMemoryError"
    raw = MODEL.with_name(MODEL.name + "_unpacked")
    subprocess.run(
        [
            str(
                ROOT
                / "exporter/build/install/popy-cubism-exporter/bin/popy-cubism-exporter.bat"
            ),
            str(BASE / "rig/POPY.rig.json"),
            str(raw),
            "--no-cmo",
            "--deduplicate-positions",
        ],
        env=env,
        check=True,
    )
    copy_assets(SOURCE, raw)
    copy_assets(SOURCE, MODEL)
    pack(
        raw / "POPY.moc3",
        MODEL / "POPY.moc3",
        BASE / "pack.json",
        runtime_extents(raw / "POPY.moc3"),
    )


def validate():
    from .cli import run

    run(
        f"{PREFIX}_equivalence",
        [
            ROOT / "rust/target/release/batch_equivalence.exe",
            SOURCE / "POPY.model3.json",
            MODEL / "POPY.model3.json",
            BASE / "mapping.json",
            ROOT / "artifacts/perf_opt/equivalence_cases.json",
            BASE / "equivalence.json",
        ],
    )
    env = os.environ.copy()
    env["POPY_WGPU_BACKEND"] = "dx12"
    models = [
        {"label": label, "manifest": str(folder / "POPY.model3.json")}
        for label, folder in [("baseline", SOURCE), ("candidate", MODEL)]
    ]
    jsonwrite(BASE / "models.json", models)
    for model in models:
        label = model["label"]
        run(
            f"{PREFIX}_{label}_rgba362",
            [
                ROOT / "rust/target/release/popy-validate.exe",
                model["manifest"],
                ROOT / "artifacts/fullbody/cases.json",
                BASE / label,
                "--images-only",
            ],
            env,
        )
        run(
            f"{PREFIX}_{label}_stats",
            [
                ROOT / "rust/target/release/model_stats.exe",
                model["manifest"],
                BASE / f"{label}_stats.json",
            ],
        )
    run(
        f"{PREFIX}_cpu",
        [
            ROOT / "rust/target/release/drive_benchmark.exe",
            BASE / "models.json",
            ROOT / "artifacts/perf_opt/timing_cases.json",
            BASE / "cpu.json",
        ],
    )
    run(
        f"{PREFIX}_gpu",
        [
            ROOT / "rust/target/release/render_benchmark.exe",
            models[0]["manifest"],
            models[1]["manifest"],
            ROOT / "artifacts/perf_opt/timing_cases.json",
            BASE / "gpu.json",
        ],
    )


def report():
    from PIL import Image

    cases_path = ROOT / "artifacts/fullbody/cases.json"
    cases = json.loads(cases_path.read_text())
    pixels = 0
    for case in cases:
        a, b = [
            np.asarray(
                Image.open(BASE / label / (case["name"] + ".png")).convert("RGBA")
            )
            for label in ["baseline", "candidate"]
        ]
        pixels += int(np.any(a != b, axis=2).sum())
    if pixels:
        raise ValueError(f"Rejected: {pixels} changed pixels")
    equivalence = json.loads((BASE / "equivalence.json").read_text())
    if equivalence["different_vertices"]:
        raise ValueError("Rejected: changed geometry")
    # All textures, motion/exp3/CDI metadata must remain the same bytes.
    resources = [resource_sha(folder) for folder in [SOURCE, MODEL]]
    if resources[0] != resources[1]:
        raise ValueError("Runtime resources changed")
    stats = {
        label: json.loads((BASE / f"{label}_stats.json").read_text())
        for label in ["baseline", "candidate"]
    }
    perf = json.loads((ROOT / "artifacts/performance.json").read_text())
    memory = {
        label: next(
            p["peak_process_tree_rss_bytes"]
            for p in reversed(perf)
            if p["stage"] == f"{PREFIX}_{label}_rgba362"
        )
        for label in stats
    }
    timing = {}
    for name, key in [("cpu", "driver_ms"), ("gpu", "pose_to_gpu_complete_ms")]:
        rows = json.loads((BASE / f"{name}.json").read_text())["samples"]
        timing[name] = {
            label: {
                "samples": len(values),
                "mean_ms": float(np.mean(values)),
                "median_ms": float(np.median(values)),
                "p95_ms": float(np.percentile(values, 95)),
            }
            for label in stats
            for values in [[r[key] for r in rows if r["label"] == label]]
        }
    result = {
        "statistics": stats,
        "equivalence": equivalence,
        "rendered_poses": len(cases),
        "different_pixels": pixels,
        "timing": timing,
        "peak_render_process_rss_bytes": memory,
        "source_sha256": sha(SOURCE / "POPY.moc3"),
        "candidate_sha256": sha(MODEL / "POPY.moc3"),
        "mapping_sha256": sha(BASE / "mapping.json"),
        "image_cases_sha256": sha(cases_path),
        "geometry_cases_sha256": sha(
            ROOT / "artifacts/perf_opt/equivalence_cases.json"
        ),
        "runtime_assets_sha256": resources[0],
        "no_new_quality_loss": "Exact vertex float bits, same triangle order within each adjacent equal-depth run, identical scalar programs and clipping, unchanged original mask sources and resource bytes.",
        "limits": "Relative to the delivered lossy performance edition; its existing errors remain. Finite runtime/render verification is not official Cubism Core or VTube Studio validation. Native DX12 paired benchmark, both models resident; desktop not isolated.",
    }
    jsonwrite(BASE / "report.json", result)
    print(
        json.dumps(
            {
                "statistics": stats,
                "timing": timing,
                "peak_rss_mib": {k: v / 2**20 for k, v in memory.items()},
            }
        )
    )
    return result


def package():
    result = report()
    destination = ROOT / "delivery/POPY_VTS_Exact"
    shutil.copytree(MODEL, destination, dirs_exist_ok=True)
    shutil.copy2(BASE / "report.json", destination / "performance_report.json")
    shutil.copy2(BASE / "candidate/natural.png", destination / "preview.png")
    (destination / "README.txt").write_text(
        "POPY VTS 性能版·后续无损优化\n\n"
        "以此前 POPY_VTS_Performance 为基准，不新增拟合误差，不降纹理/遮罩分辨率。\n"
        "原版已有的复合动作误差依然保留；本包没有将它们修回高质量 teacher。\n\n"
        "将整个文件夹放入 VTube Studio_Data/StreamingAssets/Live2DModels/，\n"
        "加载 POPY.model3.json，首次 Auto-Setup。仍需 256 masks，每组至少 512x512。\n"
        "保留 80 组遮罩及全部牙齿修复、39 参数、58 动作、24 表情。\n\n"
        "优化：仅合并严格相同材质/遮挡/参数/外观函数的相邻网格，遮罩来源不合并；\n"
        "位置和颜色数据按原始浮点位模式复用，并裁去运行时不读取的填充。\n"
        "1803 姿态所有顶点逐位一致，362 姿态 RGBA 逐像素一致。\n"
        "部分内部 ArtMesh ID 合并，旧 VTS 的逐 ArtMesh 自定义设置可能需要重设。\n"
        "VTube Studio 实机加载、FPS 和内存未验证；包内数据为原生 Ayagami DX12。\n"
        "没有新 CMO3。旧包保留可切回。\n",
        encoding="utf-8-sig",
    )
    archive = destination.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as z:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(destination.parent))
    print(f"Delivered {archive}: {archive.stat().st_size} bytes")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["all", "rig", "export", "validate", "report", "package"]
    )
    args = parser.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    stages = (
        [rig, export, validate, package]
        if args.stage == "all"
        else [globals()[args.stage]]
    )
    for stage in stages:
        stage()


if __name__ == "__main__":
    main()
