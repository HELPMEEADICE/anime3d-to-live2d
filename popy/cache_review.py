"""Lossless cache/storage layout profile, keeping the runtime Glue-free."""

import argparse
import json
import shutil
import subprocess
import zipfile

import numpy as np
from PIL import Image

from . import vts_exact as exact
from .cache_layout import cache_order, first_use, pack_positions, paired_permutations
from .compile_ir import ROOT, jsonwrite
from .representation_review import resource_sha, sha
from .trajectory_pool import program
from .vts_mask_exact import invisible

BASE = ROOT / "artifacts/cache_layout"
SOURCE = ROOT / "output/POPY_VTS_Exact2"
MODEL = ROOT / "output/POPY_VTS_Cache"
STORAGE_ONLY = False


def configure():
    exact.BASE, exact.SOURCE, exact.MODEL, exact.PREFIX = (
        BASE,
        SOURCE,
        MODEL,
        "vts_overlap" if STORAGE_ONLY else "vts_cache",
    )


def build():
    source = ROOT / "artifacts/vts_mask_exact/rig"
    folder = BASE / "rig"
    folder.mkdir(parents=True, exist_ok=True)
    ir = json.loads((source / "POPY.rig.json").read_text())
    data = np.memmap(source / ir["keyform_data"], "<f4", mode="r")
    sources = {s for m in ir["meshes"] for s in m["masked_by"]}
    opaque = []
    for name in ir["textures"]:
        with Image.open(source / name) as image:
            opaque.append(
                image.convert("RGBA").getchannel("A").getextrema() == (255, 255)
            )
    eligible = [
        m
        for m in ir["meshes"]
        if m["id"] in sources
        and invisible(m)
        and not m["masked_by"]
        and opaque[m["texture"]]
    ]
    triangle_orders = {}
    for mesh in [] if STORAGE_ONLY else eligible:
        indices, order = cache_order(mesh["indices"], len(mesh["positions"]) // 2)
        mesh["indices"] = indices
        triangle_orders[mesh["id"]] = order
    vertex_orders, pairs = paired_permutations(eligible, data, program)
    jsonwrite(BASE / "pairs.json", pairs)
    mapping = []
    cursor = 0
    with (folder / "keyforms.f32").open("wb") as stream:

        def emit(values):
            nonlocal cursor
            a = np.asarray(values, dtype="<f4").ravel()
            begin = cursor
            stream.write(a.tobytes())
            cursor += len(a)
            return begin

        for mesh in ir["meshes"]:
            n = len(mesh["positions"]) // 2
            order = vertex_orders.get(
                mesh["id"],
                list(range(n)) if STORAGE_ONLY else first_use(mesh["indices"], n),
            )
            inverse = np.empty(n, dtype=int)
            inverse[order] = np.arange(n)
            mesh["positions"] = (
                np.array(mesh["positions"]).reshape(n, 2)[order].ravel().tolist()
            )
            mesh["uvs"] = np.array(mesh["uvs"]).reshape(n, 2)[order].ravel().tolist()
            mesh["indices"] = inverse[mesh["indices"]].tolist()
            row = {
                "source": mesh["id"],
                "target": mesh["id"],
                "vertex_start": 0,
                "vertex_count": n,
                "index_start": 0,
                "index_count": len(mesh["indices"]),
                "vertex_order": order,
            }
            if mesh["id"] in triangle_orders:
                row["opaque_mask_triangle_order"] = triangle_orders[mesh["id"]]
            mapping.append(row)
            for cell in mesh["cells"]:
                offset = cell["offset"]
                cell["offset"] = emit(
                    data[offset : offset + 2 * n].reshape(n, 2)[order]
                )
            for blend in mesh["blend_shapes"]:
                blend["offsets"] = [
                    None
                    if off is None
                    else emit(data[off : off + 2 * n].reshape(n, 2)[order])
                    for off in blend["offsets"]
                ]
    for name in ir["textures"]:
        shutil.copy2(source / name, folder / name)
    jsonwrite(folder / "POPY.rig.json", ir)
    jsonwrite(BASE / "mapping.json", mapping)
    jsonwrite(
        BASE / "build.json",
        {
            "mapped_meshes": len(mapping),
            "vertex_permuted_meshes": sum(
                r["vertex_order"] != list(range(r["vertex_count"])) for r in mapping
            ),
            "opaque_cache_ordered_meshes": len(triangle_orders),
            "paired_meshes": 2 * len(pairs),
            "potential_overlap_vertices": sum(p["shared"] for p in pairs),
            "glues": len(ir["glues"]),
            "storage_only": STORAGE_ONLY,
        },
    )
    print(json.loads((BASE / "build.json").read_text()), flush=True)


def export():
    configure()
    exact.export()
    layout = BASE / "form_layout.json"
    subprocess.run(
        [
            str(ROOT / "rust/target/release/form_layout.exe"),
            str(MODEL / "POPY.model3.json"),
            str(layout),
        ],
        check=True,
    )
    out = MODEL / "POPY.overlap.moc3"
    result = pack_positions(
        MODEL / "POPY.moc3",
        out,
        json.loads(layout.read_text()),
        json.loads((BASE / "pairs.json").read_text()),
        BASE / "overlap.json",
    )
    out.replace(MODEL / "POPY.moc3")
    print(result, flush=True)


def fingerprints():
    return {
        "source": sha(SOURCE / "POPY.moc3"),
        "candidate": sha(MODEL / "POPY.moc3"),
        "mapping": sha(BASE / "mapping.json"),
        "geometry_cases": sha(ROOT / "artifacts/perf_opt/equivalence_cases.json"),
        "image_cases": sha(ROOT / "artifacts/fullbody/cases.json"),
        "timing_cases": sha(ROOT / "artifacts/perf_opt/timing_cases.json"),
        "assets_source": resource_sha(SOURCE),
        "assets_candidate": resource_sha(MODEL),
    }


def validate():
    configure()
    before = fingerprints()
    exact.validate()
    if before != fingerprints():
        raise ValueError("Validation inputs changed")
    jsonwrite(BASE / "validated_inputs.json", before)


def report():
    configure()
    if fingerprints() != json.loads((BASE / "validated_inputs.json").read_text()):
        raise ValueError("Stale validation")
    r = exact.report()
    r["build"] = json.loads((BASE / "build.json").read_text())
    r["position_overlap"] = json.loads((BASE / "overlap.json").read_text())
    r["no_new_quality_loss"] = (
        "Every original vertex float bit and UV preserved by explicit permutation. Visible primitive order unchanged; only alpha-255 constant-zero-opacity mask sources may reorder the same oriented triangles. Native position blocks share byte-identical overlaps. No new Glue or fitting."
    )
    r["limits"] = (
        "Relative to Exact2; previous lossy-fit errors remain. Native unmodified Ayagami DX12, not VTube Studio/Core/Editor verification."
    )
    r["validated_inputs"] = fingerprints()
    repeats = json.loads((BASE / "memory_repeats.json").read_text())
    if repeats["inputs"] != fingerprints():
        raise ValueError("Stale memory inputs")
    r["repeated_single_model_memory"] = {
        "method": repeats["sequence"],
        "rows": repeats["rows"],
        "summary": {
            label: {
                "samples": len(values),
                "median_peak_rss_mib": float(np.median(values)) / 2**20,
                "min_peak_rss_mib": min(values) / 2**20,
                "max_peak_rss_mib": max(values) / 2**20,
            }
            for label in ["baseline", "candidate"]
            for values in [
                [x["peak_rss_bytes"] for x in repeats["rows"] if x["label"] == label]
            ]
        },
    }
    r["mask_contexts"] = {
        label: len(
            {
                tuple(sorted(m["clips"]))
                for m in json.loads((BASE / label / "model_inventory.json").read_text())
                if m["clips"]
            }
        )
        for label in ["baseline", "candidate"]
    }
    if r["mask_contexts"] != {"baseline": 80, "candidate": 80}:
        raise ValueError("Mask contexts changed")
    jsonwrite(BASE / "report.json", r)
    return r


def memory():
    import os

    from .cli import run

    configure()
    before = fingerprints()
    if before != json.loads((BASE / "validated_inputs.json").read_text()):
        raise ValueError("Validate first")
    env = os.environ.copy()
    env["POPY_WGPU_BACKEND"] = "dx12"
    rows = []
    for i, label in enumerate(
        [
            "baseline",
            "candidate",
            "candidate",
            "baseline",
            "candidate",
            "baseline",
            "baseline",
            "candidate",
        ]
    ):
        stage = f"{exact.PREFIX}_memory_{i}_{label}"
        folder = SOURCE if label == "baseline" else MODEL
        run(
            stage,
            [
                ROOT / "rust/target/release/popy-validate.exe",
                folder / "POPY.model3.json",
                ROOT / "artifacts/perf_opt/timing_cases.json",
                BASE / f"memory_{i}_{label}",
                "--timing-only",
            ],
            env,
        )
        row = json.loads((ROOT / "artifacts/performance.json").read_text())[-1]
        rows.append(
            {"label": label, "peak_rss_bytes": row["peak_process_tree_rss_bytes"]}
        )
    if before != fingerprints():
        raise ValueError("Memory inputs changed")
    jsonwrite(
        BASE / "memory_repeats.json",
        {
            "inputs": before,
            "sequence": "ABBA BAAB; 8 independent single-model native DX12 processes, same 65 poses, includes loading/warmup, no PNG/readback",
            "rows": rows,
        },
    )


def package():
    report()
    destination = ROOT / "delivery" / MODEL.name
    shutil.copytree(MODEL, destination, dirs_exist_ok=True)
    shutil.copy2(BASE / "report.json", destination / "performance_report.json")
    shutil.copy2(BASE / "candidate/natural.png", destination / "preview.png")
    (destination / "README.txt").write_text(
        "POPY VTS 顶点/存储布局无损优化版\n\n相对 Exact2 不新增画质损失，旧拟合误差保留。\n顶点重排和位置块字节重叠共享；运行时仍零 Glue。\n"
        + (
            "本存储配置保留全部原三角形顺序，没有进行三角缓存重排。\n"
            if STORAGE_ONLY
            else "仅全不透明隐藏遮罩进行了三角缓存重排。\n"
        )
        + "安装至 VTube Studio_Data/StreamingAssets/Live2DModels/，加载 POPY.model3.json。\n仍需 256 masks，每组至少 512x512；80 组遮罩和牙齿修复保留。\n39 参数、58 动作、24 表情、贴图字节不改变。内部顶点编号改变，ArtMesh ID 保留；逐顶点附着设置可能需要重建。\n报告为原生 Ayagami DX12 数据，未进行 VTS 实机或官方 Core/Editor 验证，无新 CMO3。\n本轮主要收益是文件/内存，不能由文件变小比例推导帧率提升。\n",
        encoding="utf-8-sig",
    )
    archive = destination.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as z:
        for p in sorted(destination.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(destination.parent))
    print(f"Delivered {archive}: {archive.stat().st_size} bytes")


def main():
    global BASE, MODEL, STORAGE_ONLY
    p = argparse.ArgumentParser()
    p.add_argument(
        "stage",
        choices=["all", "build", "export", "validate", "memory", "report", "package"],
    )
    p.add_argument("--storage-only", action="store_true")
    a = p.parse_args()
    if a.storage_only:
        STORAGE_ONLY = True
        BASE = ROOT / "artifacts/overlap_layout"
        MODEL = ROOT / "output/POPY_VTS_Overlap"
    BASE.mkdir(parents=True, exist_ok=True)
    for f in (
        [build, export, validate, memory, package]
        if a.stage == "all"
        else [globals()[a.stage]]
    ):
        f()


if __name__ == "__main__":
    main()
