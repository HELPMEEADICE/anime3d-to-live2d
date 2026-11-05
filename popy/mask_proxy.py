"""Batch raw mask coverage independently of visible painter ordering.

Visible meshes keep their complete original programs. Hidden proxies concatenate
equal geometry programs; alpha-255 source coverage is idempotent union. Opaque
texture pages and source order can change, while every primitive remains. No
opacity fitting, Glue or topology simplification is used.
"""

import argparse
import copy
import json
import os
import shutil
import subprocess
import zipfile
from collections import defaultdict

from PIL import Image

from . import cache_review
from . import vts_exact as exact
from .cache_layout import pack_positions
from .compile_ir import ROOT, jsonwrite
from .trajectory_pool import program
from .vts_mask_exact import invisible

BASE = ROOT / "artifacts/mask_proxy"
SOURCE = ROOT / "output/POPY_VTS_Overlap"
MODEL = ROOT / "output/POPY_VTS_MaskBatch"


def ordered_runs(context, lookup, vertex_limit=60000):
    """Partition a source list without exchanging any primitive's position."""
    runs, previous, size = [], None, 0
    for name in context:
        m = lookup[name]
        key = (program(m), m["texture"], m["culling"])
        n = len(m["positions"]) // 2
        if runs and key == previous and size + n <= vertex_limit:
            runs[-1].append(name)
            size += n
        else:
            runs.append([name])
            size = n
        previous = key
    return runs


def opaque_runs(context, lookup, vertex_limit=60000):
    """Binary coverage is idempotent union; group equal programs across order.

    Caller must prove every sampled texture alpha is exactly one. Texture RGB,
    source opacity, painter order and the source's own mask do not enter raw
    mask coverage. Each original oriented primitive is still retained.
    """
    cohorts = defaultdict(list)
    for name in context:
        m = lookup[name]
        cohorts[(program(m), m["culling"])].append(name)
    return [
        run
        for group in cohorts.values()
        for run in ordered_runs_without_texture(group, lookup, vertex_limit)
    ]


def require_opaque_sources(folder, textures, pages):
    for page in pages:
        with Image.open(folder / textures[page]) as image:
            if image.convert("RGBA").getchannel("A").getextrema() != (255, 255):
                raise ValueError("Opaque union profile requires all source alpha = 255")


def ordered_runs_without_texture(names, lookup, vertex_limit):
    runs, size = [], 0
    for name in names:
        n = len(lookup[name]["positions"]) // 2
        if runs and size + n <= vertex_limit:
            runs[-1].append(name)
            size += n
        else:
            runs.append([name])
            size = n
    return runs


def configure():
    exact.BASE, exact.SOURCE, exact.MODEL, exact.PREFIX = (
        BASE,
        SOURCE,
        MODEL,
        "vts_mask_proxy",
    )
    cache_review.BASE, cache_review.SOURCE, cache_review.MODEL = BASE, SOURCE, MODEL


def build():
    configure()
    source = ROOT / "artifacts/overlap_layout/rig"
    ir = json.loads((source / "POPY.rig.json").read_text())
    lookup = {m["id"]: m for m in ir["meshes"]}
    contexts = sorted({tuple(m["masked_by"]) for m in ir["meshes"] if m["masked_by"]})
    source_pages = {lookup[n]["texture"] for ctx in contexts for n in ctx}
    require_opaque_sources(source, ir["textures"], source_pages)
    replacements, sequences = {}, {}
    for context in contexts:
        output = []
        for run in opaque_runs(context, lookup):
            if len(run) == 1:
                output.extend(run)
            else:
                key = tuple(run)
                if key not in sequences:
                    name = f"MaskProxy_{len(sequences):03}"
                    if name in lookup:
                        raise ValueError("Proxy ID collides with an existing mesh")
                    sequences[key] = name
                output.append(sequences[key])
        replacements[context] = output
        expanded = {name: list(seq) for seq, name in sequences.items()}
        if sorted(n for s in output for n in expanded.get(s, [s])) != sorted(context):
            raise ValueError("Mask primitive multiset changed")
    groups = []
    for m in ir["meshes"]:
        if m["masked_by"]:
            m["masked_by"] = replacements[tuple(m["masked_by"])]
        groups.append([m])
    aliases = {}
    # Preserve the geometry program, including neutral/null BlendShape pattern,
    # but never copy visible opacity, painter depth or a source's own clipping.
    # A source's own clipping is not applied to its raw mask alpha in Cubism.
    for names, proxy_id in sequences.items():
        group = []
        for i, name in enumerate(names):
            clone = copy.deepcopy(lookup[name])
            clone["id"] = proxy_id if i == 0 else f"{proxy_id}_member_{i}"
            aliases[clone["id"]] = name
            clone.update(opacity=0.0, masked_by=[], invert_mask=False)
            for cell in clone["cells"]:
                cell.update(order=0.0, opacity=0.0)
            for blend in clone["blend_shapes"]:
                blend["orders"] = [0.0] * len(blend["keys"])
                blend["opacities"] = [0.0] * len(blend["keys"])
            group.append(clone)
        groups.append(group)
    intermediate = BASE / "proxy_rig"
    rows = exact.emit_rig(ir, source, intermediate, groups)
    original = [r for r in rows if r["source"] not in aliases]
    proxies = [
        {**r, "source": aliases[r["source"]], "mask_proxy": True}
        for r in rows
        if r["source"] in aliases
    ]
    # Proxies release the former source meshes from mask membership. Remove
    # newly dead invisible copies and batch the released adjacent visible runs.
    ir = json.loads((intermediate / "POPY.rig.json").read_text())
    used = {s for m in ir["meshes"] for s in m["masked_by"]}
    removed = {
        m["id"]
        for m in ir["meshes"]
        if invisible(m) and m["id"] not in used and not m["masked_by"]
    }
    mesh_groups = exact.groups([m for m in ir["meshes"] if m["id"] not in removed])
    second = exact.emit_rig(ir, intermediate, BASE / "rig", mesh_groups)
    by_id = {r["source"]: r for r in second}

    def compose(row):
        if row["target"] in removed:
            return {"source": row["source"], "removed": True}
        parent = by_id[row["target"]]
        return {
            **row,
            "target": parent["target"],
            "vertex_start": parent["vertex_start"] + row["vertex_start"],
            "index_start": parent["index_start"] + row["index_start"],
        }

    original = [compose(r) for r in original]
    proxies = [compose(r) for r in proxies]
    jsonwrite(
        BASE / "mapping.json",
        {
            "rows": original,
            "mask_proxies": proxies,
            "opaque_mask_union": True,
        },
    )
    jsonwrite(
        BASE / "build.json",
        {
            "proxies": len(sequences),
            "proxy_vertices": sum(r["vertex_count"] for r in proxies),
            "context_source_draws_before": sum(map(len, contexts)),
            "context_source_draws_after": sum(map(len, replacements.values())),
            "contexts": len(contexts),
            "sequences": {name: list(seq) for seq, name in sequences.items()},
            "geometry_arithmetic_changed": False,
            "opaque_mask_union": True,
            "removed_dead_originals": len(removed),
            "additional_adjacent_batches": sum(len(g) - 1 for g in mesh_groups),
        },
    )


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
    temporary = MODEL / "POPY.overlap.moc3"
    native_layout = json.loads(layout.read_text())
    retained = {m["id"] for m in native_layout["meshes"]}
    pairs = [
        p
        for p in json.loads((ROOT / "artifacts/overlap_layout/pairs.json").read_text())
        if p["a"] in retained and p["b"] in retained
    ]
    pack_positions(
        MODEL / "POPY.moc3",
        temporary,
        native_layout,
        pairs,
        BASE / "overlap.json",
    )
    temporary.replace(MODEL / "POPY.moc3")


def validate():
    configure()
    before = cache_review.fingerprints()
    exact.validate()
    if before != cache_review.fingerprints():
        raise ValueError("Validation inputs changed")
    jsonwrite(BASE / "validated_inputs.json", before)


def timing():
    configure()
    from .cli import run

    before = cache_review.fingerprints()
    run(
        "vts_mask_proxy_timing",
        [
            ROOT / "rust/target/release/render_benchmark.exe",
            SOURCE / "POPY.model3.json",
            MODEL / "POPY.model3.json",
            ROOT / "artifacts/perf_opt/timing_cases.json",
            BASE / "gpu_trial.json",
        ],
    )
    if before != cache_review.fingerprints():
        raise ValueError("Timing inputs changed")
    jsonwrite(BASE / "gpu_trial_inputs.json", before)


def memory():
    configure()
    from .cli import run

    before = cache_review.fingerprints()
    if before != json.loads((BASE / "validated_inputs.json").read_text()):
        raise ValueError("Validate these inputs before memory measurement")
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
        folder = SOURCE if label == "baseline" else MODEL
        run(
            f"vts_mask_proxy_memory_{i}_{label}",
            [
                ROOT / "rust/target/release/popy-validate.exe",
                folder / "POPY.model3.json",
                ROOT / "artifacts/perf_opt/timing_cases.json",
                BASE / f"memory_{i}_{label}",
                "--timing-only",
            ],
            env,
        )
        sample = json.loads((ROOT / "artifacts/performance.json").read_text())[-1]
        rows.append(
            {"label": label, "peak_rss_bytes": sample["peak_process_tree_rss_bytes"]}
        )
    if before != cache_review.fingerprints():
        raise ValueError("Memory inputs changed")
    jsonwrite(
        BASE / "memory_repeats.json",
        {
            "inputs": before,
            "sequence": "ABBA BAAB; single-model processes; same 65 poses; loading and warmup included; no readback/PNG",
            "rows": rows,
        },
    )


def report():
    configure()
    if cache_review.fingerprints() != json.loads(
        (BASE / "validated_inputs.json").read_text()
    ):
        raise ValueError("Stale validation")
    r = exact.report()
    r["build"] = json.loads((BASE / "build.json").read_text())
    r["validated_inputs"] = cache_review.fingerprints()
    r["no_new_quality_loss"] = (
        "Visible mesh programs remain unchanged. Mask proxies concatenate identical "
        "geometry programs. Alpha-255 texture coverage is an idempotent union, "
        "allowing source order and opaque texture page substitution; the exact "
        "per-context primitive multiset, UVs, culling, parent and interpolation "
        "are preserved. No Glue."
    )
    r["limits"] = (
        "Relative to Overlap; existing lossy-fit errors remain. Unmodified native "
        "Ayagami DX12 timings, not VTS/Core/Editor validation. Hidden proxy geometry "
        "is duplicated to trade a small CPU/memory cost for fewer mask draw calls."
    )
    import numpy as np

    samples = json.loads((BASE / "gpu.json").read_text())["samples"]
    r["frame_stages_median_ms"] = {
        label: {
            key: float(np.median([s[key] for s in samples if s["label"] == label]))
            for key in ["prepare_ms", "encode_ms", "gpu_submit_and_wait_ms"]
        }
        for label in ["baseline", "candidate"]
    }
    if (
        json.loads((BASE / "gpu_trial_inputs.json").read_text())
        != cache_review.fingerprints()
    ):
        raise ValueError("Stale timing trial")
    r["independent_timing_trial"] = json.loads((BASE / "gpu_trial.json").read_text())
    repeats = json.loads((BASE / "memory_repeats.json").read_text())
    if repeats["inputs"] != cache_review.fingerprints():
        raise ValueError("Stale memory inputs")
    r["repeated_single_model_memory"] = {
        "method": repeats["sequence"],
        "rows": repeats["rows"],
        "summary": {
            label: {
                "median_peak_rss_mib": float(np.median(values)) / 2**20,
                "min_peak_rss_mib": min(values) / 2**20,
                "max_peak_rss_mib": max(values) / 2**20,
            }
            for label in ["baseline", "candidate"]
            for values in [
                [r["peak_rss_bytes"] for r in repeats["rows"] if r["label"] == label]
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
        raise ValueError("Mask context count changed")
    jsonwrite(BASE / "report.json", r)
    return r


def package():
    r = report()
    if (
        r["timing"]["gpu"]["candidate"]["median_ms"]
        >= r["timing"]["gpu"]["baseline"]["median_ms"] * 0.97
    ):
        raise ValueError("Insufficient measured frame-time benefit to promote")
    destination = ROOT / "delivery" / MODEL.name
    shutil.copytree(MODEL, destination, dirs_exist_ok=True)
    shutil.copy2(BASE / "report.json", destination / "performance_report.json")
    shutil.copy2(BASE / "candidate/natural.png", destination / "preview.png")
    (destination / "README.txt").write_text(
        "POPY VTS 遮罩绘制批处理版\n\n"
        "相对 Overlap 不新增拟合误差，保留原画面及牙齿修复。\n"
        "全不透明遮罩源按几何程序合并为隐藏代理；可见画面的参数程序保留。\n"
        "增加部分隐藏顶点换取更少绘制次数；没有新增 Glue。\n"
        "仍有 80 组遮罩、39 参数、58 动作、24 表情。贴图不变。\n"
        "安装到 VTube Studio_Data/StreamingAssets/Live2DModels/，加载 POPY.model3.json。\n"
        "仍需设置 256 masks，每组至少 512x512。\n"
        "部分内部网格已合并，逐 ArtMesh 自定义配置可能需要重设。\n"
        "运行时单独隐藏源网格或替换为带透明像素的贴图时，遮罩代理也需同步重建。\n"
        "旧拟合误差保留；无新 CMO3。性能报告为原生 Ayagami DX12，非 VTS 实机。\n",
        encoding="utf-8-sig",
    )
    with zipfile.ZipFile(
        destination.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED, compresslevel=5
    ) as z:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(destination.parent))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "stage",
        choices=[
            "all",
            "build",
            "export",
            "timing",
            "validate",
            "memory",
            "report",
            "package",
        ],
    )
    a = p.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    for f in (
        [build, export, timing, validate, memory, package]
        if a.stage == "all"
        else [globals()[a.stage]]
    ):
        f()


if __name__ == "__main__":
    main()
