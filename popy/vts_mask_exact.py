"""Further exact optimization: invisible dependency pruning and mask batching.

Mask chains must be consecutive in EVERY context, with identical interpolation
programs and pipeline state. Their original primitive order is retained.
"""

import argparse
import collections
import hashlib
import json
import shutil
import zipfile

import numpy as np

from . import vts_exact as exact
from .compile_ir import ROOT, jsonwrite
from .representation_review import resource_sha, sha

BASE = ROOT / "artifacts/vts_mask_exact"
SOURCE = ROOT / "output/POPY_VTS_Exact"
MODEL = ROOT / "output/POPY_VTS_Exact2"


def invisible(mesh):
    return (
        bool(mesh["cells"])
        and all(c.get("opacity", mesh["opacity"]) == 0 for c in mesh["cells"])
        and all(
            b.get("opacities") is not None
            and len(b["opacities"]) == len(b["keys"])
            and all(v == 0 for v in b["opacities"])
            for b in mesh["blend_shapes"]
        )
    )


def geometry_signature(mesh):
    value = {
        k: mesh.get(k)
        for k in [
            "axes",
            "texture",
            "deformer",
            "projection_warp",
            "masked_by",
            "invert_mask",
            "culling",
        ]
    }
    value["coordinates"] = [c["coordinate"] for c in mesh["cells"]]
    value["blends"] = [
        {k: v for k, v in b.items() if k not in ["offsets", "orders"]}
        | {"nulls": [v is None for v in b["offsets"]]}
        for b in mesh["blend_shapes"]
    ]
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).digest()


def chains(meshes, vertex_limit=60000):
    lookup = {m["id"]: m for m in meshes}
    contexts = sorted({tuple(m["masked_by"]) for m in meshes if m["masked_by"]})
    users = collections.defaultdict(list)
    for i, context in enumerate(contexts):
        for name in context:
            users[name].append(i)
    keys = {
        name: geometry_signature(lookup[name])
        for name in users
        if invisible(lookup[name]) and not lookup[name]["masked_by"]
    }
    successor = {}
    for name, key in keys.items():
        following = []
        for i in users[name]:
            context = contexts[i]
            j = context.index(name)
            following.append(context[j + 1] if j + 1 < len(context) else None)
        if len(set(following)) == 1:
            next_name = following[0]
            if (
                next_name in keys
                and key == keys[next_name]
                and users[name] == users[next_name]
            ):
                successor[name] = next_name
    result = []
    for name in sorted(set(keys) - set(successor.values())):
        chain = [name]
        while chain[-1] in successor:
            chain.append(successor[chain[-1]])
        part, vertices = [], 0
        for member in chain:
            n = len(lookup[member]["positions"]) // 2
            if part and vertices + n > vertex_limit:
                result.append(part)
                part, vertices = [], 0
            part.append(member)
            vertices += n
        result.append(part)
    if sum(map(len, result)) != len(keys):
        raise ValueError("Cyclic or incomplete mask chain")
    return [group for group in result if len(group) > 1]


def trim_indices(indices):
    triangles = np.asarray(indices, dtype=np.uint32).reshape(-1, 3)
    valid = (
        (triangles[:, 0] != triangles[:, 1])
        & (triangles[:, 1] != triangles[:, 2])
        & (triangles[:, 0] != triangles[:, 2])
    )
    keep = np.flatnonzero(valid).tolist()
    # Keep the existing source's enabled/empty-index status and target bounds.
    if not keep and len(triangles):
        keep = [0]
    return triangles[keep].ravel().tolist(), keep


def configure():
    exact.BASE, exact.SOURCE, exact.MODEL, exact.PREFIX = (
        BASE,
        SOURCE,
        MODEL,
        "vts_mask_exact",
    )


def rig():
    configure()
    source = ROOT / "artifacts/vts_exact/rig"
    ir = json.loads((source / "POPY.rig.json").read_text())
    sources = {name for m in ir["meshes"] for name in m["masked_by"]}
    chain_list = chains(ir["meshes"])
    alias = {name: group[0] for group in chain_list for name in group}
    member_chain = {name: group for group in chain_list for name in group}
    removed = {
        m["id"]
        for m in ir["meshes"]
        if invisible(m) and m["id"] not in sources and not m["masked_by"]
    }
    kept, original_triangles = {}, 0
    for m in ir["meshes"]:
        original_triangles += len(m["indices"]) // 3
        if m["id"] in sources and invisible(m) and not m["masked_by"]:
            indices, which = trim_indices(m["indices"])
            if len(indices) != len(m["indices"]):
                kept[m["id"]] = which
                m["indices"] = indices
        expanded = [alias.get(s, s) for s in m["masked_by"]]
        replaced = [s for i, s in enumerate(expanded) if i == 0 or s != expanded[i - 1]]
        # Enforce a reversible source-list substitution. No alpha source may
        # be inserted, deleted, duplicated, or moved across a foreign source.
        expansion = {group[0]: group for group in chain_list}
        if [n for s in replaced for n in expansion.get(s, [s])] != m["masked_by"]:
            raise ValueError("Mask sequence changed")
        m["masked_by"] = replaced
    lookup = {m["id"]: m for m in ir["meshes"]}
    grouped, seen = [], set()
    for m in ir["meshes"]:
        name = m["id"]
        if name in removed or name in seen:
            continue
        group = member_chain.get(name, [name])
        grouped.append([lookup[n] for n in group])
        seen.update(group)
    mapping = exact.emit_rig(ir, source, BASE / "rig", grouped)
    for row in mapping:
        name = row["source"]
        if name in kept:
            row["kept_triangles"] = kept[name]
        if name in member_chain:
            row["ignore_hidden_depth"] = True
    mapping.extend({"source": name, "removed": True} for name in sorted(removed))
    jsonwrite(BASE / "mapping.json", mapping)
    summary = {
        "removed_unobservable_meshes": len(removed),
        "mask_source_chains": chain_list,
        "batched_mask_sources": sum(len(c) - 1 for c in chain_list),
        "triangles_before": original_triangles,
        "triangles_after": sum(
            len(m["indices"]) // 3 for group in grouped for m in group
        ),
        "vertices_before": sum(len(m["positions"]) // 2 for m in lookup.values()),
        "vertices_after": sum(
            len(m["positions"]) // 2 for group in grouped for m in group
        ),
    }
    # The triangle difference also includes only the separately proven dead
    # drawables; report those counts separately, avoiding a false raster claim.
    removed_dead_triangles = sum(len(lookup[n]["indices"]) // 3 for n in removed)
    summary["dead_drawable_triangles_removed"] = removed_dead_triangles
    summary["degenerate_source_triangles_removed"] = (
        original_triangles - summary["triangles_after"] - removed_dead_triangles
    )
    jsonwrite(BASE / "build.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "mask_source_chains"}))


def fingerprints():
    return {
        "source": sha(SOURCE / "POPY.moc3"),
        "candidate": sha(MODEL / "POPY.moc3"),
        "mapping": sha(BASE / "mapping.json"),
        "geometry_cases": sha(ROOT / "artifacts/perf_opt/equivalence_cases.json"),
        "image_cases": sha(ROOT / "artifacts/fullbody/cases.json"),
        "timing_cases": sha(ROOT / "artifacts/perf_opt/timing_cases.json"),
        "source_assets": resource_sha(SOURCE),
        "candidate_assets": resource_sha(MODEL),
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
    if json.loads((BASE / "validated_inputs.json").read_text()) != fingerprints():
        raise ValueError("Run validation for these exact inputs before packaging")
    result = exact.report()
    result["build"] = json.loads((BASE / "build.json").read_text())
    result["mask_contexts"] = {
        label: len(
            {
                tuple(sorted(m["clips"]))
                for m in json.loads((BASE / label / "model_inventory.json").read_text())
                if m["clips"]
            }
        )
        for label in ["baseline", "candidate"]
    }
    if result["mask_contexts"] != {"baseline": 80, "candidate": 80}:
        raise ValueError("Mask context count changed")
    result["no_new_quality_loss"] = (
        "Only constant-zero unreferenced unclipped drawables removed; only repeated-index zero-area primitives removed; invisible mask source batches expand to exact original source order. Every retained vertex is bit-exact; resource bytes unchanged."
    )
    result["limits"] = (
        "Relative to POPY_VTS_Exact; previous lossy-fit errors remain. Not VTube Studio or official Cubism Core validation. Native DX12, paired timing, desktop not isolated."
    )
    result["validated_inputs"] = fingerprints()
    repeated = json.loads((BASE / "memory_repeats.json").read_text())
    if repeated["inputs"] != fingerprints():
        raise ValueError("Memory samples belong to other inputs")
    result["repeated_single_model_memory"] = {
        "method": repeated["sequence"],
        "rows": repeated["rows"],
        "summary": {
            label: {
                "samples": len(values),
                "median_peak_rss_mib": float(np.median(values)) / 2**20,
                "min_peak_rss_mib": min(values) / 2**20,
                "max_peak_rss_mib": max(values) / 2**20,
            }
            for label in ["baseline", "candidate"]
            for values in [
                [r["peak_rss_bytes"] for r in repeated["rows"] if r["label"] == label]
            ]
        },
        "caution": "Use these repeats for memory conclusions. Single image-process peaks fluctuate more and do not establish a stable 20% memory improvement.",
    }
    jsonwrite(BASE / "report.json", result)
    return result


def memory():
    import os

    from .cli import run

    before = fingerprints()
    if json.loads((BASE / "validated_inputs.json").read_text()) != before:
        raise ValueError("Validate these models first")
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
        stage = f"vts_mask_exact_memory_{i}_{label}"
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
            {
                "label": label,
                "stage": stage,
                "peak_rss_bytes": row["peak_process_tree_rss_bytes"],
            }
        )
    if before != fingerprints():
        raise ValueError("Memory benchmark inputs changed")
    jsonwrite(
        BASE / "memory_repeats.json",
        {
            "inputs": before,
            "sequence": "ABBA BAAB; independent single-model native DX12 processes, same 65 poses, no PNG/readback, includes loading/warmup",
            "rows": rows,
        },
    )


def package():
    report()
    destination = ROOT / "delivery/POPY_VTS_Exact2"
    shutil.copytree(MODEL, destination, dirs_exist_ok=True)
    shutil.copy2(BASE / "report.json", destination / "performance_report.json")
    shutil.copy2(BASE / "candidate/natural.png", destination / "preview.png")
    (destination / "README.txt").write_text(
        "POPY VTS 后续无损优化第二版\n\n"
        "相对 POPY_VTS_Exact 不新增画质损失，旧的有损拟合误差保留。\n"
        "移除永久不可见且无依赖的网格、永远零面积的索引三角形，\n"
        "合批每个遮罩列表中始终连续且同序的隐藏来源；覆盖、剔除和透明度保持。\n"
        "保留 80 组遮罩、牙齿修复、39 参数、58 动作、24 表情，贴图不改变。\n\n"
        "安装：整个文件夹放入 VTube Studio_Data/StreamingAssets/Live2DModels/，\n"
        "加载 POPY.model3.json，首次 Auto-Setup。仍设 256 masks，每组至少 512x512。\n"
        "内部 ArtMesh ID 有合并，逐 ArtMesh 自定义效果可能需要重设。\n"
        "实测记录在 performance_report.json；非 VTS 实机 FPS/内存，无新增 CMO3。\n",
        encoding="utf-8-sig",
    )
    archive = destination.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as z:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(destination.parent))
    print(f"Delivered {archive}: {archive.stat().st_size} bytes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=["all", "rig", "export", "validate", "memory", "report", "package"],
    )
    args = parser.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    configure()
    stages = (
        [rig, exact.export, validate, memory, package]
        if args.stage == "all"
        else [exact.export if args.stage == "export" else globals()[args.stage]]
    )
    for stage in stages:
        stage()


if __name__ == "__main__":
    main()
