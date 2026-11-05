"""Lossy sparse functional distillation of the existing VTS model.

The teacher's complete Glue program runs OFFLINE. The delivered MOC contains
small local tensor grids, additive single-parameter residuals and two rotations.
No upstream runtime or original asset is modified.
"""

import argparse
import itertools
import json
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from .compile_ir import ROOT, jsonwrite, project

BASE = ROOT / "artifacts/vts_fast"
TEACHER = ROOT / "delivery/POPY_VTS_TeethFixed"
MODEL = ROOT / "output/POPY_VTS_Performance"


def call(command, **kwargs):
    subprocess.run([str(x) for x in command], check=True, **kwargs)


def sample(cases, out):
    out.mkdir(parents=True, exist_ok=True)
    call(
        [
            ROOT / "rust/target/release/distill_samples.exe",
            TEACHER / "POPY.model3.json",
            cases,
            out,
        ]
    )


def case_key(values):
    return tuple(sorted((p, float(v)) for p, v in values.items()))


def keys_for(p):
    lo, hi, default = p["min"], p["max"], p["default"]
    # Fine elbow/raise curves retain an articulated arc rather than one chord.
    count = (
        5 if any(s in p["id"] for s in ["AngleX", "Raise", "Elbow", "Forearm"]) else 3
    )
    return sorted({float(v) for v in [*np.linspace(lo, hi, count), default, 0]})


def pivots():
    manifest = json.loads((ROOT / "artifacts/samples/manifest.json").read_text())
    bones = {
        b["name"]: np.array(b["matrix_local"])[:3, 3]
        for b in json.loads((ROOT / "artifacts/samples/skeleton.json").read_text())
    }
    return {
        group: project(
            np.array([bones[manifest["semantics"]["bones"][group]]]), manifest["camera"]
        )[0]
        * manifest["config"]["canvas"]
        for group in ["head", "body"]
    }


def local(points, values, head, pivot):
    q = points.astype(np.float64)
    for param, center in [("ParamBodyAngleZ", pivot["body"])] + (
        [("ParamAngleZ", pivot["head"])] if head else []
    ):
        angle = np.radians(values.get(param, 0))
        c, s = np.cos(angle), np.sin(angle)
        # Undo the screen-space parent rotation before fitting local residuals.
        q = (q - center) @ np.array([[c, -s], [s, c]]) + center
    return q - pivot["head" if head else "body"]


def probe():
    BASE.mkdir(parents=True, exist_ok=True)
    path = BASE / "probe_cases.json"
    jsonwrite(path, [{"name": "neutral", "parameters": {}}])
    sample(path, BASE / "probe")


def scan():
    metadata = json.loads((BASE / "probe/metadata.json").read_text())
    cases = [{"name": "neutral", "parameters": {}}]
    for p in metadata["parameters"]:
        for i, value in enumerate(keys_for(p)):
            cases.append({"name": p["id"] + f"_{i}", "parameters": {p["id"]: value}})
    path = BASE / "scan_cases.json"
    jsonwrite(path, cases)
    sample(path, BASE / "scan")


def load_samples(folder, cases):
    meta = json.loads((folder / "metadata.json").read_text())
    if meta.get("format_version", 1) == 1:
        # Upgrade the initial development capture without running the teacher
        # again: its Y axis and integer depth byte encoding were inverted/raw.
        legacy = np.memmap(
            folder / "positions.f32",
            "<f4",
            mode="r+",
            shape=(len(cases), meta["vertices"], 2),
        )
        for i in range(len(cases)):
            legacy[i, :, 1] = meta["canvas"][1] - legacy[i, :, 1]
        legacy.flush()
        del legacy
        ch = np.memmap(
            folder / "channels.f32",
            "<f4",
            mode="r+",
            shape=(len(cases), len(meta["meshes"]), 2),
        )
        ch[:, :, 0] = ch[:, :, 0].view("<i4").astype("<f4")
        ch.flush()
        del ch
        meta["format_version"] = 2
        jsonwrite(folder / "metadata.json", meta)
    points = np.memmap(
        folder / "positions.f32",
        "<f4",
        mode="r",
        shape=(len(cases), meta["vertices"], 2),
    )
    channels = np.memmap(
        folder / "channels.f32",
        "<f4",
        mode="r",
        shape=(len(cases), len(meta["meshes"]), 2),
    )
    return meta, points, channels


def plan():
    cases = json.loads((BASE / "scan_cases.json").read_text())
    meta, points, _ = load_samples(BASE / "scan", cases)
    parameters = {p["id"]: p for p in meta["parameters"]}
    byparam = {
        p: [i for i, c in enumerate(cases) if p in c["parameters"]] for p in parameters
    }
    rows = []
    combined = list(cases)
    lookup = {case_key(c["parameters"]): i for i, c in enumerate(combined)}
    for m in meta["meshes"]:
        a, b = m["start"], m["start"] + m["vertices"]
        neutral = points[0, a:b]
        sensitivity = {
            p: float(np.linalg.norm(points[indices, a:b] - neutral, axis=2).max())
            for p, indices in byparam.items()
        }
        arm = {
            s: max(
                sensitivity[p]
                for p in parameters
                if any(
                    f in p
                    for f in [
                        f"Arm{s}",
                        f"Elbow{s}",
                        f"Wrist{s}",
                        f"Hand{s}",
                        f"Forearm{s}",
                    ]
                )
            )
            for s in ["L", "R"]
        }
        leg = {
            s: max(
                sensitivity[p]
                for p in parameters
                if any(f in p for f in [f"Leg{s}", f"Knee{s}", f"Ankle{s}"])
            )
            for s in ["L", "R"]
        }
        head = sensitivity["ParamAngleX"] > max(0.15, max(arm.values()) * 0.3)
        if head:
            names = ["ParamAngleX", "ParamAngleY"]
            expression = [
                p
                for p in parameters
                if any(
                    s in p
                    for s in ["EyeLOpen", "EyeROpen", "MouthOpen", "MouthForm", "Brow"]
                )
            ]
            best = max(expression, key=sensitivity.get)
            if sensitivity[best] > 0.4:
                names.append(best)
        elif max(arm.values()) > max(leg.values()) and max(arm.values()) > 0.5:
            side = max(arm, key=arm.get)
            third = max(
                [f"ParamArm{side}Depth", f"ParamForearm{side}Turn"], key=sensitivity.get
            )
            names = [f"ParamArm{side}Raise", f"ParamElbow{side}", third]
        elif max(leg.values()) > 0.5:
            side = max(leg, key=leg.get)
            names = [
                f"ParamLeg{side}Swing",
                f"ParamLeg{side}Spread",
                f"ParamKnee{side}",
            ]
        else:
            names = ["ParamBodyAngleX", "ParamBodyAngleY"]
        names = [p for p in names if sensitivity[p] > 0.12]
        axes = [{"parameter": p, "keys": keys_for(parameters[p])} for p in names]
        indices = []
        coordinates = list(itertools.product(*(range(len(a["keys"])) for a in axes)))
        for coord in coordinates:
            values = {ax["parameter"]: ax["keys"][k] for ax, k in zip(axes, coord)}
            key = case_key(values)
            if key not in lookup:
                lookup[key] = len(combined)
                combined.append(
                    {"name": f"tensor_{len(combined)}", "parameters": values}
                )
            indices.append(lookup[key])
        rows.append(
            {
                "id": m["id"],
                "head": head,
                "axes": axes,
                "samples": indices,
                "coordinates": coordinates,
                "residuals": [
                    p for p in parameters if p not in names and sensitivity[p] > 0.12
                ],
            }
        )
    jsonwrite(BASE / "plan.json", rows)
    path = BASE / "fit_cases.json"
    jsonwrite(path, combined)
    print(f"{len(rows)} meshes, {len(combined)} offline poses", flush=True)
    sample(path, BASE / "fit")


def rig():
    cases = json.loads((BASE / "fit_cases.json").read_text())
    meta, points, channels = load_samples(BASE / "fit", cases)
    rows = json.loads((BASE / "plan.json").read_text())
    params = {p["id"]: p for p in meta["parameters"]}
    indices = {case_key(c["parameters"]): i for i, c in enumerate(cases)}
    pivot = pivots()
    folder = BASE / "rig"
    folder.mkdir(parents=True, exist_ok=True)
    corners = [0, 0, 768, 0, 0, 1024, 768, 1024]
    ir = {
        "schema": "popy.rig/1",
        "name": "POPY",
        "canvas": meta["canvas"],
        "baked_runtime": True,
        "geometry_encoding": "parent_absolute",
        "parameters": list(params.values()),
        "keyform_data": "keyforms.f32",
        "meshes": [],
        "glues": [],
        "textures": [],
        "physics": [],
        "root_warp": {"axes": [], "cells": [{"coordinate": [], "points": corners}]},
        "head_warp": {"axes": [], "cells": [{"coordinate": [], "points": corners}]},
        "body_rotation": {"pivot": pivot["body"].tolist()},
        "head_rotation": {
            "axes": [{"parameter": "ParamAngleZ", "keys": [-30, 0, 30]}],
            "cells": [
                {
                    "coordinate": [i],
                    "origin": (pivot["head"] - pivot["body"]).tolist(),
                    "angle": z,
                }
                for i, z in enumerate([-30, 0, 30])
            ],
        },
        "diagnostics": {
            "method": "Sparse local tensor + additive functional ANOVA residuals; offline evaluated teacher, zero runtime Glue",
            "lossy": True,
            "limitations": "Higher-order interactions outside each local tensor are approximated; mask topology transitions may differ.",
        },
    }
    cursor = 0
    form_count = 0
    with (folder / "keyforms.f32").open("wb") as stream:

        def emit(array):
            nonlocal cursor
            data = np.asarray(array, dtype="<f4").ravel()
            offset = cursor
            stream.write(data.tobytes())
            cursor += data.size
            return offset

        for mi, (m, row) in enumerate(zip(meta["meshes"], rows)):
            a, b = m["start"], m["start"] + m["vertices"]

            def value(i, a=a, b=b, row=row, mi=mi):
                q = local(points[i, a:b], cases[i]["parameters"], row["head"], pivot)
                return np.c_[q, np.broadcast_to(channels[i, mi], (b - a, 2))]

            neutral = value(0)
            correction = np.zeros_like(neutral)
            residual = []
            for p in row["residuals"]:
                keys = keys_for(params[p])
                data = [value(indices[case_key({p: k})]) for k in keys]
                zero = data[keys.index(0.0)]
                delta = [v - zero for v in data]
                correction += delta[keys.index(float(params[p]["default"]))]
                # Do not retain subpixel no-op roll residuals or scalar noise.
                if (
                    max(float(np.abs(v[:, :2]).max()) for v in delta) > 0.06
                    or max(float(np.abs(v[:, 2:]).max()) for v in delta) > 0.001
                ):
                    residual.append((p, keys, delta))
                else:
                    correction -= delta[keys.index(float(params[p]["default"]))]
            reference = neutral - correction
            cells = []
            for ci, coord in zip(row["samples"], row["coordinates"]):
                state = value(ci) - correction
                cells.append(
                    {
                        "coordinate": coord,
                        "offset": emit(state[:, :2]),
                        "order": float(state[0, 2]),
                        "opacity": float(state[0, 3]),
                    }
                )
            blends = []
            for p, keys, delta in residual:
                offsets, orders, opacity = [], [], []
                for key, d in zip(keys, delta):
                    offsets.append(
                        None if key == 0 else emit(reference[:, :2] + d[:, :2])
                    )
                    orders.append(float(reference[0, 2] + d[0, 2]))
                    opacity.append(float(reference[0, 3] + d[0, 3]))
                blends.append(
                    {
                        "parameter": p,
                        "keys": keys,
                        "neutral": keys.index(0.0),
                        "offsets": offsets,
                        "orders": orders,
                        "opacities": opacity,
                    }
                )
            ir["meshes"].append(
                {
                    "id": m["id"],
                    "positions": [0.0] * (2 * m["vertices"]),
                    "uvs": m["uvs"],
                    "indices": m["indices"],
                    "texture": m["texture"],
                    "axes": row["axes"],
                    "cells": cells,
                    "blend_shapes": blends,
                    "opacity": float(reference[0, 3]),
                    "deformer": "head" if row["head"] else "body",
                    "masked_by": m["clips"],
                    "invert_mask": m["invert_mask"],
                    "culling": m["culling"],
                    "source": {
                        "teacher": "POPY_VTS_TeethFixed",
                        "artmesh": m["id"],
                        "original_uv_topology": True,
                    },
                }
            )
            form_count += len(cells) + sum(
                sum(o is not None for o in b["offsets"]) for b in blends
            )
    ir["parts"] = [{"id": "P_Distilled", "mesh_ids": [m["id"] for m in ir["meshes"]]}]
    manifest = json.loads((TEACHER / "POPY.model3.json").read_text())
    for texture in manifest["FileReferences"]["Textures"]:
        name = Path(texture).name
        image = Image.open(TEACHER / texture)
        # Keep eye/face detail; halve the large body/clothes/hair atlas dimensions.
        if not any(s in name for s in ["Eye", "Face"]):
            image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
        image.save(folder / name)
        ir["textures"].append(name)
    welding = weld_mask_corners(ir, folder / "keyforms.f32")
    jsonwrite(folder / "POPY.rig.json", ir)
    jsonwrite(
        BASE / "build_summary.json",
        {
            "meshes": len(rows),
            "vertices": welding["vertices_after"],
            "authored_forms_including_blends": form_count,
            "glues": 0,
            "deformers": 2,
            "mask_contexts": len(
                {tuple(sorted(m["clips"])) for m in meta["meshes"] if m["clips"]}
            ),
            "offline_poses": len(cases),
            "lossy": True,
            "mask_corner_welding": welding,
        },
    )
    print(
        f"Rig: {form_count} forms, {cursor * 4} position bytes, zero Glue", flush=True
    )


def weld_mask_corners(ir, path):
    """Weld duplicate mask corners only when their complete fitted tracks agree.

    This compares ALL authored main/residual forms, not just neutral positions.
    The fully opaque texture requirement makes UV seams irrelevant to masks.
    Visible surfaces retain their original UVs and topology.
    """
    alpha_opaque = []
    for texture in ir["textures"]:
        with Image.open(path.parent / texture) as image:
            alpha_opaque.append(
                image.convert("RGBA").getchannel("A").getextrema() == (255, 255)
            )
    old = np.memmap(path, "<f4", mode="r")
    temporary = path.with_suffix(".welded")
    cursor = 0
    before = after = welded = 0
    with temporary.open("wb") as stream:
        for mesh in ir["meshes"]:
            n = len(mesh["positions"]) // 2
            cells = mesh["cells"]
            entries = [(c, "offset", None) for c in cells]
            for blend in mesh["blend_shapes"]:
                entries += [
                    (blend, "offsets", k)
                    for k, offset in enumerate(blend["offsets"])
                    if offset is not None
                ]
            offsets = [o[key] if k is None else o[key][k] for o, key, k in entries]
            data = np.stack(
                [old[offset : offset + 2 * n].reshape(n, 2) for offset in offsets]
            )
            selected = np.arange(n)
            before += n
            if (
                mesh["id"].startswith("Mask")
                and alpha_opaque[mesh["texture"]]
                and n > 3
            ):
                features = np.rint(data.transpose(1, 0, 2).reshape(n, -1) * 64).astype(
                    "i8"
                )
                _, selected, inverse = np.unique(
                    features, axis=0, return_index=True, return_inverse=True
                )
                mesh["indices"] = inverse[np.array(mesh["indices"])].tolist()
                mesh["uvs"] = (
                    np.array(mesh["uvs"]).reshape(n, 2)[selected].ravel().tolist()
                )
                mesh["positions"] = [0.0] * (2 * len(selected))
                mesh["source"]["mask_corner_weld_grid_px"] = 1 / 64
                welded += int(len(selected) < n)
            after += len(selected)
            for (obj, key, k), form in zip(entries, data):
                if k is None:
                    obj[key] = cursor
                else:
                    obj[key][k] = cursor
                payload = form[selected].astype("<f4")
                stream.write(payload.tobytes())
                cursor += payload.size
    del old
    temporary.replace(path)
    return {
        "vertices_before": before,
        "vertices_after": after,
        "welded_mask_meshes": welded,
        "comparison": "Every fitted main and residual form; 1/64 canvas-pixel coordinate bins; opaque alpha only",
    }


def export():
    env = os.environ.copy()
    env["JAVA_HOME"] = r"C:\lib\java"
    env["JAVA_OPTS"] = "-Xmx12g -XX:+UseG1GC -XX:+ExitOnOutOfMemoryError"
    call(
        [
            ROOT / "vendor/psd2live/gradlew.bat",
            "-p",
            ROOT / "exporter",
            "installDist",
            "--console=plain",
        ],
        env=env,
    )
    call(
        [
            ROOT
            / "exporter/build/install/popy-cubism-exporter/bin/popy-cubism-exporter.bat",
            BASE / "rig/POPY.rig.json",
            MODEL,
            "--no-cmo",
            "--deduplicate-positions",
        ],
        env=env,
    )
    # Keep the existing user-facing parameter IDs, motion and expression entries.
    for name in ["POPY.model3.json", "emotion_catalog.json"]:
        shutil.copy2(TEACHER / name, MODEL / name)
    for name in ["motions", "expressions"]:
        shutil.copytree(TEACHER / name, MODEL / name, dirs_exist_ok=True)


def review():
    """A small explicit quality/performance review, not full compatibility QA."""
    from .cli import run

    cases = json.loads((ROOT / "artifacts/fullbody/packing_cases.json").read_text())
    for name, values in [
        (
            "turn_left_blink",
            {
                "ParamAngleX": -45,
                "ParamAngleY": 15,
                "ParamEyeLOpen": 0,
                "ParamEyeROpen": 0,
            },
        ),
        (
            "turn_right_mouth",
            {
                "ParamAngleX": 45,
                "ParamAngleY": -15,
                "ParamMouthOpenY": 1,
                "ParamMouthForm": 1,
            },
        ),
        ("roll_mouth", {"ParamAngleZ": -30, "ParamAngleY": 15, "ParamMouthOpenY": 1}),
        ("wink", {"ParamEyeLOpen": 0}),
        ("down_mouth", {"ParamAngleY": 30, "ParamMouthOpenY": 1}),
        (
            "body_head",
            {"ParamBodyAngleX": 15, "ParamBodyAngleY": -10, "ParamAngleX": -30},
        ),
        (
            "legs",
            {
                "ParamLegLSwing": 20,
                "ParamLegLSpread": 15,
                "ParamKneeL": 30,
                "ParamLegRSwing": -15,
                "ParamKneeR": 15,
            },
        ),
    ]:
        cases.append({"name": name, "parameters": values})
    path = BASE / "review_cases.json"
    jsonwrite(path, cases)
    models = [
        {"label": label, "manifest": str(folder / "POPY.model3.json")}
        for label, folder in [("baseline", TEACHER), ("candidate", MODEL)]
    ]
    jsonwrite(BASE / "models.json", models)
    env = os.environ.copy()
    env["POPY_WGPU_BACKEND"] = "dx12"
    for model in models:
        label = model["label"]
        run(
            f"vts_fast_{label}_images",
            [
                ROOT / "rust/target/release/popy-validate.exe",
                model["manifest"],
                path,
                BASE / label,
                "--images-only",
            ],
            env,
        )
        run(
            f"vts_fast_{label}_stats",
            [
                ROOT / "rust/target/release/model_stats.exe",
                model["manifest"],
                BASE / f"{label}_stats.json",
            ],
        )
    run(
        "vts_fast_paired_cpu",
        [
            ROOT / "rust/target/release/drive_benchmark.exe",
            BASE / "models.json",
            path,
            BASE / "paired_cpu.json",
        ],
    )
    run(
        "vts_fast_paired_gpu",
        [
            ROOT / "rust/target/release/render_benchmark.exe",
            models[0]["manifest"],
            models[1]["manifest"],
            path,
            BASE / "paired_gpu.json",
        ],
    )
    report()


def report():
    from .compare import composite, ssim

    stats = {
        label: json.loads((BASE / f"{label}_stats.json").read_text())
        for label in ["baseline", "candidate"]
    }
    timing = {}
    for filename, key in [
        ("paired_cpu.json", "driver_ms"),
        ("paired_gpu.json", "pose_to_gpu_complete_ms"),
    ]:
        data = json.loads((BASE / filename).read_text())
        timing[filename] = {
            label: {
                "mean_ms": float(np.mean(values)),
                "median_ms": float(np.median(values)),
                "p95_ms": float(np.percentile(values, 95)),
                "samples": len(values),
            }
            for label in stats
            for values in [[r[key] for r in data["samples"] if r["label"] == label]]
        }
    perf = json.loads((ROOT / "artifacts/performance.json").read_text())
    memory = {
        label: next(
            r["peak_process_tree_rss_bytes"]
            for r in reversed(perf)
            if r["stage"] == f"vts_fast_{label}_images"
        )
        for label in stats
    }
    quality = []
    for case in json.loads((BASE / "review_cases.json").read_text()):
        name = case["name"]
        a, b = [
            np.array(
                Image.open(BASE / label / f"{name}.png").convert("RGBA"), dtype=float
            )
            / 255
            for label in stats
        ]
        aa, bb = a[:, :, 3] > 0.5, b[:, :, 3] > 0.5
        union = aa | bb
        ac, bc = composite(a), composite(b)
        quality.append(
            {
                "name": name,
                "foreground_mae": float(np.abs(ac - bc)[union].mean()),
                "silhouette_iou": float((aa & bb).sum() / max(1, union.sum())),
                "foreground_ssim": ssim(ac, bc, union),
            }
        )
    output = {
        "statistics": stats,
        "timing": timing,
        "peak_render_process_rss_bytes": memory,
        "quality_against_teacher": quality,
        "review_poses": len(quality),
        "vts_app_tested": False,
        "renderer": "Unmodified Ayagami 0d1d7aa, native DX12",
        "limits": "12 finite poses, GPU benchmark 4 rounds of 7 timed poses after 5 warmups; both models resident, desktop not isolated. Not VTS frame times or VTS memory measurements. Higher-order interactions are approximate.",
    }
    jsonwrite(BASE / "performance_report.json", output)
    old, new = stats["baseline"], stats["candidate"]
    cpu, gpu = timing["paired_cpu.json"], timing["paired_gpu.json"]
    rows = [
        ("MOC bytes", old["moc_bytes"], new["moc_bytes"]),
        ("Glue", old["glues"], new["glues"]),
        ("Main ArtMesh keyforms", old["artmesh_keyforms"], new["artmesh_keyforms"]),
        ("Vertices including masks", old["vertices"], new["vertices"]),
        (
            "CPU driver median ms",
            cpu["baseline"]["median_ms"],
            cpu["candidate"]["median_ms"],
        ),
        (
            "Pose-to-GPU-complete median ms",
            gpu["baseline"]["median_ms"],
            gpu["candidate"]["median_ms"],
        ),
        (
            "Render process peak RSS MiB",
            memory["baseline"] / 2**20,
            memory["candidate"] / 2**20,
        ),
    ]
    text = "# POPY VTS performance edition\n\nLossy local tensor / additive residual fit; original 80 mask contexts retained.\n\n| Metric | Tooth-fixed teacher | Performance edition |\n|---|---:|---:|\n"
    text += "\n".join(f"| {name} | {a:.3f} | {b:.3f} |" for name, a, b in rows)
    text += "\n\n39 parameters, 58 motions, 24 expressions. Face/eye textures 2048; other atlases 1024.\n\n"
    text += (
        output["limits"]
        + "\n\nThese are native Ayagami measurements; VTube Studio application loading, memory and FPS are not verified.\n\n"
    )
    text += "Quality compares this edition against the existing Live2D teacher, not new Blender ground truth. The original teacher's remaining defects may be inherited.\n\n"
    text += "| Pose | Foreground MAE | Silhouette IoU | SSIM |\n|---|---:|---:|---:|\n"
    text += "\n".join(
        f"| {r['name']} | {r['foreground_mae']:.6f} | {r['silhouette_iou']:.6f} | {r['foreground_ssim']:.6f} |"
        for r in quality
    )
    (BASE / "performance_report.md").write_text(text, encoding="utf8")
    print(
        json.dumps(
            {
                "timing": timing,
                "peak_rss_mib": {k: v / 2**20 for k, v in memory.items()},
                "worst_quality": max(quality, key=lambda r: r["foreground_mae"]),
            }
        ),
        flush=True,
    )


def package():
    destination = ROOT / "delivery/POPY_VTS_Performance"
    shutil.copytree(MODEL, destination, dirs_exist_ok=True)
    shutil.copy2(BASE / "build_summary.json", destination / "performance_build.json")
    for name in ["performance_report.json", "performance_report.md"]:
        if (BASE / name).exists():
            shutil.copy2(BASE / name, destination / name)
    if (BASE / "candidate/natural.png").exists():
        shutil.copy2(BASE / "candidate/natural.png", destination / "preview.png")
    (destination / "README.txt").write_text(
        "POPY VTS 性能版（有损拟合）\n\n"
        "将整个文件夹放入 VTube Studio_Data/StreamingAssets/Live2DModels/，\n"
        "加载 POPY.model3.json，首次使用 Auto-Setup。Live2D Mask Config 设为\n"
        "256 masks，每组至少 512x512；本模型保留 80 组遮罩和牙齿遮挡修复。\n\n"
        "全部 Glue 已离线求值，运行时改用局部稀疏关键形、独立参数残差和头/身体两层旋转。\n"
        "保留 39 参数、58 动作、24 表情，删除不再参与渲染或遮罩的隐藏运算载体。\n"
        "眼睛与脸部贴图保持原分辨率，身体/服装/头发贴图降到 1024。\n\n"
        "画质取舍：张量之外的多参数联动采用加法近似，极端转身/手腕/表情组合可能\n"
        "产生位置误差、遮挡偏差或接缝。原版保留，可按需切回。无新增 CMO3。\n"
        "未完成 VTube Studio 实机兼容性验证。性能数字以包内报告的实际测量范围为准。\n",
        encoding="utf-8-sig",
    )
    archive = destination.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as z:
        for file in sorted(destination.rglob("*")):
            if file.is_file():
                z.write(file, file.relative_to(destination.parent))
    print(f"Delivery: {archive} ({archive.stat().st_size} bytes)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "all",
            "probe",
            "scan",
            "plan",
            "rig",
            "export",
            "review",
            "report",
            "package",
        ],
    )
    args = parser.parse_args()
    stages = (
        [probe, scan, plan, rig, export, review, package]
        if args.stage == "all"
        else [globals()[args.stage]]
    )
    for stage in stages:
        start = time.perf_counter()
        stage()
        print(f"{stage.__name__}: {time.perf_counter() - start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
