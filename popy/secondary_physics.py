# SPDX-License-Identifier: MIT
"""Low-rank secondary motion, preserving existing mesh programs and mask copies."""

import argparse
import json
import os
import shutil
import subprocess
import zipfile
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from .compile_ir import ROOT, jsonwrite
from .representation_review import sha

BASE = ROOT / "artifacts/secondary_physics"
SOURCE = ROOT / "output/POPY_VTS_MaskBatch"
MODEL = ROOT / "output/POPY_VTS_Physics"
IDS = ["ParamHairSway", "ParamHairFlutter", "ParamSkirtSway", "ParamSkirtFlutter"]


def kind(name):
    if "M_Hair" in name:
        return 1
    if any(s in name for s in ["M_Skirts", "M_Necktie", "M_Bow"]):
        return 2
    return 0


def smooth(x):
    t = np.clip(x, 0, 1)
    return t * t * (3 - 2 * t)


def modes(features):
    """Rest-space bending modes: fixed roots, shared continuous seam field."""
    x, y, k = features.T
    result = np.zeros((4, len(x), 2), dtype=np.float32)
    hair = smooth((y - 130) / 140) * (k == 1)
    # Reduce movement near the eyes; long side locks retain full tip motion.
    hair *= 0.2 + 0.8 * np.maximum(
        smooth((np.abs(x - 384) - 25) / 40), smooth((y - 190) / 65)
    )
    result[0, :, 0] = 7 * hair
    result[0, :, 1] = -0.012 * (x - 384) * hair
    result[1, :, 0] = 1.8 * np.tanh((x - 384) / 55) * hair
    result[1, :, 1] = 2.5 * hair
    skirt = smooth((y - 350) / 235) ** 2 * (k == 2)
    result[2, :, 0] = 10 * skirt
    result[2, :, 1] = -4 * (x - 384) / 125 * skirt
    result[3, :, 0] = 4 * (x - 384) / 125 * skirt
    result[3, :, 1] = -2 * skirt
    return result


def reference(mesh, data, defaults):
    cell = next(
        c
        for c in mesh["cells"]
        if all(
            a["keys"][k] == defaults[a["parameter"]]
            for a, k in zip(mesh["axes"], c["coordinate"])
        )
    )
    n = len(mesh["positions"])
    return cell, data[cell["offset"] : cell["offset"] + n].reshape(-1, 2).copy()


def provenance():
    """Recover per-corner semantic identity through all exact batch/permutation maps.

    The old mask welding lost indices. Resolve those against the *same* offline
    teacher pose and original union leaves, rejecting ambiguous or distant matches.
    All later copies use exact saved vertex mappings, never screen segmentation.
    """
    folder = ROOT / "artifacts/vts_fast/rig"
    ir = json.loads((folder / "POPY.rig.json").read_text())
    data = np.memmap(folder / "keyforms.f32", "<f4", mode="r")
    defaults = {p["id"]: p["default"] for p in ir["parameters"]}
    meta = json.loads((ROOT / "artifacts/vts_fast/fit/metadata.json").read_text())
    teacher = np.memmap(
        ROOT / "artifacts/vts_fast/fit/positions.f32", "<f4", mode="r"
    ).reshape(-1, 2)[: meta["vertices"]]
    original = {m["id"]: m for m in meta["meshes"]}
    unions = {
        u["id"]: u["sources"]
        for u in json.loads(
            (ROOT / "artifacts/mask_batch_final/plan.json").read_text()
        )["unions"]
    }

    def leaves(name):
        return (
            [leaf for child in unions[name] for leaf in leaves(child)]
            if name in unions
            else [name]
        )

    features, worst = {}, 0.0
    for m in ir["meshes"]:
        _, ref = reference(m, data, defaults)
        points = ref.copy()
        for b in m["blend_shapes"]:
            off = b["offsets"][b["keys"].index(defaults[b["parameter"]])]
            if off is not None:
                points += data[off : off + ref.size].reshape(-1, 2) - ref
        points += np.array(ir["body_rotation"]["pivot"]) + (
            np.array(ir["head_rotation"]["cells"][1]["origin"])
            if m["deformer"] == "head"
            else 0
        )
        names = leaves(m["id"])
        classes = {kind(n) for n in names}
        labels = np.full(len(points), next(iter(classes)))
        if len(classes) > 1:
            distances = []
            for label in sorted(classes):
                refs = [
                    teacher[
                        original[n]["start"] : original[n]["start"]
                        + original[n]["vertices"]
                    ]
                    for n in names
                    if kind(n) == label
                ]
                distances.append(cKDTree(np.concatenate(refs)).query(points)[0])
            distances = np.array(distances)
            nearest = distances.min(axis=0)
            worst = max(worst, float(nearest.max()))
            if nearest.max() > 0.04:
                raise ValueError(f"Unresolved welded source {m['id']}: {nearest.max()}")
            labels = np.array(sorted(classes))[distances.argmin(axis=0)]
            # At coincident mixed-class boundaries use a fixed common root.
            ambiguous = (distances <= nearest + 1e-6).sum(axis=0) > 1
            if np.any(
                ambiguous
                & (
                    np.linalg.norm(modes(np.c_[points, labels]), axis=2).max(axis=0)
                    > 0.01
                )
            ):
                raise ValueError("Ambiguous moving source corner")
        features[m["id"]] = np.c_[points, labels]
    for path in ["vts_exact", "vts_mask_exact", "overlap_layout", "mask_proxy"]:
        mapping = json.loads((ROOT / f"artifacts/{path}/mapping.json").read_text())
        rows = (
            mapping
            if isinstance(mapping, list)
            else mapping["rows"] + mapping["mask_proxies"]
        )
        pieces = defaultdict(list)
        for row in rows:
            if row.get("removed"):
                continue
            values = features[row["source"]]
            if "vertex_order" in row:
                values = values[row["vertex_order"]]
            assert len(values) == row["vertex_count"]
            pieces[row["target"]].append((row["vertex_start"], values))
        features = {}
        for name, parts in pieces.items():
            offset = 0
            for start, values in sorted(parts, key=lambda r: r[0]):
                assert start == offset
                offset += len(values)
            features[name] = np.concatenate(
                [v for _, v in sorted(parts, key=lambda r: r[0])]
            )
    jsonwrite(
        BASE / "provenance.json",
        {
            "worst_mixed_source_match_px": worst,
            "method": "Original 3D surface IDs, teacher union leaves, exact batching/copy/permutation maps",
        },
    )
    return features


def physics():
    def target(id):
        return {"Target": "Parameter", "Id": id}

    def setting(id, inputs, outputs, radius, mobility, delay):
        return {
            "Id": id,
            "Input": [
                {"Source": target(p), "Weight": w, "Type": t, "Reflect": False}
                for p, w, t in inputs
            ],
            "Output": [
                {
                    "Destination": target(p),
                    "VertexIndex": i + 1,
                    "Scale": scale,
                    "Weight": 100,
                    "Type": "Angle",
                    "Reflect": False,
                }
                for i, (p, scale) in enumerate(outputs)
            ],
            "Vertices": [
                {
                    "Position": {"X": 0, "Y": y},
                    "Mobility": mobility,
                    "Delay": delay,
                    "Acceleration": 1.3,
                    "Radius": r,
                }
                for y, r in [(0, 0), (radius, radius), (radius * 1.7, radius * 0.7)]
            ],
            "Normalization": {
                "Position": {"Minimum": -8, "Default": 0, "Maximum": 8},
                "Angle": {"Minimum": -10, "Default": 0, "Maximum": 10},
            },
        }

    settings = [
        setting(
            "Hair",
            [
                ("ParamAngleX", 70, "X"),
                ("ParamBodyAngleX", 30, "X"),
                ("ParamAngleZ", 70, "Angle"),
                ("ParamBodyAngleZ", 30, "Angle"),
            ],
            [(IDS[0], 3), (IDS[1], 2)],
            8,
            0.88,
            0.8,
        ),
        setting(
            "Skirt",
            [
                ("ParamBodyAngleX", 50, "X"),
                ("ParamLegLSwing", 25, "X"),
                ("ParamLegRSwing", 25, "X"),
                ("ParamBodyAngleZ", 100, "Angle"),
            ],
            [(IDS[2], 3), (IDS[3], 1.8)],
            12,
            0.9,
            0.65,
        ),
    ]
    return {
        "Version": 3,
        "Meta": {
            "PhysicsSettingCount": 2,
            "TotalInputCount": 8,
            "TotalOutputCount": 4,
            "VertexCount": 6,
            "Fps": 60,
            "EffectiveForces": {"Gravity": {"X": 0, "Y": -1}, "Wind": {"X": 0, "Y": 0}},
            "PhysicsDictionary": [{"Id": s["Id"], "Name": s["Id"]} for s in settings],
        },
        "PhysicsSettings": settings,
    }


def build():
    BASE.mkdir(parents=True, exist_ok=True)
    folder = BASE / "rig"
    folder.mkdir(exist_ok=True)
    source = ROOT / "artifacts/mask_proxy/rig"
    ir = json.loads((source / "POPY.rig.json").read_text())
    features = provenance()
    data = np.memmap(source / "keyforms.f32", "<f4", mode="r")
    defaults = {p["id"]: p["default"] for p in ir["parameters"]}
    shutil.copy2(source / "keyforms.f32", folder / "keyforms.f32")
    cursor = len(data)
    counts = dict.fromkeys(IDS, 0)
    with (folder / "keyforms.f32").open("ab") as stream:
        for m in ir["meshes"]:
            assert len(features[m["id"]]) * 2 == len(m["positions"])
            cell, ref = reference(m, data, defaults)
            for id, delta in zip(IDS, modes(features[m["id"]])):
                if not np.any(delta):
                    continue
                offsets = []
                for sign in [-1, 0, 1]:
                    offsets.append(cursor if sign else None)
                    if sign:
                        a = (ref + sign * delta).astype("<f4")
                        stream.write(a.tobytes())
                        cursor += a.size
                m["blend_shapes"].append(
                    {
                        "parameter": id,
                        "keys": [-1, 0, 1],
                        "neutral": 1,
                        "offsets": offsets,
                        "orders": [cell["order"]] * 3,
                        "opacities": [cell["opacity"]] * 3,
                    }
                )
                counts[id] += 1
    ir["parameters"] += [
        {"id": id, "min": -1, "max": 1, "default": 0, "kind": "blend"} for id in IDS
    ]
    ir["physics"] = physics()
    for t in ir["textures"]:
        shutil.copy2(source / t, folder / t)
    jsonwrite(folder / "POPY.rig.json", ir)
    jsonwrite(
        BASE / "build.json",
        {
            "bindings": counts,
            "additional_authored_forms": sum(counts.values()) * 2,
            "added_position_bytes": (cursor - len(data)) * 4,
            "glues": len(ir["glues"]),
            "mask_contexts": len(
                {tuple(m["masked_by"]) for m in ir["meshes"] if m["masked_by"]}
            ),
            "source_moc_sha256": sha(SOURCE / "POPY.moc3"),
        },
    )


def export():
    from . import vts_exact as exact
    from .cache_layout import pack_positions

    exact.BASE, exact.SOURCE, exact.MODEL = BASE, SOURCE, MODEL
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
    native = json.loads(layout.read_text())
    by_id = {m["id"]: m for m in native["meshes"]}
    pairs = [
        p
        for p in json.loads((ROOT / "artifacts/overlap_layout/pairs.json").read_text())
        if p["a"] in by_id
        and p["b"] in by_id
        and len(by_id[p["a"]]["forms"]) == len(by_id[p["b"]]["forms"])
    ]
    temp = MODEL / "POPY.overlap.moc3"
    pack_positions(MODEL / "POPY.moc3", temp, native, pairs, BASE / "overlap.json")
    temp.replace(MODEL / "POPY.moc3")
    jsonwrite(MODEL / "POPY.physics3.json", physics())
    manifest = json.loads((MODEL / "POPY.model3.json").read_text())
    manifest["FileReferences"]["Physics"] = "POPY.physics3.json"
    jsonwrite(MODEL / "POPY.model3.json", manifest)
    # Keep generated display parameters: copying old resources retains an old CDI.
    cdi = json.loads((MODEL / "POPY.cdi3.json").read_text())
    cdi["Parameters"] = [p for p in cdi["Parameters"] if p["Id"] not in IDS] + [
        {"Id": id, "GroupId": "SecondaryPhysics", "Name": name}
        for id, name in zip(
            IDS, ["头发惯性摆动", "发梢回弹", "裙摆惯性摆动", "裙摆回弹"]
        )
    ]
    cdi["ParameterGroups"].append(
        {"Id": "SecondaryPhysics", "GroupId": "", "Name": "头发与裙摆物理"}
    )
    jsonwrite(MODEL / "POPY.cdi3.json", cdi)


def validate():
    env = os.environ.copy()
    env["POPY_WGPU_BACKEND"] = "dx12"
    from .cli import run

    run(
        "secondary_physics_trace",
        [
            ROOT / "rust/target/release/physics_trace.exe",
            MODEL / "POPY.model3.json",
            BASE / "trace.json",
            BASE / "trace_cases.json",
        ],
    )
    cases = json.loads((ROOT / "artifacts/fullbody/cases.json").read_text())
    # Include all existing supplied screenshot angles and expressions in the neutral gate.
    jsonwrite(BASE / "neutral_cases.json", cases)
    for label, folder in [("before", SOURCE), ("after", MODEL)]:
        run(
            f"physics_neutral_{label}",
            [
                ROOT / "rust/target/release/popy-validate.exe",
                folder / "POPY.model3.json",
                BASE / "neutral_cases.json",
                BASE / label,
                "--images-only",
            ],
            env,
        )
    changed = 0
    for c in cases:
        a, b = [
            np.array(Image.open(BASE / label / f"{c['name']}.png").convert("RGBA"))
            for label in ["before", "after"]
        ]
        changed += int(np.any(a != b, axis=2).sum())
    if changed:
        raise ValueError(f"Neutral regression: {changed} pixels")
    run(
        "physics_temporal_render",
        [
            ROOT / "rust/target/release/popy-validate.exe",
            MODEL / "POPY.model3.json",
            BASE / "trace_cases.json",
            BASE / "animation",
            "--images-only",
        ],
        env,
    )
    check_modes()
    for label, folder in [("before", SOURCE), ("after", MODEL)]:
        run(
            f"physics_stats_{label}",
            [
                ROOT / "rust/target/release/model_stats.exe",
                folder / "POPY.model3.json",
                BASE / f"{label}_stats.json",
            ],
        )
    run(
        "physics_frame_cost",
        [
            ROOT / "rust/target/release/render_benchmark.exe",
            SOURCE / "POPY.model3.json",
            MODEL / "POPY.model3.json",
            ROOT / "artifacts/perf_opt/timing_cases.json",
            BASE / "timing.json",
        ],
        env,
    )
    jsonwrite(
        BASE / "validation.json",
        {
            "candidate_sha256": sha(MODEL / "POPY.moc3"),
            "physics_sha256": sha(MODEL / "POPY.physics3.json"),
            "neutral_poses": len(cases),
            "different_pixels": changed,
            "limits": "Native Ayagami; VTS/Core not verified. Procedural secondary modes, not 3D cloth collision simulation.",
        },
    )
    frames = [
        Image.open(BASE / "animation" / (c["name"] + ".png")).convert("RGBA")
        for c in json.loads((BASE / "trace_cases.json").read_text())
    ]
    # Palette PNG frames -> browser preview, using a neutral opaque background.
    previews = []
    for frame in frames:
        bg = Image.new("RGBA", frame.size, (35, 37, 42, 255))
        bg.alpha_composite(frame)
        previews.append(bg.convert("RGB").resize((384, 512)))
    previews[0].save(
        BASE / "physics_preview.gif",
        save_all=True,
        append_images=previews[1:],
        duration=100,
        loop=0,
    )


def check_modes():
    """Verify each delivered control actually moves its intended visible surfaces."""
    from .cli import run

    env = os.environ.copy()
    env["POPY_WGPU_BACKEND"] = "dx12"
    cases = [{"name": "rest", "parameters": {}}] + [
        {"name": f"{id}_{label}", "parameters": {id: value}}
        for id in IDS
        for label, value in [("negative", -1), ("positive", 1)]
    ]
    jsonwrite(BASE / "mode_cases.json", cases)
    run(
        "physics_mode_geometry",
        [
            ROOT / "rust/target/release/popy-validate.exe",
            MODEL / "POPY.model3.json",
            BASE / "mode_cases.json",
            BASE / "modes",
        ],
        env,
    )
    rest = {
        m["id"]: m for m in json.loads((BASE / "modes/rest.geometry.json").read_text())
    }
    rows = []
    for c in cases[1:]:
        current = json.loads(
            (BASE / "modes" / (c["name"] + ".geometry.json")).read_text()
        )
        moved, worst = 0, 0.0
        for m in current:
            old = rest[m["id"]]
            assert m["depth"] == old["depth"] and m["opacity"] == old["opacity"]
            delta = (
                np.linalg.norm(np.array(m["vertices"]) - old["vertices"], axis=1) * 384
            )
            if m["visible"] and m["opacity"] > 0:
                moved += int((delta > 0.1).sum())
                worst = max(worst, float(delta.max()))
        assert moved > 10 and worst > 1, "Disconnected mesh physics control"
        rows.append(
            {
                "case": c["name"],
                "visible_vertices_moved_over_0_1px": moved,
                "max_visible_displacement_px": worst,
            }
        )
    jsonwrite(BASE / "modes.json", rows)


def package():
    check = json.loads((BASE / "validation.json").read_text())
    assert check["candidate_sha256"] == sha(MODEL / "POPY.moc3")
    assert check["physics_sha256"] == sha(MODEL / "POPY.physics3.json")
    manifest = json.loads((MODEL / "POPY.model3.json").read_text())
    for texture in manifest["FileReferences"]["Textures"]:
        assert sha(MODEL / texture) == sha(SOURCE / texture)
    for directory in ["motions", "expressions"]:
        for path in (SOURCE / directory).glob("*.json"):
            assert sha(path) == sha(MODEL / directory / path.name)
    target = ROOT / "delivery" / MODEL.name
    shutil.copytree(MODEL, target, dirs_exist_ok=True)
    shutil.copy2(BASE / "physics_preview.gif", target / "physics_preview.gif")
    (target / "README.txt").write_text(
        "POPY 头发与裙摆物理版\n加载 POPY.model3.json，在 VTube Studio 中开启物理。\n头部转向/倾斜驱动发梢惯性回弹；身体转动/腿部动作驱动裙摆。\n新增四个 ParamHair*/ParamSkirt* 参数由物理输出控制，请勿重复绑定追踪输入。\n保留现有牙齿修复、贴图、动作、表情；零 Glue、80 遮罩组。\nVTS 继续设置 256 masks，每组至少 512x512。\n摆动是低秩附加形变，不是三维布料碰撞；极端姿势仍可能穿插。\n未在 VTS/Cubism Editor 实机验收；无新 CMO3。\n",
        encoding="utf-8-sig",
    )
    report = {
        "validation": check,
        "build": json.loads((BASE / "build.json").read_text()),
        "trace": json.loads((BASE / "trace.json").read_text()),
        "mode_geometry": json.loads((BASE / "modes.json").read_text()),
        "timing": json.loads((BASE / "timing.json").read_text()),
    }
    jsonwrite(target / "physics_report.json", report)
    jsonwrite(ROOT / "benchmarks/popy/fullbody/secondary_physics.json", report)
    with zipfile.ZipFile(
        target.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED, compresslevel=5
    ) as z:
        for p in sorted(target.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(target.parent))
    with zipfile.ZipFile(target.with_suffix(".zip")) as z:
        assert z.testzip() is None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["all", "build", "export", "validate", "package"]
    )
    a = parser.parse_args()
    for f in (
        [build, export, validate, package] if a.stage == "all" else [globals()[a.stage]]
    ):
        f()


if __name__ == "__main__":
    main()
