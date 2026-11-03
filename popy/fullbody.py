"""Reproducible articulated POPY conversion and real timeline image regression."""

import argparse
import hashlib
import html
import json
import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

from .animation import author_assets, evaluate, segments
from .cli import run
from .compare import composite, ssim
from .compile_ir import ROOT, jsonwrite, project
from .face_repair import EXPORTER, RENDERER
from .kinematics import Skeleton, garment_order_bias

BASE = ROOT / "artifacts/fullbody"
RIG = BASE / "rig/POPY.rig.json"
MODEL = ROOT / "output/POPY_fullbody"
CONFIG = ROOT / "config/fullbody.json"


def reference_audit(reference):
    reference = Path(reference)
    model = next(reference.glob("*.model3.json"))
    cdi = next(reference.glob("*.cdi3.json"))
    info = json.loads(cdi.read_text(encoding="utf-8-sig"))
    kinds, curve_count = {}, 0
    for path in sorted((reference / "motions").glob("*.motion3.json")):
        motion = json.loads(path.read_text(encoding="utf-8-sig"))
        for curve in motion["Curves"]:
            curve_count += 1
            for kind, _ in segments(curve):
                kinds[str(kind)] = kinds.get(str(kind), 0) + 1
    result = {
        "reference": reference.name,
        "parameters": len(info["Parameters"]),
        "motions": len(list((reference / "motions").glob("*.motion3.json"))),
        "expressions": len(list((reference / "expressions").glob("*.exp3.json"))),
        "curve_count": curve_count,
        "segment_types": kinds,
        "limb_controls": [
            {"id": p["Id"], "name": p["Name"]}
            for p in info["Parameters"]
            if any(x in p["Id"].lower() for x in ["arm", "hand", "leg", "shoulder"])
        ],
        "sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [model, cdi]
        },
        "reuse": "Control/action/expression semantics only. No Anon MOC, textures, keyforms, motion curves or expression files are copied. Anon upper/forearm numeric domains are model-specific; POPY uses independently authored degrees and real skin weights. No independent Anon leg motion parameters were found.",
    }
    jsonwrite(BASE / "anon_audit.json", result)
    return result


def assets():
    ir = json.loads(RIG.read_text())
    jsonwrite(BASE / "parameters.json", ir["parameters"])
    motions, expressions = author_assets(MODEL, ir["parameters"])
    manifest = json.loads((MODEL / "POPY.model3.json").read_text())
    manifest["FileReferences"].update(Motions=motions, Expressions=expressions)
    (MODEL / "POPY.model3.json").write_text(json.dumps(manifest, indent=2))
    groups = [
        {"Id": k, "GroupId": "", "Name": v}
        for k, v in [
            ("Arms", "Arms and hands"),
            ("Legs", "Hips, knees and ankles"),
            ("Face", "Head and expression"),
            ("Body", "Body and breathing"),
        ]
    ]
    display = {
        "Version": 3,
        "ParameterGroups": groups,
        "Parameters": [],
        "Parts": [{"Id": p["id"], "Name": p["id"]} for p in ir["parts"]],
    }
    for p in ir["parameters"]:
        k = p["id"]
        group = (
            "Arms"
            if any(s in k for s in ["Arm", "Elbow", "Wrist", "Hand"])
            else "Legs"
            if any(s in k for s in ["Leg", "Knee", "Ankle"])
            else "Body"
            if any(s in k for s in ["Body", "Breath"])
            else "Face"
        )
        display["Parameters"].append(
            {"Id": k, "Name": p.get("name", k), "GroupId": group}
        )
    (MODEL / "POPY.cdi3.json").write_text(json.dumps(display, indent=2))
    cases = [
        {"name": "natural", "parameters": {}},
        {
            "name": "source_T_pose",
            "parameters": {
                c["id"]: 0 for c in json.loads(CONFIG.read_text())["controls"]
            },
        },
    ]
    for c in json.loads(CONFIG.read_text())["controls"]:
        for label, v in [
            ("min", c["min"]),
            ("max", c["max"]),
            ("heldout", (c["keys"][0] + c["keys"][1]) / 2),
        ]:
            cases.append({"name": c["id"] + "_" + label, "parameters": {c["id"]: v}})
    for e in expressions:
        expression = json.loads((MODEL / e["File"]).read_text())
        for yaw in [0, 30]:
            values = evaluate(ir["parameters"], expression=expression)
            values["ParamAngleX"] = yaw
            cases.append(
                {
                    "name": "expression_" + e["Name"] + "_" + str(yaw),
                    "parameters": values,
                }
            )
    for name, refs in motions.items():
        motion = json.loads((MODEL / refs[0]["File"]).read_text())
        duration = motion["Meta"]["Duration"]
        for label, fraction in [
            ("quarter", 0.25),
            ("middle", 0.5),
            ("threequarter", 0.75),
        ]:
            cases.append(
                {
                    "name": "motion_" + name + "_" + label,
                    "parameters": evaluate(
                        ir["parameters"], motion, duration * fraction
                    ),
                    "motion": name,
                    "time": duration * fraction,
                }
            )
    for name in ["wave_L", "walk_in_place"]:
        motion = json.loads((MODEL / motions[name][0]["File"]).read_text())
        for i, t in enumerate(np.arange(0, motion["Meta"]["Duration"] + 0.05, 0.1)):
            cases.append(
                {
                    "name": "sequence_" + name + "_" + str(i).zfill(3),
                    "parameters": evaluate(ir["parameters"], motion, float(t)),
                    "motion": name,
                    "time": float(t),
                }
            )
    for side in ["L", "R"]:
        cases.append(
            {
                "name": "compound_" + side,
                "parameters": {
                    "ParamArm" + side + "Raise": -25,
                    "ParamArm" + side + "Depth": 15,
                    "ParamElbow" + side: 75,
                    "ParamWrist" + side: 17.5,
                    "ParamHand" + side + "Pose": 0.5,
                    "ParamBodyAngleX": 7.5,
                    "ParamBodyAngleY": -7.5,
                    "ParamAngleX": -22.5,
                    "ParamEyeLOpen": 0.3,
                    "ParamMouthOpenY": 0.6,
                    "ParamLeg" + side + "Swing": 12.5,
                    "ParamKnee" + side: 15,
                },
            }
        )
    neutral_limbs = {c["id"]: 0 for c in json.loads(CONFIG.read_text())["controls"]}
    preserve = [
        {"name": "head_preserve_front", "parameters": {}},
        {
            "name": "head_preserve_turn",
            "parameters": {
                "ParamAngleX": 45,
                "ParamAngleY": -15,
                "ParamEyeLOpen": 0,
                "ParamMouthOpenY": 0.6,
            },
        },
        {
            "name": "head_preserve_combo",
            "parameters": {
                "ParamAngleX": -30,
                "ParamAngleY": 15,
                "ParamAngleZ": -15,
                "ParamBodyAngleX": 15,
                "ParamBodyAngleY": -15,
                "ParamBrowLForm": 1,
            },
        },
    ]
    jsonwrite(BASE / "head_baseline_cases.json", preserve)
    cases.extend(
        {**c, "parameters": {**neutral_limbs, **c["parameters"]}} for c in preserve
    )
    for side in ["L", "R"]:
        cases.append(
            {
                "name": "heldout_leg_" + side,
                "parameters": {
                    "ParamLeg" + side + "Swing": 6.25,
                    "ParamKnee" + side: 7.5,
                    "ParamLeg" + side + "Spread": 7.5,
                    "ParamAnkle" + side: 10,
                },
            }
        )
        cases.append(
            {
                "name": "heldout_fingers_" + side,
                "parameters": {
                    "ParamHand" + side + "Pose": 0.25,
                    "ParamArm" + side + "Raise": -55,
                    "ParamWrist" + side: 17.5,
                },
            }
        )
    for side, sign in [("L", 1), ("R", -1)]:
        cases.append(
            {
                "name": "stress_arm_" + side,
                "parameters": {
                    "ParamArm" + side + "Raise": 50,
                    "ParamArm" + side + "Depth": -30 * sign,
                    "ParamElbow" + side: 120,
                    "ParamWrist" + side: 35,
                    "ParamHand" + side + "Pose": 1,
                    "ParamBodyAngleX": 15 * sign,
                    "ParamBodyAngleY": 15,
                    "ParamBodyAngleZ": -15 * sign,
                    "ParamAngleX": 30 * sign,
                    "ParamAngleY": -15,
                    "ParamAngleZ": 15 * sign,
                    "ParamEyeLOpen": 0,
                    "ParamMouthOpenY": 1,
                },
            }
        )
    cases.append(
        {
            "name": "stress_all",
            "parameters": {
                "ParamArmLRaise": 50,
                "ParamArmRRaise": 25,
                "ParamArmLDepth": -30,
                "ParamArmRDepth": 30,
                "ParamElbowL": 120,
                "ParamElbowR": 90,
                "ParamWristL": 35,
                "ParamWristR": -35,
                "ParamHandLPose": 1,
                "ParamHandRPose": -1,
                "ParamLegLSwing": 25,
                "ParamLegRSwing": -25,
                "ParamLegLSpread": 15,
                "ParamLegRSpread": 15,
                "ParamKneeL": 60,
                "ParamKneeR": 45,
                "ParamAnkleL": 20,
                "ParamAnkleR": -20,
                "ParamBodyAngleX": 15,
                "ParamBodyAngleY": -15,
                "ParamBodyAngleZ": 15,
                "ParamAngleX": -45,
                "ParamAngleY": 30,
                "ParamAngleZ": -30,
                "ParamEyeLOpen": 0.3,
                "ParamEyeROpen": 0.3,
                "ParamMouthOpenY": 1,
                "ParamMouthForm": 0.8,
            },
        }
    )
    jsonwrite(BASE / "cases.json", cases)
    probe_names = {
        "natural",
        "source_T_pose",
        "compound_L",
        "compound_R",
        "motion_wave_L_middle",
        "motion_thinking_middle",
        "motion_walk_in_place_quarter",
        "motion_leg_lift_L_middle",
        "ParamHandLPose_min",
        "ParamHandRPose_max",
        "ParamArmLDepth_max",
        "ParamArmRRaise_max",
        "ParamKneeL_max",
        "ParamAnkleR_max",
    }
    probe_names.update(c["name"] for c in preserve)
    probe_names.update(
        "heldout_" + part + "_" + side
        for part in ["leg", "fingers"]
        for side in ["L", "R"]
    )
    probe_names.update(
        ["motion_idle_middle", "stress_arm_L", "stress_arm_R", "stress_all"]
    )
    jsonwrite(BASE / "probes.json", [c for c in cases if c["name"] in probe_names])
    print(
        f"Authored {len(motions)} motions, {len(expressions)} expressions, {len(cases)} test poses",
        flush=True,
    )


def export():
    env = os.environ.copy()
    env.setdefault("JAVA_HOME", r"C:\lib\java")
    env.setdefault("JAVA_OPTS", "-Xmx20g -XX:+UseG1GC -XX:+ExitOnOutOfMemoryError")
    run(
        "fullbody_export",
        [EXPORTER, RIG, MODEL, "--no-cmo", "--deduplicate-positions"],
        env,
    )


def ordering():
    """Update constant garment-layer bias without resampling source geometry."""
    ir = json.loads(RIG.read_text())
    native = ir["diagnostics"]["fullbody"]["native_skinning"]
    previous = native.get("runtime_order_bias", {})
    desired = json.loads(CONFIG.read_text()).get("runtime_order_bias", {})
    changed = 0
    skeleton = Skeleton(ROOT / "artifacts/samples", json.loads(CONFIG.read_text()))
    config = json.loads(CONFIG.read_text())
    for mesh in ir["meshes"]:
        if not mesh.get("articulated"):
            continue
        obj = mesh["source"]["object"]
        bias = garment_order_bias(mesh, skeleton.object(obj)[0], config)
        difference = bias - mesh.get("runtime_order_bias_applied", previous.get(obj, 0))
        if difference:
            for c in mesh["cells"]:
                c["order"] += difference
            changed += 1
        mesh["runtime_order_bias_applied"] = bias
    native["runtime_order_bias"] = desired
    native["runtime_inner_caps"] = config.get("runtime_inner_caps", {})
    jsonwrite(RIG, ir)
    print("UPDATED_GARMENT_ORDER", changed, desired, flush=True)


def reference(blender):
    run(
        "fullbody_reference",
        [
            blender,
            "--background",
            ROOT / json.loads(CONFIG.read_text())["source"],
            "--python-exit-code",
            "1",
            "--python",
            ROOT / "blender/fullbody.py",
            "--",
            "--cases",
            BASE / "cases.json",
            "--out",
            BASE / "reference",
        ],
    )


def validate():
    run(
        "fullbody_runtime",
        [
            RENDERER,
            MODEL / "POPY.model3.json",
            BASE / "cases.json",
            BASE / "runtime",
            "--images-only",
        ],
    )
    run(
        "fullbody_geometry",
        [RENDERER, MODEL / "POPY.model3.json", BASE / "probes.json", BASE / "geometry"],
    )
    run(
        "fullbody_head_baseline",
        [
            RENDERER,
            ROOT / "output/POPY_face_fixed/POPY.model3.json",
            BASE / "head_baseline_cases.json",
            BASE / "head_baseline",
        ],
    )


def packing():
    """Compare optimized and original position packing using the same final IR."""
    env = os.environ.copy()
    env.setdefault("JAVA_HOME", r"C:\lib\java")
    env.setdefault("JAVA_OPTS", "-Xmx20g -XX:+UseG1GC -XX:+ExitOnOutOfMemoryError")
    unpacked = BASE / "unpacked_model"
    run("fullbody_unpacked_export", [EXPORTER, RIG, unpacked, "--no-cmo"], env)
    names = {
        "natural",
        "motion_wave_L_middle",
        "compound_L",
        "heldout_leg_L",
        "heldout_fingers_L",
    }
    cases = [
        c for c in json.loads((BASE / "cases.json").read_text()) if c["name"] in names
    ]
    casefile = BASE / "packing_cases.json"
    jsonwrite(casefile, cases)
    for name, model in [("unpacked", unpacked), ("packed", MODEL)]:
        run(
            "fullbody_packing_" + name,
            [
                RENDERER,
                model / "POPY.model3.json",
                casefile,
                BASE / ("packing_" + name),
            ],
        )
    rows = []
    for case in cases:
        name = case["name"]
        a = np.array(Image.open(BASE / "packing_unpacked" / (name + ".png")))
        b = np.array(Image.open(BASE / "packing_packed" / (name + ".png")))
        pa = BASE / "packing_unpacked" / (name + ".geometry.json")
        pb = BASE / "packing_packed" / (name + ".geometry.json")
        equal = (
            hashlib.sha256(pa.read_bytes()).digest()
            == hashlib.sha256(pb.read_bytes()).digest()
        )
        difference = int(np.count_nonzero(a != b))
        if difference or not equal:
            raise AssertionError("Position packing changed native evaluation: " + name)
        rows.append(
            {
                "name": name,
                "different_channels": difference,
                "all_vertices_bit_exact": equal,
            }
        )
    with (MODEL / "POPY.moc3").open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    jsonwrite(
        BASE / "dedup_verification.json",
        {
            "unpacked_bytes": (unpacked / "POPY.moc3").stat().st_size,
            "packed_bytes": (MODEL / "POPY.moc3").stat().st_size,
            "packed_moc_sha256": digest,
            "render_checks": rows,
        },
    )


def report():
    ir = json.loads(RIG.read_text())
    cases = json.loads((BASE / "cases.json").read_text())
    camera = json.loads((BASE / "reference/camera.json").read_text())
    if not np.allclose(
        camera["projection"], ir["camera"]["projection"], atol=1e-6
    ) or not np.allclose(
        camera["matrix_world"], ir["camera"]["matrix_world"], atol=1e-6
    ):
        raise AssertionError("Independent Blender/native camera mismatch")
    out = BASE / "report"
    out.mkdir(exist_ok=True)
    measurements = []
    skeleton = Skeleton(ROOT / "artifacts/samples", json.loads(CONFIG.read_text()))
    shoe_rest = np.load(ROOT / "artifacts/samples/M_Shoes_source.npz")["positions"]
    calf = (shoe_rest[:, 2] > 0.12) & (shoe_rest[:, 2] < 0.36)
    for case in cases:
        name = case["name"]
        a = (
            np.array(
                Image.open(BASE / "reference" / (name + ".png")).convert("RGBA"), float
            )
            / 255
        )
        b = (
            np.array(
                Image.open(BASE / "runtime" / (name + ".png")).convert("RGBA"), float
            )
            / 255
        )
        ma, mb = a[..., 3] > 0.5, b[..., 3] > 0.5
        union = ma | mb
        ac, bc = composite(a), composite(b)
        error = np.abs(ac - bc).mean(2)
        item = {
            "name": name,
            "parameters": case["parameters"],
            "silhouette_iou": float((ma & mb).sum() / max(union.sum(), 1)),
            "mae_foreground": float(error[union].mean()),
            "ssim_foreground": ssim(ac, bc, union),
        }
        shoe_pose = np.load(BASE / "reference" / (name + "__M_Shoes.npy"))
        box = project(shoe_pose[calf], camera) * ir["canvas"]
        lo = np.maximum(np.floor(box.min(0)).astype(int) - 1, 0)
        hi = np.minimum(np.ceil(box.max(0)).astype(int) + 2, ir["canvas"])
        roi = np.zeros(ma.shape, bool)
        roi[lo[1] : hi[1], lo[0] : hi[0]] = True
        white_a = (
            (np.min(a[:, :, :3], axis=2) > 0.95)
            & (np.ptp(a[:, :, :3], axis=2) < 0.05)
            & ma
            & roi
        )
        white_b = (
            (np.min(b[:, :, :3], axis=2) > 0.95)
            & (np.ptp(b[:, :, :3], axis=2) < 0.05)
            & mb
        )
        item["stocking_white_reference_pixels"] = int(white_a.sum())
        item["stocking_white_recall_1px"] = float(
            (white_a & binary_dilation(white_b, iterations=1)).sum()
            / max(white_a.sum(), 1)
        )
        cap = shoe_rest[:, 2] > 0.36
        box = project(shoe_pose[cap], camera) * ir["canvas"]
        lo = np.maximum(np.floor(box.min(0)).astype(int) - 2, 0)
        hi = np.minimum(np.ceil(box.max(0)).astype(int) + 3, ir["canvas"])
        roi[:] = False
        roi[lo[1] : hi[1], lo[0] : hi[0]] = True

        def skin_mask(image):
            return (
                (image[..., 0] > image[..., 1] + 0.025)
                & (image[..., 0] > image[..., 2] + 0.025)
                & (image[..., 1] > 0.65)
                & (image[..., 3] > 0.5)
            )

        skin_a, skin_b = skin_mask(a) & roi, skin_mask(b)
        item["stocking_cap_skin_reference_pixels"] = int(skin_a.sum())
        item["stocking_cap_skin_recall_1px"] = float(
            (skin_a & binary_dilation(skin_b, iterations=1)).sum()
            / max(skin_a.sum(), 1)
        )
        path = BASE / "geometry" / (name + ".geometry.json")
        if path.exists():
            native = {m["id"]: m for m in json.loads(path.read_text())}
            distances, analytical, original_distances = [], [], []
            cache = {}
            worst = []
            for m in ir["meshes"]:
                if not m.get("articulated") or m.get("mask_only"):
                    continue
                obj = m["source"]["object"]
                if obj not in cache:
                    v = np.load(BASE / "reference" / (name + "__" + obj + ".npy"))
                    cache[obj] = project(v, camera) * ir["canvas"]
                    actual = skeleton.evaluate(obj, case["parameters"])
                    analytical.extend(np.linalg.norm(actual - v, axis=1).tolist())
                ref = cache[obj][m["source"]["vertices"]]
                pred = (
                    np.array(native[m["id"]]["vertices"]) * ir["canvas"][0] / 2
                    + np.array(ir["canvas"]) / 2
                )
                d = np.linalg.norm(pred - ref, axis=1)
                distances.extend(d.tolist())
                if m.get("native_original_target"):
                    original_distances.extend(d.tolist())
                worst.append((float(d.max()), m["id"]))
            d = np.array(distances)
            item.update(
                articulated_rmse_px=float(np.sqrt((d * d).mean())),
                articulated_p95_px=float(np.percentile(d, 95)),
                articulated_max_px=float(d.max()),
                skinning_max_error_m=float(max(analytical)),
                worst_articulated_meshes=sorted(worst, reverse=True)[:5],
            )
            if original_distances:
                d0 = np.array(original_distances)
                item.update(
                    original_target_rmse_px=float(np.sqrt((d0 * d0).mean())),
                    original_target_max_px=float(d0.max()),
                )
        measurements.append(item)
        panel = Image.new("RGB", (768 * 3, 1060), "#252525")
        panel.paste(Image.fromarray(np.uint8(ac * 255)), (0, 36))
        panel.paste(Image.fromarray(np.uint8(bc * 255)), (768, 36))
        heat = np.stack(
            [np.clip(error * 4, 0, 1), np.clip(error, 0, 1), np.zeros_like(error)],
            axis=2,
        )
        panel.paste(Image.fromarray(np.uint8(heat * 255)), (1536, 36))
        ImageDraw.Draw(panel).text(
            (10, 10),
            f"{name} | Blender / Ayagami / error x4 | IoU {item['silhouette_iou']:.4f}",
            fill="white",
        )
        panel.save(out / (name + ".jpg"), quality=88)
    timing = json.loads((BASE / "runtime/runtime_report.json").read_text())
    ms = [x["pose_to_gpu_complete_ms"] for x in timing["cases"]]
    result = {
        "reference": json.loads((BASE / "anon_audit.json").read_text()),
        "pose_count": len(cases),
        "geometry_probe_count": sum("articulated_rmse_px" in m for m in measurements),
        "parameters": len(ir["parameters"]),
        "motions": len(
            json.loads((MODEL / "POPY.model3.json").read_text())["FileReferences"][
                "Motions"
            ]
        ),
        "expressions": len(
            json.loads((MODEL / "POPY.model3.json").read_text())["FileReferences"][
                "Expressions"
            ]
        ),
        "statistics": ir["diagnostics"]["fullbody"],
        "moc_bytes": (MODEL / "POPY.moc3").stat().st_size,
        "moc_sha256": hashlib.file_digest(
            (MODEL / "POPY.moc3").open("rb"), "sha256"
        ).hexdigest(),
        "render_median_ms": float(np.median(ms)),
        "render_max_ms": float(max(ms)),
        "cases": measurements,
        "limitations": [
            "Finite pose coverage, no proof of collision-free arbitrary motion",
            "Full-body clothing uses source skinning, no cloth/body collision simulation or foot-contact IK",
            "Mean-depth ArtMesh ordering and inherited head masks approximate 3D occlusion",
            "Expression presets use existing audited facial controls; Anon-specific tears/hearts/sweat/shadows are unsupported",
            "Large MOC/order-keyform memory cost; single-parameter matrix fields retain finite interpolation and float error",
            "No new CMO3 and no official Cubism Editor/Core verification",
            "CPU motion evaluator implements curves, looping, expression blend and clamping; host-level motion crossfades are not part of this offline harness",
        ],
    }
    unchanged = {
        m["id"]
        for m in ir["meshes"]
        if not m.get("articulated")
        and not m["id"].startswith("Full_")
        and not m.get("native_skinning")
    }
    transform = ir["diagnostics"]["fullbody"]["camera_transform"]
    head_checks = []
    for case in json.loads((BASE / "head_baseline_cases.json").read_text()):
        name = case["name"]
        a = {
            m["id"]: m
            for m in json.loads(
                (BASE / "head_baseline" / (name + ".geometry.json")).read_text()
            )
        }
        b = {
            m["id"]: m
            for m in json.loads(
                (BASE / "geometry" / (name + ".geometry.json")).read_text()
            )
        }
        error = []
        for k in sorted(unchanged):
            expected = (np.array(a[k]["vertices"]) * 384 + [384, 512]) * transform[
                "scale"
            ] + transform["translation"]
            actual = np.array(b[k]["vertices"]) * 384 + [384, 512]
            error.extend(np.linalg.norm(expected - actual, axis=1).tolist())
        head_checks.append(
            {
                "name": name,
                "unchanged_meshes": len(unchanged),
                "vertex_max_px": float(max(error)),
                "vertex_rmse_px": float(np.sqrt(np.mean(np.array(error) ** 2))),
            }
        )
    result["head_baseline_preservation"] = head_checks
    result["blender_verification"] = json.loads(
        (BASE / "blender_verification.json").read_text()
    )
    result["position_packing_verification"] = json.loads(
        (BASE / "dedup_verification.json").read_text()
    )
    if (
        result["position_packing_verification"]["packed_moc_sha256"]
        != result["moc_sha256"]
    ):
        raise AssertionError("Position packing verification belongs to another MOC")
    dense = BASE / "dense_baseline_metrics.json"
    if dense.exists():
        old_cases = {c["name"]: c for c in json.loads(dense.read_text())["cases"]}
        result["native_skinning_comparison"] = [
            {
                "name": c["name"],
                "dense_rmse_px": old_cases[c["name"]]["articulated_rmse_px"],
                "native_same_targets_rmse_px": c["original_target_rmse_px"],
                "dense_max_px": old_cases[c["name"]]["articulated_max_px"],
                "native_same_targets_max_px": c["original_target_max_px"],
            }
            for c in measurements
            if "original_target_rmse_px" in c
            and c["name"] in old_cases
            and "articulated_rmse_px" in old_cases[c["name"]]
        ]
    old = BASE / "coarse_leg_baseline/metrics.json"
    if old.exists():
        before = json.loads(old.read_text())
        old_cases = {c["name"]: c for c in before["cases"]}
        result["articulation_refinement"] = [
            {
                "name": c["name"],
                "before_rmse_px": old_cases[c["name"]]["articulated_rmse_px"],
                "after_rmse_px": c["articulated_rmse_px"],
                "before_max_px": old_cases[c["name"]]["articulated_max_px"],
                "after_max_px": c["articulated_max_px"],
            }
            for c in measurements
            if c["name"].startswith("compound_")
        ]
    jsonwrite(out / "metrics.json", result)
    public = ROOT / "benchmarks/popy/fullbody"
    public.mkdir(exist_ok=True, parents=True)
    jsonwrite(public / "metrics.json", result)
    perf = json.loads((ROOT / "artifacts/performance.json").read_text())
    latest = {
        p["stage"]: {k: v for k, v in p.items() if k != "command"}
        for p in perf
        if p["stage"].startswith("fullbody_")
    }
    jsonwrite(
        public / "performance.json",
        {
            "stages": list(latest.values()),
            "note": "Latest separately measured processes; desktop was not isolated. RSS is a sampled process-tree peak, not total host memory. Native timing includes parameter evaluation and GPU completion. Timeline render tests are separate from the comparison corpus.",
        },
    )
    selected = [
        "natural",
        "motion_wave_L_middle",
        "motion_point_R_middle",
        "motion_thinking_middle",
        "motion_leg_lift_L_middle",
        "motion_stretch_middle",
        "expression_smile_0",
        "expression_surprised_0",
    ]
    contact = Image.new("RGB", (384 * 4, 530 * 2), "#333333")
    for i, name in enumerate(selected):
        im = (
            Image.open(BASE / "runtime" / (name + ".png"))
            .convert("RGBA")
            .resize((384, 512))
        )
        contact.paste(im, ((i % 4) * 384, (i // 4) * 530), im)
        ImageDraw.Draw(contact).text(
            ((i % 4) * 384 + 5, (i // 4) * 530 + 512), name, fill="white"
        )
    contact.save(out / "overview.jpg", quality=94)
    html_rows = "".join(
        f'<details><summary>{html.escape(m["name"])} — IoU {m["silhouette_iou"]:.4f}; MAE {m["mae_foreground"]:.4f}</summary><img loading="lazy" width="100%" src="{m["name"]}.jpg"><pre>{html.escape(json.dumps(m, indent=2))}</pre></details>'
        for m in measurements
    )
    overview = {
        k: v for k, v in result.items() if k not in ["cases", "reference", "statistics"]
    }
    (out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>POPY articulated regression</title><style>body{background:#222;color:#eee;font:16px system-ui;max-width:1600px;margin:auto;padding:24px}summary{padding:12px}pre{white-space:pre-wrap}img{max-width:100%}</style><h1>POPY full body / motions / expressions</h1><img src="overview.jpg"><pre>'
        + html.escape(json.dumps(overview, indent=2))
        + "</pre>"
        + html_rows,
        encoding="utf8",
    )
    print(json.dumps(overview, indent=2), flush=True)


def package():
    delivery = ROOT / "delivery"
    delivery.mkdir(exist_ok=True)
    (MODEL / "README.txt").write_text(
        "POPY full body research model\nOpen POPY.model3.json with the adjacent textures. 37 parameters, 15 motion3 assets, 16 exp3 presets.\nThe Anon model was used as a semantic reference; its assets are not bundled.\nL/R means character left/right. Hand pose: -1 point, 0 open, +1 fist.\nMotion playback depends on the host. Offline native preview: python -m popy.fullbody frames --motion wave_L --expression smile --fps 15\nEditable 3D animation: separate delivery/POPY_fullbody.blend. No new CMO3.\nLarge research model: native matrix fields, mean-depth ordering, no cloth collision or foot-contact IK. Read docs/FULLBODY.md and the generated comparison report.\n",
        encoding="utf8",
    )
    metrics = json.loads((BASE / "report/metrics.json").read_text())
    log = (ROOT / "artifacts/logs/fullbody_tests.log").read_text()
    match = re.search(r"Ran (\d+) tests", log)
    if not match or not log.rstrip().endswith("OK"):
        raise AssertionError("Package requires a passing fullbody regression log")
    validation = {
        "python_tests": int(match.group(1)),
        "passed": True,
        "moc_sha256": metrics["moc_sha256"],
        "geometry_probes": metrics["geometry_probe_count"],
        "image_poses": metrics["pose_count"],
        "blender_clips": metrics["blender_verification"]["clips"],
        "blender_reopen_geometry_checks": len(
            metrics["blender_verification"]["evaluated_checks"]
        ),
        "official_cubism_editor_verified": False,
    }
    jsonwrite(ROOT / "benchmarks/popy/fullbody/validation.json", validation)
    with zipfile.ZipFile(
        delivery / "POPY_fullbody.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=3
    ) as z:
        for p in sorted(MODEL.rglob("*")):
            if p.is_file():
                z.write(p, Path("POPY") / p.relative_to(MODEL))
    with zipfile.ZipFile(
        delivery / "POPY_fullbody_report.zip",
        "w",
        zipfile.ZIP_DEFLATED,
        compresslevel=3,
    ) as z:
        for p in sorted((BASE / "report").rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(BASE / "report"))
    hashes = {}
    for path in [
        delivery / "POPY_fullbody.zip",
        delivery / "POPY_fullbody_report.zip",
        delivery / "POPY_fullbody.blend",
        MODEL / "POPY.moc3",
    ]:
        with path.open("rb") as stream:
            hashes[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    jsonwrite(delivery / "SHA256_fullbody.json", hashes)
    for path in [delivery / "POPY_fullbody.zip", delivery / "POPY_fullbody_report.zip"]:
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None:
                raise AssertionError("ZIP CRC failed: " + str(path))
    print("PACKAGED_FULLBODY", json.dumps(hashes), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "all",
            "build",
            "audit",
            "rig",
            "export",
            "ordering",
            "assets",
            "reference",
            "validate",
            "packing",
            "report",
            "test",
            "package",
            "blender",
            "frames",
        ],
    )
    parser.add_argument("--reference-model", default=str(ROOT.parent / "anon"))
    parser.add_argument(
        "--blender",
        default=r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
    )
    parser.add_argument("--motion", default="wave_L")
    parser.add_argument("--expression")
    parser.add_argument("--fps", type=float, default=15)
    args = parser.parse_args()
    BASE.mkdir(exist_ok=True)
    stages = (
        [
            "build",
            "audit",
            "rig",
            "export",
            "assets",
            "reference",
            "blender",
            "validate",
            "packing",
            "report",
            "test",
            "package",
        ]
        if args.stage == "all"
        else [args.stage]
    )
    source = ROOT / json.loads(CONFIG.read_text())["source"]
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    for stage in stages:
        if stage == "build":
            run(
                "fullbody_exporter_patch", [sys.executable, "-m", "popy.setup_exporter"]
            )
            env = os.environ.copy()
            env.setdefault("JAVA_HOME", r"C:\lib\java")
            run(
                "fullbody_exporter_build",
                [
                    ROOT / "vendor/psd2live/gradlew.bat",
                    "-p",
                    ROOT / "exporter",
                    "installDist",
                ],
                env,
            )
        elif stage == "audit":
            reference_audit(args.reference_model)
            run(
                "fullbody_source_audit",
                [
                    args.blender,
                    "--background",
                    source,
                    "--python-exit-code",
                    "1",
                    "--python",
                    ROOT / "blender/fullbody_audit.py",
                ],
            )
        elif stage == "rig":
            run("fullbody_rig", [sys.executable, "-u", "-m", "popy.fullbody_ir"])
        elif stage == "export":
            export()
        elif stage == "ordering":
            ordering()
        elif stage == "assets":
            assets()
        elif stage == "blender":
            run(
                "fullbody_actions",
                [
                    args.blender,
                    "--background",
                    source,
                    "--python-exit-code",
                    "1",
                    "--python",
                    ROOT / "blender/fullbody_actions.py",
                ],
            )
            run(
                "fullbody_actions_verify",
                [
                    args.blender,
                    "--background",
                    ROOT / "delivery/POPY_fullbody.blend",
                    "--python-exit-code",
                    "1",
                    "--python",
                    ROOT / "blender/fullbody_verify.py",
                ],
            )
        elif stage == "reference":
            reference(args.blender)
        elif stage == "validate":
            validate()
        elif stage == "packing":
            packing()
        elif stage == "report":
            report()
        elif stage == "test":
            run(
                "fullbody_tests",
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "tests",
                    "-p",
                    "test_fullbody.py",
                    "-v",
                ],
            )
        elif stage == "package":
            package()
        elif stage == "frames":
            if not 0 < args.fps <= 120:
                raise ValueError("fps must be in (0,120]")
            parameters = json.loads((BASE / "parameters.json").read_text())
            motion = json.loads(
                (MODEL / "motions" / (args.motion + ".motion3.json")).read_text()
            )
            expression = (
                json.loads(
                    (
                        MODEL / "expressions" / (args.expression + ".exp3.json")
                    ).read_text()
                )
                if args.expression
                else None
            )
            frames = [
                {
                    "name": f"frame_{i:05}",
                    "parameters": evaluate(parameters, motion, float(t), expression),
                    "time": float(t),
                }
                for i, t in enumerate(
                    np.arange(0, motion["Meta"]["Duration"], 1 / args.fps)
                )
            ]
            path = BASE / (args.motion + "_timeline.json")
            jsonwrite(path, frames)
            run(
                "fullbody_timeline",
                [
                    RENDERER,
                    MODEL / "POPY.model3.json",
                    path,
                    BASE / (args.motion + "_frames"),
                    "--images-only",
                ],
            )
            imgs = [
                Image.open(
                    BASE / (args.motion + "_frames") / (f["name"] + ".png")
                ).convert("RGBA")
                for f in frames
            ]
            imgs[0].save(
                BASE / (args.motion + ".webp"),
                save_all=True,
                append_images=imgs[1:],
                duration=round(1000 / args.fps),
                loop=0,
                lossless=False,
                quality=85,
            )
    if hashlib.sha256(source.read_bytes()).hexdigest() != original:
        raise AssertionError("Source snapshot changed")


if __name__ == "__main__":
    main()
