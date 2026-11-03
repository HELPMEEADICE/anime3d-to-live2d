"""Measure eye visibility, geometric seams and frame-to-frame excess changes against Blender."""

import argparse
import json
import os
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion

from .compile_ir import ROOT, jsonwrite
from .visibility import weld_sources


def eye_palette(model, destination):
    model, destination = Path(model).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    value = json.loads(model.read_text())
    source = json.loads((ROOT / "artifacts/samples/manifest.json").read_text())
    eye_object = source["semantics"]["objects"]["gaze"]
    eye_textures = set(
        next(o for o in source["objects"] if o["name"] == eye_object)["textures"]
    )
    refs = value["FileReferences"]
    refs["Moc"] = os.path.relpath(model.parent / refs["Moc"], destination)
    for i, texture in enumerate(refs["Textures"]):
        image = np.asarray(Image.open(model.parent / texture).convert("RGBA")).copy()
        image[:, :, :3] = 255 if Path(texture).name in eye_textures else 0
        name = f"palette_{i}.png"
        Image.fromarray(image).save(destination / name)
        refs["Textures"][i] = name
    refs.pop("DisplayInfo", None)
    out = destination / "POPY.model3.json"
    jsonwrite(out, value)
    return out


def rgba(path):
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.float32) / 255


def rgb(path):
    a = rgba(path)
    return a[:, :, :3] * a[:, :, 3:4] + 0.7 * (1 - a[:, :, 3:4])


def assess(ir_path, runtime, eye_runtime, destination):
    ir = json.loads(Path(ir_path).read_text())
    if not ir.get("glues"):
        weld_sources(ir, ROOT / "artifacts/samples")
    runtime, eye_runtime, destination = map(Path, [runtime, eye_runtime, destination])
    destination.mkdir(parents=True, exist_ok=True)
    for stale in destination.glob("*_eye_errors.png"):
        stale.unlink()
    reference = ROOT / "artifacts/visibility_reference"
    cases = json.loads((ROOT / "artifacts/visibility_cases.json").read_text())
    result = []
    sequences = defaultdict(list)
    for case in cases:
        name = case["name"]
        a, b = rgba(reference / f"{name}_eyes.png"), rgba(eye_runtime / f"{name}.png")
        ref_eye = (a[:, :, :3].mean(2) * a[:, :, 3]) > 0.5
        actual_eye = (b[:, :, :3].mean(2) * b[:, :, 3]) > 0.5
        leak = actual_eye & ~binary_dilation(ref_eye, iterations=1)
        missing = ref_eye & ~binary_dilation(actual_eye, iterations=1)
        mask_union = ref_eye | actual_eye
        regular_ref, regular_actual = (
            rgba(reference / f"{name}.png"),
            rgba(runtime / f"{name}.png"),
        )
        opaque_interior = binary_erosion(regular_ref[:, :, 3] > 0.99, iterations=3)
        cracks = opaque_interior & (regular_actual[:, :, 3] < 0.25)
        geometry = {
            m["id"]: np.asarray(m["vertices"])
            for m in json.loads((runtime / f"{name}.geometry.json").read_text())
        }
        seam = []
        for g in ir["glues"]:
            if not g["id"].startswith("G_Source_"):
                continue
            pairs = np.asarray(g["pairs"])
            seam.extend(
                np.linalg.norm(
                    geometry[g["mesh_a"]][pairs[:, 0].astype(int)]
                    - geometry[g["mesh_b"]][pairs[:, 1].astype(int)],
                    axis=1,
                ).tolist()
            )
        seam = np.asarray(seam) * (ir["canvas"][0] / 2)
        aa, bb = rgb(reference / f"{name}.png"), rgb(runtime / f"{name}.png")
        union = (regular_ref[:, :, 3] > 0.5) | (regular_actual[:, :, 3] > 0.5)
        record = {
            "name": name,
            "parameters": case["parameters"],
            "eye_leak_pixels": int(leak.sum()),
            "eye_missing_pixels": int(missing.sum()),
            "eye_visible_ref": int(ref_eye.sum()),
            "eye_visible_actual": int(actual_eye.sum()),
            "eye_iou": float((ref_eye & actual_eye).sum() / max(mask_union.sum(), 1)),
            "interior_crack_pixels": int(cracks.sum()),
            "source_seam_max_px": float(seam.max()) if len(seam) else 0,
            "source_seam_p95_px": float(np.percentile(seam, 95)) if len(seam) else 0,
            "mae_foreground": float(np.mean(np.abs(aa - bb)[union])),
        }
        result.append(record)
        if "sequence" in case:
            sequences[case["sequence"]].append((case["frame"], name))
        if record["eye_leak_pixels"] or record["eye_missing_pixels"]:
            overlay = np.uint8(bb * 255)
            overlay[leak] = [255, 30, 30]
            overlay[missing] = [30, 100, 255]
            Image.fromarray(overlay).save(destination / f"{name}_eye_errors.png")
    transitions = []
    for sequence, frames in sequences.items():
        frames.sort()
        for (_, previous), (_, current) in pairwise(frames):
            ra, rb = (
                rgb(reference / f"{previous}.png"),
                rgb(reference / f"{current}.png"),
            )
            aa, ab = rgb(runtime / f"{previous}.png"), rgb(runtime / f"{current}.png")
            source_change = np.mean(np.abs(rb - ra), axis=2)
            error = np.mean(np.abs((ab - aa) - (rb - ra)), axis=2)
            false_jump = (error > 0.20) & (source_change < 0.05)
            transitions.append(
                {
                    "sequence": sequence,
                    "from": previous,
                    "to": current,
                    "false_jump_pixels": int(false_jump.sum()),
                    "temporal_residual_mae": float(error.mean()),
                }
            )
    summary = {
        "poses": len(result),
        "transitions": len(transitions),
        "eye_leak_total": sum(r["eye_leak_pixels"] for r in result),
        "eye_leak_max": max(r["eye_leak_pixels"] for r in result),
        "eye_missing_total": sum(r["eye_missing_pixels"] for r in result),
        "interior_crack_total": sum(r["interior_crack_pixels"] for r in result),
        "source_seam_max_px": max(r["source_seam_max_px"] for r in result),
        "false_jump_total": sum(r["false_jump_pixels"] for r in transitions),
        "false_jump_max": max(r["false_jump_pixels"] for r in transitions),
        "method": "Blender depth-rendered eye-only RGB visibility, 1px boundary allowance; source-identity seam constraints; alpha cracks exclude 3px silhouette border; frame residual subtracts real 3D motion",
    }
    report = {"summary": summary, "cases": result, "transitions": transitions}
    jsonwrite(destination / "metrics.json", report)
    print(json.dumps(summary, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    palette = sub.add_parser("palette")
    palette.add_argument("model")
    palette.add_argument("destination")
    check = sub.add_parser("assess")
    check.add_argument("--rig", required=True)
    check.add_argument("--runtime", required=True)
    check.add_argument("--eyes", required=True)
    check.add_argument("--out", required=True)
    args = p.parse_args()
    if args.command == "palette":
        print(eye_palette(args.model, args.destination))
    else:
        assess(args.rig, args.runtime, args.eyes, args.out)
