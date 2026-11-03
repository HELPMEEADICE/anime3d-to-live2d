"""Independent eye/face/headwear coverage checks for hair repair regressions."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from .compile_ir import ROOT, jsonwrite
from .surface_regression import diagnostic_rig


def feature_labels(rig):
    ir = json.loads(rig.read_text())
    source = ROOT / "artifacts/samples"
    result = {
        n: np.zeros(len(np.load(source / (n + "_source.npz"))["triangles"]), int)
        for n in ["M_Face", "M_Eye", "M_Headwear"]
    }
    eyes = set(ir["visibility"]["eye_targets"])
    skin = set(ir["visibility"]["skin_surface_meshes"])
    for mesh in ir["meshes"]:
        if mesh.get("mask_only"):
            continue
        name = mesh["source"]["object"]
        value = (
            1
            if name == "M_Eye" or mesh["id"] in eyes
            else 2
            if mesh["id"] in skin
            else 3
            if name == "M_Headwear"
            else 0
        )
        if name in result:
            result[name][mesh["source"]["triangles"]] = value
    return result


def assess(reference, runtime, out, case_file=ROOT / "artifacts/hair_cases.json"):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in json.loads(case_file.read_text()):
        name = case["name"]
        a = np.array(Image.open(reference / (name + "_surfaces.png")).convert("RGBA"))
        b = np.array(Image.open(runtime / (name + ".png")).convert("RGBA"))
        row = {"name": name, "parameters": case["parameters"]}
        for channel, kind in enumerate(["eyes", "face", "headwear"]):
            aa = (a[:, :, channel] > 128) & (a[:, :, 3] > 128)
            bb = (b[:, :, channel] > 128) & (b[:, :, 3] > 128)
            row[kind] = {
                "missing": int((aa & ~binary_dilation(bb, iterations=1)).sum()),
                "leak": int((bb & ~binary_dilation(aa, iterations=1)).sum()),
                "reference": int(aa.sum()),
                "runtime": int(bb.sum()),
            }
        rows.append(row)
    result = {
        "poses": len(rows),
        "summary": {
            k: {
                m: sum(r[k][m] for r in rows)
                for m in ["missing", "leak", "reference", "runtime"]
            }
            for k in ["eyes", "face", "headwear"]
        },
        "cases": rows,
        "method": "Native diagnostic MOC with source eye/sclera, face-skin and headwear IDs versus an independent Blender depth-rendered pass, original alpha retained, 1px boundary tolerance. Includes projection interpolation error.",
    }
    jsonwrite(out / "metrics.json", result)
    print(json.dumps(result["summary"]), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prepare", "assess"])
    p.add_argument(
        "--rig", type=Path, default=ROOT / "artifacts/hair_clearance_rig/POPY.rig.json"
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--runtime", type=Path)
    p.add_argument(
        "--reference", type=Path, default=ROOT / "artifacts/hair_feature_reference"
    )
    a = p.parse_args()
    if a.command == "prepare":
        labels = feature_labels(ROOT / "artifacts/rig/POPY.rig.json")
        jsonwrite(
            ROOT / "artifacts/hair_feature_labels.json",
            {k: v.tolist() for k, v in labels.items()},
        )
        print(diagnostic_rig(a.rig, a.out, labels))
    else:
        assess(a.reference, a.runtime, a.out)
