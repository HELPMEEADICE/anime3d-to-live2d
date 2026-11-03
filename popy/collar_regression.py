"""Independent Blender/native neck ID comparisons and continuous angle sweeps."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from .compile_ir import ROOT, jsonwrite, project
from .surface_regression import diagnostic_rig


def cases():
    rows = [
        {
            "name": "reported_neck",
            "parameters": {"ParamAngleX": 15, "ParamAngleY": 11, "ParamBreath": 0.5},
        }
    ]
    for x in [0, 7.5, 15, 22.5, 30]:
        for y in [0, 7.5, 11, 15, 22.5]:
            if x == 15 and y == 11:
                continue
            rows.append(
                {
                    "name": f"near_{x}_{y}",
                    "parameters": {
                        "ParamAngleX": x,
                        "ParamAngleY": y,
                        "ParamBreath": 0.5,
                    },
                }
            )
    for axis, lo, hi, fixed in [
        ("Y", -30, 30, {"ParamAngleX": 15}),
        ("X", -45, 45, {"ParamAngleY": 11}),
    ]:
        for i, value in enumerate(np.linspace(lo, hi, 21)):
            rows.append(
                {
                    "name": f"neck_{axis}_{i:02}",
                    "parameters": dict(fixed, **{"ParamAngle" + axis: float(value)}),
                    "sequence": "neck_" + axis,
                    "frame": i,
                }
            )
    for x in [-45, 45]:
        for y in [-30, 0, 30]:
            for sign in [-1, 1]:
                rows.append(
                    {
                        "name": f"compound_{x}_{y}_{sign}",
                        "parameters": {
                            "ParamAngleX": x,
                            "ParamAngleY": y,
                            "ParamAngleZ": 30 * sign,
                            "ParamBodyAngleX": 15 * sign,
                            "ParamBodyAngleY": -15 * sign,
                            "ParamBodyAngleZ": 15 * sign,
                            "ParamMouthOpenY": 0.7,
                            "ParamEyeLOpen": 0,
                            "ParamBreath": 1,
                        },
                    }
                )
    return rows


def labels():
    result = {}
    for name, value in [
        ("M_Body", 1),
        ("M_Top", 2),
        ("M_Skirts", 2),
        ("M_Bow", 2),
        ("M_Necktie", 2),
        ("M_Face", 3),
    ]:
        with np.load(ROOT / "artifacts/samples" / (name + "_source.npz")) as source:
            result[name] = np.full(len(source["triangles"]), value, dtype=int)
    return result


def assess(reference, runtime, out, case_file):
    out.mkdir(parents=True, exist_ok=True)
    camera = json.loads((ROOT / "artifacts/samples/manifest.json").read_text())[
        "camera"
    ]
    with np.load(ROOT / "artifacts/samples/M_Body_source.npz") as source:
        rest = source["positions"]
    neck = (rest[:, 2] > 1.30) & (rest[:, 2] < 1.38) & (np.abs(rest[:, 0]) < 0.065)
    rows, previous, transitions = [], {}, []
    for case in json.loads(case_file.read_text()):
        name = case["name"]
        a = np.array(Image.open(reference / (name + "_surfaces.png")).convert("RGBA"))
        b = np.array(Image.open(runtime / (name + ".png")).convert("RGBA"))
        h, w = a.shape[:2]
        points = project(np.load(reference / (name + "__M_Body.npy"))[neck], camera) * [
            w,
            h,
        ]
        lo = np.maximum(np.floor(points.min(0) - 5), 0).astype(int)
        hi = np.minimum(np.ceil(points.max(0) + 6), [w, h]).astype(int)
        roi = np.zeros((h, w), bool)
        roi[lo[1] : hi[1], lo[0] : hi[0]] = True
        aa = (a[:, :, 0] > 128) & (a[:, :, 3] > 128)
        bb = (b[:, :, 0] > 128) & (b[:, :, 3] > 128)
        leak = bb & ~binary_dilation(aa, iterations=1) & roi
        missing = aa & ~binary_dilation(bb, iterations=1) & roi
        row = {
            "name": name,
            "parameters": case["parameters"],
            "leak": int(leak.sum()),
            "missing": int(missing.sum()),
            "reference_pixels": int((aa & roi).sum()),
            "runtime_pixels": int((bb & roi).sum()),
        }
        rows.append(row)
        heat = np.zeros_like(a)
        heat[leak] = [255, 50, 0, 255]
        heat[missing] = [0, 120, 255, 255]
        Image.fromarray(heat).save(out / (name + "_errors.png"))
        sequence = case.get("sequence")
        if sequence in previous:
            pa, pb, proi = previous[sequence]
            region = roi | proi
            false_change = (bb ^ pb) & ~binary_dilation(aa ^ pa, iterations=1) & region
            transitions.append({"name": name, "false_change": int(false_change.sum())})
        if sequence:
            previous[sequence] = (aa, bb, roi)
    result = {
        "poses": len(rows),
        "summary": {
            k: sum(row[k] for row in rows)
            for k in ["leak", "missing", "reference_pixels", "runtime_pixels"]
        },
        "cases": rows,
        "transitions": transitions,
        "temporal_false_change": sum(t["false_change"] for t in transitions),
        "method": "Native skin-ID MOC versus independent Blender source depth pass, projected original neck bounds +5px ROI, 1px boundary tolerance; includes projection and support-plane approximation error.",
    }
    jsonwrite(out / "metrics.json", result)
    print(result["summary"], flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare", "assess"])
    p.add_argument("--rig", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--runtime", type=Path)
    p.add_argument(
        "--cases", type=Path, default=ROOT / "artifacts/neck_repair/cases.json"
    )
    a = p.parse_args()
    if a.stage == "prepare":
        jsonwrite(a.cases, cases())
        jsonwrite(
            ROOT / "artifacts/neck_repair/labels.json",
            {k: v.tolist() for k, v in labels().items()},
        )
        diagnostic_rig(a.rig, a.out, labels())
    else:
        assess(a.reference, a.runtime, a.out, a.cases)
