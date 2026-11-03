"""Shoulder-wide hair occlusion regression, including user-reported poses."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from .compile_ir import ROOT, jsonwrite, project


def cases():
    rows = [
        {
            "name": "reported_left",
            "parameters": {
                "ParamAngleX": -45,
                "ParamAngleY": -6,
                "ParamAngleZ": 13,
                "ParamBreath": 0.86,
            },
        },
        {
            "name": "reported_right",
            "parameters": {"ParamAngleX": 45, "ParamAngleY": -5, "ParamBreath": 0.56},
        },
        {
            "name": "reported_roll",
            "parameters": {
                "ParamAngleX": -5,
                "ParamAngleY": -4,
                "ParamAngleZ": -30,
                "ParamBreath": 0.78,
            },
        },
    ]
    for sequence, key, values, fixed in [
        (
            "hair_yaw",
            "ParamAngleX",
            np.linspace(-45, 45, 31),
            {"ParamAngleY": -5, "ParamAngleZ": 13},
        ),
        (
            "hair_roll",
            "ParamAngleZ",
            np.linspace(-30, 30, 31),
            {"ParamAngleX": -5, "ParamAngleY": -4},
        ),
    ]:
        for i, value in enumerate(values):
            rows.append(
                {
                    "name": f"{sequence}_{i:03}",
                    "parameters": {**fixed, key: float(value)},
                    "sequence": sequence,
                    "frame": i,
                }
            )
    for yaw in [-45, 45]:
        for pitch in [-30, 30]:
            for roll in [-30, 0, 30]:
                rows.append(
                    {
                        "name": f"hair_body_{yaw}_{pitch}_{roll}",
                        "parameters": {
                            "ParamAngleX": yaw,
                            "ParamAngleY": pitch,
                            "ParamAngleZ": roll,
                            "ParamBodyAngleX": 15,
                            "ParamBodyAngleY": -15,
                            "ParamBodyAngleZ": 15,
                        },
                    }
                )
    return rows


def assess(case_file, reference, runtime, destination):
    destination.mkdir(parents=True, exist_ok=True)
    camera = json.loads((ROOT / "artifacts/samples/manifest.json").read_text())[
        "camera"
    ]
    rest = np.load(ROOT / "artifacts/samples/M_Top_source.npz")["positions"]
    # Anatomical shoulder/upper-chest envelope, deliberately wider than old neck ROI.
    use = (rest[:, 2] >= 1.14) & (rest[:, 2] <= 1.42) & (abs(rest[:, 0]) <= 0.30)
    rows, sequences = [], {}
    for case in json.loads(case_file.read_text()):
        name = case["name"]
        a = np.array(Image.open(reference / f"{name}_surfaces.png").convert("RGBA"))
        b = np.array(Image.open(runtime / f"{name}.png").convert("RGBA"))
        h, w = a.shape[:2]
        points = project(np.load(reference / f"{name}__M_Top.npy")[use], camera) * [
            w,
            h,
        ]
        lo = np.maximum(np.floor(points.min(0) - 8), 0).astype(int)
        hi = np.minimum(np.ceil(points.max(0) + 8), [w, h]).astype(int)
        roi = np.zeros((h, w), bool)
        roi[lo[1] : hi[1], lo[0] : hi[0]] = True
        aa = (a[:, :, 2] > 128) & (a[:, :, 3] > 128)
        bb = (b[:, :, 2] > 128) & (b[:, :, 3] > 128)
        row = {**case, "roi": [*lo.tolist(), *hi.tolist()]}
        overlay = np.uint8(b[:, :, :3] * (b[:, :, 3:4] / 255))
        for radius, suffix in [(1, ""), (3, "_far")]:
            leak = bb & ~binary_dilation(aa, iterations=radius) & roi
            missing = aa & ~binary_dilation(bb, iterations=radius) & roi
            row["leak" + suffix] = int(leak.sum())
            row["missing" + suffix] = int(missing.sum())
            if radius == 1:
                overlay[leak] = [255, 80, 20]
                overlay[missing] = [20, 180, 255]
        Image.fromarray(overlay).save(destination / f"{name}_errors.png")
        rows.append(row)
        if "sequence" in case:
            sequences.setdefault(case["sequence"], []).append(
                (case["frame"], name, aa, bb, roi)
            )
    transitions = []
    for sequence, states in sequences.items():
        states.sort(key=lambda r: r[0])
        for before, after in itertools.pairwise(states):
            ground = before[2] ^ after[2]
            actual = before[3] ^ after[3]
            transitions.append(
                {
                    "sequence": sequence,
                    "before": before[1],
                    "after": after[1],
                    "false_change": int(
                        (
                            actual
                            & ~binary_dilation(ground, iterations=1)
                            & (before[4] | after[4])
                        ).sum()
                    ),
                }
            )
    report = {
        "cases": rows,
        "poses": len(rows),
        "summary": {
            k: sum(r[k] for r in rows)
            for k in ["leak", "missing", "leak_far", "missing_far"]
        },
        "transitions": transitions,
        "false_change_total": sum(t["false_change"] for t in transitions),
        "method": "Real Blender depth surface IDs versus Ayagami native mask rendering; source-clothing shoulder/chest envelope z=1.14..1.42m, abs(x)<=0.30m, 8px margin; 1/3px boundary allowance. Includes projection interpolation error. False change counts adjacent-frame runtime changes outside the 1px-dilated Blender change mask; includes motion interpolation differences.",
    }
    jsonwrite(destination / "metrics.json", report)
    print(json.dumps({"poses": len(rows), **report["summary"]}), flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["cases", "assess"])
    p.add_argument("--cases", type=Path, default=ROOT / "artifacts/hair_cases.json")
    p.add_argument("--reference", type=Path, default=ROOT / "artifacts/hair_reference")
    p.add_argument("--runtime", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    if a.command == "cases":
        jsonwrite(a.cases, cases())
    else:
        assess(a.cases, a.reference, a.runtime, a.out)
