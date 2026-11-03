"""RGB surface-ID passes retain real texture alpha, masks and native runtime sorting."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from .compile_ir import ROOT, jsonwrite, project
from .visibility import component_labels

COLORS = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]


def labels(samples=ROOT / "artifacts/samples"):
    config = json.loads((ROOT / "config/surface_visibility.json").read_text())
    result = {}
    for i, kind in enumerate(["brows", "lashes", "hair"], 1):
        spec = config[kind]
        name = spec["object"]
        source = np.load(samples / f"{name}_source.npz")
        _, components = component_labels(source)
        result.setdefault(name, np.zeros(len(components), dtype=int))
        use = (
            np.isin(components, spec["components"])
            if "components" in spec
            else np.ones(len(components), bool)
        )
        result[name][use] = i
    return result


def cases():
    rows = json.loads((ROOT / "artifacts/visibility_cases.json").read_text())
    # Keep every existing eye/body regression. Additional held-out eyebrow/neck combinations.
    for x in [-45, -37.5, -22.5, 22.5, 37.5, 45]:
        for y in [-30, 0, 30]:
            for expression in ["angry", "raised"]:
                p = {"ParamAngleX": x, "ParamAngleY": y}
                p.update(
                    {"ParamBrowLForm": 1, "ParamBrowRForm": 1}
                    if expression == "angry"
                    else {"ParamBrowLY": 1, "ParamBrowRY": 1}
                )
                rows.append({"name": f"feature_{x}_{y}_{expression}", "parameters": p})
    # Held-out continuous controls: never used to select critical source patches.
    for yaw in [-45, 45]:
        for i in range(21):
            t = i / 20
            rows.append(
                {
                    "name": f"continuous_blink_{yaw}_{i:03}",
                    "parameters": {
                        "ParamAngleX": yaw,
                        "ParamAngleY": -30,
                        "ParamEyeLOpen": t,
                        "ParamEyeROpen": t,
                        "ParamMouthOpenY": 0.6,
                    },
                    "sequence": f"continuous_blink_{yaw}",
                    "frame": i,
                }
            )
    for i in range(21):
        t = i / 20
        rows.append(
            {
                "name": f"continuous_brow_{i:03}",
                "parameters": {
                    "ParamAngleX": 30,
                    "ParamAngleY": 15,
                    "ParamAngleZ": 15,
                    "ParamBrowLForm": t,
                    "ParamBrowRForm": t,
                    "ParamBrowLY": 0.6 * (1 - t),
                    "ParamBrowRY": 0.6 * (1 - t),
                },
                "sequence": "continuous_brow",
                "frame": i,
            }
        )
    return rows


def diagnostic_rig(rig, destination):
    """A real second MOC with identical geometry, alpha and masks; only texture RGB changes.

    Give each class its own texture pages so shared UV regions never relabel another part.
    Hidden helpers stay hidden and retain their original alpha in native clipping passes.
    """
    rig, destination = Path(rig).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for stale in destination.glob("*_errors.png"):
        stale.unlink()
    ir = json.loads(rig.read_text())
    classes = labels()
    original = ir["textures"][:]
    ir["textures"] = []
    for label, color in enumerate(COLORS):
        for ti, texture in enumerate(original):
            image = Image.open(rig.parent / texture).convert("RGBA")
            rgb = Image.new("RGBA", image.size, (*color, 255))
            rgb.putalpha(image.getchannel("A"))
            name = f"surface_{label}_{ti}.png"
            rgb.save(destination / name)
            ir["textures"].append(name)
    for mesh in ir["meshes"]:
        source = mesh["source"]
        selected = classes.get(source["object"])
        values = (
            np.unique(selected[source["triangles"]]) if selected is not None else [0]
        )
        if len(values) != 1:
            raise ValueError(f"Mixed semantic surface in {mesh['id']}: {values}")
        mesh["texture"] += int(values[0]) * len(original)
    ir["keyform_data"] = os.path.relpath(rig.parent / ir["keyform_data"], destination)
    out = destination / "POPY.rig.json"
    jsonwrite(out, ir)
    return out


def assess(runtime, destination, reference=ROOT / "artifacts/surface_reference"):
    runtime, destination, reference = map(Path, [runtime, destination, reference])
    destination.mkdir(parents=True, exist_ok=True)
    camera = json.loads((ROOT / "artifacts/samples/manifest.json").read_text())[
        "camera"
    ]
    spec = json.loads((ROOT / "config/surface_visibility.json").read_text())["neck"]
    rest = np.load(ROOT / "artifacts/samples" / f"{spec['object']}_source.npz")[
        "positions"
    ]
    use = (
        (rest[:, 2] >= spec["rest_z"][0])
        & (rest[:, 2] <= spec["rest_z"][1])
        & (rest[:, 0] >= spec["rest_x"][0])
        & (rest[:, 0] <= spec["rest_x"][1])
    )
    result = []
    for case in cases():
        name = case["name"]
        a = np.array(Image.open(reference / f"{name}_surfaces.png").convert("RGBA"))
        b = np.array(Image.open(runtime / f"{name}.png").convert("RGBA"))
        h, w = a.shape[:2]
        neck = project(
            np.load(reference / f"{name}__{spec['object']}.npy")[use], camera
        ) * [w, h]
        x0, y0 = np.maximum(np.floor(neck.min(0) - 8), 0).astype(int)
        x1, y1 = np.minimum(np.ceil(neck.max(0) + 8), [w, h]).astype(int)
        roi = np.zeros((h, w), bool)
        roi[y0:y1, x0:x1] = True
        row = {
            "name": name,
            "parameters": case["parameters"],
            "neck_roi": [int(x0), int(y0), int(x1), int(y1)],
        }
        overlay = np.uint8(b[:, :, :3] * (b[:, :, 3:4] / 255))
        for channel, kind in enumerate(["brows", "lashes", "neck_hair"]):
            aa = (a[:, :, channel] > 128) & (a[:, :, 3] > 128)
            bb = (b[:, :, channel] > 128) & (b[:, :, 3] > 128)
            leak = bb & ~binary_dilation(aa, iterations=1)
            missing = aa & ~binary_dilation(bb, iterations=1)
            if kind == "neck_hair":
                leak &= roi
                missing &= roi
                aa &= roi
                bb &= roi
            leak_far = bb & ~binary_dilation(aa, iterations=3)
            missing_far = aa & ~binary_dilation(bb, iterations=3)
            if kind == "neck_hair":
                leak_far &= roi
                missing_far &= roi
            row[kind] = {
                "leak": int(leak.sum()),
                "missing": int(missing.sum()),
                "leak_far": int(leak_far.sum()),
                "missing_far": int(missing_far.sum()),
                "reference": int(aa.sum()),
                "runtime": int(bb.sum()),
            }
            overlay[leak] = [255, 80, 20]
            overlay[missing] = [20, 180, 255]
        if any(
            row[k][m]
            for k in ["brows", "lashes", "neck_hair"]
            for m in ["leak", "missing"]
        ):
            Image.fromarray(overlay).save(destination / f"{name}_errors.png")
        result.append(row)
    summary = {
        k: {
            m: sum(r[k][m] for r in result)
            for m in [
                "leak",
                "missing",
                "leak_far",
                "missing_far",
                "reference",
                "runtime",
            ]
        }
        for k in ["brows", "lashes", "neck_hair"]
    }
    report = {
        "summary": summary,
        "poses": len(result),
        "cases": result,
        "method": "Independent Blender depth-rendered RGB surface IDs versus an identical real Ayagami MOC; original texture alpha retained, per-class pages avoid UV aliasing; leak/missing use 1px boundary allowance, *_far use 3px. Both include geometry interpolation/alignment error and are not pure depth-sort counts. Neck ROI follows source body vertices.",
    }
    jsonwrite(destination / "metrics.json", report)
    print(json.dumps({"poses": len(result), **summary}, indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("diagnostic")
    d.add_argument("rig", type=Path)
    d.add_argument("destination", type=Path)
    c = sub.add_parser("cases")
    c.add_argument("destination", type=Path)
    a = sub.add_parser("assess")
    a.add_argument("runtime", type=Path)
    a.add_argument("destination", type=Path)
    args = p.parse_args()
    if args.command == "diagnostic":
        print(diagnostic_rig(args.rig, args.destination))
    elif args.command == "cases":
        jsonwrite(args.destination, cases())
        print(len(cases()), "surface visibility cases")
    else:
        assess(args.runtime, args.destination)


if __name__ == "__main__":
    main()
