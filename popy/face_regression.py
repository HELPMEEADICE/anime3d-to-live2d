"""Independent dense head sweeps and source-ID visibility measurements."""

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_fill_holes, label

from .compile_ir import ROOT, jsonwrite, project
from .visibility import component_labels


def cases():
    rows = [
        {
            "name": "reported_right",
            "parameters": {
                "ParamAngleX": 45,
                "ParamAngleY": -2,
                "ParamAngleZ": 1,
                "ParamBreath": 0.05,
            },
        },
        {
            "name": "reported_front",
            "parameters": {"ParamAngleX": -11, "ParamAngleY": 2, "ParamBreath": 0.43},
        },
    ]
    for x in np.linspace(-45, 45, 13):
        for y in np.linspace(-30, 30, 9):
            rows.append(
                {
                    "name": f"grid_{x:g}_{y:g}",
                    "parameters": {"ParamAngleX": float(x), "ParamAngleY": float(y)},
                }
            )
    for axis, lo, hi, fixed in [
        ("ParamAngleX", -45, 45, {"ParamAngleY": -2, "ParamAngleZ": 1}),
        ("ParamAngleY", -30, 30, {"ParamAngleX": 45}),
        ("ParamAngleZ", -30, 30, {"ParamAngleX": 15, "ParamAngleY": 11}),
    ]:
        for i, value in enumerate(np.linspace(lo, hi, 21)):
            rows.append(
                {
                    "name": f"sweep_{axis}_{i:02}",
                    "parameters": {**fixed, axis: float(value)},
                    "sequence": axis,
                    "frame": i,
                }
            )
    expressions = [
        {"ParamEyeLOpen": 0, "ParamEyeROpen": 1},
        {"ParamEyeLOpen": 0, "ParamEyeROpen": 0, "ParamEyeLSmile": 0.7},
        {"ParamMouthOpenY": 1, "ParamMouthForm": 1, "ParamBrowLY": 1},
        {
            "ParamEyeLOpen": 0.35,
            "ParamEyeROpen": 0.6,
            "ParamMouthOpenY": 0.7,
            "ParamBrowRForm": 0.8,
        },
    ]
    for x in [-45, 45]:
        for y in [-30, 0, 30]:
            for i, expression in enumerate(expressions):
                rows.append(
                    {
                        "name": f"expression_{x}_{y}_{i}",
                        "parameters": {
                            "ParamAngleX": x,
                            "ParamAngleY": y,
                            **expression,
                        },
                    }
                )
            for s in [-1, 1]:
                rows.append(
                    {
                        "name": f"compound_{x}_{y}_{s}",
                        "parameters": {
                            "ParamAngleX": x,
                            "ParamAngleY": y,
                            "ParamAngleZ": s * 30,
                            "ParamBodyAngleX": s * 15,
                            "ParamBodyAngleY": -s * 15,
                            "ParamBodyAngleZ": s * 15,
                            "ParamMouthOpenY": 0.7,
                            "ParamBreath": 1,
                        },
                    }
                )
    return rows


def labels():
    result = {}
    face = np.load(ROOT / "artifacts/samples/M_Face_source.npz")
    _, components = component_labels(face)
    result["M_Face"] = np.where(components == 0, 1, np.where(components == 22, 3, 0))
    hair = np.load(ROOT / "artifacts/samples/M_Hair_source.npz")
    result["M_Hair"] = np.full(len(hair["triangles"]), 2, dtype=int)
    return result


def assess(reference, runtime, destination, case_file):
    reference, runtime, destination = map(Path, [reference, runtime, destination])
    destination.mkdir(parents=True, exist_ok=True)
    rows, masks, temporal = [], {}, 0
    for case in json.loads(Path(case_file).read_text()):
        name = case["name"]
        a = np.array(Image.open(reference / (name + "_surfaces.png")).convert("RGBA"))
        b = np.array(Image.open(runtime / (name + ".png")).convert("RGBA"))
        row = {"name": name, "parameters": case["parameters"]}
        overlay = np.uint8(b[:, :, :3] * (b[:, :, 3:] / 255))
        for channel, kind in enumerate(["skin", "hair", "nose"]):
            aa = (a[:, :, channel] > 128) & (a[:, :, 3] > 128)
            bb = (b[:, :, channel] > 128) & (b[:, :, 3] > 128)
            # Hair is assessed only near the *independently visible* face. The
            # original lower-hair clearance is intentionally outside this ROI.
            roi = binary_dilation(binary_fill_holes(a[:, :, 0] > 128), iterations=5)
            le = bb & ~binary_dilation(aa) & roi
            mi = aa & ~binary_dilation(bb) & roi
            aa, bb = aa & roi, bb & roi
            components, _ = label(le)
            sizes = np.bincount(components.ravel())[1:]
            row[kind] = {
                "leak": int(le.sum()),
                "missing": int(mi.sum()),
                "leak_far": int((bb & ~binary_dilation(aa, iterations=3)).sum()),
                "missing_far": int((aa & ~binary_dilation(bb, iterations=3)).sum()),
                "reference": int(aa.sum()),
                "runtime": int(bb.sum()),
                "small_islands": int(((sizes > 0) & (sizes <= 16)).sum()),
            }
            overlay[le] = [255, 50, 30]
            overlay[mi] = [40, 100, 255]
            if kind == "skin":
                masks[name] = (aa, bb)
        Image.fromarray(overlay).save(destination / (name + "_errors.png"))
        rows.append(row)
    sequences = {}
    for case in json.loads(Path(case_file).read_text()):
        if "sequence" in case:
            sequences.setdefault(case["sequence"], []).append(case)
    transitions = 0
    for sequence in sequences.values():
        sequence.sort(key=lambda c: c["frame"])
        for p, q in pairwise(sequence):
            a, b = masks[p["name"]], masks[q["name"]]
            temporal += int(((a[1] ^ b[1]) & ~binary_dilation(a[0] ^ b[0])).sum())
            transitions += 1
    result = {
        "poses": len(rows),
        "summary": {
            k: {m: sum(r[k][m] for r in rows) for m in rows[0][k]}
            for k in ["skin", "hair", "nose"]
        },
        "cases": rows,
        "transitions": transitions,
        "temporal_false_change": temporal,
        "method": "Independent Blender source-ID depth render vs real native diagnostic MOC, original texture alpha retained. 1px tolerance (far: 3px), visible reference face with interior eye/mouth holes filled then dilated 5px as ROI. Pixel counts include projection/interpolation errors; small islands count connected leak components <=16px; finite coverage is not a proof for all real-valued poses.",
    }
    jsonwrite(destination / "metrics.json", result)
    print(json.dumps(result["summary"]), flush=True)
    return result


def assess_teeth(reference, runtime, case_file):
    """POPY-specific white-tooth diagnostic inside independently projected bounds.

    This is a texture/ROI proxy, additional to the source-ID face/hair passes;
    it is not a general semantic segmentation method for arbitrary characters.
    """
    source = np.load(ROOT / "artifacts/samples/M_Face_source.npz")
    _, components = component_labels(source)
    vertices = np.unique(source["triangles"][np.isin(components, [7, 8, 9, 10])])
    camera = json.loads((ROOT / "artifacts/samples/manifest.json").read_text())[
        "camera"
    ]
    rows = []
    for case in json.loads(Path(case_file).read_text()):
        name = case["name"]
        a = np.array(Image.open(Path(reference) / (name + ".png")).convert("RGBA"))
        b = np.array(Image.open(Path(runtime) / (name + ".png")).convert("RGBA"))
        h, w = a.shape[:2]
        points = project(
            np.load(Path(reference) / (name + "__M_Face.npy"))[vertices], camera
        ) * [w, h]
        lo = np.maximum(np.floor(points.min(0) - 2), 0).astype(int)
        hi = np.minimum(np.ceil(points.max(0) + 2), [w, h]).astype(int)
        roi = np.zeros((h, w), bool)
        roi[lo[1] : hi[1], lo[0] : hi[0]] = True

        def white(image, roi=roi):
            rgb = image[:, :, :3].astype(int)
            return (
                (rgb.min(2) >= 248)
                & (np.ptp(rgb, axis=2) <= 7)
                & (image[:, :, 3] > 200)
                & roi
            )

        aa, bb = white(a), white(b)
        rows.append(
            {
                "name": name,
                "missing": int((aa & ~binary_dilation(bb)).sum()),
                "leak": int((bb & ~binary_dilation(aa)).sum()),
                "reference": int(aa.sum()),
                "runtime": int(bb.sum()),
            }
        )
    return {
        "poses": len(rows),
        "summary": {
            k: sum(r[k] for r in rows)
            for k in ["missing", "leak", "reference", "runtime"]
        },
        "cases": rows,
        "method": "POPY-specific white texture threshold minRGB>=248, saturation<=7 and alpha>200 inside an independent Blender projection of the actual original tooth vertices, padded 2px; 1px boundary tolerance. Additional proxy only; nearby white surfaces can confound it.",
    }


def reported_spots(base):
    """Canvas-specific acceptance regions for the attached screenshot defects."""
    result = {}
    for label_name, case, channel, rectangle in [
        ("ear_white_speck", "reported_right", 0, [312, 367, 323, 378]),
        ("cheek_slit", "reported_right", 0, [361, 404, 370, 417]),
        ("hair_near_nose", "reported_right", 1, [473, 403, 481, 414]),
    ]:
        ref = np.array(
            Image.open(base / "reference" / (case + "_surfaces.png")).convert("RGBA")
        )
        expected = (ref[:, :, channel] > 128) & (ref[:, :, 3] > 128)
        row = {"case": case, "rectangle": rectangle}
        x0, y0, x1, y1 = rectangle
        for side in ["before", "after"]:
            im = np.array(
                Image.open(base / ("surfaces_" + side) / (case + ".png")).convert(
                    "RGBA"
                )
            )
            actual = (im[:, :, channel] > 128) & (im[:, :, 3] > 128)
            row[side] = int((actual & ~binary_dilation(expected))[y0:y1, x0:x1].sum())
        result[label_name] = row
    return result


def oral_labels():
    source = np.load(ROOT / "artifacts/samples/M_Face_source.npz")
    _, components = component_labels(source)
    values = np.zeros(len(components), dtype=int)
    values[np.isin(components, [7, 8, 9, 10])] = 1
    values[np.isin(components, [19, 23, 24])] = 2
    values[components == 0] = 3
    return {"M_Face": values}


def assess_oral_ids(reference, runtime, case_file):
    rows = []
    for case in json.loads(Path(case_file).read_text()):
        name = case["name"]
        a = np.array(
            Image.open(Path(reference) / (name + "_surfaces.png")).convert("RGBA")
        )
        b = np.array(Image.open(Path(runtime) / (name + ".png")).convert("RGBA"))
        aa = (a[:, :, 0] > 128) & (a[:, :, 3] > 128)
        bb = (b[:, :, 0] > 128) & (b[:, :, 3] > 128)
        rows.append(
            {
                "name": name,
                "missing": int((aa & ~binary_dilation(bb)).sum()),
                "leak": int((bb & ~binary_dilation(aa)).sum()),
                "missing_far": int((aa & ~binary_dilation(bb, iterations=3)).sum()),
                "leak_far": int((bb & ~binary_dilation(aa, iterations=3)).sum()),
                "reference": int(aa.sum()),
                "runtime": int(bb.sum()),
            }
        )
    return {
        "poses": len(rows),
        "summary": {
            k: sum(r[k] for r in rows)
            for k in [
                "missing",
                "leak",
                "missing_far",
                "leak_far",
                "reference",
                "runtime",
            ]
        },
        "cases": rows,
        "method": "Actual Blender tooth source-ID depth pass versus real diagnostic MOC red channel, original alpha retained, 1px boundary tolerance (far: 3px); no white-texture threshold or projected ROI.",
    }
