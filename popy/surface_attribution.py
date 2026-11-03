"""Attribute measured visibility residuals to the original source triangles."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from .compile_ir import ROOT, jsonwrite
from .surface_regression import labels


def attribute(rig, runtime, surfaces, metrics, destination):
    ir = json.loads(Path(rig).read_text())
    rows = json.loads(Path(metrics).read_text())["cases"]
    classes = labels()
    results = {}
    for channel, kind in enumerate(["brows", "lashes", "neck_hair"]):
        meshes = [
            m
            for m in ir["meshes"]
            if not m.get("mask_only")
            and m["source"]["object"] in classes
            and np.all(
                classes[m["source"]["object"]][m["source"]["triangles"]] == channel + 1
            )
        ]
        scores = Counter()
        triangle_scores = Counter()
        all_covering = Counter()
        for case in rows:
            if case["name"].startswith("continuous_"):
                continue
            if not case[kind]["leak"]:
                continue
            p = case["parameters"]
            if kind != "neck_hair" and any(
                p.get(k, 0) not in keys
                for k, keys in {
                    "ParamAngleX": [-45, -30, -15, 0, 15, 30, 45],
                    "ParamAngleY": [-30, -15, 0, 15, 30],
                    "ParamAngleZ": [-30, 0, 30],
                    "ParamBodyAngleX": [-15, 0, 15],
                    "ParamBodyAngleY": [-15, 0, 15],
                }.items()
            ):
                continue
            n = case["name"]
            a = np.array(
                Image.open(ROOT / "artifacts/surface_reference" / f"{n}_surfaces.png")
            )
            b = np.array(Image.open(Path(surfaces) / f"{n}.png"))
            error = (b[:, :, channel] > 128) & ~binary_dilation(
                a[:, :, channel] > 128, iterations=3 if kind != "neck_hair" else 1
            )
            if kind == "neck_hair":
                x0, y0, x1, y1 = case["neck_roi"]
                error[:y0] = 0
                error[y1:] = 0
                error[:, :x0] = 0
                error[:, x1:] = 0
            yy, xx = np.where(error)
            if not len(xx):
                continue
            geometry = {
                m["id"]: m
                for m in json.loads((Path(runtime) / f"{n}.geometry.json").read_text())
            }
            owners = np.full(len(xx), -1)
            owner_tri = np.full(len(xx), -1)
            depth = np.full(len(xx), -1.0)
            for i, mesh in enumerate(meshes):
                state = geometry[mesh["id"]]
                v = (
                    np.array(state["vertices"]) * (ir["canvas"][0] / 2)
                    + np.array(ir["canvas"]) / 2
                )
                low, high = v.min(0), v.max(0)
                ix = np.where(
                    (xx >= low[0]) & (xx <= high[0]) & (yy >= low[1]) & (yy <= high[1])
                )[0]
                if not len(ix):
                    continue
                hit = np.zeros(len(ix), bool)
                tri_hit = np.full(len(ix), -1)
                for ti, tri in enumerate(np.array(mesh["indices"]).reshape(-1, 3)):
                    c, e, f = v[tri]
                    u, e2 = e - c, f - c
                    det = u[0] * e2[1] - u[1] * e2[0]
                    if det <= 1e-10:
                        continue
                    q = np.c_[xx[ix] + 0.5, yy[ix] + 0.5] - c
                    s = (q[:, 0] * e2[1] - q[:, 1] * e2[0]) / det
                    t = (u[0] * q[:, 1] - u[1] * q[:, 0]) / det
                    inside = (s >= 0) & (t >= 0) & (s + t <= 1)
                    if inside.any():
                        all_covering[
                            (mesh["source"]["object"], mesh["source"]["triangles"][ti])
                        ] += int(inside.sum())
                    hit |= inside
                    tri_hit[inside] = mesh["source"]["triangles"][ti]
                replace = hit & (state["depth"] >= depth[ix])
                owners[ix[replace]] = i
                owner_tri[ix[replace]] = tri_hit[replace]
                depth[ix[replace]] = state["depth"]
            scores.update(meshes[i]["id"] for i in owners if i >= 0)
            triangle_scores.update(
                (meshes[i]["source"]["object"], int(t))
                for i, t in zip(owners, owner_tri)
                if i >= 0 and t >= 0
            )
        results[kind] = [
            {
                "mesh": m,
                "pixels": int(count),
                "object": next(x for x in meshes if x["id"] == m)["source"]["object"],
                "triangles": next(x for x in meshes if x["id"] == m)["source"][
                    "triangles"
                ],
            }
            for m, count in scores.most_common()
        ]
        results[kind + "_triangles"] = [
            {"object": n, "triangle": t, "pixels": int(count)}
            for (n, t), count in triangle_scores.most_common()
        ]
        results[kind + "_all_covering"] = [
            {"object": n, "triangle": t, "pixels": int(count)}
            for (n, t), count in all_covering.most_common()
        ]
    result = {
        "source_sha256": {
            n: hashlib.sha256(
                (ROOT / "artifacts/samples" / f"{n}_source.npz").read_bytes()
            ).hexdigest()
            for n in classes
        },
        "attribution": results,
        "method": "3px residual on exact head/body grid for linework; 1px neck ROI residual, native topmost triangle ownership; no image-derived topology",
    }
    jsonwrite(Path(destination), result)
    print(
        {
            k: [(r.get("mesh", r.get("triangle")), r["pixels"]) for r in rows[:12]]
            for k, rows in results.items()
        },
        flush=True,
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("rig")
    p.add_argument("runtime")
    p.add_argument("surfaces")
    p.add_argument("metrics")
    p.add_argument("destination")
    a = p.parse_args()
    attribute(a.rig, a.runtime, a.surfaces, a.metrics, a.destination)
