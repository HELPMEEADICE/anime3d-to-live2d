"""List original-surface triangles covering one evaluated runtime pixel."""

import argparse
import json
from pathlib import Path

import numpy as np

from .compile_ir import ROOT
from .visibility import component_labels


def probe(rig, geometry, x, y):
    ir = json.loads(Path(rig).read_text())
    evaluated = {m["id"]: m for m in json.loads(Path(geometry).read_text())}
    w, h = ir["canvas"]
    sources = {}
    hits = []
    for mesh in ir["meshes"]:
        g = evaluated[mesh["id"]]
        if mesh.get("mask_only") or not g["visible"] or g["opacity"] < 0.01:
            continue
        points = np.array(g["vertices"]) * (w / 2) + [w / 2, h / 2]
        for ti, tri in enumerate(np.array(mesh["indices"]).reshape(-1, 3)):
            a, b, c = points[tri]
            transform = np.array([a - c, b - c]).T
            area = np.linalg.det(transform)
            if area <= 1e-8:
                continue
            u, v = np.linalg.solve(transform, [x, y] - c)
            if u >= 0 and v >= 0 and u + v <= 1:
                name = mesh["source"]["object"]
                if name not in sources:
                    sources[name] = component_labels(
                        np.load(ROOT / "artifacts/samples" / f"{name}_source.npz")
                    )[1]
                source_tri = mesh["source"]["triangles"][ti]
                hits.append(
                    {
                        "id": mesh["id"],
                        "order": g["depth"],
                        "component": int(sources[name][source_tri]),
                        "source_triangle": source_tri,
                    }
                )
    print(json.dumps(sorted(hits, key=lambda hit: hit["order"]), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("rig")
    p.add_argument("geometry")
    p.add_argument("x", type=float)
    p.add_argument("y", type=float)
    a = p.parse_args()
    probe(a.rig, a.geometry, a.x, a.y)
