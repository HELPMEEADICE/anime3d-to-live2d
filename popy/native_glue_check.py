"""Cross-check exported/evaluated native mask Glue against independent IR interpolation."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .compile_ir import ROOT, jsonwrite


def check(rig, runtime, destination, case_file=ROOT / "artifacts/surface_cases.json"):
    ir = json.loads(Path(rig).read_text())
    runtime = Path(runtime)
    targets = {
        m["id"]
        for m in ir["meshes"]
        if m.get("mask_only") and "barycentric_sources" in m["source"]
    }
    inventory = runtime / "model_inventory.json"
    if inventory.exists():
        decoded = {m["id"]: m for m in json.loads(inventory.read_text())}
        for mesh in ir["meshes"]:
            entry = decoded[mesh["id"]]
            assert entry["indices"] == mesh["indices"], (
                "Native topology mismatch: " + mesh["id"]
            )
            assert set(entry["clips"]) == set(mesh.get("masked_by", [])), (
                "Native clipping references mismatch: " + mesh["id"]
            )
            assert entry["vertices"] == len(mesh["positions"]) // 2
            assert entry["culling"] == mesh.get("culling", True), (
                "Native culling flag mismatch: " + mesh["id"]
            )
    cases = {
        r["name"]: r["parameters"] for r in json.loads(Path(case_file).read_text())
    }
    defaults = {p["id"]: p["default"] for p in ir["parameters"]}
    results = []
    for name in ["front", "all_extreme", "eye_45_-30_open", "continuous_blink_-45_013"]:
        parameters = {**defaults, **cases[name]}
        actual = {
            m["id"]: np.array(m["vertices"])
            for m in json.loads((runtime / f"{name}.geometry.json").read_text())
        }
        expected = {k: np.zeros_like(actual[k]) for k in targets}
        for g in ir["glues"]:
            if not g["id"].startswith("G_Depth_"):
                continue
            intensity = g.get("intensity", 1)
            if g.get("axes"):
                data = np.array(g["intensities"]).reshape(
                    tuple(len(a["keys"]) for a in g["axes"])
                )
                brackets = []
                for a in g["axes"]:
                    keys = a["keys"]
                    v = parameters[a["parameter"]]
                    i = max(
                        0,
                        min(len(keys) - 2, np.searchsorted(keys, v, side="right") - 1),
                    )
                    t = np.clip((v - keys[i]) / (keys[i + 1] - keys[i]), 0, 1)
                    brackets.append([(i, 1 - t), (i + 1, t)])
                intensity = sum(
                    data[tuple(i for i, w in c)] * np.prod([w for i, w in c])
                    for c in itertools.product(*brackets)
                )
            pairs = np.array(g["pairs"])
            a = pairs[:, 0].astype(int)
            b = pairs[:, 1].astype(int)
            assert np.all(pairs[:, 2] == 0), (
                "Mask Glue must never move a source surface"
            )
            result = expected[g["mesh_b"]]
            source = actual[g["mesh_a"]]
            result[b] += (source[a] - result[b]) * (pairs[:, 3] * intensity)[:, None]
        errors = np.concatenate(
            [
                np.linalg.norm(expected[k] - actual[k], axis=1) * (ir["canvas"][0] / 2)
                for k in targets
            ]
        )
        results.append(
            {
                "name": name,
                "max_px": float(errors.max()),
                "rmse_px": float(np.sqrt(np.mean(errors**2))),
            }
        )
    report = {
        "method": "Independent IR scalar-grid interpolation and ordered barycentric lerps compared with actual pinned Ayagami-evaluated MOC vertices, after original source welds",
        "cases": results,
        "max_px": max(r["max_px"] for r in results),
        "helpers": len(targets),
        "native_topology_and_masks_checked": inventory.exists(),
    }
    jsonwrite(Path(destination), report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("rig")
    p.add_argument("runtime")
    p.add_argument("destination")
    a = p.parse_args()
    check(a.rig, a.runtime, a.destination)
