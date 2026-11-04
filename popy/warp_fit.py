"""Fit real sampled geometry to bilinear spatial deformers; retain rejected grids.

This is a numeric admission experiment, not a claim of Cubism compatibility.
Residuals are measured at every stored corner/pose, rather than silhouettes.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from .compile_ir import ROOT, jsonwrite
from .glue_arena import initial_overwrites


def fit(points, poses):
    points = np.asarray(points, dtype=np.float64)
    poses = np.asarray(poses, dtype=np.float64)
    lo, hi = points.min(0), points.max(0)
    if np.any(hi - lo < 1e-9):
        return None
    uv = (points - lo) / (hi - lo)
    u, v = uv.T
    basis = np.column_stack([(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v])
    if np.linalg.matrix_rank(basis) < 4:
        return None
    lattice = np.einsum("kv,pvd->pkd", np.linalg.pinv(basis), poses).astype("f4")
    predicted = np.einsum("vk,pkd->pvd", basis, lattice)
    error = np.linalg.norm(predicted - poses, axis=2)
    return uv, lattice, error


def audit(source, output, budget_px=0.01):
    ir = json.loads(Path(source).read_text())
    data = np.memmap(Path(source).parent / ir["keyform_data"], "<f4", mode="r")
    killed = initial_overwrites(ir["glues"])
    rows = []
    for m in ir["meshes"]:
        n = len(m["positions"]) // 2
        if n < 8 or len(m["cells"]) < 2 or m.get("blend_shapes"):
            continue
        if all(killed.get((m["id"], i), False) for i in range(n)):
            continue
        offsets = np.array([c["offset"] for c in m["cells"]])
        poses = data[offsets[:, None] + np.arange(2 * n)].reshape(-1, n, 2)
        result = fit(poses[0], poses)
        if result is None:
            continue
        _uv, lattice, error = result
        # Conservative proxy only; admission still needs actual parent-Jacobian
        # and native image checks. No untested fit replaces the source rig.
        maximum = float(error.max()) * max(ir["canvas"]) * 4
        rows.append(
            {
                "id": m["id"],
                "vertices": n,
                "poses": len(poses),
                "position_floats_before": int(poses.size),
                "lattice_floats": int(lattice.size),
                "conservative_screen_proxy_px": maximum,
                "numeric_candidate": maximum <= budget_px,
            }
        )
    report = {
        "source": str(source),
        "budget_px": budget_px,
        "fits": rows,
        "candidates": sum(r["numeric_candidate"] for r in rows),
        "tested_meshes": len(rows),
        "candidate_float_savings": sum(
            r["position_floats_before"] - r["lattice_floats"]
            for r in rows
            if r["numeric_candidate"]
        ),
        "status": "Admission audit only; parent Jacobian, flips, held-out poses and rendered quality still required. No source geometry or Glue removed.",
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    jsonwrite(output, report)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source",
        type=Path,
        default=ROOT / "artifacts/representation_final/rig/POPY.rig.json",
    )
    p.add_argument(
        "--out", type=Path, default=ROOT / "artifacts/deformer_fit/audit.json"
    )
    p.add_argument("--emit", type=Path)
    a = p.parse_args()
    r = audit(a.source, a.out)
    if a.emit:
        emit(a.source, a.emit, r)
    print({k: v for k, v in r.items() if k != "fits"})


def emit(source, destination, report):
    source, destination = Path(source), Path(destination)
    if source.parent.resolve() == destination.resolve():
        raise ValueError("Separate candidate required")
    ir = json.loads(source.read_text())
    data = np.memmap(source.parent / ir["keyform_data"], "<f4", mode="r")
    destination.mkdir(parents=True, exist_ok=True)
    for name in [ir["keyform_data"], *ir["textures"]]:
        shutil.copy2(source.parent / name, destination / name)
    selected = {r["id"] for r in report["fits"] if r["numeric_candidate"]}
    owners = {n: p["id"] for p in ir["parts"] for n in p["mesh_ids"]}
    with (destination / ir["keyform_data"]).open("ab") as out:
        for m in ir["meshes"]:
            if m["id"] not in selected:
                continue
            size = len(m["positions"])
            offsets = np.array([c["offset"] for c in m["cells"]])
            poses = data[offsets[:, None] + np.arange(size)].reshape(-1, size // 2, 2)
            uv, lattice, _ = fit(poses[0], poses)
            parent = (
                "D_HeadProjection"
                if m.get("deformer") == "head"
                else m.get("projection_warp", "D_GlobalProjection")
            )
            name = "D_Fitted_" + m["id"]
            part = "P_Fitted_" + m["id"]
            cells = m["cells"]
            axes = m["axes"]
            ir["body_warps"].append(
                {
                    "id": name,
                    "parent": parent,
                    "strength": 1.0,
                    "axes": axes,
                    "cells": [
                        {"coordinate": c["coordinate"], "points": p.ravel().tolist()}
                        for c, p in zip(cells, lattice)
                    ],
                }
            )
            ir["order_groups"].append(
                {
                    "id": part,
                    "group_id": "G_" + part,
                    "org_parent": owners[m["id"]],
                    "mesh_ids": [m["id"]],
                    "axes": axes,
                    "cells": [
                        {"coordinate": c["coordinate"], "order": c["order"]}
                        for c in cells
                    ],
                }
            )
            off = out.tell() // 4
            out.write(uv.astype("<f4").tobytes())
            m["axes"] = []
            m["cells"] = [{"coordinate": [], "offset": off, "order": 500.0}]
            m["deformer"] = "body"
            m["projection_warp"] = name
    ir["diagnostics"]["spatial_warp_fit"] = report
    jsonwrite(destination / "POPY.rig.json", ir)


if __name__ == "__main__":
    main()
