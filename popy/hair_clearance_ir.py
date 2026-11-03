"""Compile source-strand clearance into rear-hair/outfit native masking.

Only rear lower-hair clipping changes. Front hair, face, eyes, headwear, all
geometry and all draw-order tracks retain the previously validated rig values.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from .compile_ir import ROOT, jsonwrite
from .form_storage import compact_forms
from .hair_semantics import policy, strand_sides


def apply_clearance(ir, samples):
    if ir["diagnostics"].get("hair_clearance_routing"):
        return
    config = policy()
    matrix = np.array(ir["camera"]["projection"]) @ np.linalg.inv(
        np.array(ir["camera"]["matrix_world"])
    )
    if not np.allclose(matrix[3, :3], 0) or np.max(np.abs(matrix[[0, 1], 1])) > 1e-5:
        raise ValueError(
            "Hair depth routing requires the audited orthographic frontal camera"
        )
    with np.load(samples / "M_Hair_source.npz") as source:
        positions = source["positions"]
        sides = strand_sides(
            positions, source["triangles"], config["strand_classification_below_z"]
        )
    outfit = set(config["outfit_mask_objects"])
    masks = [
        m["id"]
        for m in ir["meshes"]
        if not m.get("mask_only") and m["source"]["object"] in outfit
    ]
    rear = []
    front_count = 0
    for mesh in ir["meshes"]:
        if mesh.get("mask_only") or mesh["source"]["object"] != "M_Hair":
            continue
        ids = mesh["source"]["vertices"]
        side = np.unique(sides[ids])
        if len(side) != 1:
            raise ValueError("ArtMesh spans opposite collision routes: " + mesh["id"])
        if positions[ids, 2].min() >= config["lower_patch_min_z"]:
            continue
        if side[0] > 0:
            rear.append(mesh["id"])
            mesh["masked_by"] = masks.copy()
            mesh["invert_mask"] = True
        else:
            front_count += 1
    # Retain existing front-tip masks too: removing them can change chin/face
    # coverage at compounded poses even without changing any order keyforms.
    removed = {
        m["id"]
        for m in ir["meshes"]
        if m["id"].startswith("MaskNeckDepth_")
        and m["source"].get("depth_target") in rear
    }
    ir["meshes"] = [m for m in ir["meshes"] if m["id"] not in removed]
    ir["glues"] = [
        g
        for g in ir["glues"]
        if g.get("mesh_a") not in removed and g.get("mesh_b") not in removed
    ]
    ir["diagnostics"]["hair_clearance_routing"] = {
        "rear": len(rear),
        "front_preserved": front_count,
        "front_reordered": 0,
        "removed_rear_helpers": len(removed),
        "outfit_mask_sources": len(masks),
        "source_sha256": hashlib.sha256(
            (samples / "M_Hair_source.npz").read_bytes()
        ).hexdigest(),
        "policy": config,
        "method": "Connected source rear strands use one shared outfit inverted alpha mask; front masks, head-feature ordering, all geometry and all draw-order tracks retained",
        "limitations": "View-specific conservative approximation of the Blender closed-envelope clearance; compounded head/body interpolation and grazing contacts remain; no hair self collision",
    }
    ir["surface_visibility"]["neck_depth"] = [
        d
        for d in ir["surface_visibility"].get("neck_depth", [])
        if d["target"] not in rear
    ]
    retained_helpers = [m for m in ir["meshes"] if m["id"].startswith("MaskNeckDepth_")]
    ir["diagnostics"]["neck_depth_masks"] = {
        "retained_front_helpers": len(retained_helpers),
        "removed_rear_helpers": len(removed),
        "method": "Original front-tip depth masks retained; rear targets superseded by shared outfit clipping",
    }
    ir["parts"] = [
        {
            **p,
            "mesh_ids": [
                m["id"]
                for m in ir["meshes"]
                if m["source"]["object"] == p["source_object"]
            ],
        }
        for p in ir["parts"]
    ]


def convert(source, destination):
    if source.parent.resolve() == destination.resolve():
        raise ValueError(
            "Choose a separate output directory; the baseline rig is retained"
        )
    destination.mkdir(parents=True, exist_ok=True)
    ir = json.loads(source.read_text())
    for name in ir["textures"]:
        shutil.copy2(source.parent / name, destination / Path(name).name)
    ir["textures"] = [Path(t).name for t in ir["textures"]]
    binary = destination / "keyforms.f32"
    shutil.copy2(source.parent / ir["keyform_data"], binary)
    ir["keyform_data"] = binary.name
    apply_clearance(ir, ROOT / "artifacts/samples")
    ir["diagnostics"]["keyform_bytes"] = compact_forms(ir, binary)
    ir["diagnostics"].update(
        artmesh_count=len(ir["meshes"]),
        vertices=sum(len(m["positions"]) // 2 for m in ir["meshes"]),
        triangles=sum(len(m["indices"]) // 3 for m in ir["meshes"]),
        glues=len(ir["glues"]),
        animated_glues=sum(bool(g.get("axes")) for g in ir["glues"]),
        masked_drawables=sum(bool(m.get("masked_by")) for m in ir["meshes"]),
    )
    jsonwrite(destination / "POPY.rig.json", ir)
    jsonwrite(destination / "statistics.json", ir["diagnostics"])
    print(ir["diagnostics"]["hair_clearance_routing"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=ROOT / "artifacts/rig/POPY.rig.json")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/hair_clearance_rig")
    a = p.parse_args()
    convert(a.source.resolve(), a.out.resolve())
