"""Prevent skinned neck patches from drawing through the source collar.

Native inverse masks retain source cloth alpha and parameterized 3D depth
intersections. Neither visible geometry nor global draw-order is replaced.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from .compile_ir import ROOT, jsonwrite
from .form_storage import compact_forms
from .neck_masks import build_depth_masks


def apply_collar_clearance(ir, samples, binary):
    if ir["diagnostics"].get("collar_clearance"):
        return
    config = json.loads((ROOT / "config/collar_clearance.json").read_text())
    camera = np.array(ir["camera"]["projection"]) @ np.linalg.inv(
        np.array(ir["camera"]["matrix_world"])
    )
    if not np.allclose(camera[3, :3], 0) or np.max(np.abs(camera[[0, 1], 1])) > 1e-5:
        raise ValueError(
            "Collar support planes require the audited frontal orthographic camera"
        )
    meshes = [m for m in ir["meshes"] if not m.get("mask_only")]
    names = {config["target_object"], *config["occluder_objects"]}
    rest = {}
    for name in names:
        with np.load(samples / (name + "_source.npz")) as source:
            rest[name] = source["positions"]
    targets, occluders = [], []
    for mesh in meshes:
        name = mesh["source"]["object"]
        if name not in rest:
            continue
        points = rest[name][mesh["source"]["vertices"]]
        if name == config["target_object"]:
            zlo, zhi = config["target_rest_z"]
            if (
                points[:, 2].min() >= zlo
                and points[:, 2].max() <= zhi
                and np.abs(points[:, 0]).max() <= config["target_rest_abs_x"]
            ):
                targets.append(mesh)
        elif points[:, 2].max() >= config["occluder_min_rest_z"]:
            occluders.append(mesh)
    if not targets or not occluders:
        raise ValueError("No source neck/collar geometry matches the configured region")
    manifest = json.loads((samples / "manifest.json").read_text())
    build_depth_masks(
        ir,
        samples,
        targets,
        occluders,
        binary,
        manifest["head_poses"],
        "Collar",
        "collar_clearance",
        "collar_clearance",
        merge_masks=True,
    )
    ir["diagnostics"]["collar_clearance"]["config"] = config
    for part in ir["parts"]:
        part["mesh_ids"] = [
            m["id"]
            for m in ir["meshes"]
            if m["source"]["object"] == part["source_object"]
        ]


def convert(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source.parent == destination:
        raise ValueError("Choose a separate output directory to preserve the baseline")
    destination.mkdir(parents=True, exist_ok=True)
    ir = json.loads(source.read_text())
    for name in ir["textures"]:
        shutil.copy2(source.parent / name, destination / Path(name).name)
    ir["textures"] = [Path(name).name for name in ir["textures"]]
    binary = destination / "keyforms.f32"
    shutil.copy2(source.parent / ir["keyform_data"], binary)
    ir["keyform_data"] = binary.name
    apply_collar_clearance(ir, ROOT / "artifacts/samples", binary)
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
    return destination / "POPY.rig.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "artifacts/hair_clearance_rig/POPY.rig.json",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/neck_repair/rig")
    args = parser.parse_args()
    convert(args.source, args.out)
