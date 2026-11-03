"""Screen complete source pose grids for per-ArtMesh depth conflicts in Rust."""

import hashlib
import itertools
import json
import subprocess
from pathlib import Path

import numpy as np

from .compile_ir import ROOT, jsonwrite, project
from .visibility import component_labels


def screen(rig, destination):
    ir = json.loads(Path(rig).read_text())
    camera = np.array(ir["camera"]["projection"]) @ np.linalg.inv(
        np.array(ir["camera"]["matrix_world"])
    )
    if not np.allclose(camera[3, :3], 0) or np.max(np.abs(camera[[0, 1], 1])) > 1e-5:
        raise ValueError(
            "Source depth screening requires the audited frontal orthographic camera"
        )
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    _, components = component_labels(
        np.load(ROOT / "artifacts/samples/M_Face_source.npz")
    )
    excluded = json.loads((ROOT / "config/surface_visibility.json").read_text())[
        "lashes"
    ]["components"]
    meshes = [
        m
        for m in ir["meshes"]
        if not m.get("mask_only")
        and (
            m["source"]["object"] == "M_Hair"
            or (
                m["source"]["object"] == "M_Face"
                and not np.any(np.isin(components[m["source"]["triangles"]], excluded))
            )
        )
    ]
    manifest = json.loads((ROOT / "artifacts/samples/manifest.json").read_text())
    defaults = {p["id"]: p["default"] for p in ir["parameters"]}
    lookup = [{tuple(c["coordinate"]): c["order"] for c in m["cells"]} for m in meshes]
    records = []
    names = ["M_Face", "M_Hair"]
    neutral = next(
        p
        for p in manifest["head_poses"]
        if all(v == 0 for v in p["parameters"].values())
    )
    archive = np.load(ROOT / "artifacts/samples" / neutral["file"])
    rest = {n: archive[n] for n in names}
    body = {}
    for bx, by in itertools.product([-15, 0, 15], repeat=2):
        data = np.load(ROOT / "artifacts/samples" / f"body_{bx}_{by}.npz")
        a, b = np.radians([bx, by])
        rotation = np.array(
            [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
        ) @ np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
        body[(bx, by)] = ({n: data[n] for n in names}, rotation)
    for pi, pose in enumerate(manifest["head_poses"]):
        archive = np.load(ROOT / "artifacts/samples" / pose["file"])
        head = {n: archive[n] for n in names}
        expressions = [
            {},
            {"ParamMouthOpenY": 1, "ParamMouthForm": 1},
            {
                "ParamEyeLOpen": 0,
                "ParamEyeROpen": 0,
                "ParamEyeLSmile": 1,
                "ParamEyeRSmile": 1,
            },
        ]
        for ei, expression in enumerate(expressions):
            expressed = {n: head[n].copy() for n in names}
            for pid, value in expression.items():
                spec = manifest["semantics"]["shape_parameters"][pid]
                expressed[spec["object"]] += archive[spec["object"] + "__" + pid] * (
                    1 - value if spec.get("invert") else value
                )
            for (bx, by), (base, rotation) in body.items():
                world = {
                    n: base[n] + (expressed[n] - rest[n]) @ rotation.T for n in names
                }
                projected = {
                    name: np.c_[
                        project(world[name], ir["camera"]) * ir["canvas"],
                        world[name][:, 1],
                    ]
                    for name in names
                }
                values = {
                    **defaults,
                    **pose["parameters"],
                    **expression,
                    "ParamBodyAngleX": bx,
                    "ParamBodyAngleY": by,
                }
                triangles = []
                for mi, m in enumerate(meshes):
                    coord = tuple(
                        a["keys"].index(values[a["parameter"]]) for a in m["axes"]
                    )
                    t = np.array(m["indices"]).reshape(-1, 3)
                    p = projected[m["source"]["object"]][m["source"]["vertices"]][
                        t
                    ].reshape(-1, 9)
                    triangles.append(
                        np.c_[
                            p,
                            np.full(len(t), lookup[mi][coord]),
                            np.full(len(t), mi),
                            m["source"]["triangles"],
                        ]
                    )
                filename = f"screen_{pi:03}_{bx}_{by}_e{ei}.f32"
                np.concatenate(triangles).astype("<f4").tofile(destination / filename)
                records.append({"name": filename, "file": filename})
    jsonwrite(
        destination / "input.json",
        {
            "width": ir["canvas"][0],
            "height": ir["canvas"][1],
            "meshes": [
                {"id": m["id"], "kind": int(m["source"]["object"] == "M_Hair")}
                for m in meshes
            ],
            "cases": records,
        },
    )
    subprocess.run(
        [
            ROOT / "rust/target/release/face_conflicts.exe",
            destination / "input.json",
            destination / "conflicts.json",
        ],
        check=True,
    )
    result = json.loads((destination / "conflicts.json").read_text())
    result["source_ir_sha256"] = hashlib.sha256(Path(rig).read_bytes()).hexdigest()
    result["samples_manifest_sha256"] = hashlib.sha256(
        (ROOT / "artifacts/samples/manifest.json").read_bytes()
    ).hexdigest()
    jsonwrite(destination / "conflicts.json", result)
    print(
        sorted(result["meshes"], key=lambda m: m["wrong_cover"], reverse=True)[:25],
        flush=True,
    )
    return result


if __name__ == "__main__":
    screen(
        ROOT / "artifacts/neck_repair/rig/POPY.rig.json",
        ROOT / "artifacts/face_repair/screen",
    )
