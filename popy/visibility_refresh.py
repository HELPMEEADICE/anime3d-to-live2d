"""Rebuild native masks from an existing sampled rig, retaining its geometry/order fit."""

import argparse
import json
from pathlib import Path

import numpy as np

from .compile_ir import ROOT, jsonwrite
from .visibility import protect_eyes


def refresh(folder):
    folder = Path(folder)
    path = folder / "POPY.rig.json"
    ir = json.loads(path.read_text())
    ir["meshes"] = [m for m in ir["meshes"] if not m.get("mask_only")]
    binary = folder / ir["keyform_data"]
    protect_eyes(
        ir,
        ROOT / "artifacts/samples",
        json.loads((ROOT / "config/semantics.json").read_text()),
        binary,
    )
    # Remove superseded helpers' storage. Each keyform offset counts f32 values.
    data = np.fromfile(binary, "<f4")
    compact = bytearray()
    offsets = {}

    def move(offset, size):
        if offset is None:
            return None
        key = (offset, size)
        if key not in offsets:
            offsets[key] = len(compact) // 4
            compact.extend(data[offset : offset + size].tobytes())
        return offsets[key]

    for mesh in ir["meshes"]:
        size = len(mesh["positions"])
        for cell in mesh["cells"]:
            cell["offset"] = move(cell["offset"], size)
        for blend in mesh["blend_shapes"]:
            blend["offsets"] = [move(o, size) for o in blend["offsets"]]
    binary.write_bytes(compact)
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
    ir["diagnostics"].update(
        artmesh_count=len(ir["meshes"]),
        vertices=sum(len(m["positions"]) // 2 for m in ir["meshes"]),
        triangles=sum(len(m["indices"]) // 3 for m in ir["meshes"]),
        authored_keyforms=sum(len(m["cells"]) for m in ir["meshes"]),
        keyform_bytes=len(compact),
        mask_artmesh_count=sum(bool(m.get("mask_only")) for m in ir["meshes"]),
        visible_artmesh_count=sum(not m.get("mask_only", False) for m in ir["meshes"]),
    )
    jsonwrite(path, ir)
    jsonwrite(folder / "statistics.json", ir["diagnostics"])
    print("Native masks refreshed", path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rig-dir", type=Path, default=ROOT / "artifacts/rig")
    refresh(p.parse_args().rig_dir)
