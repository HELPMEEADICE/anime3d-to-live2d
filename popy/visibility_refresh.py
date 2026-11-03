"""Rebuild native masks from an existing sampled rig, retaining its geometry/order fit."""

import argparse
import json
from pathlib import Path

from .compile_ir import ROOT, jsonwrite
from .visibility import protect_eyes, protect_surfaces, weld_sources


def refresh(folder):
    folder = Path(folder)
    path = folder / "POPY.rig.json"
    ir = json.loads(path.read_text())
    ir["meshes"] = [m for m in ir["meshes"] if not m.get("mask_only")]
    ir["glues"] = []
    for mesh in ir["meshes"]:
        mesh["masked_by"] = []
        mesh["invert_mask"] = False
    binary = folder / ir["keyform_data"]
    semantics = json.loads((ROOT / "config/semantics.json").read_text())
    from .linework_partition import partition_linework
    partition_linework(ir, ROOT / "artifacts/samples", semantics, binary)
    protect_eyes(
        ir,
        ROOT / "artifacts/samples",
        json.loads((ROOT / "config/semantics.json").read_text()),
        binary,
    )
    protect_surfaces(ir, ROOT / "artifacts/samples", json.loads((ROOT / "config/semantics.json").read_text()), binary)
    weld_sources(ir, ROOT / "artifacts/samples")
    # Remove superseded helpers' storage. Each keyform offset counts f32 values.
    from .form_storage import compact_forms
    packed_bytes=compact_forms(ir,binary)
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
        keyform_bytes=packed_bytes,
        mask_artmesh_count=sum(bool(m.get("mask_only")) for m in ir["meshes"]),
        visible_artmesh_count=sum(not m.get("mask_only", False) for m in ir["meshes"]),
        glues=len(ir.get("glues", [])),
        animated_glues=sum(bool(g.get("axes")) for g in ir.get("glues", [])),
        glue_keyforms=sum(len(g.get('intensities',g.get('cells',[None]))) for g in ir.get('glues',[])),
        masked_drawables=sum(bool(m.get("masked_by")) for m in ir["meshes"]),
    )
    jsonwrite(path, ir)
    jsonwrite(folder / "statistics.json", ir["diagnostics"])
    print("Native masks refreshed", path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rig-dir", type=Path, default=ROOT / "artifacts/rig")
    refresh(p.parse_args().rig_dir)
