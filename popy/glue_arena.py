"""Pack dead-initial-geometry registers to expose cross-mesh Glue batching."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from .compile_ir import ROOT, jsonwrite
from .performance_opt import compact_glues, factor_native_constants


def initial_overwrites(glues):
    first = {}
    for g in glues:
        one = not g.get("axes") and g.get("intensity", 1) == 1
        for ia, ib, wa, wb in g["pairs"]:
            a, b = (g["mesh_a"], ia), (g["mesh_b"], ib)
            first.setdefault(a, one and wa == 1 and wb == 0 and a != b)
            first.setdefault(b, one and wb == 1 and wa == 0 and a != b)
    return first


def assign_arenas(meshes, glues, capacity=60000):
    first = initial_overwrites(glues)
    counts = {m["id"]: len(m["positions"]) // 2 for m in meshes}
    selected = {
        name
        for name, count in counts.items()
        if 3 <= count <= capacity
        and all(first.get((name, i), False) for i in range(count))
    }
    neighbors = {name: set() for name in selected}
    for g in glues:
        a, b = g["mesh_a"], g["mesh_b"]
        if a in selected and b in selected:
            if a == b:
                raise ValueError("In-place Glue is not supported by pinned runtime")
            neighbors[a].add(b)
            neighbors[b].add(a)
    bins, mapping = [], {}
    for name in sorted(selected, key=lambda n: (-len(neighbors[n]), -counts[n], n)):
        for i, group in enumerate(bins):
            if group["vertices"] + counts[name] <= capacity and not group[
                "members"
            ].intersection(neighbors[name]):
                break
        else:
            i = len(bins)
            bins.append({"id": f"GlueArena_{i}", "vertices": 0, "members": set()})
        group = bins[i]
        mapping[name] = {
            "arena": group["id"],
            "start": group["vertices"],
            "vertices": counts[name],
        }
        group["vertices"] += counts[name]
        group["members"].add(name)
    for g in glues:
        a = mapping.get(g["mesh_a"], {"arena": g["mesh_a"]})["arena"]
        b = mapping.get(g["mesh_b"], {"arena": g["mesh_b"]})["arena"]
        if a == b:
            raise ValueError("Arena assignment merged Glue endpoints")
    return bins, mapping


def compile_rig(source, output, precision_native=False):
    source, output = Path(source), Path(output)
    if source.parent.resolve() == output.resolve():
        raise ValueError("Separate output required")
    ir = json.loads(source.read_text())
    factored_constants = 0
    if precision_native:
        ir["glues"], factored_constants = factor_native_constants(ir["glues"])
    groups, mapping = assign_arenas(ir["meshes"], ir["glues"])
    print(f"Assigned {len(mapping)} registers to {len(groups)} arenas", flush=True)
    rewritten = []
    for g in ir["glues"]:
        a = mapping.get(g["mesh_a"], {"arena": g["mesh_a"], "start": 0})
        b = mapping.get(g["mesh_b"], {"arena": g["mesh_b"], "start": 0})
        rewritten.append(
            {
                **g,
                "mesh_a": a["arena"],
                "mesh_b": b["arena"],
                "pairs": [
                    [ia + a["start"], ib + b["start"], wa, wb]
                    for ia, ib, wa, wb in g["pairs"]
                ],
            }
        )
    del ir["glues"]
    optimized, diagnostics = compact_glues(rewritten)
    del rewritten
    print(json.dumps(diagnostics), flush=True)
    copies = []
    for name, m in mapping.items():
        copies.append(
            {
                "id": "G_ArenaOutput_" + name,
                "mesh_a": m["arena"],
                "mesh_b": name,
                "pairs": [[m["start"] + i, i, 0.0, 1.0] for i in range(m["vertices"])],
                "intensity": 1.0,
            }
        )
    ir["glues"] = optimized + copies
    output.mkdir(parents=True, exist_ok=True)
    for name in [ir["keyform_data"], *ir["textures"]]:
        shutil.copy2(source.parent / name, output / name)
    binary = output / ir["keyform_data"]
    offset = binary.stat().st_size // 4
    with binary.open("ab") as stream:
        for group in groups:
            count = group["vertices"]
            stream.write(np.zeros(2 * count, dtype="<f4").tobytes())
            ir["meshes"].append(
                {
                    "id": group["id"],
                    "positions": [0.0] * (2 * count),
                    "uvs": [0.0] * (2 * count),
                    "indices": [0, 1, 2],
                    "texture": 0,
                    "deformer": "body",
                    "projection_warp": "D_NativeSkeletonProjection",
                    "axes": [],
                    "cells": [{"coordinate": [], "offset": offset, "order": 0.0}],
                    "blend_shapes": [],
                    "opacity": 0.0,
                    "mask_only": True,
                    "culling": False,
                    "masked_by": [],
                    "invert_mask": False,
                    "source": {
                        "object": "GlueArena",
                        "vertices": list(range(count)),
                        "triangles": [],
                        "correspondence": "Computation registers; original output meshes and their source provenance retained",
                    },
                }
            )
            offset += 2 * count
    ir["parts"].append(
        {
            "id": "GlueArenas",
            "source_object": "GlueArena",
            "mesh_ids": [g["id"] for g in groups],
        }
    )
    plan = {
        "native_constants_factored": factored_constants,
        "mapping": mapping,
        "added_auxiliary_mesh_ids": [g["id"] for g in groups],
        "arena_vertices": sum(g["vertices"] for g in groups),
        "arena_count": len(groups),
        "input_glues": diagnostics["input_glues"],
        "batched_glues": diagnostics["output_glues"],
        "output_copy_glues": len(copies),
        "final_glues": len(ir["glues"]),
        "input_pairs": diagnostics["input_pairs"],
        "output_pairs": sum(len(g["pairs"]) for g in ir["glues"]),
        "method": "Graph-colored bounded register arenas, original per-vertex arithmetic DAG retained, field/endpoint batching, final canonical copies to untouched original mesh identities",
    }
    ir["diagnostics"]["glue_arena"] = {k: v for k, v in plan.items() if k != "mapping"}
    jsonwrite(output / "POPY.rig.json", ir)
    jsonwrite(output / "arena_plan.json", plan)
    source_plan = source.parent / "plan.json"
    if source_plan.exists():
        combined = json.loads(source_plan.read_text())
        combined["arena"] = plan
        jsonwrite(output / "plan.json", combined)
    print(
        json.dumps({k: v for k, v in plan.items() if k != "mapping"}, indent=2),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source",
        type=Path,
        default=ROOT / "artifacts/order_factor/rig_owned/POPY.rig.json",
    )
    p.add_argument("--precision-native", action="store_true")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/glue_arena/rig")
    a = p.parse_args()
    compile_rig(a.source, a.out, a.precision_native)


if __name__ == "__main__":
    main()
