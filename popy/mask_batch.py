"""SPDX-License-Identifier: MIT. Lossless opaque mask-union common subexpressions.

Build shared geometry for frequently repeated source pairs. This changes the
MOC itself, and works with the unmodified runtime. No smaller textures or masks.
"""

import argparse
import heapq
import itertools
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .compile_ir import ROOT, jsonwrite

BASE = ROOT / "artifacts/mask_batch"


def plan(inventory, vertex_budget=200000, min_uses=3, helper_penalty=0):
    if (
        vertex_budget < 0
        or min_uses < 2
        or helper_penalty < 0
        or not math.isfinite(helper_penalty)
    ):
        raise ValueError("Require nonnegative vertex budget and at least two uses")
    lookup = {m["id"]: m for m in inventory}
    if len(lookup) != len(inventory) or any(s.startswith("MaskBatch_") for s in lookup):
        raise ValueError("Duplicate or reserved mesh ID")
    groups = list(dict.fromkeys(tuple(m["clips"]) for m in inventory if m["clips"]))
    if any(len(set(g)) != len(g) for g in groups):
        raise ValueError("Duplicate mask sources require separate analysis")
    names = sorted({i for g in groups for i in g})
    if set(names) - lookup.keys():
        raise ValueError("Unknown mask source")
    indices = {s: i for i, s in enumerate(names)}
    occurrences = [0] * len(names)
    sizes = [lookup[s]["vertices"] for s in names]
    if any(s <= 0 for s in sizes):
        raise ValueError("Empty mask source")
    culling = [lookup[s]["culling"] for s in names]
    for i, g in enumerate(groups):
        for s in g:
            occurrences[indices[s]] |= 1 << i
    pairs = set()
    for g in groups:
        for a, b in itertools.combinations(sorted(indices[s] for s in g), 2):
            if culling[a] == culling[b]:
                pairs.add((a, b))
    heap = []

    def offer(a, b):
        count = (occurrences[a] & occurrences[b]).bit_count()
        size = sizes[a] + sizes[b]
        if count >= min_uses and size <= 60000:
            heapq.heappush(heap, (-count / (size + helper_penalty), -count, a, b))

    for a, b in sorted(pairs):
        offer(a, b)
    del pairs
    unions = []
    spent = saved = 0
    while heap:
        _, old_count, a, b = heapq.heappop(heap)
        selected = occurrences[a] & occurrences[b]
        count = selected.bit_count()
        if count < min_uses:
            continue
        if count != -old_count:
            offer(a, b)
            continue
        size = sizes[a] + sizes[b]
        if spent + size > vertex_budget:
            continue
        node = len(names)
        name = f"MaskBatch_{len(unions)}"
        unions.append(
            {
                "id": name,
                "sources": [names[a], names[b]],
                "vertices": size,
                "culling": culling[a],
                "uses": count,
            }
        )
        occurrences[a] &= ~selected
        occurrences[b] &= ~selected
        names.append(name)
        occurrences.append(selected)
        sizes.append(size)
        culling.append(culling[a])
        spent += size
        saved += count
        for i in range(node):
            if culling[i] == culling[node] and occurrences[i] & selected:
                offer(i, node)
    final = []
    for i, g in enumerate(groups):
        final.append([names[j] for j, occ in enumerate(occurrences) if occ & (1 << i)])
    before = sum(map(len, groups))
    after = sum(map(len, final))
    assert before - after == saved
    result = {
        "unions": unions,
        "groups": [{"before": list(a), "after": b} for a, b in zip(groups, final)],
        "input_mask_draws": before,
        "output_mask_draws": after,
        "added_vertices": spent,
        "helper_meshes": len(unions),
        "vertex_budget": vertex_budget,
        "min_uses": min_uses,
        "helper_penalty": helper_penalty,
        "method": "Common-subexpression factorization of opaque alpha unions; original triangles and screen-space mask resolution retained",
    }
    validate_design(inventory, result)
    return result


def validate_design(inventory, design):
    """Expand the DAG and prove source multiplicity, culling, sizes and budget."""
    lookup = {m["id"]: m for m in inventory}
    leaves = {name: Counter([name]) for name in lookup}
    sizes = {name: m["vertices"] for name, m in lookup.items()}
    culling = {name: m["culling"] for name, m in lookup.items()}
    spent = 0
    for node in design["unions"]:
        name = node["id"]
        if name in leaves or len(node["sources"]) != 2:
            raise ValueError("Invalid union DAG")
        a, b = node["sources"]
        if a not in leaves or b not in leaves:
            raise ValueError("Union must follow its operands")
        size = sizes[a] + sizes[b]
        if size != node["vertices"] or size > 60000:
            raise ValueError("Invalid union vertex count")
        if not culling[a] == culling[b] == node["culling"]:
            raise ValueError("Mixed culling in union")
        leaves[name] = leaves[a] + leaves[b]
        sizes[name], culling[name] = size, culling[a]
        spent += size
    expected = {tuple(m["clips"]) for m in inventory if m["clips"]}
    if expected != {tuple(g["before"]) for g in design["groups"]}:
        raise ValueError("Mask groups do not match source")
    for group in design["groups"]:
        actual = Counter()
        for name in group["after"]:
            actual.update(leaves[name])
        if actual != Counter(group["before"]):
            raise ValueError("Union lost or duplicated source triangles")
    if spent != design["added_vertices"] or spent > design["vertex_budget"]:
        raise ValueError("Vertex budget violated")
    if len(design["unions"]) != design["helper_meshes"]:
        raise ValueError("Helper count mismatch")
    for field, side in [("input_mask_draws", "before"), ("output_mask_draws", "after")]:
        if design[field] != sum(len(g[side]) for g in design["groups"]):
            raise ValueError("Draw count mismatch")


def compile_plan(source, destination, design):
    source, destination = Path(source), Path(destination)
    if source.resolve().parent == destination.resolve():
        raise ValueError("Separate destination required")
    ir = json.loads(source.read_text())
    lookup = {m["id"]: m for m in ir["meshes"]}
    validate_design(
        [
            {
                "id": m["id"],
                "vertices": len(m["positions"]) // 2,
                "culling": m.get("culling", True),
                "clips": m.get("masked_by", []),
            }
            for m in ir["meshes"]
        ],
        design,
    )
    parameters = {p["id"]: p for p in ir["parameters"]}
    for m in ir["meshes"]:
        for axis in m["axes"]:
            p = parameters[axis["parameter"]]
            if axis["keys"][0] > p["min"] or axis["keys"][-1] < p["max"]:
                raise ValueError("Cannot prove constant source visibility: " + m["id"])
    for texture in ir["textures"]:
        with Image.open(source.parent / texture) as im:
            if im.convert("RGBA").getchannel("A").getextrema() != (255, 255):
                raise ValueError("Mask batching requires fully opaque texture alpha")
    destination.mkdir(parents=True, exist_ok=True)
    for name in [ir["keyform_data"]] + ir["textures"]:
        shutil.copy2(source.parent / name, destination / name)
    binary = destination / ir["keyform_data"]
    cursor = binary.stat().st_size // 4
    helpers = []
    glues = []
    with binary.open("ab") as stream:
        for node in design["unions"]:
            positions = []
            triangles = []
            provenance = []
            for name in node["sources"]:
                source_mesh = lookup[name]
                count = len(source_mesh["positions"]) // 2
                start = len(positions) // 2
                if source_mesh.get("culling", True) != node["culling"]:
                    raise ValueError("Mixed culling")
                positions.extend([0.0] * (2 * count))
                triangles.extend(start + i for i in source_mesh["indices"])
                glues.append(
                    {
                        "id": f"G_MaskBatch_{len(glues)}",
                        "mesh_a": name,
                        "mesh_b": node["id"],
                        "pairs": [[i, start + i, 0.0, 1.0] for i in range(count)],
                        "intensity": 1.0,
                    }
                )
                provenance.append({"mesh": name, "start": start, "vertices": count})
            n = len(positions) // 2
            mesh = {
                "id": node["id"],
                "positions": positions,
                "uvs": [0.5, 0.5] * n,
                "indices": triangles,
                "texture": 0,
                "projection_warp": "D_NativeSkeletonProjection",
                "deformer": "body",
                "axes": [],
                "cells": [{"coordinate": [], "offset": cursor, "order": 0.0}],
                "blend_shapes": [],
                "opacity": 0.0,
                "mask_only": True,
                "culling": node["culling"],
                "masked_by": [],
                "invert_mask": False,
                "source": {
                    "object": "MaskBatch",
                    "vertices": list(range(n)),
                    "triangles": list(range(len(triangles) // 3)),
                    "barycentric_sources": provenance,
                    "correspondence": "Identity copies of already evaluated original source corners; exact opaque union",
                },
            }
            stream.write(np.zeros(2 * n, dtype="<f4").tobytes())
            cursor += 2 * n
            helpers.append(mesh)
            lookup[node["id"]] = mesh
    replacements = {tuple(g["before"]): g["after"] for g in design["groups"]}
    for m in ir["meshes"]:
        if m.get("masked_by"):
            m["masked_by"] = replacements[tuple(m["masked_by"])]
    ir["meshes"].extend(helpers)
    ir["glues"].extend(glues)
    ir["parts"].append(
        {
            "id": "MaskBatch",
            "source_object": "MaskBatch",
            "mesh_ids": [m["id"] for m in helpers],
        }
    )
    ir["diagnostics"]["mask_batch"] = {
        k: v for k, v in design.items() if k not in ["groups", "unions"]
    }
    jsonwrite(destination / "POPY.rig.json", ir)
    jsonwrite(destination / "plan.json", design)
    print(json.dumps(ir["diagnostics"]["mask_batch"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["plan", "compile"])
    parser.add_argument("--budget", type=int, default=200000)
    parser.add_argument("--helper-penalty", type=float, default=0)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "artifacts/perf_opt/exact_rig/POPY.rig.json",
    )
    parser.add_argument("--out", type=Path, default=BASE / "rig")
    args = parser.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    suffix = f"_p{args.helper_penalty:g}" if args.helper_penalty else ""
    path = BASE / f"plan_{args.budget}{suffix}.json"
    if args.stage == "plan":
        inventory = json.loads(
            (ROOT / "artifacts/perf_opt/images_before/model_inventory.json").read_text()
        )
        result = plan(inventory, args.budget, helper_penalty=args.helper_penalty)
        jsonwrite(path, result)
        print(
            json.dumps(
                {k: v for k, v in result.items() if k not in ["groups", "unions"]},
                indent=2,
            )
        )
    else:
        compile_plan(args.source, args.out, json.loads(path.read_text()))


if __name__ == "__main__":
    main()
