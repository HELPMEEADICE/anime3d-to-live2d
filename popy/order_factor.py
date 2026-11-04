"""Factor scalar draw-order tensors out of fully driven zero-geometry ArtMeshes.

The original Glue program already overwrites their positions. A non-offscreen Part can own
the identical scalar grid, while the child has one static keyform. Only adjacent
children with identical order functions share a Part, preserving stable sorting.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from .compile_ir import ROOT, jsonwrite


def scalar_signature(mesh):
    axes = mesh["axes"]
    shape = tuple(len(a["keys"]) for a in axes)
    values = np.empty(shape, dtype="<f4")
    if int(np.prod(shape)) != len(mesh["cells"]):
        raise ValueError("Incomplete scalar grid")
    seen = set()
    for cell in mesh["cells"]:
        coordinate = tuple(cell["coordinate"])
        if (
            coordinate in seen
            or len(coordinate) != len(shape)
            or any(i < 0 or i >= n for i, n in zip(coordinate, shape))
        ):
            raise ValueError("Invalid or duplicated scalar coordinate")
        seen.add(coordinate)
        values[coordinate] = cell["order"]
    signature = hashlib.sha256()
    for axis in axes:
        signature.update(axis["parameter"].encode() + b"\0")
        signature.update(np.asarray(axis["keys"], dtype="<f4").tobytes())
    signature.update(values.tobytes())
    return signature.digest()


def head_visual_parents(ir):
    """Mirror only the visual arithmetic of the audited projection ancestry.

    All bridge-created ancestors have unit per-form visual channels and no
    organizational Part. The head chain is Warp -> Rotation -> Rotation. Its
    pure-visual mirror uses Rotation -> Rotation -> Rotation, avoiding a second
    Warp-parent application in the runtime's rotation geometry calculation.
    Original geometry is irrelevant only for first-write-killed consumers.
    """
    if not all(k in ir for k in ["head_warp", "head_rotation", "body_rotation"]):
        return [], {}
    definitions = [
        (
            "BodyRoll",
            None,
            {
                "axes": [{"parameter": "ParamBodyAngleZ", "keys": [-15.0, 0.0, 15.0]}],
                "cells": [{"coordinate": [i]} for i in range(3)],
            },
        ),
        ("HeadRoll", "BodyRoll", ir["head_rotation"]),
        ("HeadProjection", "HeadRoll", ir["head_warp"]),
    ]
    nodes = []
    for label, parent, original in definitions:
        node = {
            "id": "D_ChannelVisual_" + label,
            "kind": "channel_rotation",
            "axes": original["axes"],
            "cells": [{"coordinate": c["coordinate"]} for c in original["cells"]],
        }
        if parent is not None:
            node["parent"] = "D_ChannelVisual_" + parent
        nodes.append(node)
    return nodes, {"D_HeadProjection": "D_ChannelVisual_HeadProjection"}


def factor(ir, data, native_only=False, rotation_carriers=False):
    if ir.get("order_groups"):
        raise ValueError("Order grids already factored")
    # A vertex's initial geometric field is dead only if its first use is an
    # unconditional endpoint copy that does not read it on the opposite side.
    overwritten = {}
    for g in ir["glues"]:
        one = not g.get("axes") and g.get("intensity", 1) == 1
        for ia, ib, wa, wb in g["pairs"]:
            a, b = (g["mesh_a"], ia), (g["mesh_b"], ib)
            overwritten.setdefault(a, one and wa == 1 and wb == 0 and a != b)
            overwritten.setdefault(b, one and wb == 1 and wa == 0 and a != b)
    candidates = {}
    org_parent = {name: p["id"] for p in ir["parts"] for name in p["mesh_ids"]}
    old = sum(len(m["cells"]) for m in ir["meshes"])
    for mesh in ir["meshes"]:
        if len(mesh["cells"]) <= 1 or mesh.get("blend_shapes"):
            continue
        if native_only and not mesh.get("native_vertex_driver"):
            continue
        if all(
            overwritten.get((mesh["id"], i), False)
            for i in range(len(mesh["positions"]) // 2)
        ):
            candidates[mesh["id"]] = (scalar_signature(mesh), org_parent[mesh["id"]])
    groups, fields, carriers, static_orders = [], {}, {}, []
    visual_nodes, visual_parent_map = (
        head_visual_parents(ir) if rotation_carriers else ([], {})
    )
    roots = {
        n["id"]
        for n in ir["body_warps"]
        if n.get("kind") == "rotation" and n.get("root") and not n.get("axes")
    }
    previous = None
    old_axes = {}
    for mesh in ir["meshes"]:
        name = mesh["id"]
        signature = candidates.get(name)
        if signature is None:
            previous = None
            continue
        axes = mesh["axes"]
        old_axes[name] = axes
        parent = (
            "D_HeadProjection"
            if mesh.get("deformer") == "head"
            else mesh.get("projection_warp", "D_GlobalProjection")
        )
        carrier_key = (parent, json.dumps(axes, sort_keys=True))
        if carrier_key not in carriers:
            # Preserve the ORIGINAL floating-point partition-of-unity program,
            # including opacity/tint roundoff and parameter invalidation. Its
            # geometric output is immaterial because every child corner is
            # overwritten before any original Glue can consume it.
            carriers[carrier_key] = {
                "id": f"D_ChannelCarrier_{len(carriers)}",
                "parent": parent,
                "strength": 1.0,
                "axes": axes,
                "cells": [
                    {"coordinate": c["coordinate"], "points": [0.0] * 8}
                    for c in mesh["cells"]
                ],
            }
            if rotation_carriers and (parent in roots or parent in visual_parent_map):
                carriers[carrier_key]["kind"] = "channel_rotation"
                carriers[carrier_key]["parent"] = visual_parent_map.get(parent, parent)
        carrier = carriers[carrier_key]
        order_zero = all(np.float32(c["order"]) == 0 for c in mesh["cells"])
        if order_zero:
            static_orders.append(name)
            previous = None
        else:
            if signature != previous:
                if signature not in fields:
                    fields[signature] = {
                        "id": f"P_FactoredOrder_{len(fields)}",
                        "org_parent": org_parent[name],
                        "axes": axes,
                        "cells": [
                            {"coordinate": c["coordinate"], "order": c["order"]}
                            for c in mesh["cells"]
                        ],
                    }
                groups.append(
                    {
                        **fields[signature],
                        "group_id": f"G_FactoredOrder_{len(groups)}",
                        "mesh_ids": [],
                    }
                )
            group = groups[-1]
            group["mesh_ids"].append(name)
            mesh["factored_order_part"] = group["id"]
            previous = signature
        mesh["axes"] = []
        mesh["cells"] = [
            {
                "coordinate": [],
                "offset": mesh["cells"][0]["offset"],
                "order": 0.0 if order_zero else 500.0,
            }
        ]
        mesh["deformer"] = "body"
        mesh["projection_warp"] = carrier["id"]
    static_redirects = []
    for mesh in ir["meshes"]:
        if (
            mesh["id"] in candidates
            or mesh.get("blend_shapes")
            or len(mesh["cells"]) != 1
        ):
            continue
        parent = (
            "D_HeadProjection"
            if mesh.get("deformer") == "head"
            else mesh.get("projection_warp", "D_GlobalProjection")
        )
        if parent in visual_parent_map and all(
            overwritten.get((mesh["id"], i), False)
            for i in range(len(mesh["positions"]) // 2)
        ):
            mesh["deformer"] = "body"
            mesh["projection_warp"] = visual_parent_map[parent]
            static_redirects.append(mesh["id"])
    if not static_redirects and not any(
        g["parent"] in visual_parent_map.values() for g in carriers.values()
    ):
        visual_nodes = []
    ir["body_warps"].extend(visual_nodes)
    ir["body_warps"].extend(carriers.values())
    ir["order_groups"] = groups
    result = {
        "old_artmesh_keyforms": old,
        "new_artmesh_keyforms": sum(len(m["cells"]) for m in ir["meshes"]),
        "new_part_keyforms": sum(len(g["cells"]) for g in fields.values()),
        "new_warp_keyforms": sum(
            len(g["cells"])
            for g in carriers.values()
            if g.get("kind") != "channel_rotation"
        ),
        "new_rotation_keyforms": sum(
            len(g["cells"])
            for g in list(carriers.values()) + visual_nodes
            if g.get("kind") == "channel_rotation"
        ),
        "visual_parent_proxies": len(visual_nodes),
        "static_geometry_visual_redirects": len(static_redirects),
        "order_groups": len(groups),
        "shared_order_parts": len(fields),
        "shared_visual_carriers": len(carriers),
        "rotation_visual_carriers": sum(
            g.get("kind") == "channel_rotation" for g in carriers.values()
        ),
        "factored_meshes": len(candidates),
        "dependency_edges": 0,
        "vertices_removed": 0,
        "original_glues_changed": 0,
        "method": "Dead initial geometry elimination, shared floating-point channel carriers, and exact scalar draw-order Parts; no original Glue, topology, UV or parameter reduction",
    }
    ir["diagnostics"]["order_factorization"] = result
    return {
        **result,
        "groups": [
            {"id": g["id"], "group_id": g["group_id"], "mesh_ids": g["mesh_ids"]}
            for g in groups
        ],
        "static_order_meshes": static_orders,
        "static_visual_redirect_meshes": static_redirects,
        "factored_meshes_ids": list(candidates),
        "carriers": [
            {
                "id": g["id"],
                "parent": g["parent"],
                "parameters": [a["parameter"] for a in g["axes"]],
            }
            for g in carriers.values()
        ],
    }


def compile_rig(source, output, rotation_carriers=False):
    source, output = Path(source), Path(output)
    if source.parent.resolve() == output.resolve():
        raise ValueError("Separate output required")
    ir = json.loads(source.read_text())
    data = np.memmap(source.parent / ir["keyform_data"], "<f4", mode="r")
    result = factor(ir, data, rotation_carriers=rotation_carriers)
    output.mkdir(parents=True, exist_ok=True)
    for name in [ir["keyform_data"], *ir["textures"]]:
        shutil.copy2(source.parent / name, output / name)
    jsonwrite(output / "POPY.rig.json", ir)
    jsonwrite(output / "plan.json", result)
    print(
        json.dumps({k: v for k, v in result.items() if k != "groups"}, indent=2),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source",
        type=Path,
        default=ROOT / "artifacts/mask_batch/rig_p1024/POPY.rig.json",
    )
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/order_factor/rig")
    p.add_argument("--rotation-carriers", action="store_true")
    a = p.parse_args()
    compile_rig(a.source, a.out, a.rotation_carriers)


if __name__ == "__main__":
    main()
