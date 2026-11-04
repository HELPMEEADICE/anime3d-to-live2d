"""Audit Cartesian keyform multiplication and reusable scalar/position fields."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .compile_ir import ROOT, jsonwrite


def audit(source, output):
    source, output = Path(source), Path(output)
    ir = json.loads(source.read_text())
    data = np.memmap(source.parent / ir["keyform_data"], "<f4", mode="r")
    by_object = defaultdict(Counter)
    axes_histogram = Counter()
    positions = {}
    form_groups = defaultdict(list)
    rows = []
    for mesh in ir["meshes"]:
        cells = mesh["cells"]
        obj = mesh.get("source", {}).get("object", "unknown")
        by_object[obj].update(
            meshes=1, keyforms=len(cells), vertices=len(mesh["positions"]) // 2
        )
        axes_histogram[len(mesh["axes"])] += len(cells)
        size = len(mesh["positions"])
        geometry = []
        for offset in dict.fromkeys(c["offset"] for c in cells):
            key = (offset, size)
            if key not in positions:
                values = data[offset : offset + size]
                positions[key] = hashlib.sha256(values.tobytes()).digest()
            geometry.append(positions[key])
        geom_lookup = dict(zip(dict.fromkeys(c["offset"] for c in cells), geometry))
        orders = np.asarray([c["order"] for c in cells], dtype="<f4")
        signature = hashlib.sha256()
        signature.update(
            json.dumps(
                [
                    mesh["axes"],
                    size,
                    mesh.get("opacity", 1),
                    mesh.get("deformer"),
                    mesh.get("projection_warp"),
                ],
                sort_keys=True,
            ).encode()
        )
        for c, order in zip(cells, orders):
            signature.update(geom_lookup[c["offset"]])
            signature.update(order.tobytes())
        form_groups[signature.hexdigest()].append(mesh["id"])
        shape = tuple(len(a["keys"]) for a in mesh["axes"])
        invariant_order_axes = []
        if np.prod(shape, dtype=np.int64) == len(cells):
            order_grid = orders.reshape(shape)
            for axis, definition in enumerate(mesh["axes"]):
                if np.all(
                    order_grid == np.expand_dims(np.take(order_grid, 0, axis), axis)
                ):
                    invariant_order_axes.append(definition["parameter"])
        rows.append(
            {
                "id": mesh["id"],
                "object": obj,
                "keyforms": len(cells),
                "axes": [
                    {"parameter": a["parameter"], "keys": len(a["keys"])}
                    for a in mesh["axes"]
                ],
                "unique_geometry_forms": len(set(geometry)),
                "unique_orders": len(np.unique(orders)),
                "invariant_order_axes": invariant_order_axes,
                "native_skinning": bool(mesh.get("native_skinning")),
                "has_final_vertex_driver": bool(mesh.get("native_vertex_driver")),
                "vertices": size // 2,
            }
        )
    rows.sort(key=lambda r: r["keyforms"], reverse=True)
    scalar_counts = Counter()
    scalar_sizes = {}
    glue_groups = defaultdict(Counter)
    for glue in ir["glues"]:
        prefix = glue["id"].split("_")[0]
        values = np.asarray(
            glue.get("intensities", [glue.get("intensity", 1)]), dtype="<f4"
        )
        axes = json.dumps(glue.get("axes", []), sort_keys=True).encode()
        digest = hashlib.sha256(axes + values.tobytes()).hexdigest()
        scalar_counts[digest] += 1
        scalar_sizes[digest] = len(values)
        glue_groups[prefix].update(
            glues=1,
            keyforms=len(values),
            pairs=len(glue["pairs"]),
            animated=bool(glue.get("axes")),
        )
    counts = {m["id"]: len(m["cells"]) for m in ir["meshes"]}
    duplicated = [names for names in form_groups.values() if len(names) > 1]
    result = {
        "source": str(source.relative_to(ROOT)),
        "artmesh_keyforms": sum(counts.values()),
        "by_object": {k: dict(v) for k, v in by_object.items()},
        "keyforms_by_axis_count": dict(axes_histogram),
        "potential_identical_main_form_block_savings": sum(
            counts[names[0]] * (len(names) - 1) for names in duplicated
        ),
        "identical_form_block_groups": duplicated,
        "glue_groups": {k: dict(v) for k, v in glue_groups.items()},
        "scalar_fields": sum(scalar_counts.values()),
        "unique_scalar_fields": len(scalar_counts),
        "scalar_keyforms": sum(scalar_sizes[k] * v for k, v in scalar_counts.items()),
        "unique_scalar_keyforms": sum(scalar_sizes.values()),
        "most_shared_scalar_fields": [
            {"uses": n, "keyforms": scalar_sizes[k]}
            for k, n in scalar_counts.most_common(20)
        ],
        "meshes": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    jsonwrite(output, result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ["meshes", "identical_form_block_groups"]
            },
            indent=2,
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source",
        type=Path,
        default=ROOT / "artifacts/perf_opt/exact_rig/POPY.rig.json",
    )
    p.add_argument(
        "--out", type=Path, default=ROOT / "artifacts/representation/audit.json"
    )
    args = p.parse_args()
    audit(args.source, args.out)


if __name__ == "__main__":
    main()
