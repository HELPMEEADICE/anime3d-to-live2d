"""Reduce scalar grids exactly, or within an estimated screen-space budget."""

import argparse
import json
from pathlib import Path

import numpy as np

from .compile_ir import ROOT, jsonwrite


def reduce_grid(axes, values, defaults, effect_px=0.0, budget_px=0.0):
    axes = [{**a, "keys": list(a["keys"])} for a in axes]
    values = np.asarray(values).copy()
    spent = 0.0
    removed_axes = removed_keys = 0
    for axis in reversed(range(len(axes))):
        keys = axes[axis]["keys"]
        default = defaults.get(axes[axis]["parameter"], keys[0])
        index = min(range(len(keys)), key=lambda i: abs(keys[i] - default))
        first = np.take(values, index, axis=axis)
        error = float(np.max(np.abs(values - np.expand_dims(first, axis))))
        cost = 0.0 if error <= 1e-12 else error * effect_px
        if error <= 1e-12 or (effect_px > 0 and spent + cost <= budget_px):
            values = first
            axes.pop(axis)
            spent += cost
            removed_axes += 1
            continue
        i = 1
        while i < len(keys) - 1:
            t = (keys[i] - keys[i - 1]) / (keys[i + 1] - keys[i - 1])
            predicted = (
                np.take(values, i - 1, axis=axis) * (1 - t)
                + np.take(values, i + 1, axis=axis) * t
            )
            error = float(np.max(np.abs(np.take(values, i, axis=axis) - predicted)))
            cost = 0.0 if error <= 1e-12 else error * effect_px
            if error <= 1e-12 or (effect_px > 0 and spent + cost <= budget_px):
                values = np.delete(values, i, axis=axis)
                keys.pop(i)
                spent += cost
                removed_keys += 1
            else:
                i += 1
    return axes, values, spent, removed_axes, removed_keys


def _warp_lipschitz(node):
    result = 0.0
    for cell in node["cells"]:
        points = np.array(cell["points"]).reshape(-1, 2)
        if len(points) != 4:
            return float("inf")
        for u, v in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            dx = (points[1] - points[0]) * (1 - v) + (points[3] - points[2]) * v
            dy = (points[2] - points[0]) * (1 - u) + (points[3] - points[1]) * u
            result = max(result, float(np.linalg.norm(np.array([dx, dy]).T, 2)))
    return result


def compress_glue_fields(ir, binary=None, max_vertex_error_px=0.05):
    data = np.memmap(binary, dtype="<f4", mode="r") if binary is not None else None
    meshes = {m["id"]: m for m in ir.get("meshes", [])}
    warps = {n["id"]: n for n in ir.get("body_warps", [])}
    limits = {}
    defaults = {p["id"]: p["default"] for p in ir.get("parameters", [])}

    def diameter(mesh_id):
        if mesh_id in limits:
            return limits[mesh_id]
        m = meshes[mesh_id]
        size = len(m["positions"])
        offsets = np.array([c["offset"] for c in m["cells"]])
        points = data[offsets[:, None] + np.arange(size)].reshape(-1, size // 2, 2)
        bound = float(np.max(np.linalg.norm(points.max(1) - points.min(1), axis=1)))
        reference = np.array(m["positions"]).reshape(-1, 2) / np.array(ir["canvas"])
        for blend in m["blend_shapes"]:
            deltas = [
                np.max(
                    np.linalg.norm(
                        data[o : o + size].reshape(-1, 2) - reference, axis=1
                    )
                )
                for o in blend["offsets"]
                if o is not None
            ]
            if deltas:
                bound += 2 * max(deltas)
        warp = (
            ir["head_warp"]
            if m.get("deformer") == "head"
            else warps.get(m.get("projection_warp"), ir["root_warp"])
        )
        # Extra allowance for native float arithmetic and source-identity welds
        # crossing different parent warp groups; actual native deltas are also
        # checked over the full regression cohort after export.
        limits[mesh_id] = bound * _warp_lipschitz(warp) * 1.5 + 2.0
        return limits[mesh_id]

    before = after = 0
    removed = removed_keys = 0
    max_bound = 0.0
    for glue in ir.get("glues", []):
        if not glue.get("axes") or "intensities" not in glue:
            continue
        if data is not None and "compression_error_bound_px" in glue:
            continue
        axes = list(glue["axes"])
        values = np.asarray(glue["intensities"]).reshape(
            tuple(len(a["keys"]) for a in axes)
        )
        before += values.size
        effect = (
            diameter(glue["mesh_a"]) * max(p[3] for p in glue["pairs"])
            if data is not None
            else 0.0
        )
        axes, values, error, ra, rk = reduce_grid(
            axes,
            values,
            defaults,
            effect,
            max_vertex_error_px / 2 if data is not None else 0.0,
        )
        removed += ra
        removed_keys += rk
        max_bound = max(max_bound, error)
        if data is not None:
            glue["compression_error_bound_px"] = error
        after += values.size
        if axes:
            glue["axes"] = axes
            glue["intensities"] = values.ravel().tolist()
        else:
            glue["intensity"] = float(values)
            del glue["axes"]
            del glue["intensities"]
    noops = sum(
        not g.get("axes") and g.get("intensity", 1) == 0 for g in ir.get("glues", [])
    )
    ir["glues"] = [
        g for g in ir.get("glues", []) if g.get("axes") or g.get("intensity", 1) != 0
    ]
    if before == 0 and not noops and "scalar_grid_compression" in ir["diagnostics"]:
        return ir["diagnostics"]["scalar_grid_compression"]
    ir["diagnostics"]["scalar_grid_compression"] = {
        "input_glue_keyforms": int(before),
        "output_glue_keyforms": int(after),
        "removed_axes": removed,
        "removed_keys": removed_keys,
        "removed_noop_glues": noops,
        "max_glue_error_estimate_px": max_bound,
        "vertex_budget_px": max_vertex_error_px if data is not None else 0.0,
        "method": "Remove invariant/redundant axes and keys using source-master pose diameter, parent Warp Lipschitz and a weld allowance. At most two variable lerps per mask vertex; verify actual native displacement over the regression cohort.",
    }
    return ir["diagnostics"]["scalar_grid_compression"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rig-dir", type=Path, default=ROOT / "artifacts/rig")
    a = p.parse_args()
    path = a.rig_dir / "POPY.rig.json"
    ir = json.loads(path.read_text())
    result = compress_glue_fields(ir, a.rig_dir / ir["keyform_data"])
    ir["diagnostics"].update(
        glue_keyforms=sum(
            len(g.get("intensities", g.get("cells", [None]))) for g in ir["glues"]
        ),
        glues=len(ir["glues"]),
        animated_glues=sum(bool(g.get("axes")) for g in ir["glues"]),
        masked_drawables=sum(bool(m.get("masked_by")) for m in ir["meshes"]),
    )
    jsonwrite(path, ir)
    jsonwrite(a.rig_dir / "statistics.json", ir["diagnostics"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
