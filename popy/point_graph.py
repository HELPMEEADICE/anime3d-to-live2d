"""Audit immutable point-expression DAGs before choosing a native lowering.

This is not a floating-point equivalence proof: endpoint identities can change
signed zero. A future lowered model needs the same native/Blender regressions.
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from .compile_ir import ROOT, jsonwrite
from .glue_arena import initial_overwrites


class PointGraph:
    def __init__(self, canonical=False):
        self.nodes = [("undefined",)]
        self.interned = {self.nodes[0]: 0}
        self.fields = []
        self.field_ids = {}
        self.bits = {0: 0, 1: 1065353216}
        self.canonical = canonical
        self.folded_rounding = 0

    def node(self, key):
        existing = self.interned.get(key)
        if existing is not None:
            return existing
        result = len(self.nodes)
        self.nodes.append(key)
        self.interned[key] = result
        return result

    def constant(self, a, b, weight):
        if weight == 0:
            return a
        if weight == 1065353216:
            return b
        if self.canonical and a == b:
            self.folded_rounding += 1
            return a
        return self.node(("constant", a, b, weight))

    def blend(self, a, b, field, constant):
        if constant is not None:
            return self.constant(a, b, constant)
        if self.canonical and a == b:
            self.folded_rounding += 1
            return a
        return self.node(("field", a, b, field))

    def weight_bits(self, w):
        if w not in self.bits:
            self.bits[w] = struct.unpack("<I", struct.pack("<f", w))[0]
        return self.bits[w]

    def field(self, g):
        axes = g.get("axes", [])
        if not axes:
            return None, self.weight_bits(g.get("intensity", 1))
        if "intensities" in g:
            values = np.asarray(g["intensities"], dtype="<f4")
        else:
            values = np.empty(tuple(len(a["keys"]) for a in axes), dtype="<f4")
            for c in g["cells"]:
                values[tuple(c["coordinate"])] = c["intensity"]
            values = values.ravel()
        key = (json.dumps(axes, sort_keys=True), values.tobytes())
        if key not in self.field_ids:
            self.field_ids[key] = len(self.fields)
            self.fields.append({"axes": axes, "values": values})
        return self.field_ids[key], None


def build(ir, data, canonical=False):
    graph = PointGraph(canonical)
    first = initial_overwrites(ir["glues"])
    states, initial, variables, aliases = {}, {}, {}, {}
    for mi, m in enumerate(ir["meshes"]):
        count = len(m["positions"]) // 2
        source = []
        rows = np.asarray([c["offset"] for c in m["cells"]], dtype=np.int64)
        parent = (
            "D_HeadProjection"
            if m.get("deformer") == "head"
            else m.get("projection_warp", "D_GlobalProjection")
        )
        header = json.dumps([m["axes"], parent], sort_keys=True).encode()
        for i in range(count):
            if first.get((m["id"], i), False):
                source.append(0)
                continue
            # Only exactly identical initial geometric fields share an input.
            if not m.get("blend_shapes"):
                values = np.column_stack([data[rows + 2 * i], data[rows + 2 * i + 1]])
                key = hashlib.sha256(header + values.tobytes()).digest()
            else:
                key = (mi, i)
            if key not in aliases:
                node = graph.node(("input", mi, i))
                aliases[key] = node
                variables[node] = {"mesh": m["id"], "vertex": i}
            source.append(aliases[key])
        states[m["id"]] = source.copy()
        initial[m["id"]] = source
    for i, g in enumerate(ir["glues"]):
        field, constant = graph.field(g)
        a, b = states[g["mesh_a"]], states[g["mesh_b"]]
        for ia, ib, wa, wb in g["pairs"]:
            va, vb = a[ia], b[ib]
            a[ia] = graph.blend(
                va, graph.constant(va, vb, graph.weight_bits(wa)), field, constant
            )
            b[ib] = graph.blend(
                vb, graph.constant(vb, va, graph.weight_bits(wb)), field, constant
            )
        if i % 100000 == 0:
            print(
                f"{i}/{len(ir['glues'])}: {len(graph.nodes)} expression nodes",
                flush=True,
            )
    masks = {name for m in ir["meshes"] for name in m.get("masked_by", [])}
    outputs = {
        m["id"]: states[m["id"]]
        for m in ir["meshes"]
        if m.get("opacity", 1) != 0 or m["id"] in masks
    }
    todo = [n for values in outputs.values() for n in values]
    live = set()
    while todo:
        n = todo.pop()
        if n in live:
            continue
        live.add(n)
        node = graph.nodes[n]
        if node[0] in ["constant", "field"]:
            todo.extend(node[1:3])
    if 0 in live:
        raise AssertionError(
            "A supposedly dead initial value reached observable output"
        )
    depths = [0] * len(graph.nodes)
    kinds, fields = {}, set()
    for i, node in enumerate(graph.nodes):
        if node[0] in ["constant", "field"]:
            depths[i] = 1 + max(depths[node[1]], depths[node[2]])
        if i in live:
            kinds[node[0]] = kinds.get(node[0], 0) + 1
            if node[0] == "field":
                fields.add(node[3])
    result = {
        "audit_only_not_a_lowered_model_proof": True,
        "endpoint_identity_requires_signed_zero_validation": True,
        "canonical_real_arithmetic": canonical,
        "original_glues": len(ir["glues"]),
        "original_pairs": sum(len(g["pairs"]) for g in ir["glues"]),
        "all_expression_nodes": len(graph.nodes),
        "live_expression_nodes": len(live),
        "live_nodes_by_kind": kinds,
        "live_scalar_fields": len(fields),
        "live_scalar_keyforms": sum(len(graph.fields[i]["values"]) for i in fields),
        "maximum_expression_depth": max(depths[i] for i in live),
        "folded_self_interpolations": graph.folded_rounding,
        "observable_meshes": len(outputs),
        "observable_vertices": sum(map(len, outputs.values())),
        "initial_geometric_fields": len(variables),
    }
    return graph, states, initial, outputs, live, result


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source",
        type=Path,
        default=ROOT / "artifacts/order_factor/rig_owned/POPY.rig.json",
    )
    p.add_argument(
        "--out", type=Path, default=ROOT / "artifacts/point_graph/audit.json"
    )
    p.add_argument("--canonical", action="store_true")
    a = p.parse_args()
    ir = json.loads(a.source.read_text())
    data = np.memmap(a.source.parent / ir["keyform_data"], "<f4", mode="r")
    *_, result = build(ir, data, a.canonical)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    jsonwrite(a.out, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
