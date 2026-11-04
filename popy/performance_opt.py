"""Algebraic, topology-preserving Glue program compaction.

The original rig and model are never overwritten by this command. This module
does not import a Cubism binary codec; it only rewrites the neutral Rig IR.
"""

import argparse
import hashlib
import json
import shutil
import struct
from collections import Counter, defaultdict, deque
from pathlib import Path

from .compile_ir import ROOT, jsonwrite


def field_key(glue):
    """Only bit-identical scalar fields may share an evaluation."""
    axes = glue.get("axes", [])
    if "cells" in glue:
        return None
    if axes:
        return (
            json.dumps(axes, sort_keys=True),
            b"".join(struct.pack("<f", v) for v in glue["intensities"]),
        )
    return ("constant", struct.pack("<f", glue.get("intensity", 1.0)))


def compact_glues(glues):
    """Topologically batch identical fields without changing vertex programs.

    A zero side weight is *not* generally read-only in float arithmetic:
    L(A,A,t) can round. Only endpoint compatibility 1 permits a read-only side.
    All read/write, write/read and write/write hazards are retained per vertex.
    """
    successors = [[] for _ in glues]
    pending = [0] * len(glues)
    writers, readers = {}, defaultdict(set)
    signatures = []
    for index, glue in enumerate(glues):
        key = field_key(glue)
        signature = (glue["mesh_a"], glue["mesh_b"], key)
        if key is None or glue["mesh_a"] == glue["mesh_b"]:
            signature += (index,)
        signatures.append(signature)
        reads, writes = set(), set()
        endpoint = not glue.get("axes") and glue.get("intensity", 1) == 1
        for a, b, wa, wb in glue["pairs"]:
            va, vb = (glue["mesh_a"], a), (glue["mesh_b"], b)
            reads.update([va, vb])
            if wa != 0 or not endpoint:
                writes.add(va)
            if wb != 0 or not endpoint:
                writes.add(vb)
        dependencies = {writers[v] for v in reads if v in writers}
        for v in writes:
            dependencies.update(readers[v])
        for previous in dependencies:
            successors[previous].append(index)
        pending[index] = len(dependencies)
        for v in reads - writes:
            readers[v].add(index)
        for v in writes:
            writers[v] = index
            readers.pop(v, None)
    ready = defaultdict(deque)
    queue = deque()
    active = set()

    def schedule(index):
        signature = signatures[index]
        ready[signature].append(index)
        if signature not in active:
            queue.append(signature)
            active.add(signature)

    for index, count in enumerate(pending):
        if count == 0:
            schedule(index)
    result, scheduled = [], 0
    while queue:
        signature = queue.popleft()
        batch = None
        # Completing a node may release another node with the same signature.
        while ready[signature]:
            index = ready[signature].popleft()
            glue = glues[index]
            if batch is None:
                batch = {**glue, "pairs": list(glue["pairs"])}
            else:
                batch["pairs"].extend(glue["pairs"])
            scheduled += 1
            for nxt in successors[index]:
                pending[nxt] -= 1
                if pending[nxt] == 0:
                    schedule(nxt)
        active.remove(signature)
        result.append(batch)
    if scheduled != len(glues):
        raise ValueError("Cycle in Glue program")
    return result, {
        "input_glues": len(glues),
        "output_glues": len(result),
        "constant_lerps_folded": 0,
        "dependency_batches_merged": len(glues) - len(result),
        "input_pairs": sum(len(g["pairs"]) for g in glues),
        "output_pairs": sum(len(g["pairs"]) for g in result),
    }


def factor_native_constants(glues):
    """Real-arithmetic factorization, deliberately limited to the 3D skeleton.

    L(L(B,A,1),t) = L(L(B,A,t),1). Source weight is zero in both.
    Unlike the exact pass, this removes L(A,A,t) source-rounding drift. The
    result must be checked against independently sampled Blender geometry;
    bitwise equivalence to the old, rounded program is not claimed.
    """
    result, count = [], 0
    for g in glues:
        t = struct.unpack("<f", struct.pack("<f", g.get("intensity", 1)))[0]
        if (
            g["id"].startswith("NativeG_")
            and not g.get("axes")
            and "cells" not in g
            and "intensities" not in g
            and 0 <= t < 1
            and all(p[2:] == [0, 1] for p in g["pairs"])
        ):
            result.append(
                {
                    **g,
                    "intensity": 1.0,
                    "pairs": [[p[0], p[1], 0.0, t] for p in g["pairs"]],
                }
            )
            count += 1
        else:
            result.append(g)
    return result, count


def optimize(source, destination, mode="exact"):
    source, destination = Path(source), Path(destination)
    if source.resolve().parent == destination.resolve():
        raise ValueError("Optimization requires a separate destination")
    ir = json.loads(source.read_text())
    print("Loaded source IR", flush=True)
    before = Counter(g["id"].split("_")[0] for g in ir["glues"])
    folded = 0
    if mode == "precision":
        ir["glues"], folded = factor_native_constants(ir["glues"])
    elif mode != "exact":
        raise ValueError("Unknown optimization mode")
    ir["glues"], report = compact_glues(ir["glues"])
    report["constant_lerps_folded"] = folded
    report.update(
        source_ir_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        groups_before=dict(before),
        groups_after=dict(Counter(g["id"].split("_")[0] for g in ir["glues"])),
        method=(
            "Native skeleton constant-lerp factorization, then per-vertex dependency DAG batching; changes floating-point drift, requires Blender ground-truth verification"
            if mode == "precision"
            else "Per-vertex dependency DAG and bit-identical scalar-field batching; no lerp factorization, curve fitting, vertex reduction, texture or order changes"
        ),
        topology_unchanged=True,
        parameter_dependent_scalar_fields_unchanged=True,
        mode=mode,
    )
    ir["diagnostics"]["algebraic_compaction"] = report
    native = ir["diagnostics"].get("fullbody", {}).get("native_skinning")
    if native is not None:
        operators = [g for g in ir["glues"] if g["id"].startswith("NativeG_")]
        native["operator_glues_before_compaction"] = native["operator_glues"]
        native["operator_glues"] = len(operators)
        native["canonical_glue_weights"] = all(
            p[2:] == [0, 1] for g in operators for p in g["pairs"]
        )
        native["scalar_keyforms"] = sum(
            len(g.get("intensities", [])) for g in operators
        )
    destination.mkdir(parents=True, exist_ok=True)
    for name in [ir["keyform_data"]] + ir["textures"]:
        shutil.copy2(source.parent / name, destination / name)
    jsonwrite(destination / "POPY.rig.json", ir)
    jsonwrite(destination / "optimization.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source", type=Path, default=ROOT / "artifacts/fullbody/rig/POPY.rig.json"
    )
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/perf_opt/rig")
    p.add_argument("--mode", choices=["exact", "precision"], default="exact")
    args = p.parse_args()
    optimize(args.source, args.out, args.mode)


if __name__ == "__main__":
    main()
