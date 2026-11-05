"""Intern complete IEEE float32 vertex tracks into shared native carriers.

Research candidate: a small constant Glue copy layer trades metadata/work for
less duplicate geometry evaluation and storage. Acceptance is measured, never
inferred from the amount of deduplicated input data.
"""

import argparse
import collections
import copy
import json
import struct
from pathlib import Path

import numpy as np

from . import vts_exact as exact
from .compile_ir import ROOT, jsonwrite
from .moc_pack import pack, runtime_extents
from .vts_mask_exact import invisible

BASE = ROOT / "artifacts/trajectory_pool"
SOURCE = ROOT / "output/POPY_VTS_Exact2"
MODEL = ROOT / "output/POPY_VTS_Pooled"


def program(mesh):
    return json.dumps(
        {
            "axes": mesh["axes"],
            "coords": [c["coordinate"] for c in mesh["cells"]],
            "parent": mesh["deformer"],
            "projection": mesh.get("projection_warp"),
            "blends": [
                {
                    k: v
                    for k, v in b.items()
                    if k not in ["offsets", "orders", "opacities"]
                }
                | {"nulls": [o is None for o in b["offsets"]]}
                for b in mesh["blend_shapes"]
            ],
        },
        sort_keys=True,
    )


def build():
    source = ROOT / "artifacts/vts_mask_exact/rig"
    folder = BASE / "rig"
    folder.mkdir(parents=True, exist_ok=True)
    ir = json.loads((source / "POPY.rig.json").read_text())
    data = np.memmap(source / ir["keyform_data"], "<f4", mode="r")
    sources = {name for m in ir["meshes"] for name in m["masked_by"]}
    groups = collections.defaultdict(list)
    for mesh in ir["meshes"]:
        if mesh["id"] in sources and invisible(mesh) and not mesh["masked_by"]:
            groups[program(mesh)].append(mesh)
    # Keep small cohorts unchanged: a carrier needs to amortize Glue overhead.
    selected = {
        key: group
        for key, group in groups.items()
        if sum(len(m["positions"]) // 2 for m in group) >= 256
    }
    replaced = {m["id"] for group in selected.values() for m in group}
    mapping = [
        {
            "source": m["id"],
            "target": m["id"],
            "vertex_start": 0,
            "vertex_count": len(m["positions"]) // 2,
            "index_start": 0,
            "index_count": len(m["indices"]),
            "ignore_hidden_depth": m["id"] in replaced,
            "ignore_hidden_color": m["id"] in replaced,
        }
        for m in ir["meshes"]
    ]
    cursor = 0
    carriers, glues, stats = [], [], []
    with (folder / "keyforms.f32").open("wb") as stream:

        def emit(array):
            nonlocal cursor
            a = np.asarray(array, dtype="<f4").ravel()
            offset = cursor
            stream.write(a.tobytes())
            cursor += a.size
            return offset

        def clone_fields(mesh):
            n = len(mesh["positions"])
            for cell in mesh["cells"]:
                off = cell["offset"]
                cell["offset"] = emit(data[off : off + n])
            for blend in mesh["blend_shapes"]:
                blend["offsets"] = [
                    None if off is None else emit(data[off : off + n])
                    for off in blend["offsets"]
                ]

        for gi, cohort in enumerate(selected.values()):
            chunks = []
            for mesh in cohort:
                n = len(mesh["positions"]) // 2
                offsets = [c["offset"] for c in mesh["cells"]] + [
                    o
                    for b in mesh["blend_shapes"]
                    for o in b["offsets"]
                    if o is not None
                ]
                chunks.append(
                    np.stack([data[o : o + 2 * n].reshape(n, 2) for o in offsets])
                    .transpose(1, 0, 2)
                    .reshape(n, -1)
                    .copy()
                )
            tracks = np.concatenate(chunks)
            unique_bits, inverse = np.unique(
                tracks.view("<u4"), axis=0, return_inverse=True
            )
            unique = unique_bits.view("<f4").reshape(len(unique_bits), -1, 2)
            example = cohort[0]
            names = []
            for ai, start in enumerate(range(0, len(unique), 60000)):
                values = unique[start : start + 60000]
                n = len(values)
                item = copy.deepcopy(example)
                name = f"TrackPool_{gi}_{ai}"
                names.append(name)
                item.update(
                    id=name,
                    positions=[0.0] * (2 * n),
                    uvs=[0.0] * (2 * n),
                    indices=[0, 1, 2] if n >= 3 else [],
                    texture=0,
                    culling=False,
                    invert_mask=False,
                    masked_by=[],
                    opacity=0.0,
                    source={"complete_track_pool": True},
                )
                fi = 0
                for cell in item["cells"]:
                    cell.update(offset=emit(values[:, fi]), order=0.0, opacity=0.0)
                    fi += 1
                for blend in item["blend_shapes"]:
                    blend["orders"] = [0.0] * len(blend["keys"])
                    blend["opacities"] = [0.0] * len(blend["keys"])
                    for k, off in enumerate(blend["offsets"]):
                        if off is not None:
                            blend["offsets"][k] = emit(values[:, fi])
                            fi += 1
                carriers.append(item)
            start = 0
            for mesh in cohort:
                n = len(mesh["positions"]) // 2
                refs = inverse[start : start + n]
                start += n
                for ai, name in enumerate(names):
                    vertices = np.flatnonzero(
                        (refs >= ai * 60000) & (refs < (ai + 1) * 60000)
                    )
                    if len(vertices):
                        glues.append(
                            {
                                "id": f"CopyTrack_{len(glues)}",
                                "mesh_a": name,
                                "mesh_b": mesh["id"],
                                "intensity": 1.0,
                                "pairs": [
                                    [int(refs[v] - ai * 60000), int(v), 0.0, 1.0]
                                    for v in vertices
                                ],
                            }
                        )
                mesh["world_zero"] = True
                mesh["axes"] = []
                mesh["cells"] = [
                    {
                        "coordinate": [],
                        "offset": emit(np.zeros((n, 2))),
                        "order": 0.0,
                        "opacity": 0.0,
                    }
                ]
                mesh["blend_shapes"] = []
                mesh["opacity"] = 0.0
            stats.append(
                {
                    "original_vertices": len(tracks),
                    "unique_tracks": len(unique),
                    "input_track_bytes": tracks.nbytes,
                    "pooled_track_bytes": unique.nbytes,
                    "meshes": len(cohort),
                }
            )
        for mesh in ir["meshes"]:
            if mesh["id"] not in replaced:
                clone_fields(mesh)
    ir["meshes"].extend(carriers)
    ir["glues"] = glues
    ir["parts"] = [{"id": "P_Distilled", "mesh_ids": [m["id"] for m in ir["meshes"]]}]
    for texture in ir["textures"]:
        import shutil

        shutil.copy2(source / texture, folder / texture)
    jsonwrite(folder / "POPY.rig.json", ir)
    jsonwrite(
        BASE / "mapping.json",
        {"rows": mapping, "extra_unobservable": [m["id"] for m in carriers]},
    )
    jsonwrite(
        BASE / "build.json",
        {
            "groups": stats,
            "zero_destinations": sorted(replaced),
            "glues": len(glues),
            "pairs": sum(len(g["pairs"]) for g in glues),
            "carriers": len(carriers),
        },
    )
    print(
        {
            "carriers": len(carriers),
            "glues": len(glues),
            "track_bytes_saved": sum(
                r["input_track_bytes"] - r["pooled_track_bytes"] for r in stats
            ),
        }
    )


def world_zero(source, target, names):
    """Give root-parent destinations a fresh immutable +0 vector block.

    Appending instead of editing a deduplicated block cannot corrupt another
    owner's geometry. Parent/position semantics are independently checked by
    the unchanged runtime during full vertex equivalence verification.
    """
    raw = Path(source).read_bytes()
    if raw[:8] != b"MOC3\x05\0\0\0":
        raise ValueError("Audited MOC5 only")
    offsets = np.frombuffer(raw, "<u4", 152, 64).copy()
    starts = sorted({int(v) for v in offsets if v})
    bounds = dict(zip(starts, starts[1:] + [len(raw)]))
    counts = np.frombuffer(raw, "<u4", 64, int(offsets[0])).copy()
    ids = [
        raw[int(offsets[33]) + i * 64 : int(offsets[33]) + (i + 1) * 64]
        .split(b"\0")[0]
        .decode()
        for i in range(int(counts[4]))
    ]
    form_start = np.frombuffer(raw, "<u4", counts[4], int(offsets[35]))
    form_count = np.frombuffer(raw, "<u4", counts[4], int(offsets[36]))
    refs = np.frombuffer(raw, "<u4", counts[9], int(offsets[70])).copy()
    npos = int(counts[10])
    # The common block covers the longest destination; readers consume prefixes.
    ir = json.loads((BASE / "rig/POPY.rig.json").read_text())
    lengths = {m["id"]: len(m["positions"]) for m in ir["meshes"] if m["id"] in names}
    if set(lengths) != set(names):
        raise ValueError("Missing zero destinations")
    for name in names:
        i = ids.index(name)
        if int(form_count[i]) != 1:
            raise ValueError("Zero destination must have one form")
        refs[int(form_start[i])] = npos
    payload = raw[int(offsets[71]) : int(offsets[71]) + 4 * npos] + bytes(
        4 * max(lengths.values())
    )
    counts[10] += max(lengths.values())
    changes = {
        int(offsets[0]): counts.tobytes(),
        int(offsets[70]): refs.tobytes(),
        int(offsets[71]): payload,
    }
    result = bytearray(raw[: starts[0]])
    relocated = {}
    for begin in starts:
        result.extend(bytes(-len(result) % 64))
        relocated[begin] = len(result)
        result.extend(changes.get(begin, raw[begin : bounds[begin]]))
    result.extend(bytes(-len(result) % 64))
    for i, begin in enumerate(offsets):
        struct.pack_into(
            "<I", result, 64 + 4 * i, relocated[int(begin)] if begin else 0
        )
    Path(target).write_bytes(result)


def export():
    exact.BASE, exact.SOURCE, exact.MODEL = BASE, SOURCE, MODEL
    exact.export()
    build = json.loads((BASE / "build.json").read_text())
    patched = MODEL / "POPY.zero.moc3"
    world_zero(MODEL / "POPY.moc3", patched, set(build["zero_destinations"]))
    # The trim tool uses the manifest's Moc entry; first install the patched
    # candidate, then discover its extents and repack into a separate file.
    patched.replace(MODEL / "POPY.moc3")
    packed = MODEL / "POPY.packed.moc3"
    pack(
        MODEL / "POPY.moc3",
        packed,
        BASE / "pack.json",
        runtime_extents(MODEL / "POPY.moc3"),
    )
    packed.replace(MODEL / "POPY.moc3")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["build", "export"])
    args = parser.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    globals()[args.stage]()


if __name__ == "__main__":
    main()
