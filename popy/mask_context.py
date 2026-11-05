"""Mask-context budget experiment: real MOC rewrite, always a separate candidate.

Material unions deliberately require rendered acceptance: fewer contexts alone
does not imply correct depth-dependent occlusion.
"""

import argparse
import collections
import json
import struct
from pathlib import Path

import numpy as np

from .compile_ir import ROOT, jsonwrite


def audit_inventory(inventory, limit=36):
    contexts = {tuple(sorted(m["clips"])) for m in inventory if m["clips"]}
    return {
        "mask_contexts": len(contexts),
        "masked_artmeshes": sum(bool(m["clips"]) for m in inventory),
        "largest_source_list": max((len(m["clips"]) for m in inventory), default=0),
        "budget": limit,
        "within_budget": len(contexts) <= limit,
    }


def group_inventory(inventory, centers=None, vts=False):
    groups = collections.defaultdict(list)
    for m in inventory:
        if m["clips"]:
            name = m["id"].split("_")
            semantic = "_".join(name[:2]) if name[0] == "M" else name[0]
            groups[(semantic, m["invert_mask"])].append(m)
    if centers is None:
        if vts:
            raise ValueError("VTS packing requires spatial centers")
        return list(groups.values())
    from scipy.cluster.vq import kmeans2

    result = []
    budgets = {"M_Body": 1, "M_Eye": 2, "M_Brow": 2, "M_Face": 10, "M_Hair": 21}
    if vts:
        budgets = {"M_Body": 1, "M_Eye": 3, "M_Brow": 3, "M_Face": 24, "M_Hair": 33}
    for (semantic, _), members in groups.items():
        count = min(budgets.get(semantic, 1), len(members))
        if count == 1 and not vts:
            result.append(members)
            continue
        points = np.array([centers[m["id"]] for m in members])
        if count == 1:
            centroids, labels = (
                points.mean(0, keepdims=True),
                np.zeros(len(points), dtype=int),
            )
        else:
            centroids, labels = kmeans2(points, count, iter=40, minit="++", seed=811)
        if vts:
            # Preserve all original sources and forbid newly introduced self
            # masks. Overflow is explicit rather than deleting occluders.
            bins = [[] for _ in range(count)]
            unions = [set() for _ in range(count)]
            names = [set() for _ in range(count)]
            ids = {m["id"] for m in members}
            for i in sorted(
                range(len(members)),
                key=lambda i: -len(ids.intersection(members[i]["clips"])),
            ):
                m = members[i]
                sources = set(m["clips"])
                choices = [
                    j
                    for j in range(len(bins))
                    if m["id"] not in unions[j] and not sources.intersection(names[j])
                ]
                if choices:
                    j = min(
                        choices,
                        key=lambda j: float(
                            np.linalg.norm(
                                points[i] - centroids[min(j, len(centroids) - 1)]
                            )
                        ),
                    )
                else:
                    j = len(bins)
                    bins.append([])
                    unions.append(set())
                    names.append(set())
                bins[j].append(m)
                unions[j].update(sources)
                names[j].add(m["id"])
            result.extend(g for g in bins if g)
            continue
        result.extend(
            [
                [m for m, k in zip(members, labels) if k == i]
                for i in range(count)
                if np.any(labels == i)
            ]
        )
    return result


def rewrite(
    source, target, inventory, report, centers=None, exclude_members=False, vts=False, design=None, clip_overrides=None
):
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve():
        raise ValueError("Separate candidate required")
    raw = source.read_bytes()
    if raw[:8] != b"MOC3\x05\0\0\0":
        raise ValueError("Audited MOC5 only")
    offsets = np.frombuffer(raw, "<u4", 152, 64).copy()
    starts = sorted({int(s) for s in offsets if s})
    bounds = dict(zip(starts, starts[1:] + [len(raw)]))
    counts = np.frombuffer(raw, "<u4", 64, int(offsets[0])).copy()
    if len(inventory) != counts[4]:
        raise ValueError("Inventory mismatch")
    # Bind public inventory to the actual IDs and actual original clip lists.
    ids = [
        raw[int(offsets[33]) + i * 64 : int(offsets[33]) + (i + 1) * 64]
        .split(b"\0")[0]
        .decode()
        for i in range(len(inventory))
    ]
    if ids != [m["id"] for m in inventory]:
        raise ValueError("Mesh identity mismatch")

    def array(section, count):
        return np.frombuffer(raw, "<u4", int(count), int(offsets[section]))

    refs = array(80, counts[17])
    oldstart = array(47, counts[4])
    oldcount = array(48, counts[4])
    for i, m in enumerate(inventory):
        if [ids[j] for j in refs[oldstart[i] : oldstart[i] + oldcount[i]]] != m[
            "clips"
        ]:
            raise ValueError("Clip inventory mismatch")
    lookup = {n: i for i, n in enumerate(ids)}
    replacements = {}
    removed = 0
    for group in (design if design is not None else group_inventory(inventory, centers, vts)):
        union = sorted({s for m in group for s in m["clips"]})
        if exclude_members:
            members = {m["id"] for m in group}
            removed += sum(s in members for s in union)
            union = [s for s in union if s not in members]
        for m in group:
            replacements[m["id"]] = union
    if clip_overrides is not None:
        replacements={m['id']:clip_overrides.get(m['id'],m['clips']) for m in inventory}
    pool = []
    spans = {}
    newstart = []
    newcount = []
    for m in inventory:
        names = tuple(replacements.get(m["id"], []))
        if names not in spans:
            spans[names] = len(pool)
            pool.extend(lookup[s] for s in names)
        newstart.append(spans[names])
        newcount.append(len(names))
    counts[17] = len(pool)
    changes = {
        int(offsets[0]): counts.tobytes(),
        int(offsets[47]): np.array(newstart, dtype="<u4").tobytes(),
        int(offsets[48]): np.array(newcount, dtype="<u4").tobytes(),
        int(offsets[80]): np.array(pool, dtype="<u4").tobytes(),
    }
    result = bytearray(raw[: starts[0]])
    relocated = {}
    for start in starts:
        result.extend(b"\0" * (-len(result) % 64))
        relocated[start] = len(result)
        result.extend(changes.get(start, raw[start : bounds[start]]))
    result.extend(b"\0" * (-len(result) % 64))
    for i, s in enumerate(offsets):
        struct.pack_into("<I", result, 64 + 4 * i, relocated[int(s)] if s else 0)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(result)
    r = {
        "before_contexts": len(
            {tuple(sorted(m["clips"])) for m in inventory if m["clips"]}
        ),
        "after_contexts": len(spans) - int(() in spans),
        "original_glue_program_unchanged": True,
        "union_member_sources_removed": removed,
        "original_sources_preserved": all(
            set(m["clips"]).issubset(replacements.get(m["id"], []))
            for m in inventory
            if m["clips"]
        ),
        "self_masks_introduced": sum(
            m["id"] in replacements.get(m["id"], []) and m["id"] not in m["clips"]
            for m in inventory
        ),
        "status": "Experimental semantic unions; may over-occlude. Must not replace default without rendered acceptance.",
    }
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    jsonwrite(report, r)
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source", type=Path, default=ROOT / "output/POPY_channelpacked/POPY.moc3"
    )
    p.add_argument(
        "--out", type=Path, default=ROOT / "output/POPY_mask_context_trial/POPY.moc3"
    )
    p.add_argument(
        "--inventory",
        type=Path,
        default=ROOT / "artifacts/representation_final/candidate/model_inventory.json",
    )
    p.add_argument("--spatial-rig", type=Path)
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--limit", type=int, choices=[36, 64, 256])
    p.add_argument("--exclude-members", action="store_true")
    p.add_argument("--vts", action="store_true")
    a = p.parse_args()
    if a.audit_only:
        result = audit_inventory(
            json.loads(a.inventory.read_text()), a.limit or (64 if a.vts else 36)
        )
        print(json.dumps(result))
        if not result["within_budget"]:
            raise SystemExit(1)
        return
    if a.vts and a.exclude_members:
        p.error(
            "VTS mode preserves original sources; excluded-member experiment is not supported"
        )
    if a.vts and not a.spatial_rig:
        p.error("VTS packing requires --spatial-rig")
    centers = None
    if a.spatial_rig:
        ir = json.loads(a.spatial_rig.read_text())
        centers = {}
        sources = {}
        for m in ir["meshes"]:
            if not m.get("masked_by"):
                continue
            name = m["source"]["object"]
            if name not in sources:
                with np.load(
                    ROOT / "artifacts/samples" / (name + "_source.npz")
                ) as archive:
                    sources[name] = archive["positions"]
            centers[m["id"]] = sources[name][m["source"]["vertices"]].mean(0).tolist()
    print(
        rewrite(
            a.source,
            a.out,
            json.loads(a.inventory.read_text()),
            ROOT / "artifacts/deformer_fit/mask_trial.json",
            centers,
            a.exclude_members,
            a.vts,
        )
    )


if __name__ == "__main__":
    main()
