"""Cache-aware opaque mask indexing and immutable position-block path packing.

Vertex/triangle permutations retain coordinates and primitive winding exactly.
Storage paths share byte-identical suffix/prefix regions, with no runtime Glue.
"""

import heapq
import json
import struct
from collections import Counter, defaultdict

import numpy as np


def cache_order(indices, vertices, capacity=32):
    triangles = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    adjacency = [set() for _ in range(vertices)]
    for i, tri in enumerate(triangles):
        for v in tri:
            adjacency[v].add(i)
    live = np.array([len(a) for a in adjacency])
    positions = [-1] * vertices
    scores = np.zeros(vertices)
    versions = np.zeros(len(triangles), dtype=np.int64)
    emitted = np.zeros(len(triangles), bool)
    cache = []

    def score(v):
        if live[v] == 0:
            return -1.0
        valence = 2.0 / np.sqrt(live[v])
        p = positions[v]
        return valence + (
            0 if p < 0 else 0.75 if p < 3 else (1 - (p - 3) / (capacity - 3)) ** 1.5
        )

    for v in range(vertices):
        scores[v] = score(v)
    heap = [(-float(scores[t].sum()), i, 0) for i, t in enumerate(triangles)]
    heapq.heapify(heap)
    order = []
    while heap:
        _, ti, version = heapq.heappop(heap)
        if emitted[ti] or version != versions[ti]:
            continue
        emitted[ti] = True
        order.append(ti)
        tri = triangles[ti].tolist()
        changed = set(cache) | set(tri)
        for v in set(tri):
            adjacency[v].remove(ti)
            live[v] -= 1
        cache = (list(dict.fromkeys(tri)) + [v for v in cache if v not in tri])[
            :capacity
        ]
        for v in changed:
            positions[v] = -1
        for i, v in enumerate(cache):
            positions[v] = i
        touched = set()
        for v in changed:
            scores[v] = score(v)
            touched.update(adjacency[v])
        for i in touched:
            versions[i] += 1
            heapq.heappush(
                heap, (-float(scores[triangles[i]].sum()), i, int(versions[i]))
            )
    return triangles[order].ravel().tolist(), order


def first_use(indices, vertices):
    return list(dict.fromkeys([*indices, *range(vertices)]))


def overlap_paths(blocks, edges):
    """Greedy weighted acyclic path cover of immutable byte strings."""
    parent = list(range(len(blocks)))
    before, after = {}, {}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for overlap, a, b in sorted(set(edges), reverse=True):
        if a == b or a in after or b in before or root(a) == root(b):
            continue
        if (
            overlap <= 0
            or overlap > min(len(blocks[a]), len(blocks[b]))
            or overlap % 8
            or blocks[a][-overlap:] != blocks[b][:overlap]
        ):
            raise ValueError("Nonidentical or unaligned overlap")
        after[a] = (b, overlap)
        before[b] = a
        parent[root(b)] = root(a)
    payload = bytearray()
    offsets = {}
    for start in range(len(blocks)):
        if start in before:
            continue
        offsets[start] = len(payload)
        payload.extend(blocks[start])
        node = start
        while node in after:
            nxt, overlap = after[node]
            offsets[nxt] = len(payload) - overlap
            payload.extend(blocks[nxt][overlap:])
            node = nxt
    if len(offsets) != len(blocks):
        raise ValueError("Incomplete storage graph")
    return bytes(payload), offsets, len(after)


def pack_positions(source, target, layout, pairs, report_path):
    raw = source.read_bytes()
    if raw[:8] != b"MOC3\x05\0\0\0":
        raise ValueError("Audited MOC5 only")
    offsets = np.frombuffer(raw, "<u4", 152, 64).copy()
    starts = sorted({int(p) for p in offsets if p})
    bounds = dict(zip(starts, starts[1:] + [len(raw)]))
    counts = np.frombuffer(raw, "<u4", 64, int(offsets[0])).copy()
    if counts[7] or counts[10] == 0:
        raise ValueError("This profile has rotation parents only")
    refs = np.frombuffer(raw, "<u4", counts[9], int(offsets[70])).copy()
    blocks, block_id, by_form = [], {}, {}
    by_mesh = {m["id"]: m for m in layout["meshes"]}
    for mesh in layout["meshes"]:
        size = mesh["vertices"] * 8
        for form in mesh["forms"]:
            if not 0 <= form < len(refs):
                raise ValueError("Form outside position reference table")
            start = int(offsets[71]) + 4 * int(refs[form])
            if start + size > int(offsets[71]) + 4 * int(counts[10]):
                raise ValueError("Form position extent outside pool")
            block = raw[start : start + size]
            if block not in block_id:
                block_id[block] = len(blocks)
                blocks.append(block)
            if form in by_form and by_form[form] != block_id[block]:
                raise ValueError("Conflicting shared form extent")
            by_form[form] = block_id[block]
    if len(by_form) != counts[9]:
        raise ValueError("Incomplete form ownership")
    edges = []
    accepted = failed = 0
    for pair in pairs:
        a, b = by_mesh[pair["a"]], by_mesh[pair["b"]]
        if len(a["forms"]) != len(b["forms"]):
            raise ValueError("Program mismatch")
        overlap = pair["shared"] * 8
        for fa, fb in zip(a["forms"], b["forms"]):
            ia, ib = by_form[fa], by_form[fb]
            if blocks[ia][-overlap:] == blocks[ib][:overlap]:
                edges.append((overlap, ia, ib))
                accepted += 1
            else:
                failed += 1
    # Generic exact prefix/suffix containment also shares unchanged static rows.
    for suffix in [False, True]:
        previous = None
        for i in sorted(
            range(len(blocks)),
            key=lambda i: blocks[i][::-1] if suffix else blocks[i],
            reverse=True,
        ):
            if previous is not None:
                a, b = blocks[previous], blocks[i]
                if a.endswith(b) if suffix else a.startswith(b):
                    edges.append(
                        (len(b), previous, i) if suffix else (len(b), i, previous)
                    )
            previous = i
    payload, mapped, links = overlap_paths(blocks, edges)
    for form, bid in by_form.items():
        refs[form] = mapped[bid] // 4
    counts[10] = len(payload) // 4
    changes = {
        int(offsets[0]): counts.tobytes(),
        int(offsets[70]): refs.tobytes(),
        int(offsets[71]): payload,
    }
    result = bytearray(raw[: starts[0]])
    relocated = {}
    for start in starts:
        result.extend(bytes(-len(result) % 64))
        relocated[start] = len(result)
        result.extend(changes.get(start, raw[start : bounds[start]]))
    result.extend(bytes(-len(result) % 64))
    for i, p in enumerate(offsets):
        struct.pack_into("<I", result, 64 + 4 * i, relocated[int(p)] if p else 0)
    target.write_bytes(result)
    report = {
        "input_bytes": len(raw),
        "output_bytes": len(result),
        "unique_blocks": len(blocks),
        "unique_block_bytes": sum(map(len, blocks)),
        "pooled_position_bytes": len(payload),
        "path_links": links,
        "accepted_pair_forms": accepted,
        "unmatched_pair_forms": failed,
        "method": "Acyclic weighted suffix/prefix path cover; every shared byte compared exactly; form reads unchanged, no new arithmetic",
    }
    report_path.write_text(json.dumps(report))
    return report


def paired_permutations(meshes, data, program):
    """Disjoint weighted mesh matching; place common tracks at suffix/prefix."""
    grouped = defaultdict(list)
    for mesh in meshes:
        grouped[program(mesh)].append(mesh)
    orders, pairs = {}, []
    for cohort in grouped.values():
        chunks = []
        for mesh in cohort:
            n = len(mesh["positions"]) // 2
            offsets = [c["offset"] for c in mesh["cells"]] + [
                o for b in mesh["blend_shapes"] for o in b["offsets"] if o is not None
            ]
            chunks.append(
                np.stack([data[o : o + 2 * n].reshape(n, 2) for o in offsets])
                .transpose(1, 0, 2)
                .reshape(n, -1)
                .copy()
            )
        tracks = np.concatenate(chunks)
        _, inverse = np.unique(tracks.view("<u4"), axis=0, return_inverse=True)
        identities = []
        start = 0
        for mesh in cohort:
            n = len(mesh["positions"]) // 2
            counts = Counter()
            ids = []
            for v in inverse[start : start + n]:
                ids.append((int(v), counts[int(v)]))
                counts[int(v)] += 1
            identities.append(ids)
            start += n
        edges = []
        for a in range(len(cohort)):
            sa = set(identities[a])
            for b in range(a + 1, len(cohort)):
                common = sa.intersection(identities[b])
                if common:
                    edges.append((len(common), a, b, common))
        selected = set()
        for count, a, b, common in sorted(edges, key=lambda e: e[0], reverse=True):
            if a in selected or b in selected:
                continue
            selected.update([a, b])
            ia = {v: i for i, v in enumerate(identities[a])}
            ib = {v: i for i, v in enumerate(identities[b])}
            ca = first_use(cohort[a]["indices"], len(identities[a]))
            cb = first_use(cohort[b]["indices"], len(identities[b]))
            common_set = common
            common = [identities[a][i] for i in ca if identities[a][i] in common_set]
            orders[cohort[a]["id"]] = [
                i for i in ca if identities[a][i] not in common_set
            ] + [ia[v] for v in common]
            orders[cohort[b]["id"]] = [ib[v] for v in common] + [
                i for i in cb if identities[b][i] not in common_set
            ]
            pairs.append({"a": cohort[a]["id"], "b": cohort[b]["id"], "shared": count})
    return orders, pairs
