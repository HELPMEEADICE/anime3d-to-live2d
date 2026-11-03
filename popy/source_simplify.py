"""Conservative endpoint edge collapse with original skin/UV correspondence."""

import numpy as np


def endpoint_collapse(source, candidates, samples, pinned=None, error_m=0.0015):
    """Never invent vertices: collapse only connected edges onto an original endpoint.

    UV seams, nonmanifold links and original triangle orientation are preserved.
    A collapse is accepted only if every removed vertex stays within error_m of
    its retained source vertex in EVERY supplied 3D skinning sample. This finite
    envelope check is not a mathematical guarantee for unseen joint poses.
    """
    positions = np.asarray(source["positions"])
    triangles = np.array(source["triangles"]).copy()
    uv = np.array(source["uv"]).copy()
    n = len(positions)
    pinned = np.zeros(n, bool) if pinned is None else np.array(pinned, bool).copy()
    canonical = np.zeros((n, 2))
    seen = np.zeros(n, bool)
    for v, t in zip(triangles.ravel(), uv.reshape(-1, 2)):
        if seen[v] and np.max(np.abs(canonical[v] - t)) > 1e-6:
            pinned[v] = True
        canonical[v] = t
        seen[v] = True
    edges = np.unique(
        np.sort(
            np.concatenate(
                [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
            ),
            axis=1,
        ),
        axis=0,
    )
    edges = edges[candidates[edges].all(1) & ~pinned[edges].any(1)]
    length = np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)
    edges = edges[np.argsort(length, kind="stable")]
    parent = np.arange(n)
    members = {int(i): [int(i)] for i in np.flatnonzero(candidates & ~pinned)}
    active = np.ones(len(triangles), bool)
    accepted = 0
    maximum = 0.0

    def root(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return int(v)

    for original_a, original_b in edges:
        a, b = root(original_a), root(original_b)
        if a == b:
            continue
        merged = members[a] + members[b]
        options = []
        for target in [a, b]:
            error = float(
                np.max(
                    np.linalg.norm(
                        samples[:, merged] - samples[:, target, None], axis=2
                    )
                )
            )
            if error <= error_m:
                options.append((error, target))
        if not options:
            continue
        error, target = min(options)
        removed = b if target == a else a
        incident = np.flatnonzero(
            active & np.any((triangles == a) | (triangles == b), axis=1)
        )
        local = triangles[incident]
        common = local[np.any(local == a, axis=1) & np.any(local == b, axis=1)]
        if len(common) not in [1, 2]:
            continue
        na = set(local[np.any(local == a, axis=1)].ravel()) - {a, b}
        nb = set(local[np.any(local == b, axis=1)].ravel()) - {a, b}
        opposite = set(common.ravel()) - {a, b}
        if na & nb != opposite:
            continue
        after = local.copy()
        after[after == removed] = target
        keep = np.array([len(set(t)) == 3 for t in after])
        oldp = positions[local[keep]]
        newp = positions[after[keep]]
        oldn = np.cross(oldp[:, 1] - oldp[:, 0], oldp[:, 2] - oldp[:, 0])
        newn = np.cross(newp[:, 1] - newp[:, 0], newp[:, 2] - newp[:, 0])
        if np.any(
            np.sum(oldn * newn, axis=1)
            <= 0.1 * np.linalg.norm(oldn, axis=1) * np.linalg.norm(newn, axis=1)
        ):
            continue
        if np.any(np.linalg.norm(newn, axis=1) < 1e-12):
            continue
        triangles[incident] = after
        active[incident[~keep]] = False
        parent[removed] = target
        members[target] = merged
        del members[removed]
        accepted += 1
        maximum = max(maximum, error)
    for i in range(n):
        parent[i] = root(i)
    changed = parent[source["triangles"]] != source["triangles"]
    uv[changed] = canonical[triangles[changed]]
    result = {k: np.array(v) for k, v in source.items()}
    result["triangles"] = triangles
    result["uv"] = uv
    result["vertex_remap"] = parent
    return (
        result,
        active,
        {
            "candidate_vertices": int(np.sum(candidates)),
            "collapsed_vertices": accepted,
            "retained_vertices": len(np.unique(triangles[active])),
            "removed_triangles": int(np.sum(~active)),
            "max_sampled_source_displacement_m": maximum,
            "error_limit_m": error_m,
            "sample_count": len(samples),
            "method": "connected source endpoint collapse; UV seam/link/orientation constraints; sampled real skinning distance",
        },
    )
