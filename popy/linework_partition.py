"""Partition curved primary lashes/brows into source-connected, locally planar patches."""

import itertools
import json
from pathlib import Path

import numpy as np

from .compile_ir import ROOT
from .visibility import component_labels


def partition_linework(ir, samples, semantics, binary_path):
    config = json.loads((ROOT / "config/surface_visibility.json").read_text())
    critical_path = ROOT / "config/surface_critical.json"
    critical = (
        json.loads(critical_path.read_text()).get("linework_triangles", {})
        if critical_path.exists() and config.get("strategy") == "hybrid"
        else {}
    )
    if (
        critical_path.exists()
        and config.get("strategy") == "hybrid"
        and config.get("planar_neck")
    ):
        critical[config["hair"]["object"]] = json.loads(critical_path.read_text()).get(
            "neck_triangles", []
        )
    face = config["lashes"]["object"]
    _, components = component_labels(np.load(samples / f"{face}_source.npz"))
    manifest = json.loads((samples / "manifest.json").read_text())
    neutral = next(
        p
        for p in manifest["head_poses"]
        if all(v == 0 for v in p["parameters"].values())
    )
    archive = np.load(samples / neutral["file"])
    rest = {key: archive[key] for key in archive.files}
    head_lookup = {
        tuple(
            p["parameters"][k] for k in ["ParamAngleX", "ParamAngleY", "ParamAngleZ"]
        ): p["file"]
        for p in manifest["head_poses"]
    }
    head_cache, body_cache = {}, {}
    old = np.fromfile(binary_path, "<f4")
    cursor = len(old)
    result, diagnostics = [], []
    with Path(binary_path).open("ab") as stream:
        for mesh in ir["meshes"]:
            name = mesh["source"]["object"]
            selected = (
                name == config["brows"]["object"]
                or (
                    name == face
                    and np.all(
                        np.isin(components[mesh["source"]["triangles"]], [20, 21])
                    )
                )
                or bool(
                    set(critical.get(name, [])).intersection(
                        mesh["source"]["triangles"]
                    )
                )
            )
            if (
                mesh.get("mask_only")
                or mesh.get("linework_partition_parent")
                or not selected
            ):
                result.append(mesh)
                continue
            triangles = np.array(mesh["indices"]).reshape(-1, 3)
            source = mesh["source"]["vertices"]
            axes = [
                a
                for a in mesh["axes"]
                if a["parameter"] in semantics["shape_parameters"]
            ]
            poses = []
            for values in itertools.product(*(a["keys"] for a in axes)):
                points = rest[name][source].astype(float).copy()
                for a, value in zip(axes, values):
                    spec = semantics["shape_parameters"][a["parameter"]]
                    amount = 1 - value if spec.get("invert") else value
                    points += rest[name + "__" + a["parameter"]][source] * amount
                poses.append(points)

            def planar(group, triangles=triangles, poses=poses):
                vertices = np.unique(triangles[group])
                for points in poses:
                    p = points[vertices]
                    _, _, v = np.linalg.svd(p - p.mean(0), full_matrices=False)
                    if np.ptp(p @ v[-1]) > 0.00055:
                        return False
                    t = points[triangles[group]]
                    normals = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
                    normals /= np.maximum(
                        np.linalg.norm(normals, axis=1, keepdims=True), 1e-15
                    )
                    if np.min(normals @ normals[0]) < 0.8:
                        return False
                return True

            adjacency = {i: set() for i in range(len(triangles))}
            source_tris = np.array(source)[triangles]
            edge_uses = {}
            for i, tri in enumerate(source_tris):
                for a, b in zip(tri, np.roll(tri, 1)):
                    edge = tuple(sorted([int(a), int(b)]))
                    for other in edge_uses.get(edge, []):
                        adjacency[i].add(other)
                        adjacency[other].add(i)
                    edge_uses.setdefault(edge, []).append(i)
            unused = set(range(len(triangles)))
            groups = []
            while unused:
                group = [min(unused)]
                unused.remove(group[0])
                while len(group) < 12:
                    candidates = sorted(
                        set().union(*(adjacency[i] for i in group)) & unused
                    )
                    candidate = next(
                        (i for i in candidates if planar(group + [i])), None
                    )
                    if candidate is None:
                        break
                    group.append(candidate)
                    unused.remove(candidate)
                groups.append(group)
            if len(groups) == 1:
                result.append(mesh)
                continue
            # A child patch needs its OWN camera depth. Copying its curved
            # parent's mean depth preserves precisely the old coverage error.
            depths = []
            for c in mesh["cells"]:
                p = {
                    a["parameter"]: a["keys"][i]
                    for a, i in zip(mesh["axes"], c["coordinate"])
                }
                key = tuple(
                    p.get(k, 0) for k in ["ParamAngleX", "ParamAngleY", "ParamAngleZ"]
                )
                cache_key = (name, key)
                if cache_key not in head_cache:
                    archive = np.load(samples / head_lookup[key])
                    head_cache[cache_key] = {
                        k: archive[k]
                        for k in archive.files
                        if k == name or k.startswith(name + "__Param")
                    }
                hs = head_cache[cache_key]
                q = hs[name][source].astype(float).copy()
                for pid, spec in semantics["shape_parameters"].items():
                    if spec["object"] == name:
                        value = p.get(pid, spec["default"])
                        q += hs[name + "__" + pid][source] * (
                            1 - value if spec.get("invert") else value
                        )
                bx, by = p.get("ParamBodyAngleX", 0), p.get("ParamBodyAngleY", 0)
                if bx or by:
                    cache_key = (name, bx, by)
                    if cache_key not in body_cache:
                        body_cache[cache_key] = np.load(
                            samples / f"body_{bx}_{by}.npz"
                        )[name]
                    a, b = np.radians([bx, by])
                    rotation = np.array(
                        [
                            [np.cos(a), -np.sin(a), 0],
                            [np.sin(a), np.cos(a), 0],
                            [0, 0, 1],
                        ]
                    ) @ np.array(
                        [
                            [1, 0, 0],
                            [0, np.cos(b), -np.sin(b)],
                            [0, np.sin(b), np.cos(b)],
                        ]
                    )
                    q = (
                        body_cache[cache_key][source]
                        + (q - rest[name][source]) @ rotation.T
                    )
                depths.append(q[:, 1])
            depths = np.asarray(depths)
            for gi, group in enumerate(groups):
                used = np.unique(triangles[group])
                remap = {int(v): i for i, v in enumerate(used)}
                size = len(mesh["positions"])
                cache = {}

                def select(offset, cache=cache, size=size, used=used):
                    nonlocal cursor
                    if offset is None:
                        return None
                    if offset not in cache:
                        value = (
                            old[offset : offset + size]
                            .reshape(-1, 2)[used]
                            .astype("<f4")
                            .ravel()
                        )
                        cache[offset] = cursor
                        stream.write(value.tobytes())
                        cursor += len(value)
                    return cache[offset]

                blends = []
                for b in mesh["blend_shapes"]:
                    orders, measured = [], []
                    for value, previous in zip(
                        b["keys"], b.get("orders", [500] * len(b["keys"]))
                    ):
                        points = rest[name][source].astype(float).copy()
                        pid = b["parameter"]
                        if pid in semantics["shape_parameters"]:
                            spec = semantics["shape_parameters"][pid]
                            points += rest[name + "__" + pid][source] * (
                                1 - value if spec.get("invert") else value
                            )
                        elif value:
                            points += np.load(samples / f"{pid}_{int(value)}.npz")[
                                name
                            ][source]
                        orders.append(
                            float(
                                previous
                                - 800 * (points[used, 1].mean() - points[:, 1].mean())
                            )
                        )
                        measured.append(float(500 - 800 * points[used, 1].mean()))
                    blends.append(
                        {
                            **b,
                            "offsets": [select(o) for o in b["offsets"]],
                            "orders": orders,
                            "depth_orders": measured,
                        }
                    )
                child = {
                    **mesh,
                    "id": f"{mesh['id']}_P{gi}",
                    "linework_partition_parent": mesh["id"],
                    "positions": np.array(mesh["positions"])
                    .reshape(-1, 2)[used]
                    .ravel()
                    .tolist(),
                    "uvs": np.array(mesh["uvs"]).reshape(-1, 2)[used].ravel().tolist(),
                    "indices": [remap[int(v)] for v in triangles[group].ravel()],
                    "source": {
                        **mesh["source"],
                        "vertices": np.array(source)[used].tolist(),
                        "triangles": np.array(mesh["source"]["triangles"])[
                            group
                        ].tolist(),
                    },
                    "cells": [
                        {
                            **c,
                            "offset": select(c["offset"]),
                            "order": float(
                                c["order"]
                                - 800 * (depths[ci, used].mean() - depths[ci].mean())
                            ),
                            "depth_order": float(500 - 800 * depths[ci, used].mean()),
                        }
                        for ci, c in enumerate(mesh["cells"])
                    ],
                    "blend_shapes": blends,
                }
                result.append(child)
            diagnostics.append(
                {
                    "parent": mesh["id"],
                    "patches": len(groups),
                    "triangles": len(triangles),
                }
            )
    ir["meshes"] = result
    previous = ir["diagnostics"].get("linework_partition", {}).get("patches", [])
    combined = {p["parent"]: p for p in previous + diagnostics}
    ir["diagnostics"]["linework_partition"] = {
        "patches": list(combined.values()),
        "depth_spread_tolerance_m": 0.00055,
        "max_triangles": 12,
        "normal_dot_min": 0.8,
        "all_source_expression_extremes": True,
    }
    print(
        "Planar linework subdivision",
        sum(x["patches"] - 1 for x in diagnostics),
        "additional ArtMeshes",
        flush=True,
    )
