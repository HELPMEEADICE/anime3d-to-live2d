"""Pose-dependent collar/neck foreground clipping for actual source hair surfaces."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .compile_ir import ROOT, project
from .depth_masks import (
    barycentric_glue_weights,
    clip_barycentric_batch,
    deduplicate_barycentric,
    plane,
)


def protect_neck_depth(ir, samples, semantics, binary_path):
    config = json.loads((ROOT / "config/surface_visibility.json").read_text())
    hair = config["hair"]["object"]
    meshes = [m for m in ir["meshes"] if not m.get("mask_only")]
    refinement = json.loads((ROOT / "config/neck_refinement.json").read_text())
    if (
        hashlib.sha256((samples / f"{hair}_source.npz").read_bytes()).hexdigest()
        != refinement["source_sha256"]
    ):
        raise ValueError(
            "Neck refinement was measured against a different source topology"
        )
    selected = set(refinement["triangles"])
    if config.get("strategy") == "hybrid":
        selected = set(
            json.loads((ROOT / "config/surface_critical.json").read_text())[
                "neck_triangles"
            ]
        )
    targets = [
        m
        for m in meshes
        if m["source"]["object"] == hair
        and selected.intersection(m["source"]["triangles"])
    ]
    skin = (
        set(ir["visibility"]["skin_surface_meshes"])
        if config.get("neck_occluder_face")
        else set()
    )
    occluders = [
        m
        for m in meshes
        if m["source"]["object"] in ["M_Body", "M_Top", "M_Bow", "M_Necktie"]
        or m["id"] in skin
    ]
    manifest = json.loads((samples / "manifest.json").read_text())
    poses = [
        p
        for p in manifest["head_poses"]
        if p["parameters"]["ParamAngleX"] in [-45, -30, 0, 30, 45]
        and p["parameters"]["ParamAngleY"] in [-30, 0, 30]
    ]
    names = {m["source"]["object"] for m in targets + occluders}
    world = {n: [] for n in names}
    for p in poses:
        f = np.load(samples / p["file"])
        for n in names:
            world[n].append(f[n])
    world = {n: np.array(v) for n, v in world.items()}
    screen = {
        n: np.array([project(v, ir["camera"]) for v in a]) for n, a in world.items()
    }
    bounds = {
        m["id"]: (
            screen[m["source"]["object"]][:, m["source"]["vertices"]].min(1),
            screen[m["source"]["object"]][:, m["source"]["vertices"]].max(1),
        )
        for m in targets + occluders
    }
    axes = [
        {"parameter": p, "keys": sorted({x["parameters"][p] for x in poses})}
        for p in ["ParamAngleX", "ParamAngleY", "ParamAngleZ"]
    ]
    coordinates = [
        [a["keys"].index(p["parameters"][a["parameter"]]) for a in axes] for p in poses
    ]
    neutral = next(
        i for i, p in enumerate(poses) if all(v == 0 for v in p["parameters"].values())
    )
    cursor = Path(binary_path).stat().st_size // 4
    start_cursor = cursor
    helpers = []
    glues = []
    diagnostics = []
    with Path(binary_path).open("ab") as stream:
        for target in targets:
            points = world[hair][:, target["source"]["vertices"]]
            planes = [plane(p) for p in points]
            normals = np.array([n for n, t in planes])
            thresholds = np.array([t for n, t in planes])
            lo, hi = bounds[target["id"]]
            sources = []
            positions = []
            indices = []
            masters = []
            provenance = []
            helper_id = "MaskNeckDepth_" + target["id"]
            for occ in occluders:
                name = occ["source"]["object"]
                a, b = bounds[occ["id"]]
                if not np.any(np.all((hi + 0.035 >= a) & (b + 0.035 >= lo), axis=1)):
                    continue
                vertices = occ["source"]["vertices"]
                tri = np.array(occ["indices"]).reshape(-1, 3)
                xyz = world[name][:, vertices]
                d = np.einsum("pvi,pi->pv", xyz, normals)
                if np.all(d.max(1) <= thresholds):
                    sources.append(occ["id"])
                    continue
                projected = screen[name][:, vertices][:, tri]
                low, high = projected.min(2), projected.max(2)
                keep = np.any(
                    (d[:, tri].min(2) <= thresholds[:, None])
                    & np.all(
                        (hi[:, None] + 0.035 >= low) & (high + 0.035 >= lo[:, None]),
                        axis=2,
                    ),
                    axis=0,
                )
                if not keep.any():
                    continue
                tri = tri[keep]
                bary = clip_barycentric_batch(d[:, tri], thresholds[:, None])
                q = np.einsum(
                    "ptvi,ptic->ptvc", bary, screen[name][:, vertices][:, tri]
                )
                target_screen = screen[hair][:, target["source"]["vertices"]]
                target_tri = np.array(target["indices"]).reshape(-1, 3)
                t = target_screen[:, target_tri]
                u, v = t[:, :, 1] - t[:, :, 0], t[:, :, 2] - t[:, :, 0]
                front = np.any(
                    u[:, :, 0] * v[:, :, 1] - u[:, :, 1] * v[:, :, 0] > 1e-12, axis=1
                )
                u, v = q[:, :, 1] - q[:, :, 0], q[:, :, 2] - q[:, :, 0]
                u2, v2 = q[:, :, 2] - q[:, :, 0], q[:, :, 3] - q[:, :, 0]
                area = (u[:, :, 0] * v[:, :, 1] - u[:, :, 1] * v[:, :, 0] > 1e-12) | (
                    u2[:, :, 0] * v2[:, :, 1] - u2[:, :, 1] * v2[:, :, 0] > 1e-12
                )
                footprint = np.any(
                    np.all(
                        (hi[:, None] + 0.008 >= q.min(2))
                        & (q.max(2) + 0.008 >= lo[:, None]),
                        axis=2,
                    )
                    & area
                    & front[:, None],
                    axis=0,
                )
                if not footprint.any():
                    continue
                tri = tri[footprint]
                bary = bary[:, footprint]
                offset = len(positions)
                local = np.array(occ["positions"]).reshape(-1, 2)[tri]
                master, compact_bary, remap, selected = deduplicate_barycentric(
                    np.repeat(tri, 4, axis=0), bary.reshape(len(poses), -1, 3)
                )
                positions.extend(
                    np.einsum("tvi,tic->tvc", bary[neutral], local)
                    .reshape(-1, 2)[selected]
                    .tolist()
                )
                indices.extend(
                    [
                        offset + int(remap[4 * i + j])
                        for i in range(len(tri))
                        for j in [0, 1, 2, 0, 2, 3]
                    ]
                )
                masters.append(
                    (occ["id"], offset, master, barycentric_glue_weights(compact_bary))
                )
                provenance.append(
                    {
                        "mesh": occ["id"],
                        "mask_vertices": (remap.reshape(-1, 4) + offset).tolist(),
                        "source_triangles": np.array(occ["source"]["triangles"])[keep][
                            footprint
                        ].tolist(),
                        "original_mesh_vertices": tri.tolist(),
                        "axes": axes,
                        "barycentric": "encoded by ordered G_Depth_Neck native Glue intensity tracks; no duplicated pose geometry",
                    }
                )
            if positions:
                value = np.asarray(positions, dtype="<f4").ravel()
                form_offset = cursor
                stream.write(np.zeros_like(value).tobytes())
                cursor += len(value)
                helper = {
                    **target,
                    "id": helper_id,
                    "axes": [],
                    "cells": [{"coordinate": [], "offset": form_offset, "order": 0}],
                    "positions": value.tolist(),
                    "uvs": [0.5, 0.5] * len(positions),
                    "indices": indices,
                    "texture": 0,
                    "mask_only": True,
                    "opacity": 0.0,
                    "masked_by": [],
                    "invert_mask": False,
                    "blend_shapes": [],
                    "source": {
                        "object": hair,
                        "vertices": [target["source"]["vertices"][0]] * len(positions),
                        "triangles": [target["source"]["triangles"][0]]
                        * (len(indices) // 3),
                        "depth_target": target["id"],
                        "correspondence": "pose-dependent source barycentric Glue; head-local clip field, body geometry follows original evaluated masters",
                        "barycentric_sources": provenance,
                    },
                }
                helpers.append(helper)
                sources.append(helper_id)
                for stage in range(3):
                    for master_id, offset, master, weights in masters:
                        groups = defaultdict(list)
                        for j in range(len(master)):
                            raw = (
                                np.array([1.0])
                                if stage == 0
                                else weights[stage - 1][:, j]
                            )
                            scale = float(raw.max())
                            if scale <= 1e-10:
                                continue
                            values = tuple(np.round(raw / scale, 5))
                            groups[values].append(
                                [int(master[j, stage]), offset + j, 0.0, scale]
                            )
                        for values, pairs in groups.items():
                            g = {
                                "id": f"G_Depth_Neck_{len(glues)}",
                                "mesh_a": master_id,
                                "mesh_b": helper_id,
                                "pairs": pairs,
                                "intensity": float(values[0]),
                            }
                            if len(set(values)) > 1:
                                array = np.empty(tuple(len(a["keys"]) for a in axes))
                                for c, v in zip(coordinates, values):
                                    array[tuple(c)] = v
                                g.update(axes=axes, intensities=array.ravel().tolist())
                            glues.append(g)
            target["masked_by"] = sources
            target["invert_mask"] = True
            diagnostics.append(
                {
                    "target": target["id"],
                    "sources": len(sources),
                    "vertices": len(positions),
                }
            )
            if len(diagnostics) % 20 == 0:
                print(
                    "Neck depth progress",
                    len(diagnostics),
                    "/",
                    len(targets),
                    flush=True,
                )
    ir["meshes"].extend(helpers)
    ir.setdefault("glues", []).extend(glues)
    ir["surface_visibility"]["neck_depth"] = diagnostics
    ir["diagnostics"]["neck_depth_masks"] = {
        "targets": len(targets),
        "helpers": len(helpers),
        "glues": len(glues),
        "vertices": sum(len(m["positions"]) // 2 for m in helpers),
        "added_geometry_bytes": (cursor - start_cursor) * 4,
        "pose_clip_samples": len(poses),
        "method": "source cloth/skin clipped against per-hair-patch 3D support planes; barycentric native Glue keyed by relative head XYZ",
        "limitations": "curved target support-plane approximation; finite head-grid interpolation; clip coefficients assume common rigid body rotation while original source masters retain actual skinning",
    }
    print("Neck depth masks", ir["diagnostics"]["neck_depth_masks"], flush=True)
