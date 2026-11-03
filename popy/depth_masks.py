"""Source support planes compressed into native parameterized barycentric Glue."""

import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from .compile_ir import ROOT, project
from .visibility import clip_triangle_barycentric, component_labels


def plane(points, triangles=None, margin=0.0001):
    center = points.mean(0)
    _, _, v = np.linalg.svd(points - center, full_matrices=False)
    normal = v[-1]
    inward = np.array([0.0, 1.0, 0.0])
    if triangles is not None:
        t = points[triangles]
        # IR triangle winding was reversed from Blender for Cubism culling.
        inward = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]).sum(0)
    if normal @ inward < 0:
        normal = -normal
    return normal, float((points @ normal).min() - margin)


def clipped_barycentric(values, threshold):
    polygon = clip_triangle_barycentric(values, threshold)
    if not polygon:
        polygon = [np.eye(3)[np.argmin(values)]]
    return np.array(polygon + [polygon[-1]] * (4 - len(polygon)))


def barycentric_glue_weights(bary):
    denominator = bary[..., 0] + bary[..., 1]
    return np.divide(
        bary[..., 1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1e-10,
    ), bary[..., 2]


def clip_barycentric_batch(values, thresholds):
    """Vectorized equivalent of scalar Sutherland-Hodgman, with fixed 4 slots."""
    d = values - np.asarray(thresholds)[..., None]
    inside = d <= 0
    codes = inside @ np.array([1, 2, 4])
    result = np.empty(d.shape[:-1] + (4, 3))
    eye = np.eye(3)
    for code in range(8):
        selected = codes == code
        if not selected.any():
            continue
        distances = d[selected]
        if code == 0:
            result[selected] = np.repeat(
                eye[np.argmin(distances, axis=1)][:, None], 4, axis=1
            )
            continue
        polygon = []
        for current in range(3):
            previous = (current - 1) % 3
            a, b = bool(code & (1 << previous)), bool(code & (1 << current))
            if a != b:
                t = distances[:, previous] / (
                    distances[:, previous] - distances[:, current]
                )
                polygon.append(
                    (1 - t[:, None]) * eye[previous] + t[:, None] * eye[current]
                )
            if b:
                polygon.append(np.broadcast_to(eye[current], (len(distances), 3)))
        polygon += [polygon[-1]] * (4 - len(polygon))
        result[selected] = np.stack(polygon, axis=1)
    return result


def deduplicate_barycentric(master, bary):
    """Share original corners and shared-edge intersections across mask triangles."""
    lookup, selected, indices = {}, [], []
    rounded = np.round(bary, 7).astype("<f4")
    for j, triangle in enumerate(master):
        terms = []
        for k, vertex in enumerate(triangle):
            value = rounded[:, j, k]
            if not np.any(value):
                continue
            coefficient = (
                float(value[0]) if np.all(value == value[0]) else value.tobytes()
            )
            terms.append((int(vertex), coefficient))
        key = tuple(sorted(terms, key=lambda x: x[0]))
        if key not in lookup:
            lookup[key] = len(selected)
            selected.append(j)
        indices.append(lookup[key])
    return master[selected], bary[:, selected], np.array(indices), np.array(selected)


def protect_linework(ir, samples, semantics, binary_path):
    config = json.loads((ROOT / "config/surface_visibility.json").read_text())
    face, hair = config["lashes"]["object"], config["hair"]["object"]
    _, components = component_labels(np.load(samples / f"{face}_source.npz"))
    meshes = [m for m in ir["meshes"] if not m.get("mask_only")]
    targets = [
        m
        for m in meshes
        if m["source"]["object"] == config["brows"]["object"]
        or (
            m["source"]["object"] == face
            and np.all(np.isin(components[m["source"]["triangles"]], [20, 21]))
        )
    ]
    if config.get("strategy") == "hybrid":
        critical = json.loads((ROOT / "config/surface_critical.json").read_text())
        for name, digest in critical["source_sha256"].items():
            if (
                hashlib.sha256(
                    (samples / f"{name}_source.npz").read_bytes()
                ).hexdigest()
                != digest
            ):
                raise ValueError(
                    "Critical surface mapping source hash mismatch: " + name
                )
        selected = critical["linework_triangles"]
        targets = [
            m
            for m in meshes
            if set(selected.get(m["source"]["object"], [])).intersection(
                m["source"]["triangles"]
            )
        ]
    # A projected lash can enter an open mouth or the eyeball volume. Outer
    # skin alone has holes there; the genuine oral/scleral surfaces must also
    # participate. Exclude lash surfaces themselves to avoid self masking.
    occluders = [
        m
        for m in meshes
        if m.get("deformer") == "head"
        and (
            m["source"]["object"] in {hair, "M_Headwear", "M_Decorate"}
            or (
                m["source"]["object"] == face
                and not np.any(
                    np.isin(
                        components[m["source"]["triangles"]],
                        config["lashes"]["components"],
                    )
                )
            )
        )
    ]
    return build_local_surface_masks(
        ir, samples, semantics, binary_path, targets, occluders
    )


def build_local_surface_masks(
    ir,
    samples,
    semantics,
    binary_path,
    targets,
    occluders,
    prefix="",
    metadata_key="depth_linework",
    diagnostics_key="linework_depth_masks",
    merge_masks=False,
    support_margin=0.0001,
    include_secondary=False,
    footprint_margin=0.008,
    envelope_bounds=False,
    double_sided_masks=False,
    expression_keys=None,
):
    """Head-local source planes with independent expression-driven clip fields.

    Rigid head/body transforms preserve the signed plane intersections. Real
    evaluated source masters supply all projected motion; only local morph
    coefficients are stored in the mask Glue grids.
    """
    config = json.loads((ROOT / "config/surface_visibility.json").read_text())
    face = config["lashes"]["object"]
    manifest = json.loads((samples / "manifest.json").read_text())
    neutral = next(
        p
        for p in manifest["head_poses"]
        if all(v == 0 for v in p["parameters"].values())
    )
    archive = np.load(samples / neutral["file"])
    rest = {key: archive[key] for key in archive.files}
    shapes = semantics["shape_parameters"]
    def sample_keys(parameter, keys):
        return sorted(set(keys + [0.5] + (expression_keys or {}).get(parameter, [])))
    # POPY pages are opaque. Alpha cutouts need separate UV-aware clipping.
    for texture in {m["texture"] for m in occluders}:
        page = Image.open(Path(binary_path).parent / ir["textures"][texture]).convert(
            "RGBA"
        )
        if page.getchannel("A").getextrema() != (255, 255):
            raise ValueError("Barycentric masks require opaque occluder textures")
    # Build a broad phase directly from source samples, so a fresh conversion
    # never depends on an older exported rig or its occlusion cache.
    names = {m["source"]["object"] for m in targets + occluders}
    projected_domain = {name: [] for name in names}
    shape_projections = {
        (s["object"], pid): []
        for pid, s in shapes.items()
        if s["object"] in names
        and (include_secondary or pid not in ["ParamEyeLSmile", "ParamEyeRSmile"])
    }
    camera_matrix = np.array(ir["camera"]["projection"]) @ np.linalg.inv(
        np.array(ir["camera"]["matrix_world"])
    )
    if not np.allclose(camera_matrix[3, :3], 0):
        raise ValueError(
            "Native barycentric visibility compression currently requires orthographic projection"
        )
    derivative = np.array([camera_matrix[0, :3], -camera_matrix[1, :3]]) / (
        2 * camera_matrix[3, 3]
    )
    domain_poses = []
    joint = {
        (bx, by): np.load(samples / f"body_{bx}_{by}.npz")
        for bx, by in itertools.product([-15, 0, 15], repeat=2)
    }
    for pose in manifest["head_poses"]:
        hs = np.load(samples / pose["file"])
        head = {name: hs[name] for name in names}
        head_shapes = {key: hs[key[0] + "__" + key[1]] for key in shape_projections}
        for bx, by in joint:
            domain_poses.append(
                dict(pose["parameters"], ParamBodyAngleX=bx, ParamBodyAngleY=by)
            )
            a, b = np.radians([bx, by])
            rotation = np.array(
                [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
            ) @ np.array(
                [[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]]
            )
            for name in names:
                world_pose = (
                    joint[(bx, by)][name] + (head[name] - rest[name]) @ rotation.T
                )
                projected_domain[name].append(project(world_pose, ir["camera"]))
            for key, delta in head_shapes.items():
                shape_projections[key].append(
                    (delta @ rotation.T @ derivative.T).astype(np.float32)
                )
    projected_domain = {
        name: np.asarray(value) for name, value in projected_domain.items()
    }
    shape_projections = {
        key: np.asarray(value) for key, value in shape_projections.items()
    }

    def expressed_screen(name, vertices, params):
        screen = projected_domain[name][:, vertices].copy()
        for (obj, pid), delta in shape_projections.items():
            if obj == name:
                spec = shapes[pid]
                value = params.get(pid, spec["default"])
                amount = 1 - value if spec.get("invert") else value
                if amount:
                    screen += delta[:, vertices] * amount
        return screen

    bounds = {}
    for mesh in targets + occluders:
        p = projected_domain[mesh["source"]["object"]][:, mesh["source"]["vertices"]]
        if envelope_bounds:
            low, high = p.copy(), p.copy()
            for (obj, _), delta in shape_projections.items():
                if obj == mesh["source"]["object"]:
                    d = delta[:, mesh["source"]["vertices"]]
                    low += np.minimum(d, 0)
                    high += np.maximum(d, 0)
            bounds[mesh["id"]] = (low.min(1), high.max(1))
        else:
            bounds[mesh["id"]] = (p.min(1), p.max(1))
    domain_axes = [
        "ParamAngleX",
        "ParamAngleY",
        "ParamAngleZ",
        "ParamBodyAngleX",
        "ParamBodyAngleY",
    ]
    domain_keys = [sorted({p[a] for p in domain_poses}) for a in domain_axes]
    domain_coordinates = [
        tuple(keys.index(p[a]) for a, keys in zip(domain_axes, domain_keys))
        for p in domain_poses
    ]
    geometry = np.memmap(binary_path, dtype="<f4", mode="r")
    mesh_lookup = {m["id"]: m for m in ir["meshes"]}

    def front_domain(mesh):
        if mesh.get("face_partition_parent"):
            mesh = mesh_lookup[mesh["face_partition_parent"]]
        front = np.zeros(tuple(len(k) for k in domain_keys), bool)
        triangles = np.array(mesh["indices"]).reshape(-1, 3)
        n = len(mesh["positions"])
        for c in mesh["cells"]:
            p = {
                a["parameter"]: a["keys"][i]
                for a, i in zip(mesh["axes"], c["coordinate"])
            }
            points = geometry[c["offset"] : c["offset"] + n].reshape(-1, 2)[triangles]
            x, y = points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
            if np.any(x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0] > 1e-12):
                coord = tuple(
                    keys.index(p.get(a, 0)) for a, keys in zip(domain_axes, domain_keys)
                )
                front[coord] = True
        # A neighboring-key allowance retains narrow intermediate visibility.
        front = binary_dilation(front, iterations=1)
        return np.array([front[c] for c in domain_coordinates])

    diagnostics, helpers, glues = [], [], []
    old_count = Path(binary_path).stat().st_size // 4
    cursor = old_count

    def world(name, vertices, params):
        p = rest[name][vertices].astype(float).copy()
        for pid, shape in shapes.items():
            if shape["object"] == name:
                v = params.get(pid, shape["default"])
                amount = 1 - v if shape.get("invert") else v
                p += amount * rest[name + "__" + pid][vertices]
        return p

    with Path(binary_path).open("ab") as stream:
        for target in targets:
            name = target["source"]["object"]
            vertices = target["source"]["vertices"]
            target_triangles = np.array(target["indices"]).reshape(-1, 3)
            normal, _ = plane(
                rest[name][vertices].astype(float), target_triangles, support_margin
            )
            expression_axes = [a for a in target["axes"] if a["parameter"] in shapes]
            if include_secondary:
                expression_axes.extend(
                    {"parameter": b["parameter"], "keys": b["keys"]}
                    for b in target["blend_shapes"]
                    if b["parameter"] in shapes
                    and b["parameter"] not in {a["parameter"] for a in expression_axes}
                )
            # Only local expression grids; no copied 945-pose geometry tensors.
            expression_axes = [
                {**a, "keys": sample_keys(a["parameter"], a["keys"])} for a in expression_axes
            ]
            target_sources = []
            target_front = front_domain(target)
            helper_id = "MaskDepth_" + prefix + target["id"]
            helper_positions, indices, provenance = [], [], []
            per_master = []
            for occluder in occluders:
                obj = occluder["source"]["object"]
                if name == config["brows"]["object"] and obj == face:
                    continue
                lo, hi = bounds[target["id"]]
                a, b = bounds[occluder["id"]]
                broad_margin = footprint_margin if envelope_bounds else 0.035
                if not np.any(
                    np.all((hi + broad_margin >= a) & (b + broad_margin >= lo), axis=1)
                ):
                    continue
                axes = list(expression_axes)
                axes.extend(
                    {**a, "keys": sample_keys(a["parameter"], a["keys"])}
                    for a in occluder["axes"]
                    if a["parameter"] in shapes
                    and a["parameter"] not in {b["parameter"] for b in axes}
                )
                if include_secondary:
                    axes.extend(
                        {
                            "parameter": b["parameter"],
                            "keys": sample_keys(b["parameter"], b["keys"]),
                        }
                        for b in occluder["blend_shapes"]
                        if b["parameter"] in shapes
                        and b["parameter"] not in {a["parameter"] for a in axes}
                    )
                coordinates = list(
                    itertools.product(*(range(len(a["keys"])) for a in axes))
                )
                params = [
                    {a["parameter"]: a["keys"][c] for a, c in zip(axes, coordinate)}
                    for coordinate in coordinates
                ]
                occ_vertices = occluder["source"]["vertices"]
                points = [world(obj, occ_vertices, p) for p in params]
                planes = [
                    plane(world(name, vertices, p), target_triangles, support_margin)
                    for p in params
                ]
                thresholds = [t for n, t in planes]
                distances = [p @ n for p, (n, t) in zip(points, planes)]
                if not double_sided_masks and all(
                    d.max() <= t for d, t in zip(distances, thresholds)
                ):
                    target_sources.append(occluder["id"])
                    continue
                triangles = np.array(occluder["indices"]).reshape(-1, 3)
                keep = np.any(
                    [d[triangles].min(1) <= t for d, t in zip(distances, thresholds)],
                    axis=0,
                )
                if not keep.any():
                    continue
                projected = projected_domain[obj][:, occ_vertices][:, triangles]
                low, high = projected.min(2), projected.max(2)
                overlap = np.all(
                    (
                        hi[:, None] + (footprint_margin if envelope_bounds else 0.035)
                        >= low
                    )
                    & (
                        high + (footprint_margin if envelope_bounds else 0.035)
                        >= lo[:, None]
                    ),
                    axis=2,
                )
                overlap &= target_front[:, None]
                if obj != face and not double_sided_masks:
                    x, y = (
                        projected[:, :, 1] - projected[:, :, 0],
                        projected[:, :, 2] - projected[:, :, 0],
                    )
                    overlap &= x[:, :, 0] * y[:, :, 1] - x[:, :, 1] * y[:, :, 0] > 1e-12
                keep &= np.any(overlap, axis=0)
                if not keep.any():
                    continue
                triangles = triangles[keep]
                bary = clip_barycentric_batch(
                    np.array(distances)[:, triangles], np.array(thresholds)[:, None]
                )
                # Test the CLIPPED footprint, rather than the original triangle's
                # much larger bounds. This is essential for practical mask size.
                footprint = np.zeros(len(triangles), bool)
                for pi, parameter_values in enumerate(params):
                    source_screen = expressed_screen(
                        obj, occ_vertices, parameter_values
                    )[:, triangles]
                    q = np.einsum("tvi,ptic->ptvc", bary[pi], source_screen)
                    target_screen = expressed_screen(name, vertices, parameter_values)
                    tl, th = target_screen.min(1), target_screen.max(1)
                    t = target_screen[:, target_triangles]
                    u, v = t[:, :, 1] - t[:, :, 0], t[:, :, 2] - t[:, :, 0]
                    front = np.any(
                        u[:, :, 0] * v[:, :, 1] - u[:, :, 1] * v[:, :, 0] > 1e-12,
                        axis=1,
                    )
                    u, v = q[:, :, 1] - q[:, :, 0], q[:, :, 2] - q[:, :, 0]
                    u2, v2 = q[:, :, 2] - q[:, :, 0], q[:, :, 3] - q[:, :, 0]
                    area = (
                        u[:, :, 0] * v[:, :, 1] - u[:, :, 1] * v[:, :, 0] > 1e-12
                    ) | (u2[:, :, 0] * v2[:, :, 1] - u2[:, :, 1] * v2[:, :, 0] > 1e-12)
                    if double_sided_masks:
                        area = (
                            np.abs(u[:, :, 0] * v[:, :, 1] - u[:, :, 1] * v[:, :, 0])
                            > 1e-12
                        ) | (
                            np.abs(
                                u2[:, :, 0] * v2[:, :, 1] - u2[:, :, 1] * v2[:, :, 0]
                            )
                            > 1e-12
                        )
                    overlap = np.all(
                        (th[:, None] + footprint_margin >= q.min(2))
                        & (q.max(2) + footprint_margin >= tl[:, None]),
                        axis=2,
                    )
                    footprint |= np.any(overlap & area & front[:, None], axis=0)
                if not footprint.any():
                    continue
                triangles = triangles[footprint]
                bary = bary[:, footprint]
                offset = len(helper_positions)
                neutral_index = next(
                    i
                    for i, p in enumerate(params)
                    if all(p[k] == shapes[k]["default"] for k in p)
                )
                local_rest = np.array(occluder["positions"]).reshape(-1, 2)[triangles]
                pos = np.einsum(
                    "tvi,tic->tvc", bary[neutral_index], local_rest
                ).reshape(-1, 2)
                master = np.repeat(triangles, 4, axis=0)
                master, compact_bary, remap, selected = deduplicate_barycentric(
                    master, bary.reshape(len(params), -1, 3)
                )
                helper_positions.extend(pos[selected].tolist())
                indices.extend(
                    [
                        offset + int(remap[4 * i + j])
                        for i in range(len(triangles))
                        for j in [0, 1, 2, 0, 2, 3]
                    ]
                )
                source_triangles = np.array(occluder["source"]["triangles"])[keep][
                    footprint
                ]
                provenance.append(
                    {
                        "mesh": occluder["id"],
                        "mask_vertices": (remap.reshape(-1, 4) + offset).tolist(),
                        "source_triangles": source_triangles.tolist(),
                        "original_mesh_vertices": triangles.tolist(),
                        "axes": axes,
                        "barycentric": bary.tolist(),
                    }
                )
                weights = barycentric_glue_weights(compact_bary)
                per_master.append(
                    (occluder["id"], offset, master, weights, axes, coordinates)
                )
            if helper_positions:
                positions = np.asarray(helper_positions, dtype="<f4").ravel()
                form_offset = cursor
                # All vertices are fully copied by the first Glue stage. Keep
                # the initial parent-space state small to avoid float cancellation.
                stream.write(np.zeros_like(positions).tobytes())
                cursor += len(positions)
                helper = {
                    **target,
                    "id": helper_id,
                    "axes": [],
                    "cells": [{"coordinate": [], "offset": form_offset, "order": 0.0}],
                    "positions": positions.tolist(),
                    "uvs": [0.5, 0.5] * len(helper_positions),
                    "indices": indices,
                    "texture": 0,
                    "mask_only": True,
                    "opacity": 0.0,
                    "culling": not double_sided_masks,
                    "masked_by": [],
                    "invert_mask": False,
                    "blend_shapes": [],
                    "source": {
                        "object": name,
                        "vertices": [vertices[0]] * len(helper_positions),
                        "triangles": [target["source"]["triangles"][0]]
                        * (len(indices) // 3),
                        "depth_target": target["id"],
                        "correspondence": "source-preserving parameterized native Glue; explicit triangle barycentric provenance",
                        "barycentric_sources": provenance,
                    },
                }
                helpers.append(helper)
                target_sources.append(helper_id)
                # Source seam welds execute first. For each new vertex copy v0,
                # lerp to v1, then v2. Every source-side weight remains zero.
                for stage in range(3):
                    for (
                        master_id,
                        offset,
                        master,
                        weights,
                        axes,
                        coordinates,
                    ) in per_master:
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
                            glue = {
                                "id": f"G_Depth_{prefix}{len(glues)}",
                                "mesh_a": master_id,
                                "mesh_b": helper_id,
                                "pairs": pairs,
                                "intensity": float(values[0]),
                            }
                            if len(set(values)) > 1:
                                glue.update(
                                    axes=axes, intensities=[float(v) for v in values]
                                )
                            glues.append(glue)
            if (
                merge_masks
                and target.get("masked_by")
                and not target.get("invert_mask")
            ):
                raise ValueError("Cannot merge normal and inverse depth masks")
            target["masked_by"] = list(
                dict.fromkeys(
                    (target.get("masked_by", []) if merge_masks else [])
                    + target_sources
                )
            )
            target["invert_mask"] = True
            diagnostics.append(
                {
                    "target": target["id"],
                    "mask_sources": len(target_sources),
                    "normal": normal.tolist(),
                    "clipped_vertices": len(helper_positions),
                }
            )
            if len(diagnostics) % 20 == 0:
                print(
                    prefix + " local depth",
                    len(diagnostics),
                    "/",
                    len(targets),
                    flush=True,
                )
    ir["meshes"].extend(helpers)
    ir.setdefault("glues", []).extend(glues)
    ir["surface_visibility"][metadata_key] = diagnostics
    ir["diagnostics"][diagnostics_key] = {
        "targets": len(targets),
        "helpers": len(helpers),
        "glues": len(glues),
        "vertices": sum(len(m["positions"]) // 2 for m in helpers),
        "added_geometry_bytes": (cursor - old_count) * 4,
        "method": "source support planes, barycentric native Glue intensity grids; 3D head/body/expression geometry shared",
        "limitations": "curved target approximated by nearest support plane; clip weights sampled at 0/.5/1; opaque source occluders required; smile-plane offsets approximate",
    }
    print(prefix + " local depth masks", ir["diagnostics"][diagnostics_key], flush=True)


def stabilize_linework_order(ir):
    targets = {
        d["target"] for d in ir.get("surface_visibility", {}).get("depth_linework", [])
    }
    if ir.get("surface_visibility", {}).get("strategy") == "hybrid":
        groups = ir["surface_visibility"]["targets"]
        targets |= set(groups["brows"] + groups["lashes"])
    for mesh in ir["meshes"]:
        if mesh["id"] in targets:
            value = 800.0 if mesh["source"]["object"] == "M_Brow" else 801.0
            for c in mesh["cells"]:
                c["order"] = value
            for b in mesh["blend_shapes"]:
                b["orders"] = [value] * len(b["keys"])
