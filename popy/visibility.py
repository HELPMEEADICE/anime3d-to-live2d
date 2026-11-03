"""Native clipping and source-identity welds, independent of the Cubism binary writer."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def component_labels(source):
    triangles = source["triangles"]
    edges = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
    )
    graph = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(len(source["positions"]),) * 2,
    )
    _, labels = connected_components(graph.tocsr(), directed=False)
    return labels, labels[triangles[:, 0]]


def clip_triangle_barycentric(values, threshold):
    polygon = [np.eye(3)[i] for i in range(3)]
    result = []
    for previous, current in zip(polygon[-1:] + polygon[:-1], polygon):
        a, b = float(previous @ values - threshold), float(current @ values - threshold)
        if (a <= 0) != (b <= 0):
            result.append(previous + (current - previous) * (a / (a - b)))
        if b <= 0:
            result.append(current)
    return result


def conservative_skin_masks(
    ir,
    samples,
    semantics,
    skin_meshes,
    binary_path,
    bound_override=None,
    threshold_override=None,
    label="Skin",
):
    """Discard only skin proven behind every eye surface throughout the morph envelope.

    In this validated frontal hemisphere a ray's rest-space Y component stays positive.
    Rear skin can project into an eye hole, but cannot occlude the eye in depth. Including
    that skin in an inverse 2D mask creates false pupil clipping at intermediate yaw.
    """
    manifest = json.loads((samples / "manifest.json").read_text())
    neutral = next(
        p
        for p in manifest["head_poses"]
        if all(v == 0 for v in p["parameters"].values())
    )
    poses = np.load(samples / neutral["file"])
    face = semantics["shape_parameters"]["ParamEyeLOpen"]["object"]
    eyes = semantics["objects"]["gaze"]
    lower = poses[face][:, 1].astype(float).copy()
    for pid, shape in semantics["shape_parameters"].items():
        if shape["object"] == face:
            lower += np.minimum(poses[face + "__" + pid][:, 1], 0)
    eye_back = max(
        float(poses[key][:, 1].max())
        for key in poses.files
        if key.startswith(eyes + "__gaze_")
    )
    eye_back += 1e-4
    if bound_override is not None:
        lower = bound_override
    if threshold_override is not None:
        eye_back = threshold_override
    original_data = np.fromfile(binary_path, "<f4")
    cursor = len(original_data)
    sources = []
    helpers = []
    discarded = 0
    with binary_path.open("ab") as stream:

        def emit(data):
            nonlocal cursor
            value = np.asarray(data, dtype="<f4").ravel()
            offset = cursor
            stream.write(value.tobytes())
            cursor += len(value)
            return offset

        for mesh in skin_meshes:
            vertices = mesh["source"]["vertices"]
            triangles = np.asarray(mesh["indices"]).reshape(-1, 3)
            bounds = lower[vertices]
            front = bounds[triangles] <= eye_back
            if np.all(front):
                sources.append(mesh["id"])
                continue
            bary = []
            indices = []
            original_triangles = []
            for ti, triangle in enumerate(triangles):
                polygon = clip_triangle_barycentric(bounds[triangle], eye_back)
                if len(polygon) < 3:
                    discarded += 1
                    continue
                start = len(bary)
                for point in polygon:
                    row = np.zeros(len(vertices))
                    row[triangle] = point
                    bary.append(row)
                for k in range(1, len(polygon) - 1):
                    indices.extend([start, start + k, start + k + 1])
                    original_triangles.append(mesh["source"]["triangles"][ti])
            if not bary:
                continue
            transform = np.asarray(bary)
            n = len(vertices) * 2

            def project_offset(offset, transform=transform, n=n):
                return emit(
                    transform @ original_data[offset : offset + n].reshape(-1, 2)
                )

            helper = {
                **mesh,
                "id": "Mask" + label + "_" + mesh["id"],
                "mask_only": True,
                "opacity": 0.0,
                "masked_by": [],
                "invert_mask": False,
                "positions": (transform @ np.asarray(mesh["positions"]).reshape(-1, 2))
                .ravel()
                .tolist(),
                "uvs": (transform @ np.asarray(mesh["uvs"]).reshape(-1, 2))
                .ravel()
                .tolist(),
                "indices": indices,
                "cells": [
                    {**c, "offset": project_offset(c["offset"])} for c in mesh["cells"]
                ],
                "blend_shapes": [
                    {
                        **b,
                        "offsets": [
                            None if o is None else project_offset(o)
                            for o in b["offsets"]
                        ],
                    }
                    for b in mesh["blend_shapes"]
                ],
                "source": {
                    **mesh["source"],
                    "vertices": [vertices[int(np.argmax(row))] for row in transform],
                    "triangles": original_triangles,
                    "barycentric": "fixed barycentric clipping against conservative eye-depth envelope",
                    "barycentric_vertices": [
                        [
                            [vertices[i], float(weight)]
                            for i, weight in enumerate(row)
                            if weight > 1e-12
                        ]
                        for row in transform
                    ],
                },
            }
            helpers.append(helper)
            sources.append(helper["id"])
    ir["meshes"].extend(helpers)
    ir["diagnostics"][
        "eye_mask_rear_filter" if label == "Skin" else "eye_mask_front_hair"
    ] = {
        "eye_rest_y_back_bound": eye_back,
        "generated_masks": len(helpers),
        "discarded_rear_triangles": discarded,
        "scope": "validated orthographic frontal head/body domain; conservative full configured morph/gaze envelope",
    }
    return sources


def protect_eyes(ir, samples, semantics, binary_path):
    face = semantics["shape_parameters"]["ParamEyeLOpen"]["object"]
    eyes = semantics["objects"]["gaze"]
    src = np.load(samples / f"{face}_source.npz")
    vertex_components, triangle_components = component_labels(src)
    skin = int(np.bincount(triangle_components).argmax())
    skin_meshes = [
        m
        for m in ir["meshes"]
        if m["source"]["object"] == face
        and np.all(triangle_components[m["source"]["triangles"]] == skin)
    ]
    if not skin_meshes:
        raise ValueError("No isolated skin surface for native eye clipping")
    skin_surface_ids = [m["id"] for m in skin_meshes]
    # The same face object also contains sclera, eyelashes, teeth and mouth interiors.
    # Masking by the whole object incorrectly hides the pupils even at neutral pose.
    eye_vertices = np.load(samples / f"{eyes}_source.npz")["positions"]
    eye_zmin, eye_zmax = eye_vertices[:, 2].min(), eye_vertices[:, 2].max()
    eye_height = eye_zmax - eye_zmin
    face_manifest = next(
        o
        for o in json.loads((samples / "manifest.json").read_text())["objects"]
        if o["name"] == face
    )
    image = np.asarray(
        Image.open(samples / face_manifest["textures"][0]).convert("RGB")
    )
    uv = src["uv"].mean(axis=1)
    rgb = image[
        np.clip(
            ((1 - uv[:, 1]) * (image.shape[0] - 1)).astype(int), 0, image.shape[0] - 1
        ),
        np.clip((uv[:, 0] * (image.shape[1] - 1)).astype(int), 0, image.shape[1] - 1),
    ]
    sclera = set()
    for component in np.unique(triangle_components):
        if component == skin:
            continue
        use = triangle_components == component
        center_z = src["positions"][vertex_components == component, 2].mean()
        if (
            eye_zmin - eye_height * 0.5 <= center_z <= eye_zmax + eye_height * 0.5
            and np.all(np.median(rgb[use], axis=0) > 230)
        ):
            sclera.add(int(component))
    # Oral interiors, nose and lashes can also be in front of an eye surface.
    # In particular, an eyelid morph can project part of the sclera through the
    # mouth aperture. The mouth cavity must occlude it even when the outer skin
    # has a hole there. Exclude the actual sclera, not every non-skin component.
    face_occluders = [
        mesh
        for mesh in ir["meshes"]
        if not mesh.get("mask_only")
        and mesh["source"]["object"] == face
        and not any(
            int(c) in sclera for c in triangle_components[mesh["source"]["triangles"]]
        )
    ]
    skin_sources = conservative_skin_masks(
        ir, samples, semantics, face_occluders, binary_path
    )
    hair_sources = front_hair_masks(ir, samples, semantics, binary_path)
    skin_sources += hair_sources
    targets = []
    for mesh in ir["meshes"]:
        if mesh.get("mask_only"):
            continue
        source = mesh["source"]
        is_sclera = source["object"] == face and all(
            int(c) in sclera for c in triangle_components[source["triangles"]]
        )
        if source["object"] == eyes or is_sclera:
            mesh["masked_by"] = skin_sources
            mesh["invert_mask"] = True
            targets.append(mesh["id"])
    ir["visibility"] = {
        "skin_component": skin,
        "sclera_components": sorted(sclera),
        "skin_mask_sources": skin_sources,
        "eye_targets": targets,
        "skin_surface_meshes": skin_surface_ids,
        "front_hair_sources": hair_sources,
        "method": "native inverted foreground face/hair union, excludes sclera and conservatively clips rear surfaces; oral cavities and lashes included",
    }


def front_hair_masks(ir, samples, semantics, binary_path):
    name = semantics["objects"].get("hair")
    if not name:
        return []
    manifest = json.loads((samples / "manifest.json").read_text())
    neutral = next(
        p
        for p in manifest["head_poses"]
        if all(v == 0 for v in p["parameters"].values())
    )
    pose = np.load(samples / neutral["file"])
    eyes = semantics["objects"]["gaze"]
    front = (
        min(
            float(pose[key][:, 1].min())
            for key in pose.files
            if key.startswith(eyes + "__gaze_")
        )
        - 1e-4
    )
    vertices = np.load(samples / f"{name}_source.npz")["positions"]
    meshes = [
        m
        for m in ir["meshes"]
        if m["source"]["object"] == name and not m.get("mask_only")
    ]
    return conservative_skin_masks(
        ir,
        samples,
        semantics,
        meshes,
        binary_path,
        bound_override=vertices[:, 1],
        threshold_override=front,
        label="Hair",
    )


def source_aliases(samples, name):
    src = np.load(samples / f"{name}_source.npz")
    weights = json.loads((samples / f"{name}_weights.json").read_text())
    morph_path = samples / f"{name}_morphs.npz"
    morphs = np.load(morph_path) if morph_path.exists() else None
    morph_arrays = [morphs[k] for k in sorted(morphs.files)] if morphs else []
    lookup = {}
    aliases = []
    for i, point in enumerate(src["positions"]):
        key = (
            tuple(np.round(point.astype(float), 6)),
            tuple(sorted((n, round(w, 6)) for n, w in weights[i] if w > 1e-6)),
            hashlib.blake2b(
                b"".join(
                    np.round(a[i], 6).astype("<f4").tobytes() for a in morph_arrays
                ),
                digest_size=16,
            ).digest(),
        )
        aliases.append(lookup.setdefault(key, i))
    return aliases


def weld_sources(ir, samples):
    meshes = {m["id"]: m for m in ir["meshes"] if not m.get("mask_only")}
    aliases = {
        name: source_aliases(samples, name)
        for name in {m["source"]["object"] for m in meshes.values()}
    }
    uses = defaultdict(list)
    for mesh in meshes.values():
        name = mesh["source"]["object"]
        for local, source in enumerate(mesh["source"]["vertices"]):
            uses[(name, aliases[name][source])].append((mesh["id"], local))
    pairs = defaultdict(list)
    for candidates in uses.values():
        # A body-parent candidate exists at a head/body boundary only if this source vertex
        # does not move with the head. Its fixed trajectory is the correct weld anchor.
        rank = lambda item: (
            meshes[item[0]].get("deformer") != "body",
            -len(meshes[item[0]]["axes"]),
            item,
        )
        unique = sorted(set(candidates), key=rank)
        master = unique[0]
        for other in unique[1:]:
            if master[0] != other[0]:
                pairs[(master[0], other[0])].append([master[1], other[1], 0.0, 1.0])
    depth_glues = [g for g in ir.get("glues", []) if g["id"].startswith("G_Depth_")]
    ir["glues"] = [
        {"id": f"G_Source_{i}", "mesh_a": a, "mesh_b": b, "pairs": pair}
        for i, ((a, b), pair) in enumerate(sorted(pairs.items()))
    ]
    ir["diagnostics"]["seam_welds"] = {
        "glues": len(ir["glues"]),
        "vertex_pairs": sum(len(g["pairs"]) for g in ir["glues"]),
        "source_aliases": "position + complete skin weights + all morph deltas; same object only",
        "master_weight": 0,
        "copy_weight": 1,
    }
    ir["glues"].extend(depth_glues)


def finalize_visibility(ir, samples, semantics, binary_path):
    from .linework_partition import partition_linework
    partition_linework(ir, samples, semantics, binary_path)
    protect_eyes(ir, samples, semantics, binary_path)
    protect_surfaces(ir, samples, semantics, binary_path)
    weld_sources(ir, samples)
    correct_pitch_chord(ir, samples, semantics)


def protect_surfaces(ir, samples, semantics, binary_path):
    """Protect anatomical linework and rear neck hair with native geometry masks."""
    config = json.loads((Path(__file__).resolve().parents[1] / "config/surface_visibility.json").read_text())
    spec = config["lashes"]
    _, components = component_labels(np.load(samples / f"{spec['object']}_source.npz"))
    hair_sources = ir["visibility"]["front_hair_sources"]
    hair_sources = hair_sources + [m['id'] for m in ir['meshes'] if not m.get('mask_only') and m['source']['object'] in ['M_Headwear','M_Decorate']]
    targets = {"brows": [], "lashes": [], "rear_hair": []}
    for mesh in ir["meshes"]:
        if mesh.get("mask_only"):
            continue
        source = mesh["source"]
        kind = None
        if source["object"] == config["brows"]["object"]:
            kind = "brows"
        elif source["object"] == spec["object"] and np.all(np.isin(components[source["triangles"]], spec["components"])):
            kind = "lashes"
        if kind:
            mesh["masked_by"] = hair_sources.copy()
            mesh["invert_mask"] = True
            targets[kind].append(mesh["id"])
    if config.get("strategy") in ['partition','hybrid']:
        ir["surface_visibility"] = {"strategy":config['strategy'], "targets": targets, "neck_sources": [], "method": "source-planarity/occlusion-error partitioning, pose-local depth constraints and native foreground hair clipping"}
        if config['strategy']=='hybrid':
            from .depth_masks import protect_linework, stabilize_linework_order
            from .neck_masks import protect_neck_depth
            protect_linework(ir,samples,semantics,binary_path)
            stabilize_linework_order(ir)
            protect_neck_depth(ir,samples,semantics,binary_path)
        return
    name = config["neck"]["object"]
    source = np.load(samples / f"{name}_source.npz")
    neck = [m for m in ir["meshes"] if not m.get("mask_only") and m["source"]["object"] == name]
    sources = conservative_skin_masks(ir, samples, semantics, neck, binary_path, bound_override=source["positions"][:, 1], threshold_override=0.01, label="Neck")
    # The high collar and bow cover rear hair below the skin neck. They are
    # separate source objects, so a skin-only neck mask cannot represent them.
    for obj in ["M_Top", "M_Bow", "M_Necktie"]:
        cloth = [m for m in ir["meshes"] if not m.get("mask_only") and m["source"]["object"] == obj]
        if cloth:
            cloth_rest = np.load(samples / f"{obj}_source.npz")["positions"]
            sources += conservative_skin_masks(ir, samples, semantics, cloth, binary_path, bound_override=cloth_rest[:, 1], threshold_override=0.01, label="Neck_" + obj)
    hair_name = config["hair"]["object"]
    rest = np.load(samples / f"{hair_name}_source.npz")["positions"]
    for mesh in ir["meshes"]:
        if mesh.get("mask_only") or mesh["source"]["object"] != hair_name:
            continue
        points = rest[mesh["source"]["vertices"]]
        if points[:, 1].min() >= 0.01:
            mesh["masked_by"] = sources.copy()
            mesh["invert_mask"] = True
            targets["rear_hair"].append(mesh["id"])
    ir["surface_visibility"] = {"targets": targets, "neck_sources": sources, "method": "Native inverted foreground hair and front-neck masks; source triangle component semantics"}
    from .depth_masks import protect_linework, stabilize_linework_order
    protect_linework(ir, samples, semantics, binary_path)
    stabilize_linework_order(ir)
    from .neck_masks import protect_neck_depth
    protect_neck_depth(ir, samples, semantics, binary_path)


def chord_scale(value, keys=(-15.0, 0.0, 15.0)):
    if value <= keys[0] or value >= keys[-1]:
        return 1.0
    i = np.searchsorted(keys, value, side="right") - 1
    a, b = np.radians(keys[i : i + 2])
    t = (value - keys[i]) / (keys[i + 1] - keys[i])
    return float(
        np.hypot(
            (1 - t) * np.cos(a) + t * np.cos(b), (1 - t) * np.sin(a) + t * np.sin(b)
        )
    )


def correct_pitch_chord(ir, samples, semantics):
    """Correct orthographic chord shrink with small native warp/rotation tracks.

    Mesh geometry remains on the original sparse body grid. Native deformer controls have
    a tiny additional 1D scale curve; they do not add another Cartesian mesh tensor.
    """
    parameter = next(p for p in ir["parameters"] if p["id"] == "ParamBodyAngleY")
    manifest = json.loads((samples / "manifest.json").read_text())
    if (
        parameter.get("kind") == "blend"
        or manifest["config"]["camera"]["type"] != "ORTHO"
    ):
        return
    from .compile_ir import project

    bones = json.loads((samples / "skeleton.json").read_text())
    bind = {b["name"]: np.array(b["matrix_local"])[:3, 3] for b in bones}
    body_name = semantics["bones"]["body"]
    head_world = bind[semantics["bones"]["head"]]
    body_world = bind[body_name]
    descendants = {body_name}
    while True:
        expanded = descendants | {
            b["name"] for b in bones if b["parent"] in descendants
        }
        if expanded == descendants:
            break
        descendants = expanded
    w, h = ir["canvas"]
    canvas = np.array([w, h])
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], float)
    body_pivot = project(np.array([body_world]), ir["camera"])[0] * canvas
    head_pivot = project(np.array([head_world]), ir["camera"])[0] * canvas
    pitch_keys = [-15, -7.5, 0, 7.5, 15]
    head_cells = []
    for xi, x in enumerate([-15, 0, 15]):
        for yi, y in enumerate(pitch_keys):
            a, b = np.radians([x, y])
            rotation = np.array(
                [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
            ) @ np.array(
                [[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]]
            )
            p = (
                project(
                    np.array([(head_world - body_world) @ rotation.T + body_world]),
                    ir["camera"],
                )[0]
                * canvas
            )
            for zi, z in enumerate([-30, 0, 30]):
                for breath in [0, 1]:
                    head_cells.append(
                        {
                            "coordinate": [xi, yi, zi, breath],
                            "origin": (
                                (p - body_pivot)
                                * [1 + 0.012 * breath, 1 + 0.006 * breath]
                            ).tolist(),
                            "angle": z,
                        }
                    )
    ir["head_rotation"] = {
        "axes": [
            {"parameter": "ParamBodyAngleX", "keys": [-15, 0, 15]},
            {"parameter": "ParamBodyAngleY", "keys": pitch_keys},
            {"parameter": "ParamAngleZ", "keys": [-30, 0, 30]},
            {"parameter": "ParamBreath", "keys": [0, 1]},
        ],
        "cells": head_cells,
    }
    curve_keys = [-15, -11.25, -7.5, -3.75, 0, 3.75, 7.5, 11.25, 15]
    weights = {}
    for name in {m["source"]["object"] for m in ir["meshes"]}:
        values = json.loads((samples / f"{name}_weights.json").read_text())
        weights[name] = np.array(
            [sum(v for bone, v in vertex if bone in descendants) for vertex in values]
        )
    strengths = {0.0}
    for mesh in ir["meshes"]:
        if mesh.get("deformer") == "head":
            continue
        strength = float(
            np.clip(
                np.round(
                    np.mean(
                        weights[mesh["source"]["object"]][mesh["source"]["vertices"]]
                    )
                    * 4
                )
                / 4,
                0,
                1,
            )
        )
        mesh["body_warp_strength"] = strength
        mesh["projection_warp"] = f"D_BodyProjection_{int(strength * 4)}"
        strengths.add(strength)
    body_warps = []
    for strength in sorted(strengths):
        cells = []
        for yi, y in enumerate(curve_keys):
            correction = 1 / (1 - strength + strength * chord_scale(y))
            for breath in [0, 1]:
                cells.append(
                    {
                        "coordinate": [yi, breath],
                        "points": (
                            (corners - body_pivot)
                            * [1 + 0.012 * breath, (1 + 0.006 * breath) * correction]
                        )
                        .ravel()
                        .tolist(),
                    }
                )
        body_warps.append(
            {
                "id": f"D_BodyProjection_{int(strength * 4)}",
                "strength": strength,
                "axes": [
                    {"parameter": "ParamBodyAngleY", "keys": curve_keys},
                    {"parameter": "ParamBreath", "keys": [0, 1]},
                ],
                "cells": cells,
            }
        )
    ir["body_warps"] = body_warps
    head_cells = []
    for zi, z in enumerate([-30, 0, 30]):
        a = np.radians(z)
        rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        for yi, y in enumerate(curve_keys):
            for breath in [0, 1]:
                head_cells.append(
                    {
                        "coordinate": [zi, yi, breath],
                        "points": (
                            (
                                (corners - head_pivot)
                                @ rotation.T
                                * [
                                    1 + 0.012 * breath,
                                    (1 + 0.006 * breath) / chord_scale(y),
                                ]
                            )
                            @ rotation
                        )
                        .ravel()
                        .tolist(),
                    }
                )
    ir["head_warp"] = {
        "axes": [
            {"parameter": "ParamAngleZ", "keys": [-30, 0, 30]},
            {"parameter": "ParamBodyAngleY", "keys": curve_keys},
            {"parameter": "ParamBreath", "keys": [0, 1]},
        ],
        "cells": head_cells,
    }
    ir["diagnostics"]["pitch_chord_correction"] = {
        "method": "orthographic native scale curve weighted by actual descendant skin weights; refined head pivot track",
        "body_pitch_mesh_keys": [-15, 0, 15],
        "native_curve_keys": curve_keys,
        "warp_groups": len(body_warps),
        "approximation": "lost depth/yaw cross terms are not restored; no claim of exact 3D rotation between samples",
    }
    ir["diagnostics"]["deformers"] = len(body_warps) + 3
