"""Source-connected facial/hair surface refinement with native master tracking."""

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from .compile_ir import ROOT, jsonwrite
from .depth_masks import build_local_surface_masks
from .field_compression import compress_glue_fields
from .form_storage import compact_forms
from .mask_union import factor_inherited_masks, pack_local_sources
from .visibility import component_labels


def planar_groups(
    points, triangles, tolerance=0.0003, limit=12, normal_dot=0.9, poses=None
):
    """Partition *original* triangles by connected edges and support-plane error."""
    edges, adjacent = {}, [set() for _ in triangles]
    for i, tri in enumerate(triangles):
        for a, b in zip(tri, np.roll(tri, 1)):
            edge = tuple(sorted((int(a), int(b))))
            for j in edges.get(edge, []):
                adjacent[i].add(j)
                adjacent[j].add(i)
            edges.setdefault(edge, []).append(i)

    def valid(group):
        for state in [points] if poses is None else poses:
            q = state[np.unique(triangles[group])]
            _, _, vectors = np.linalg.svd(q - q.mean(0), full_matrices=False)
            if np.ptp(q @ vectors[-1]) > tolerance:
                return False
            t = state[triangles[group]]
            normals = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
            if np.min(normals @ normals[0]) < normal_dot:
                return False
        return True

    unused, result = set(range(len(triangles))), []
    while unused:
        group = [min(unused)]
        unused.remove(group[0])
        while len(group) < limit:
            candidates = sorted(set().union(*(adjacent[i] for i in group)) & unused)
            candidate = next((i for i in candidates if valid(group + [i])), None)
            if candidate is None:
                break
            group.append(candidate)
            unused.remove(candidate)
        result.append(group)
    return result


def tracked_partition(
    ir, targets, rest, binary, config, critical=None, expression_positions=None
):
    """Retain evaluated original meshes as hidden masters; don't copy pose tensors.

    Proxies have the original order tracks and are welded to the final evaluated
    master corners AFTER its original seam/shape Glue. Existing mask references
    still resolve to exactly the same source geometry.
    """
    children, diagnostics, glues = [], [], []
    cursor = binary.stat().st_size // 4
    with binary.open("ab") as stream:
        for mesh in targets:
            vertices = mesh["source"]["vertices"]
            triangles = np.array(mesh["indices"]).reshape(-1, 3)
            selected = (
                np.arange(len(triangles))
                if critical is None
                else np.where(np.isin(mesh["source"]["triangles"], sorted(critical)))[0]
            )
            if not len(selected):
                continue
            groups = [
                selected[g].tolist()
                for g in planar_groups(
                    rest[vertices],
                    triangles[selected],
                    poses=None
                    if expression_positions is None
                    else expression_positions[:, vertices],
                    **config,
                )
            ]
            remainder = np.setdiff1d(np.arange(len(triangles)), selected).tolist()
            corrected_groups = len(groups)
            if remainder:
                groups.append(remainder)
            for gi, group in enumerate(groups):
                used, indices = np.unique(triangles[group].ravel(), return_inverse=True)
                child = copy.deepcopy(mesh)
                child["id"] = f"{mesh['id']}_F{gi}"
                child["face_partition_parent"] = mesh["id"]
                child["face_depth_corrected"] = gi < corrected_groups
                for key in ["positions", "uvs"]:
                    child[key] = (
                        np.array(mesh[key]).reshape(-1, 2)[used].ravel().tolist()
                    )
                child["indices"] = indices.tolist()
                child["source"]["vertices"] = np.array(vertices)[used].tolist()
                child["source"]["triangles"] = np.array(mesh["source"]["triangles"])[
                    group
                ].tolist()
                for cell in child["cells"]:
                    cell["offset"] = cursor
                for blend in child["blend_shapes"]:
                    # Retain secondary draw-order/opacity bindings. Geometry
                    # stays zero until final master Glue, including these forms.
                    blend["offsets"] = [
                        None if o is None else cursor for o in blend["offsets"]
                    ]
                stream.write(np.zeros(len(used) * 2, dtype="<f4").tobytes())
                cursor += len(used) * 2
                glues.append(
                    {
                        "id": "G_FaceProxy_" + child["id"],
                        "mesh_a": mesh["id"],
                        "mesh_b": child["id"],
                        "pairs": [[int(v), i, 0.0, 1.0] for i, v in enumerate(used)],
                        "intensity": 1.0,
                    }
                )
                children.append(child)
            mesh["mask_only"] = True
            mesh["opacity"] = 0.0
            mesh["face_partition_master"] = True
            diagnostics.append(
                {
                    "master": mesh["id"],
                    "children": len(groups),
                    "corrected_triangles": len(selected),
                }
            )
    ir["meshes"].extend(children)
    ir["glues"].extend(glues)
    return diagnostics


def protect_teeth(ir, samples, binary, config):
    """Keep teeth above the oral shell, with true 3D foreground clipping.

    A single curved oral ArtMesh cannot supply the correct tooth/cavity order
    at every pixel. Stable tooth order plus local source planes handles the
    interpenetrating lip/cavity boundary continuously, including open-mouth
    and smile combinations.
    """
    source = np.load(samples / "M_Face_source.npz")
    _, components = component_labels(source)
    tooth_components = [7, 8, 9, 10]
    originals = [
        m
        for m in ir["meshes"]
        if not m.get("face_partition_parent")
        and (not m.get("mask_only") or m.get("face_partition_master"))
    ]
    teeth = [
        m
        for m in originals
        if m["source"]["object"] == "M_Face"
        and np.all(np.isin(components[m["source"]["triangles"]], tooth_components))
    ]
    if not teeth:
        raise ValueError("Audited source tooth components are absent")
    manifest = json.loads((samples / "manifest.json").read_text())
    neutral = next(
        p
        for p in manifest["head_poses"]
        if all(v == 0 for v in p["parameters"].values())
    )
    archive = np.load(samples / neutral["file"])
    poses = np.array(
        [
            archive["M_Face"]
            + a * archive["M_Face__ParamMouthOpenY"]
            + b * archive["M_Face__ParamMouthForm"]
            for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]
        ]
    )
    tracked_partition(
        ir,
        teeth,
        source["positions"],
        binary,
        config.get("teeth_partition", config["partition"]),
        expression_positions=poses
        if config.get("teeth_check_expression_planarity")
        else None,
    )
    targets = [
        m
        for m in ir["meshes"]
        if not m.get("mask_only")
        and m.get("face_partition_parent") in {t["id"] for t in teeth}
    ]
    for m in targets:
        m["face_order_override"] = 810.0
        for cell in m["cells"]:
            cell["order"] = 810.0
        for blend in m["blend_shapes"]:
            blend["orders"] = [810.0] * len(blend["keys"])
    occluders = [
        m
        for m in originals
        if m["source"]["object"] == "M_Hair"
        or (
            m["source"]["object"] == "M_Face"
            and not np.any(
                np.isin(components[m["source"]["triangles"]], tooth_components)
            )
        )
    ]
    manifest = json.loads((samples / "manifest.json").read_text())
    build_local_surface_masks(
        ir,
        samples,
        manifest["semantics"],
        binary,
        targets,
        occluders,
        "Teeth_",
        "teeth_visibility",
        "teeth_visibility",
        merge_masks=True,
        support_margin=config.get("teeth_support_margin", config["support_margin"]),
        include_secondary=True,
        double_sided_masks=True,
        expression_keys={
            p: config.get("teeth_expression_keys", [0, 0.5, 1])
            for p in ["ParamMouthOpenY", "ParamMouthForm"]
        },
        envelope_bounds=True,
        footprint_margin=config["footprint_margin"],
    )
    return {
        "targets": len(targets),
        "source_components": tooth_components,
        "native_order": 810.0,
        "method": "Source-connected tracked teeth, fixed oral hierarchy with expression-dependent 3D inverse depth masks",
    }


def compress_new_fields(ir, original_count):
    """Reduce only new scalar tensors, exactly; keep every baseline Glue intact."""
    subset = {
        "glues": ir["glues"][original_count:],
        "parameters": ir["parameters"],
        "diagnostics": {},
    }
    result = compress_glue_fields(subset)
    ir["glues"] = ir["glues"][:original_count] + subset["glues"]
    return result


def mouth_aperture(ir, samples, binary):
    """Give the actual front facial skin priority over internal teeth.

    Reuse the audited source-geometry rear filter already used for the eyes;
    full head skin would incorrectly fill the mouth opening with rear skull.
    The aperture follows native face morphology directly, without a PNG contour.
    """
    _, components = component_labels(np.load(samples / "M_Face_source.npz"))
    lookup = {m["id"]: m for m in ir["meshes"]}
    skin = [
        n
        for n in ir["visibility"]["skin_mask_sources"]
        if lookup[n]["source"]["object"] == "M_Face"
        and np.all(components[lookup[n]["source"]["triangles"]] == 0)
    ]
    targets = [
        m for m in ir["meshes"] if not m.get("mask_only") and "face_order_override" in m
    ]
    for m in targets:
        m["masked_by"] = list(dict.fromkeys(m.get("masked_by", []) + skin))
    result = factor_inherited_masks(
        ir, binary, source_groups={m["id"]: skin for m in targets}, prefix="MouthSkin"
    )
    result["front_skin_sources"] = len(skin)
    return result


def order_tracked_surfaces(ir):
    """Preserve original tie ordering against eyes/headwear/other source parts."""
    children = {}
    for m in ir["meshes"]:
        if m.get("face_partition_parent") and not m.get("mask_only"):
            children.setdefault(m["face_partition_parent"], []).append(m)
    child_ids = {m["id"] for group in children.values() for m in group}
    ordered = []
    for m in ir["meshes"]:
        if m["id"] in child_ids:
            continue
        ordered.append(m)
        ordered.extend(children.get(m["id"], []))
    ir["meshes"] = ordered


def convert(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source.parent == destination:
        raise ValueError("Choose a separate output directory to preserve the baseline")
    destination.mkdir(parents=True, exist_ok=True)
    ir = json.loads(source.read_text())
    original_glues = len(ir["glues"])
    camera = np.array(ir["camera"]["projection"]) @ np.linalg.inv(
        np.array(ir["camera"]["matrix_world"])
    )
    if not np.allclose(camera[3, :3], 0):
        raise ValueError(
            "Expression clip fields require the verified orthographic camera"
        )
    for texture in ir["textures"]:
        shutil.copy2(source.parent / texture, destination / Path(texture).name)
    ir["textures"] = [Path(t).name for t in ir["textures"]]
    binary = destination / "keyforms.f32"
    shutil.copy2(source.parent / ir["keyform_data"], binary)
    ir["keyform_data"] = binary.name
    samples = ROOT / "artifacts/samples"
    manifest = json.loads((samples / "manifest.json").read_text())
    config = json.loads((ROOT / "config/face_clearance.json").read_text())
    conflicts = json.loads(
        (ROOT / "artifacts/face_repair/screen/conflicts.json").read_text()
    )
    if conflicts["source_ir_sha256"] != hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError("Depth conflict screen belongs to a different source rig")
    if (
        conflicts["samples_manifest_sha256"]
        != hashlib.sha256((samples / "manifest.json").read_bytes()).hexdigest()
    ):
        raise ValueError("Depth conflict screen belongs to different sampling metadata")
    selected = {m["mesh"] for m in conflicts["meshes"] if m["wrong_cover"] > 0}
    hair_source = np.load(samples / "M_Hair_source.npz")
    meshes = [m for m in ir["meshes"] if not m.get("mask_only")]
    skin = [m for m in meshes if m["id"] in ir["visibility"]["skin_surface_meshes"]]
    _, components = component_labels(np.load(samples / "M_Face_source.npz"))
    excluded = json.loads((ROOT / "config/surface_visibility.json").read_text())[
        "lashes"
    ]["components"]
    face_occluders = [
        m
        for m in meshes
        if m["source"]["object"] == "M_Face"
        and not np.any(np.isin(components[m["source"]["triangles"]], excluded))
    ]
    hair = [
        m
        for m in meshes
        if m["source"]["object"] == "M_Hair"
        and hair_source["positions"][m["source"]["vertices"], 2].max()
        > config["hair_min_top_z"]
    ]
    partition = []
    for name, source_data, targets in [
        ("M_Hair", hair_source, hair),
        ("M_Face", np.load(samples / "M_Face_source.npz"), skin),
    ]:
        critical = {
            t["triangle"]
            for t in conflicts["triangles"]
            if t["mesh"].startswith(name + "_")
        }
        # A single source-vertex ring protects narrow conflicts between sampled
        # poses, including tiny triangles which have no pixel centre at a key.
        vertices = set(source_data["triangles"][sorted(critical)].ravel())
        critical.update(
            np.where(
                np.any(np.isin(source_data["triangles"], sorted(vertices)), axis=1)
            )[0].tolist()
        )
        partition.extend(
            tracked_partition(
                ir,
                [m for m in targets if set(m["source"]["triangles"]) & critical],
                source_data["positions"],
                binary,
                config["partition"],
                critical,
            )
        )
    corrected_skin = [
        m
        for m in ir["meshes"]
        if m.get("face_depth_corrected") and m["source"]["object"] == "M_Face"
    ]
    corrected_hair = [
        m
        for m in ir["meshes"]
        if m.get("face_depth_corrected") and m["source"]["object"] == "M_Hair"
    ]
    for targets, occluders, prefix in [
        (corrected_skin, hair, "FaceHair"),
        (corrected_hair, face_occluders, "HairFace"),
    ]:
        build_local_surface_masks(
            ir,
            samples,
            manifest["semantics"],
            binary,
            targets,
            occluders,
            prefix + "_",
            prefix,
            prefix,
            merge_masks=True,
            support_margin=config["support_margin"],
            include_secondary=True,
            envelope_bounds=True,
            footprint_margin=config["footprint_margin"],
        )
    # The actual black nose component is audited from source connectivity,
    # rather than confused with the stray dark rear-hair fragment near its tip.
    _, components = component_labels(np.load(samples / "M_Face_source.npz"))
    nose = []
    for mesh in meshes:
        if mesh["source"]["object"] == "M_Face" and np.all(
            components[mesh["source"]["triangles"]] == 22
        ):
            for cell in mesh["cells"]:
                cell["order"] += config["nose_order_bias"]
            nose.append(mesh["id"])
    ir["diagnostics"]["face_clearance"] = {
        "hair_partition": partition,
        "nose": nose,
        "config": config,
        "conflict_meshes": sorted(selected),
    }
    ir["diagnostics"]["face_clearance"]["mask_unions"] = factor_inherited_masks(
        ir, binary
    )
    ir["diagnostics"]["face_clearance"]["teeth"] = protect_teeth(
        ir, samples, binary, config
    )
    ir["diagnostics"]["face_clearance"]["mouth_aperture"] = mouth_aperture(
        ir, samples, binary
    )
    ir["diagnostics"]["face_clearance"]["packed_sources"] = (
        pack_local_sources(ir, binary)
        if config.get("pack_local_sources")
        else {
            "enabled": False,
            "reason": "Full-source packing increases motion-update work and model size; keep only the shared inherited unions.",
        }
    )
    ir["diagnostics"]["face_clearance"]["scalar_compression"] = compress_new_fields(
        ir, original_glues
    )
    order_tracked_surfaces(ir)
    for part in ir["parts"]:
        part["mesh_ids"] = [
            m["id"]
            for m in ir["meshes"]
            if m["source"]["object"] == part["source_object"]
        ]
    ir["diagnostics"]["keyform_bytes"] = compact_forms(ir, binary)
    ir["diagnostics"].update(
        artmesh_count=len(ir["meshes"]),
        vertices=sum(len(m["positions"]) // 2 for m in ir["meshes"]),
        triangles=sum(len(m["indices"]) // 3 for m in ir["meshes"]),
        glues=len(ir["glues"]),
        animated_glues=sum(bool(g.get("axes")) for g in ir["glues"]),
        masked_drawables=sum(bool(m.get("masked_by")) for m in ir["meshes"]),
    )
    jsonwrite(destination / "POPY.rig.json", ir)
    return destination / "POPY.rig.json"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source", type=Path, default=ROOT / "artifacts/neck_repair/rig/POPY.rig.json"
    )
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/face_repair/rig")
    args = p.parse_args()
    convert(args.source, args.out)
