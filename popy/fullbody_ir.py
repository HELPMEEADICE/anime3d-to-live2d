"""Add source-skinned limbs and previously cropped surfaces to the repaired head rig."""

import itertools
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from .compile_ir import ROOT, jsonwrite, project
from .kinematics import Skeleton


def reframe(ir, config):
    """Exact uniform camera change: parent UV coordinates and Glue stay unchanged."""
    old = json.loads((ROOT / "config/convert.json").read_text())["camera"]
    scale = old["ortho_scale"] / config["camera"]["ortho_scale"]
    size = np.asarray(ir["canvas"])
    center = size / 2
    translation = center * (1 - scale)
    # Blender ortho_scale spans the longer image dimension (the canvas height).
    translation[1] += (
        (config["camera"]["center"][2] - old["center"][2])
        * size[1]
        / config["camera"]["ortho_scale"]
    )

    def scaled(values):
        return (np.asarray(values) * scale).tolist()

    ir["body_rotation"]["pivot"] = (
        np.array(ir["body_rotation"]["pivot"]) * scale + translation
    ).tolist()
    for node in [ir["root_warp"], ir["head_warp"]] + ir["body_warps"]:
        for cell in node["cells"]:
            cell["points"] = scaled(cell["points"])
    for cell in ir["head_rotation"]["cells"]:
        cell["origin"] = scaled(cell["origin"])
    for mesh in ir["meshes"]:
        mesh["positions"] = (
            (np.array(mesh["positions"]).reshape(-1, 2) * scale + translation)
            .ravel()
            .tolist()
        )
    camera = ir["camera"]
    matrix = np.array(camera["matrix_world"])
    matrix[:3, 3] += np.array(config["camera"]["center"]) - np.array(old["center"])
    camera["matrix_world"] = matrix.tolist()
    projection = np.array(camera["projection"])
    projection[:2] *= scale
    camera["projection"] = projection.tolist()
    return {"scale": scale, "translation": translation.tolist()}


def compile_fullbody(
    config_path=ROOT / "config/fullbody.json",
    destination=ROOT / "artifacts/fullbody/rig",
):
    config = json.loads(Path(config_path).read_text())
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=True)
    baseline = ROOT / config["baseline"]
    ir = json.loads(baseline.read_text())
    skeleton = Skeleton(ROOT / "artifacts/samples", config)
    source_audit = json.loads(
        (ROOT / "artifacts/fullbody/source_audit.json").read_text()
    )
    simplification = {}
    original_cam = json.loads(json.dumps(ir["camera"]))
    size = np.array(ir["canvas"])
    old_pivot = np.array(ir["body_rotation"]["pivot"])
    original_count = len(ir["meshes"])
    controls = config["controls"]
    for c in controls:
        ir["parameters"].append(
            {k: c[k] for k in ["id", "min", "max", "default", "name"]}
        )
    # An identity canvas warp under the existing exact body-roll deformer.
    limb_warp = {
        "id": "D_ArticulatedBody",
        "strength": 1.0,
        "axes": [],
        "cells": [
            {
                "coordinate": [],
                "points": (
                    np.array([[0, 0], [size[0], 0], [0, size[1]], size]) - old_pivot
                )
                .ravel()
                .tolist(),
            }
        ],
    }
    ir["body_warps"].append(limb_warp)
    manifest = json.loads((ROOT / "artifacts/samples/manifest.json").read_text())
    groups = defaultdict(list)
    coverage = defaultdict(set)
    for mesh in ir["meshes"]:
        source = mesh["source"]
        # Hidden masters still own original source triangles used by visible proxies.
        if mesh["id"].startswith(source["object"] + "_"):
            coverage[source["object"]].update(source["triangles"])

    def dependency(name, vertex_ids):
        _, weights, deps = skeleton.object(name)
        indices = np.array(vertex_ids)
        flags = deps[indices].any(0)
        axes = [
            {"parameter": c["id"], "keys": c["keys"]}
            for c, active in zip(controls, flags)
            if active
        ]
        body_bones = []
        head_bones = []
        for i, b in enumerate(skeleton.bones):
            j = i
            ancestors = []
            while j >= 0:
                ancestors.append(skeleton.names[j])
                j = skeleton.parents[j]
            if "Spine2" in ancestors:
                body_bones.append(i)
            if "Head" in ancestors:
                head_bones.append(i)
        body_weight = weights[indices][:, body_bones].sum(1)
        if body_weight.max() > 1e-7:
            axes += [
                {"parameter": p, "keys": [-15, 0, 15]}
                for p in ["ParamBodyAngleX", "ParamBodyAngleY"]
            ]
        if body_weight.min() < 0.999999:
            axes += [{"parameter": "ParamBodyAngleZ", "keys": [-15, 0, 15]}]
        if flags.any() and weights[indices][:, head_bones].sum(1).max() > 1e-7:
            raise ValueError(
                "New articulated support overlaps head; split source support first: "
                + name
            )
        return axes, bool(flags.any())

    updated = []
    for mesh in ir["meshes"]:
        source = mesh["source"]
        name = source["object"]
        if (
            name not in {o["name"] for o in manifest["objects"]}
            or mesh.get("deformer") == "head"
        ):
            continue
        if any(
            k in source
            for k in ["barycentric_vertices", "barycentric_sources", "master_mesh"]
        ):
            continue
        # Only original identity surfaces, not parameter-dependent clipping helpers.
        if not mesh["id"].startswith(name + "_"):
            continue
        axes, active = dependency(name, source["vertices"])
        if active:
            slots = source_audit["objects"][name]["materials"]
            src = skeleton.object(name)[0]
            flags = {
                slots[int(src["material_indices"][t])]["culling"]
                for t in source["triangles"]
            }
            if len(flags) != 1:
                raise ValueError("Mixed source culling support: " + mesh["id"])
            mesh["culling"] = flags.pop()
            groups[(name, tuple(a["parameter"] for a in axes))].append((mesh, axes))
            updated.append(mesh["id"])
    # The half-body predecessor intentionally pruned unseen legs/hands/torso surfaces.
    # Recover all omitted original triangles, including backs exposed by bending.
    additions = []
    objects = {o["name"]: o for o in manifest["objects"]}
    for name, obj in objects.items():
        if name in {"M_Face", "M_Eye", "M_Brow", "M_Hair", "M_Headwear"}:
            continue
        src, _, deps = skeleton.object(name)
        valid = np.ones(len(src["triangles"]), bool)
        if name == "M_Body" and config.get("hand_simplification_error_m", 0) > 0:
            from .source_simplify import endpoint_collapse

            columns = [
                j
                for j, c in enumerate(controls)
                if c["id"] in {"ParamHandLPose", "ParamHandRPose"}
            ]
            candidates = deps[:, columns].any(1)
            pinned = np.zeros(len(src["positions"]), bool)
            covered = np.array(sorted(coverage[name]), int)
            if len(covered):
                pinned[np.unique(src["triangles"][covered])] = True
            samples = []
            for hand, wrist in itertools.product([-1, -0.5, 0, 0.5, 1], [-35, 0, 35]):
                samples.append(
                    skeleton.evaluate(
                        name,
                        {
                            "ParamHandLPose": hand,
                            "ParamHandRPose": hand,
                            "ParamWristL": wrist,
                            "ParamWristR": wrist,
                        },
                    )
                )
            src, valid, simplification[name] = endpoint_collapse(
                src,
                candidates,
                np.array(samples),
                pinned,
                error_m=config["hand_simplification_error_m"],
            )
            jsonwrite(
                out / (name + "_vertex_remap.json"),
                {
                    "original_to_retained_source_vertex": src["vertex_remap"].tolist(),
                    "retained_original_triangle_ids": np.flatnonzero(valid).tolist(),
                },
            )
        ids = np.array(
            [
                i
                for i in range(len(src["triangles"]))
                if valid[i] and i not in coverage[name]
            ],
            dtype=int,
        )
        if not len(ids):
            continue
        # Existing head surfaces were pruned over their validated turn domain.
        # Do not resurrect rear skull/neck triangles without matching head grids.
        head_bones = []
        for bi in range(len(skeleton.names)):
            j = bi
            while j >= 0 and skeleton.names[j] != "Head":
                j = skeleton.parents[j]
            if j >= 0:
                head_bones.append(bi)
        head_weight = skeleton.object(name)[1][:, head_bones].sum(1)
        ids = ids[head_weight[src["triangles"][ids]].max(1) < 1e-7]
        if not len(ids):
            continue
        tris = src["triangles"][ids]
        signatures = np.bitwise_or.reduce(
            (
                deps[tris].any(1).astype(np.uint64)
                * (np.uint64(1) << np.arange(len(controls), dtype=np.uint64))
            ),
            axis=1,
        )
        signatures |= src["material_indices"][ids].astype(np.uint64) << np.uint64(50)
        request = out / (name + "_partition_in.json")
        response = out / (name + "_partition.json")
        jsonwrite(
            request,
            {
                "triangles": tris.tolist(),
                "groups": signatures.tolist(),
                "limit": config["patch_triangles"],
            },
        )
        subprocess.run(
            [
                str(ROOT / "rust/target/release/popy-geometry.exe"),
                str(request),
                str(response),
            ],
            check=True,
        )
        for pi, patch in enumerate(json.loads(response.read_text())["patches"]):
            original = ids[patch]
            lookup, vertices, uvs, indices = {}, [], [], []
            for v, uv in zip(
                src["triangles"][original].ravel(), src["uv"][original].reshape(-1, 2)
            ):
                key = (int(v), *np.round(uv, 7))
                if key not in lookup:
                    lookup[key] = len(vertices)
                    vertices.append(int(v))
                    uvs.extend([float(uv[0]), float(1 - uv[1])])
                indices.append(lookup[key])
            texture = obj["textures"][int(src["material_indices"][original[0]])]
            if texture not in ir["textures"]:
                ir["textures"].append(texture)
            mesh = {
                "id": "Full_" + name + "_" + str(pi),
                "positions": (project(src["positions"][vertices], original_cam) * size)
                .ravel()
                .tolist(),
                "uvs": uvs,
                "indices": np.array(indices)
                .reshape(-1, 3)[:, [0, 2, 1]]
                .ravel()
                .tolist(),
                "texture": ir["textures"].index(texture),
                "culling": source_audit["objects"][name]["materials"][
                    int(src["material_indices"][original[0]])
                ]["culling"],
                "deformer": "body",
                "masked_by": [],
                "blend_shapes": [],
                "source": {
                    "object": name,
                    "vertices": vertices,
                    "triangles": original.tolist(),
                    "barycentric": "original source endpoints; conservative finger edge collapse"
                    if name in simplification
                    else "identity at original triangle corners",
                },
            }
            axes, _ = dependency(name, vertices)
            groups[(name, tuple(a["parameter"] for a in axes))].append((mesh, axes))
            additions.append(mesh)
    shutil.copy2(baseline.parent / ir["keyform_data"], out / ir["keyform_data"])
    for texture in ir["textures"]:
        src = baseline.parent / texture
        if not src.exists():
            src = ROOT / "artifacts/samples" / texture
        shutil.copy2(src, out / texture)
    matrix = np.array(original_cam["projection"]) @ np.linalg.inv(
        original_cam["matrix_world"]
    )
    cursor = (out / ir["keyform_data"]).stat().st_size // 4
    with (out / ir["keyform_data"]).open("ab") as stream:
        for gi, ((name, parameters), entries) in enumerate(groups.items()):
            axes = entries[0][1]
            coordinates = list(
                itertools.product(*(range(len(a["keys"])) for a in axes))
            )
            if len(coordinates) > config["max_grid_cells"]:
                raise ValueError(
                    f"Articulated grid budget: {name}: {len(coordinates)} {parameters}"
                )
            vertices = np.unique(
                [v for mesh, _ in entries for v in mesh["source"]["vertices"]]
            )
            _, weights, _ = skeleton.object(name)
            active = np.flatnonzero(weights[vertices].any(0))
            rest = np.c_[
                skeleton.object(name)[0]["positions"][vertices], np.ones(len(vertices))
            ]
            world = np.zeros((len(coordinates), len(vertices), 3), np.float32)
            for ci, coordinate in enumerate(coordinates):
                values = {
                    a["parameter"]: a["keys"][k] for a, k in zip(axes, coordinate)
                }
                skin = skeleton.skin_matrices(values)
                for bone in active:
                    world[ci] += (
                        (rest @ skin[bone].T)[:, :3] * weights[vertices, bone, None]
                    ).astype(np.float32)
                unbound = weights[vertices].sum(1) == 0
                world[ci, unbound] = rest[unbound, :3]
            homogeneous = np.concatenate(
                [world, np.ones(world.shape[:-1] + (1,), np.float32)], axis=2
            )
            q = homogeneous @ matrix.T
            q = q[..., :2] / q[..., 3:4]
            q = np.stack([(q[..., 0] + 1) / 2, (1 - q[..., 1]) / 2], axis=-1)
            # Undo the existing global native roll, including vertices below Spine2.
            for ci, coordinate in enumerate(coordinates):
                values = {
                    a["parameter"]: a["keys"][k] for a, k in zip(axes, coordinate)
                }
                z = np.deg2rad(values.get("ParamBodyAngleZ", 0))
                if z:
                    r = np.array([[np.cos(z), -np.sin(z)], [np.sin(z), np.cos(z)]])
                    q[ci] = ((q[ci] * size - old_pivot) @ r + old_pivot) / size
            for mesh, _ in entries:
                take = np.searchsorted(vertices, mesh["source"]["vertices"])
                forms = q[:, take].astype("<f4")
                orders = 500 - world[:, take, 1].mean(1) * 800
                stride = len(mesh["positions"])
                mesh["cells"] = [
                    {
                        "coordinate": list(c),
                        "offset": cursor + i * stride,
                        "order": float(orders[i]),
                    }
                    for i, c in enumerate(coordinates)
                ]
                stream.write(forms.tobytes())
                cursor += forms.size
                mesh["axes"] = axes
                mesh["blend_shapes"] = []
                mesh["projection_warp"] = "D_ArticulatedBody"
                mesh["deformer"] = "body"
                mesh["articulated"] = True
            print(
                "ARTICULATE",
                gi,
                name,
                len(entries),
                len(coordinates),
                len(vertices),
                parameters,
                flush=True,
            )
    ir["meshes"].extend(additions)
    transform = reframe(ir, config)
    for part in ir["parts"]:
        part["mesh_ids"].extend(
            m["id"] for m in additions if m["source"]["object"] == part["source_object"]
        )
    known = {p["source_object"] for p in ir["parts"]}
    for name in sorted({m["source"]["object"] for m in additions} - known):
        ir["parts"].append(
            {
                "id": name,
                "source_object": name,
                "mesh_ids": [
                    m["id"] for m in additions if m["source"]["object"] == name
                ],
            }
        )
    ir["diagnostics"]["fullbody"] = {
        "baseline_meshes": original_count,
        "updated_original_meshes": updated,
        "added_meshes": len(additions),
        "controls": len(controls),
        "camera_transform": transform,
        "source_simplification": simplification,
        "limitations": [
            "Sparse multilinear articulated joint grids; finite-pose validation only",
            "Mean-depth patches approximate intersecting 3D surfaces",
            "New limbs do not include breath scaling; original head and torso breath bindings are retained",
            "Original collar/hair collision trajectory limitations are retained",
        ],
    }
    # Existing grids are already compact. Preserve their Glue fields byte for byte.
    if config.get("native_skinning", False):
        from .native_skinning import lower

        lower(ir, out / "keyforms.f32", config)
        ir["diagnostics"]["fullbody"]["limitations"] = [
            "Single-parameter matrix fields retain finite interpolation and floating-point error",
            "Mean-depth patches, culling and garment order bias approximate intersecting 3D surfaces",
            "Finite-pose validation only; no cloth collision or foot-contact IK",
            "Original collar/hair collision trajectory limitations are retained",
        ]
    from .form_storage import compact_forms

    # Do not recompute screen-space bounds after changing the camera.
    for g in ir.get("glues", []):
        if g.get("axes"):
            g.setdefault("compression_error_bound_px", 0)
    compact_forms(ir, out / ir["keyform_data"])
    ir["diagnostics"]["fullbody"].update(
        meshes=len(ir["meshes"]),
        vertices=sum(len(m["positions"]) // 2 for m in ir["meshes"]),
        triangles=sum(len(m["indices"]) // 3 for m in ir["meshes"]),
        keyform_bytes=(out / ir["keyform_data"]).stat().st_size,
        authored_keyforms=sum(len(m["cells"]) for m in ir["meshes"]),
    )
    ir["diagnostics"].update(
        artmesh_count=len(ir["meshes"]),
        vertices=sum(len(m["positions"]) // 2 for m in ir["meshes"]),
        triangles=sum(len(m["indices"]) // 3 for m in ir["meshes"]),
        parameters=len(ir["parameters"]),
        authored_keyforms=sum(len(m["cells"]) for m in ir["meshes"]),
        keyform_bytes=(out / ir["keyform_data"]).stat().st_size,
        deformers=len(ir["body_warps"]) + 3,
    )
    jsonwrite(out / "POPY.rig.json", ir)
    jsonwrite(out / "statistics.json", ir["diagnostics"]["fullbody"])
    print("FULLBODY_IR", json.dumps(ir["diagnostics"]["fullbody"])[:1000], flush=True)
    return out / "POPY.rig.json"


if __name__ == "__main__":
    compile_fullbody()
