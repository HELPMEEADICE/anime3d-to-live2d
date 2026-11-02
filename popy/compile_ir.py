"""SPDX-License-Identifier: MIT
Build a neutral Rig IR from evaluated 3D samples, preserving source loop UVs and triangle IDs.
Every geometry keyform is a 3D reprojection. Facial combinations exploit linear morph/skin
superposition at EACH head pose; they do not assume screen-space expression deltas are fixed.
"""

import itertools, json, subprocess, time, shutil, os, argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def project(v, camera):
    matrix = np.array(camera["projection"]) @ np.linalg.inv(
        np.array(camera["matrix_world"])
    )
    q = np.c_[v, np.ones(len(v))] @ matrix.T
    q = q[:, :3] / q[:, 3:4]
    return np.c_[(q[:, 0] + 1) / 2, (1 - q[:, 1]) / 2].astype(np.float32)


def jsonwrite(path, data):
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf8")


def compile_ir(
    samples=ROOT / "artifacts/samples", out=ROOT / "artifacts/rig", patch_limit=None
):
    start = time.perf_counter()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    samples = Path(samples)
    m = json.loads((samples / "manifest.json").read_text())
    # An optional occluder can be assigned after sampling: all source objects were
    # already captured, unlike changing a driven bone or morph dependency.
    overrides = json.loads((ROOT / "config/semantics.json").read_text())
    hair = overrides.get("objects", {}).get("hair")
    if hair:
        if hair not in {obj["name"] for obj in m["objects"]}:
            raise ValueError(
                f"Configured hair occluder {hair!r} has no source geometry"
            )
        m["semantics"]["objects"]["hair"] = hair
    cfg = m["config"].copy()
    current = json.loads(
        Path(os.environ.get("POPY_CONFIG", ROOT / "config/convert.json")).read_text()
    )
    for key in [
        "patch_triangles",
        "max_grid_cells",
        "influence_epsilon",
        "couple_body_pitch",
    ]:
        cfg[key] = current[key]
    coupled_pitch = cfg.get("couple_body_pitch", False)
    cam = m["camera"]
    width, height = cfg["canvas"]
    canvas = np.array([width, height], np.float32)
    shape_defs = m["semantics"]["shape_parameters"]
    gaze_name = m["semantics"]["objects"]["gaze"]
    head_axes = [
        {"parameter": "ParamAngleX", "keys": cfg["head_x_keys"]},
        {"parameter": "ParamAngleY", "keys": cfg["head_y_keys"]},
        {"parameter": "ParamAngleZ", "keys": [-30, 0, 30]},
    ]
    parameters = [
        dict(id=a["parameter"], min=min(a["keys"]), max=max(a["keys"]), default=0)
        for a in head_axes
    ]
    secondary = {"ParamEyeLSmile", "ParamEyeRSmile"}
    parameters += [
        dict(
            id=p,
            min=s["min"],
            max=s["max"],
            default=s["default"],
            kind="blend" if p in secondary else "normal",
        )
        for p, s in shape_defs.items()
    ]
    parameters += [
        dict(id=p, min=-1, max=1, default=0) for p in ["ParamEyeBallX", "ParamEyeBallY"]
    ]
    parameters += [
        dict(id="ParamBodyAngleX", min=-15, max=15, default=0),
        dict(
            id="ParamBodyAngleY",
            min=-15,
            max=15,
            default=0,
            kind="normal" if coupled_pitch else "blend",
        ),
    ]
    parameters += [
        dict(id="ParamBodyAngleZ", min=-15, max=15, default=0),
        dict(id="ParamBreath", min=0, max=1, default=0),
    ]
    ir = {
        "schema": "popy.rig/1",
        "name": "POPY",
        "canvas": cfg["canvas"],
        "camera": cam,
        "parameters": parameters,
        "textures": [],
        "parts": [],
        "meshes": [],
        "keyform_data": "keyforms.f32",
        "geometry_encoding": "parent_absolute",
        "physics": [],
        "diagnostics": {
            "warnings": m["warnings"],
            "geometry": "original evaluated 3D triangles, UV-seam split; head XYZ multilinear with pose-dependent expressions",
            "occlusion": "small connected surface patches, keyed mean camera depth, backface culling; no per-pixel z-buffer",
        },
        "root_warp": {},
    }
    # Factor exact in-plane rotation out of vertex interpolation. Keep only the non-planar
    # head/body-yaw coupling as mesh residuals, avoiding chord shrink at unseen roll angles.
    bones = json.loads((samples / "skeleton.json").read_text())
    bind = {b["name"]: np.array(b["matrix_local"])[:3, 3] for b in bones}
    head_pivot_world = bind[m["semantics"]["bones"]["head"]]
    body_pivot_world = bind[m["semantics"]["bones"]["body"]]
    head_pivot = project(np.array([head_pivot_world]), cam)[0] * canvas
    body_pivot = project(np.array([body_pivot_world]), cam)[0] * canvas
    pivot = body_pivot
    ir["body_rotation"] = {
        "pivot": body_pivot.tolist(),
        "parameter": "ParamBodyAngleZ",
        "keys": [-15, 0, 15],
    }
    head_pivots = {}
    head_rot_cells = []
    pitch_keys = [-15, 0, 15] if coupled_pitch else [0]
    for i, bx in enumerate([-15, 0, 15]):
        for bi, by in enumerate(pitch_keys):
            a = np.radians(bx)
            b = np.radians(by)
            R = np.array(
                [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
            ) @ np.array(
                [[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]]
            )
            p = (
                project(
                    np.array(
                        [(head_pivot_world - body_pivot_world) @ R.T + body_pivot_world]
                    ),
                    cam,
                )[0]
                * canvas
            )
            head_pivots[(bx, by)] = p
            for j, z in enumerate([-30, 0, 30]):
                for breath in [0, 1]:
                    head_rot_cells.append(
                        {
                            "coordinate": [i, bi, j, breath]
                            if coupled_pitch
                            else [i, j, breath],
                            "origin": (
                                (p - body_pivot)
                                * [1 + 0.012 * breath, 1 + 0.006 * breath]
                            ).tolist(),
                            "angle": z,
                        }
                    )
    ir["head_rotation"] = {
        "axes": [{"parameter": "ParamBodyAngleX", "keys": [-15, 0, 15]}]
        + (
            [{"parameter": "ParamBodyAngleY", "keys": pitch_keys}]
            if coupled_pitch
            else []
        )
        + [
            {"parameter": "ParamAngleZ", "keys": [-30, 0, 30]},
            {"parameter": "ParamBreath", "keys": [0, 1]},
        ],
        "cells": head_rot_cells,
    }
    corners = np.array([[0, 0], [width, 0], [0, height], [width, height]], float)
    breath_axes = [{"parameter": "ParamBreath", "keys": [0, 1]}]
    ir["root_warp"] = {
        "axes": breath_axes,
        "cells": [
            {
                "coordinate": [i],
                "points": ((corners - body_pivot) * [1 + 0.012 * i, 1 + 0.006 * i])
                .ravel()
                .tolist(),
            }
            for i in [0, 1]
        ],
    }
    head_warp_cells = []
    for zi, z in enumerate([-30, 0, 30]):
        a = np.radians(z)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        for breath in [0, 1]:
            head_warp_cells.append(
                {
                    "coordinate": [zi, breath],
                    "points": (
                        (
                            (corners - head_pivot)
                            @ R.T
                            * [1 + 0.012 * breath, 1 + 0.006 * breath]
                        )
                        @ R
                    )
                    .ravel()
                    .tolist(),
                }
            )
    ir["head_warp"] = {
        "axes": [{"parameter": "ParamAngleZ", "keys": [-30, 0, 30]}] + breath_axes,
        "cells": head_warp_cells,
    }
    head_samples = [np.load(samples / p["file"]) for p in m["head_poses"]]
    refinement_path = ROOT / "config/patch_refinement.json"
    refinement = (
        json.loads(refinement_path.read_text()).get("objects", {})
        if refinement_path.exists()
        else {}
    )
    head_index = {
        tuple(p["parameters"][a["parameter"]] for a in head_axes): i
        for i, p in enumerate(m["head_poses"])
    }
    neutral = head_index[(0, 0, 0)]
    binary = (out / "keyforms.f32").open("wb")
    cursor = 0
    total_tris = 0
    total_vertices = 0
    authored = 0

    def emit(array):
        nonlocal cursor
        a = np.asarray(array, dtype="<f4").ravel()
        offset = cursor
        binary.write(a.tobytes())
        cursor += len(a)
        return offset

    partitioner = ROOT / "rust/target/release/popy-geometry.exe"
    if not partitioner.exists():
        raise RuntimeError(
            "Build Rust geometry first: cargo build --manifest-path rust/Cargo.toml -p popy-geometry --release"
        )
    for obj in m["objects"]:
        name = obj["name"]
        src = np.load(samples / (name + "_source.npz"))
        from .refine import fingerprint

        selected_refinement = refinement.get(name, {})
        if selected_refinement and selected_refinement["source_sha256"] != fingerprint(
            src
        ):
            selected_refinement = {}
        triangles = src["triangles"]
        triuv = src["uv"]
        object_keys = [name] + [
            name + "__" + pid for pid, s in shape_defs.items() if s["object"] == name
        ]
        if name == gaze_name:
            object_keys += [
                gaze_name + "__gaze_" + str(x) + "_" + str(y)
                for y, x in itertools.product([-1, 0, 1], repeat=2)
            ]
        object_samples = [{key: hs[key] for key in object_keys} for hs in head_samples]
        rest = object_samples[neutral][name]
        screen = project(rest, cam)
        # Retain surfaces visible anywhere in the trained head domain, including neutral hidden surfaces.
        keep = np.zeros(len(triangles), bool)
        body_data = {
            p: {
                v: np.load(samples / (p + "_" + str(v) + ".npz"))[name]
                for v in [-15, 15]
            }
            for p in ["ParamBodyAngleX", "ParamBodyAngleY", "ParamBodyAngleZ"]
        }
        body_joint_all = {
            (bx, by): np.load(samples / f"body_{bx}_{by}.npz")[name]
            for bx, by in itertools.product([-15, 0, 15], repeat=2)
        }

        def pose_domain():
            # Test the COMPLETE trained head/body domain before pruning hidden surfaces.
            # Restricting this audit to neutral body or just negative head yaw loses newly
            # exposed geometry at positive total yaw and at combined head/body pitch.
            for hs in object_samples:
                for bx, by in itertools.product([-15, 0, 15], repeat=2):
                    a, b = np.radians([bx, by])
                    R = np.array(
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
                    yield body_joint_all[(bx, by)] + (hs[name] - rest) @ R.T
            for values in body_data.values():
                for d in values.values():
                    yield rest + d

        domain = pose_domain()
        for world_domain in domain:
            q = project(world_domain, cam)[triangles]
            low = q.min(axis=1)
            high = q.max(axis=1)
            a = q[:, 1] - q[:, 0]
            b = q[:, 2] - q[:, 0]
            front = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0] < 1e-8
            keep |= (
                (low[:, 0] <= 1.03)
                & (high[:, 0] >= -0.03)
                & (low[:, 1] <= 1.03)
                & (high[:, 1] >= -0.03)
                & front
            )
        original_ids = np.flatnonzero(keep)
        triangles = triangles[keep]
        triuv = triuv[keep]
        mats = src["material_indices"][keep]
        if not len(triangles):
            continue
        active_shapes = []
        bit = np.zeros(len(triangles), np.uint64)
        for pid, s in shape_defs.items():
            if s["object"] != name:
                continue
            if pid in secondary:
                continue
            delta = object_samples[neutral][name + "__" + pid]
            affected = (
                np.max(np.linalg.norm(delta[triangles], axis=2), axis=1)
                > cfg["influence_epsilon"]
            )
            bit |= affected.astype(np.uint64) * (1 << len(active_shapes))
            active_shapes.append(pid)
        head_effect = (
            np.max(
                np.linalg.norm(object_samples[0][name] - rest, axis=1)[triangles],
                axis=1,
            )
            > cfg["influence_epsilon"]
        )
        body_effect = (
            np.max(
                np.linalg.norm(body_data["ParamBodyAngleX"][15], axis=1)[triangles],
                axis=1,
            )
            > cfg["influence_epsilon"]
        )
        pitch_effect = (
            np.max(
                np.linalg.norm(body_data["ParamBodyAngleY"][15], axis=1)[triangles],
                axis=1,
            )
            > cfg["influence_epsilon"]
            if coupled_pitch
            else np.zeros(len(triangles), bool)
        )
        body_joint = body_joint_all if coupled_pitch else {}

        # Inverse-global-roll residual: lower torso weights need not follow Spine2 rigidly.
        def undo_roll(q, degrees):
            angle = np.radians(degrees)
            R = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            return ((q * canvas - pivot) @ R + pivot) / canvas

        roll_residual = {
            v: undo_roll(project(rest + body_data["ParamBodyAngleZ"][v], cam), v)
            - screen
            for v in [-15, 15]
        }
        roll_effect = (
            np.max(np.linalg.norm(roll_residual[15], axis=1)[triangles], axis=1) > 1e-6
        )
        signatures = (
            bit
            | head_effect.astype(np.uint64) * (1 << len(active_shapes))
            | mats.astype(np.uint64) * (1 << (len(active_shapes) + 1))
            | body_effect.astype(np.uint64) * (1 << (len(active_shapes) + 3))
            | roll_effect.astype(np.uint64) * (1 << (len(active_shapes) + 4))
            | pitch_effect.astype(np.uint64) * (1 << (len(active_shapes) + 5))
        )
        fine = np.isin(original_ids, selected_refinement.get("triangles", []))
        signatures |= fine.astype(np.uint64) * (1 << 60)
        partition_in = out / (name + "_partition_in.json")
        partition_out = out / (name + "_partition.json")
        jsonwrite(
            partition_in,
            {
                "triangles": triangles.tolist(),
                "groups": signatures.tolist(),
                "limit": patch_limit or cfg["patch_triangles"],
                "limits": np.where(
                    fine,
                    selected_refinement.get("limit", 8),
                    patch_limit or cfg["patch_triangles"],
                ).tolist(),
            },
        )
        subprocess.run(
            [str(partitioner), str(partition_in), str(partition_out)], check=True
        )
        patches = json.loads(partition_out.read_text())["patches"]
        for pi, patch in enumerate(patches):
            patch = np.array(patch)
            # Static intra-patch far-to-near triangle ordering; dynamic inter-patch order is keyed.
            patch = patch[
                np.argsort(-rest[triangles[patch], 1].mean(axis=1), kind="stable")
            ]
            t = triangles[patch]
            u = triuv[patch]
            vertex_ids = []
            uv = []
            indices = []
            lookup = {}
            # Preserve loop UV seams and original triangle winding.
            for v, tex in zip(t.ravel(), u.reshape(-1, 2)):
                key = (int(v), *np.round(tex, 7))
                index = lookup.get(key)
                if index is None:
                    index = len(vertex_ids)
                    lookup[key] = index
                    vertex_ids.append(int(v))
                    uv.append([float(tex[0]), float(1 - tex[1])])
                indices.append(index)
            ids = np.array(vertex_ids)
            base = screen[ids] * canvas
            is_head = bool(head_effect[patch[0]])
            deps = [
                pid
                for b, pid in enumerate(active_shapes)
                if int(bit[patch[0]]) & (1 << b)
            ]
            axes = list(head_axes) if is_head else []
            if body_effect[patch[0]]:
                axes += [{"parameter": "ParamBodyAngleX", "keys": [-15, 0, 15]}]
            if pitch_effect[patch[0]]:
                axes += [{"parameter": "ParamBodyAngleY", "keys": [-15, 0, 15]}]
            if roll_effect[patch[0]]:
                axes += [{"parameter": "ParamBodyAngleZ", "keys": [-15, 0, 15]}]
            axes += [{"parameter": p, "keys": [0, 1]} for p in deps]
            if name == gaze_name:
                axes += [
                    {"parameter": p, "keys": [-1, 0, 1]}
                    for p in ["ParamEyeBallX", "ParamEyeBallY"]
                ]
            count = int(np.prod([len(a["keys"]) for a in axes])) if axes else 1
            if count > cfg["max_grid_cells"]:
                raise RuntimeError(
                    f"Grid budget exceeded {name} patch {pi}: {count}, axes={axes}; split semantic support or raise explicit budget"
                )
            grid = []
            for reverse_coord in itertools.product(
                *(range(len(a["keys"])) for a in axes[::-1])
            ):
                coord = list(reverse_coord[::-1])
                values = {a["parameter"]: a["keys"][c] for a, c in zip(axes, coord)}
                hi = (
                    head_index[tuple(values.get(a["parameter"], 0) for a in head_axes)]
                    if is_head
                    else neutral
                )
                hs = object_samples[hi]
                if name == gaze_name:
                    world = hs[
                        gaze_name
                        + "__gaze_"
                        + str(values["ParamEyeBallX"])
                        + "_"
                        + str(values["ParamEyeBallY"])
                    ][ids].copy()
                else:
                    world = hs[name][ids].copy()
                for pid in deps:
                    v = values[pid]
                    s = shape_defs[pid]
                    coefficient = 1 - v if s.get("invert") else v
                    world += hs[name + "__" + pid][ids] * coefficient
                bx = values.get("ParamBodyAngleX", 0)
                by = values.get("ParamBodyAngleY", 0)
                if coupled_pitch and (bx or by):
                    a = np.radians(bx)
                    b = np.radians(by)
                    R = np.array(
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
                    world = body_joint[(bx, by)][ids] + (world - rest[ids]) @ R.T
                elif bx:
                    a = np.radians(bx)
                    R = np.array(
                        [
                            [np.cos(a), -np.sin(a), 0],
                            [np.sin(a), np.cos(a), 0],
                            [0, 0, 1],
                        ],
                        np.float32,
                    )
                    world = (
                        rest[ids]
                        + body_data["ParamBodyAngleX"][bx][ids]
                        + (world - rest[ids]) @ R.T
                    )
                q = project(world, cam)
                bz = values.get("ParamBodyAngleZ", 0)
                if bz:
                    q += roll_residual[bz][ids]
                if is_head:
                    z = values.get("ParamAngleZ", 0)
                    a = np.radians(z)
                    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
                    q = ((q * canvas - head_pivots[(bx, by)]) @ R + head_pivot) / canvas
                delta = q
                order = float(
                    500 - world[:, 1].mean() * 800
                )  # front (negative world Y) draws later
                grid.append(
                    {
                        "coordinate": coord,
                        "offset": emit(delta),
                        "order": round(order, 4),
                        "depth_order": round(order, 4),
                    }
                )
            blends = []
            reference = screen[ids]
            reference_order = float(500 - rest[ids, 1].mean() * 800)
            for pid in [] if coupled_pitch else ["ParamBodyAngleY"]:
                offsets = []
                orders = []
                nonzero = False
                for v in [-15, 0, 15]:
                    if v == 0:
                        offsets.append(None)
                        orders.append(reference_order)
                        continue
                    d = body_data[pid][v][ids]
                    local = project(rest[ids] + d, cam) - screen[ids]
                    nonzero |= bool(np.max(np.abs(local)) > 1e-7)
                    offsets.append(emit(reference + local))
                    orders.append(float(reference_order - d[:, 1].mean() * 800))
                if nonzero:
                    blends.append(
                        {
                            "parameter": pid,
                            "keys": [-15, 0, 15],
                            "neutral": 1,
                            "offsets": offsets,
                            "orders": orders,
                        }
                    )
            for pid in secondary:
                if shape_defs[pid]["object"] != name:
                    continue
                d = object_samples[neutral][name + "__" + pid][ids]
                local = project(rest[ids] + d, cam) - screen[ids]
                if np.max(np.abs(local)) > 1e-7:
                    blends.append(
                        {
                            "parameter": pid,
                            "keys": [0, 1],
                            "neutral": 0,
                            "offsets": [None, emit(reference + local)],
                            "orders": [
                                reference_order,
                                float(reference_order - d[:, 1].mean() * 800),
                            ],
                        }
                    )
            texture = obj["textures"][int(mats[patch[0]])]
            if texture not in ir["textures"]:
                ir["textures"].append(texture)
                shutil.copy2(samples / texture, out / texture)
            # Cubism's drawable culling convention is opposite to Blender front-facing loops.
            indices = np.array(indices).reshape(-1, 3)[:, [0, 2, 1]].ravel().tolist()
            ir["meshes"].append(
                {
                    "id": name + "_" + str(pi),
                    "positions": base.flatten().round(5).tolist(),
                    "uvs": np.array(uv).flatten().tolist(),
                    "indices": indices,
                    "texture": ir["textures"].index(texture),
                    "deformer": "head" if is_head else "body",
                    "axes": axes,
                    "cells": grid,
                    "blend_shapes": blends,
                    "source": {
                        "object": name,
                        "vertices": vertex_ids,
                        "triangles": original_ids[patch].tolist(),
                        "barycentric": "identity at original triangle corners",
                        "semantic_parameters": deps,
                    },
                    "masked_by": [],
                }
            )
            total_tris += len(patch)
            total_vertices += len(ids)
            authored += len(grid)
        print(
            "IR",
            name,
            "patches",
            len(patches),
            "source triangles retained",
            len(triangles),
            flush=True,
        )
    binary.close()
    from .visibility import finalize_visibility

    finalize_visibility(ir, samples, m["semantics"], out / "keyforms.f32")
    ir["parts"] = [
        {
            "id": name,
            "source_object": name,
            "mesh_ids": [
                mesh["id"] for mesh in ir["meshes"] if mesh["source"]["object"] == name
            ],
        }
        for name in sorted({mesh["source"]["object"] for mesh in ir["meshes"]})
    ]
    ir["diagnostics"]["couple_body_pitch"] = coupled_pitch
    if coupled_pitch:
        ir["diagnostics"]["warnings"] = [
            w for w in ir["diagnostics"]["warnings"] if not w.startswith("Body Y")
        ] + ["Smile-eye blend residuals remain neutral-head approximations"]
    ir["diagnostics"].update(
        artmesh_count=len(ir["meshes"]),
        vertices=sum(len(mesh["positions"]) // 2 for mesh in ir["meshes"]),
        triangles=sum(len(mesh["indices"]) // 3 for mesh in ir["meshes"]),
        parameters=len(parameters),
        authored_keyforms=sum(len(mesh["cells"]) for mesh in ir["meshes"]),
        keyform_bytes=(out / "keyforms.f32").stat().st_size,
        visible_artmesh_count=sum(
            not mesh.get("mask_only", False) for mesh in ir["meshes"]
        ),
        mask_artmesh_count=sum(bool(mesh.get("mask_only")) for mesh in ir["meshes"]),
        seconds=time.perf_counter() - start,
        texture_sizes=[[2048, 2048] for _ in ir["textures"]],
        deformers=len(ir["body_warps"]) + 3 if "body_warps" in ir else 4,
        deformer_depth=3,
    )
    jsonwrite(out / "POPY.rig.json", ir)
    jsonwrite(out / "statistics.json", ir["diagnostics"])
    print(json.dumps(ir["diagnostics"], indent=2), flush=True)
    return out / "POPY.rig.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/rig")
    args = parser.parse_args()
    compile_ir(out=args.out)
