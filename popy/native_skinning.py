"""Factor 3D skeleton matrices into native single-parameter Cubism Glue arithmetic.

All vector operations use affine Glue combinations around a shared zero point.
Hidden eight-vertex ArtMeshes carry positive/negative projected matrix columns. No
runtime application callback, generated pose frames, or auxiliary user parameter
is required. This neutral implementation does not link the binary format writer.
"""

import argparse
import itertools
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from .compile_ir import ROOT, jsonwrite
from .field_compression import reduce_grid
from .form_storage import compact_forms
from .kinematics import Skeleton, garment_order_bias, rotation


class NativeGraph:
    def __init__(self, ir, binary, config):
        self.ir, self.config = ir, config
        self.binary = Path(binary)
        self.stream = self.binary.open("ab")
        self.cursor = self.binary.stat().st_size // 4
        self.meshes, self.glues = [], []
        self.parameters = {p["id"]: p for p in ir["parameters"]}
        self.size = np.array(ir["canvas"])
        self.gains = {}
        self.root = "D_NativeSkeletonProjection"
        ir["body_warps"].append(
            {
                "id": self.root,
                "root": True,
                "kind": "rotation",
                "strength": 1.0,
                "axes": [],
                "cells": [
                    {
                        "coordinate": [],
                        "points": [
                            int(v)
                            for v in [
                                0,
                                0,
                                self.size[0],
                                0,
                                0,
                                self.size[1],
                                *self.size,
                            ]
                        ],
                    }
                ],
            }
        )
        self.zero = self.node(np.zeros((4, 2)), "Zero")
        self.used_parameters = set()
        self.key_cache = {}

    def emit(self, values):
        a = np.asarray(values, dtype="<f4").ravel()
        off = self.cursor
        self.stream.write(a.tobytes())
        self.cursor += len(a)
        return off

    def node(self, points, label):
        points = np.asarray(points).reshape(-1, 2)
        name = "Native_" + str(len(self.meshes)) + "_" + label
        off = self.emit(points)
        mesh = {
            "id": name,
            "positions": points.ravel().tolist(),
            "uvs": [0.0] * (2 * len(points)),
            "indices": [0, 1, 2] if len(points) >= 3 else [],
            "texture": 0,
            "projection_warp": self.root,
            "deformer": "body",
            "axes": [],
            "cells": [{"coordinate": [], "offset": off, "order": 0.0}],
            "blend_shapes": [],
            "opacity": 0.0,
            "mask_only": True,
            "culling": False,
            "native_skinning": True,
            "masked_by": [],
            "source": {
                "object": "NativeSkinning",
                "vertices": list(range(len(points))),
                "triangles": [],
                "barycentric": "homogeneous projected matrix/vector carrier, not original surface vertices",
            },
        }
        self.meshes.append(mesh)
        return name

    def glue(self, source, target, pairs, parameter=None, values=None, keys=None):
        if not pairs:
            return
        glue = {
            "id": "NativeG_" + str(len(self.glues)),
            "mesh_a": source,
            "mesh_b": target,
            "pairs": pairs,
        }
        if parameter is not None:
            self.used_parameters.add(parameter)
        if values is not None:
            if parameter is None or np.ptp(values) < 1e-14:
                glue["intensity"] = float(values[0])
            else:
                glue["axes"] = [{"parameter": parameter, "keys": keys.tolist()}]
                glue["intensities"] = np.asarray(values).tolist()
                # Exact scalar fields; skip the unrelated spatial-mask estimator.
                glue["compression_error_bound_px"] = 0.0
        self.glues.append(glue)

    def reset(self, target, count):
        self.glue(self.zero, target, [[0, j, 0.0, 1.0] for j in range(count)])

    def right(self, source, matrices, label, parameter=None, keys=None):
        matrices = np.asarray(matrices)
        if matrices.ndim == 2:
            matrices = matrices[None]
        if np.all(matrices == np.eye(4)):
            return source
        # Signed column carriers and a fixed gain make each operation convex:
        # all Glue weights/intensities remain in [0,1], including constant maps.
        contraction = 1.1 * float(np.max(np.sum(np.abs(matrices), axis=1)))
        target = self.node(np.zeros((8, 2)), label)
        self.gains[target] = self.gains[source] / contraction
        self.reset(target, 8)
        for sign in [0, 1]:
            for col in range(4):
                denominator = (
                    1 - np.sum(np.abs(matrices[:, :, col]), axis=1) / contraction
                )
                for i in range(4):
                    for negative in [False, True]:
                        coefficient = (
                            np.maximum(
                                -matrices[:, i, col]
                                if negative
                                else matrices[:, i, col],
                                0,
                            )
                            / contraction
                        )
                        if not np.any(coefficient):
                            continue
                        denominator = denominator + coefficient
                        intensity = np.clip(coefficient / denominator, 0, 1)
                        source_index = i + (4 if (negative != bool(sign)) else 0)
                        self.glue(
                            source,
                            target,
                            [[source_index, col + 4 * sign, 0.0, 1.0]],
                            parameter,
                            intensity,
                            keys,
                        )
        return target

    def keys(self, pid):
        if pid not in self.key_cache:
            p = self.parameters[pid]
            step = 0.005 if p["max"] - p["min"] <= 2 else 0.5
            count = int(np.ceil((p["max"] - p["min"]) / step))
            self.key_cache[pid] = np.unique(
                np.r_[np.linspace(p["min"], p["max"], count + 1), p["default"], 0.0]
            )
        return self.key_cache[pid]

    def rotate(self, source, pid, axis, factor=1.0, negative_factor=None):
        keys = self.keys(pid)
        negative_factor = factor if negative_factor is None else negative_factor
        matrices = [
            rotation(axis, float(v) * (negative_factor if v < 0 else factor))
            for v in keys
        ]
        return self.right(source, matrices, pid + "_" + axis, pid, keys)

    def clock(self):
        # Ayagami skips the Glue-only update if no ArtMesh/deformer dependency is
        # dirty. A constant keyed carrier explicitly registers each real control.
        for pid in sorted(self.used_parameters):
            self.node(np.zeros((4, 2)), "Dependency_" + pid)
            mesh = self.meshes[-1]
            mesh["axes"] = [
                {
                    "parameter": pid,
                    "keys": [self.parameters[pid]["min"], self.parameters[pid]["max"]],
                }
            ]
            base = mesh["cells"][0]
            mesh["cells"] = [{**base, "coordinate": [j]} for j in range(2)]


def lower(ir, binary, config):
    if ir["diagnostics"]["fullbody"].get("native_skinning"):
        raise ValueError("Native skinning already lowered")
    skeleton = Skeleton(ROOT / "artifacts/samples", config)
    graph = NativeGraph(ir, binary, config)
    source_names = {
        o["name"]
        for o in json.loads((ROOT / "artifacts/samples/manifest.json").read_text())[
            "objects"
        ]
    }
    targets = []
    for mesh in ir["meshes"]:
        source = mesh["source"]
        name = source["object"]
        identity = (
            name in source_names
            and mesh["id"].startswith(name + "_")
            and mesh.get("deformer") != "head"
            and not any(
                k in source
                for k in ["barycentric_vertices", "barycentric_sources", "master_mesh"]
            )
        )
        if mesh.get("articulated") or identity:
            mesh["native_original_target"] = bool(mesh.get("articulated"))
            mesh["articulated"] = True
            mesh["culling"] = config.get("runtime_culling", {}).get(
                name, mesh.get("culling", True)
            )
            targets.append(mesh)
    by_object = defaultdict(list)
    for mesh in targets:
        by_object[mesh["source"]["object"]].append(mesh)
    vertices = {
        name: np.unique([v for m in meshes for v in m["source"]["vertices"]])
        for name, meshes in by_object.items()
    }
    required = set()
    for name, ids in vertices.items():
        _, weights, _ = skeleton.object(name)
        required.update(np.flatnonzero(weights[ids].any(0)).tolist())
    for i in list(required):
        while i >= 0:
            required.add(i)
            i = skeleton.parents[i]
    camera = np.array(ir["camera"]["projection"]) @ np.linalg.inv(
        ir["camera"]["matrix_world"]
    )
    if not np.allclose(camera[3], [0, 0, 0, 1]):
        raise ValueError(
            "Native affine skeleton currently requires orthographic projection"
        )
    projection = np.array(
        [
            (camera[0] + camera[3]) * ir["canvas"][0] / 2,
            (-camera[1] + camera[3]) * ir["canvas"][1] / 2,
        ]
    )
    initial_gain = float(2**20)
    root = graph.node(
        np.r_[projection.T * initial_gain, -projection.T * initial_gain], "Camera"
    )
    graph.gains[root] = initial_gain
    controls = defaultdict(list)
    for control in config["controls"]:
        for drive in control["drives"]:
            controls[drive["bone"]].append((control, drive))
    matrices = {}
    for i in sorted(required):
        name = skeleton.names[i]
        current = matrices.get(skeleton.parents[i], root)
        current = graph.right(current, skeleton.local[i], name + "_Rest")
        if name in ["Spine2", "Head"]:
            prefix = "ParamBodyAngle" if name == "Spine2" else "ParamAngle"
            q = skeleton.bind[i].copy()
            q[:3, 3] = 0
            current = graph.right(current, np.linalg.inv(q), name + "_BasisIn")
            for suffix, axis in [("Z", "Y"), ("X", "Z"), ("Y", "X")]:
                current = graph.rotate(current, prefix + suffix, axis)
            current = graph.right(current, q, name + "_BasisOut")
            if name == "Spine2":
                keys = graph.keys("ParamBreath")
                values = [np.diag([1 + 0.012 * v, 1 + 0.006 * v, 1, 1]) for v in keys]
                current = graph.right(current, values, "Breath", "ParamBreath", keys)
        else:
            for control, drive in controls.get(name, []):
                current = graph.rotate(
                    current,
                    control["id"],
                    drive["axis"],
                    drive.get("factor", 1),
                    drive.get("negative_factor"),
                )
        matrices[i] = current
    order_before = order_after = 0
    zero_offsets = {}
    ties = []
    for name, meshes in by_object.items():
        ids = vertices[name]
        source, weights, _ = skeleton.object(name)
        driver = graph.node(np.zeros((len(ids), 2)), "Vertices_" + name)
        graph.reset(driver, len(ids))
        homogeneous = np.c_[source["positions"][ids], np.ones(len(ids))]
        active = np.flatnonzero(weights[ids].any(0))
        coefficients = {
            b: (homogeneous @ skeleton.inverse[b].T) * weights[ids, b, None]
            for b in active
        }
        terms = {b: c / graph.gains[matrices[b]] for b, c in coefficients.items()}
        denominator = 1 - sum(np.abs(c).sum(1) for c in terms.values())
        if float(denominator.min()) < 0.05:
            raise ValueError(
                "Increase the camera carrier gain to keep the skinning sum convex"
            )
        for bone in active:
            for col in range(4):
                for negative in [False, True]:
                    coefficient = np.maximum(
                        -terms[bone][:, col] if negative else terms[bone][:, col], 0
                    )
                    denominator = denominator + coefficient
                    intensity = np.clip(coefficient / denominator, 0, 1)
                    # Canonical pairs always have weights A=0, B=1. Distinct
                    # per-vertex constants are encoded as Glue compatibility.
                    batches = defaultdict(list)
                    for j, value in enumerate(intensity):
                        if value > 1e-15:
                            batches[float(np.float32(value))].append(
                                [col + (4 if negative else 0), j, 0.0, 1.0]
                            )
                    for value, pairs in batches.items():
                        graph.glue(matrices[bone], driver, pairs, values=[value])
        for mesh in meshes:
            size = len(mesh["positions"])
            if size not in zero_offsets:
                zero_offsets[size] = graph.emit(np.zeros(size))
            shape = tuple(len(a["keys"]) for a in mesh["axes"])
            order = np.empty(shape)
            for c in mesh["cells"]:
                order[tuple(c["coordinate"])] = c["order"]
            bias = garment_order_bias(mesh, skeleton.object(name)[0], config)
            order += bias
            mesh["runtime_order_bias_applied"] = bias
            order_before += order.size
            axes, order, spent, _, _ = reduce_grid(
                mesh["axes"],
                order,
                {p["id"]: p["default"] for p in ir["parameters"]},
                effect_px=1.0,
                budget_px=0.5,
            )
            mesh["axes"] = axes
            mesh["cells"] = [
                {
                    "coordinate": list(c),
                    "offset": zero_offsets[size],
                    "order": float(order[c]),
                }
                for c in itertools.product(*(range(len(a["keys"])) for a in axes))
            ]
            order_after += len(mesh["cells"])
            mesh["projection_warp"] = graph.root
            mesh["blend_shapes"] = []
            mesh["native_skinning"] = True
            mesh["native_vertex_driver"] = driver
            mesh["native_order_error_units"] = spent
            take = np.searchsorted(ids, mesh["source"]["vertices"])
            ties.append(
                {
                    "id": "NativeTie_" + mesh["id"],
                    "mesh_a": driver,
                    "mesh_b": mesh["id"],
                    "pairs": [[int(v), j, 0.0, 1.0] for j, v in enumerate(take)],
                }
            )
    graph.clock()
    graph.stream.close()
    ir["meshes"].extend(graph.meshes)
    ir["glues"] = graph.glues + ties + ir.get("glues", [])
    ir["parts"].append(
        {
            "id": "NativeSkinning",
            "source_object": "NativeSkinning",
            "mesh_ids": [m["id"] for m in graph.meshes],
        }
    )
    diagnostics = {
        "method": "projected homogeneous bone-matrix products and linear skinning using native parameterized Glue",
        "matrix_bones": len(required),
        "helper_meshes": len(graph.meshes),
        "operator_glues": len(graph.glues),
        "surface_tie_glues": len(ties),
        "scalar_keyforms": sum(len(g.get("intensities", [])) for g in graph.glues),
        "order_cells_before": int(order_before),
        "order_cells_after": int(order_after),
        "order_error_budget_units": 0.5,
        "rotation_sample_step_degrees": 0.5,
        "small_parameter_step": 0.005,
        "application_pose_callback_required": False,
        "perspective_supported": False,
        "canonical_glue_weights": True,
        "all_glue_intensities_in_unit_interval": True,
        "matrix_initial_gain": initial_gain,
        "matrix_minimum_gain": min(graph.gains.values()),
        "bone_matrix_nodes": {
            skeleton.names[i]: {"mesh": node, "gain": graph.gains[node]}
            for i, node in matrices.items()
        },
        "runtime_culling_overrides": config.get("runtime_culling", {}),
        "runtime_order_bias": config.get("runtime_order_bias", {}),
        "runtime_inner_caps": config.get("runtime_inner_caps", {}),
    }
    ir["diagnostics"]["fullbody"]["native_skinning"] = diagnostics
    compact_forms(ir, binary)
    print(
        "NATIVE_SKINNING",
        json.dumps({k: v for k, v in diagnostics.items() if k != "bone_matrix_nodes"}),
        flush=True,
    )
    return diagnostics


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input", type=Path, default=ROOT / "artifacts/fullbody/rig/POPY.rig.json"
    )
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/fullbody/native_rig")
    args = p.parse_args()
    ir = json.loads(args.input.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    for name in [ir["keyform_data"]] + ir["textures"]:
        shutil.copy2(args.input.parent / name, args.out / name)
    lower(
        ir,
        args.out / ir["keyform_data"],
        json.loads((ROOT / "config/fullbody.json").read_text()),
    )
    jsonwrite(args.out / "POPY.rig.json", ir)


if __name__ == "__main__":
    main()
