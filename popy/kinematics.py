"""Source skeleton linear skinning used for articulated Cubism keyforms (MIT)."""

import json
from pathlib import Path

import numpy as np


def garment_order_bias(mesh, source, config):
    """Apply configured shell precedence while leaving horizontal inner caps behind."""
    name = mesh["source"]["object"]
    bias = config.get("runtime_order_bias", {}).get(name, 0)
    rule = config.get("runtime_inner_caps", {}).get(name)
    if rule:
        triangles = source["positions"][
            source["triangles"][mesh["source"]["triangles"]]
        ]
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        lengths = np.linalg.norm(normals, axis=1)
        vertical = np.divide(
            np.abs(normals[:, 2]),
            lengths,
            out=np.zeros_like(lengths),
            where=lengths > 1e-12,
        )
        if (
            len(triangles)
            and triangles[:, :, 2].min() >= rule["rest_min_z"]
            and triangles[:, :, 2].max() <= rule["rest_max_z"]
            and vertical.min() >= rule["min_abs_normal_z"]
        ):
            return rule["bias"]
    return bias


def rotation(axis, degrees):
    if axis not in {"X", "Y", "Z"}:
        raise ValueError("Unknown joint axis: " + axis)
    a = np.deg2rad(degrees)
    c, s = np.cos(a), np.sin(a)
    if axis == "X":
        r = [[1, 0, 0], [0, c, -s], [0, s, c]]
    elif axis == "Y":
        r = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    else:
        r = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    m = np.eye(4)
    m[:3, :3] = r
    return m


def parameter_basis(config, p):
    result = {}
    for c in config["controls"]:
        value = p.get(c["id"], c["default"])
        for drive in c["drives"]:
            bone = drive["bone"]
            factor = (
                drive.get("negative_factor", drive.get("factor", 1))
                if value < 0
                else drive.get("factor", 1)
            )
            r = rotation(drive["axis"], value * factor)
            result[bone] = result.get(bone, np.eye(4)) @ r
    return result


class Skeleton:
    def __init__(self, directory, config):
        self.directory = Path(directory)
        self.config = config
        self.bones = json.loads((self.directory / "skeleton.json").read_text())
        if not self.bones:
            raise ValueError("Source skeleton is empty")
        self.names = [b["name"] for b in self.bones]
        self.index = {n: i for i, n in enumerate(self.names)}
        self.bind = np.array([b["matrix_local"] for b in self.bones])
        self.inverse = np.linalg.inv(self.bind)
        self.parents = [self.index.get(b["parent"], -1) for b in self.bones]
        self.local = np.array(
            [
                self.inverse[parent] @ self.bind[i] if parent >= 0 else self.bind[i]
                for i, parent in enumerate(self.parents)
            ]
        )
        self.control_bones = {
            c["id"]: {d["bone"] for d in c["drives"]} for c in config["controls"]
        }
        for control, names in self.control_bones.items():
            if names - set(self.names):
                raise ValueError(
                    f"Missing source bones for {control}: {names - set(self.names)}"
                )
        if any(parent >= i for i, parent in enumerate(self.parents)):
            raise ValueError("Skeleton must be parent-before-child")
        self.dependencies = []
        for i, bone in enumerate(self.bones):
            deps = (
                set(self.dependencies[self.parents[i]])
                if self.parents[i] >= 0
                else set()
            )
            deps.update(k for k, v in self.control_bones.items() if bone["name"] in v)
            self.dependencies.append(deps)
        self.objects = {}

    def skin_matrices(self, parameters):
        basis = parameter_basis(self.config, parameters)
        # Existing head/body parameters rotate around audited world-oriented axes.
        for bone, prefix in [("Spine2", "ParamBodyAngle"), ("Head", "ParamAngle")]:
            if bone not in self.index:
                continue
            i = self.index[bone]
            q = self.bind[i].copy()
            q[:3, 3] = 0
            r = (
                rotation("Y", parameters.get(prefix + "Z", 0))
                @ rotation("Z", parameters.get(prefix + "X", 0))
                @ rotation("X", parameters.get(prefix + "Y", 0))
            )
            basis[bone] = np.linalg.inv(q) @ r @ q
            if bone == "Spine2":
                breath = parameters.get("ParamBreath", 0)
                basis[bone] = basis[bone] @ np.diag(
                    [1 + 0.012 * breath, 1 + 0.006 * breath, 1, 1]
                )
        pose = np.empty_like(self.bind)
        for i, name in enumerate(self.names):
            pose[i] = self.local[i] @ basis.get(name, np.eye(4))
            parent = self.parents[i]
            if parent >= 0:
                pose[i] = pose[parent] @ pose[i]
        return pose @ self.inverse

    def object(self, name):
        if name not in self.objects:
            with np.load(self.directory / (name + "_source.npz")) as archive:
                source = {k: archive[k] for k in archive.files}
            weights = json.loads(
                (self.directory / (name + "_weights.json")).read_text()
            )
            w = np.zeros((len(weights), len(self.names)))
            for v, entries in enumerate(weights):
                for bone, weight in entries:
                    if bone in self.index:
                        w[v, self.index[bone]] = weight
            total = w.sum(1)
            w = np.divide(
                w, total[:, None], out=np.zeros_like(w), where=total[:, None] > 0
            )
            deps = np.zeros((len(weights), len(self.config["controls"])), bool)
            for j, c in enumerate(self.config["controls"]):
                affected = [i for i, d in enumerate(self.dependencies) if c["id"] in d]
                deps[:, j] = w[:, affected].sum(1) > 1e-7
            self.objects[name] = (source, w, deps)
        return self.objects[name]

    def evaluate(self, name, parameters, vertices=None, skin=None):
        source, weights, _ = self.object(name)
        vertices = np.arange(len(weights)) if vertices is None else np.asarray(vertices)
        points = np.c_[source["positions"][vertices], np.ones(len(vertices))]
        skin = self.skin_matrices(parameters) if skin is None else skin
        active = np.flatnonzero(np.any(weights[vertices] > 0, axis=0))
        result = np.zeros((len(vertices), 3))
        for i in active:
            result += (points @ skin[i].T)[:, :3] * weights[vertices, i, None]
        zero = weights[vertices].sum(1) == 0
        result[zero] = points[zero, :3]
        return result
