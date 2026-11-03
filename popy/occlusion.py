"""SPDX-License-Identifier: MIT. Fit draw-order constraints from projected 3D overlap/depth."""

import itertools, json, subprocess, time, argparse
from pathlib import Path
import numpy as np
from .compile_ir import ROOT, project, jsonwrite


def optimize(folder=ROOT / "artifacts/rig"):
    start = time.perf_counter()
    folder = Path(folder)
    samples = ROOT / "artifacts/samples"
    path = folder / "POPY.rig.json"
    ir = json.loads(path.read_text())
    manifest = json.loads((samples / "manifest.json").read_text())
    cam = ir["camera"]
    meshes = [m for m in ir["meshes"] if not m.get("mask_only")]
    neutral_index = next(
        i
        for i, p in enumerate(manifest["head_poses"])
        if all(v == 0 for v in p["parameters"].values())
    )
    neutral = np.load(samples / manifest["head_poses"][neutral_index]["file"])
    names = sorted({m["source"]["object"] for m in meshes})
    rest = {n: neutral[n] for n in names}
    bdata = {
        v: {
            n: np.load(samples / ("ParamBodyAngleX_" + str(v) + ".npz"))[n]
            for n in names
        }
        for v in [-15, 15]
    }
    coupled = ir["diagnostics"].get("couple_body_pitch", False)
    pitch_keys = [-15, 0, 15] if coupled else [0]
    joint = (
        {
            (bx, by): {n: np.load(samples / f"body_{bx}_{by}.npz")[n] for n in names}
            for bx, by in itertools.product([-15, 0, 15], pitch_keys)
        }
        if coupled
        else {}
    )
    triangles = []
    vertex_ranges = []
    offset = 0
    for m in meshes:
        vertex_ranges.append([offset, offset + len(m["source"]["vertices"])])
        triangles.append((np.array(m["indices"]).reshape(-1, 3) + offset).tolist())
        offset += len(m["source"]["vertices"])
    poses = []
    datafile = folder / "occlusion_poses.f32"
    with datafile.open("wb") as f:
        for hp in manifest["head_poses"]:
            hs = np.load(samples / hp["file"])
            base = {n: hs[n] for n in names}
            for bx, by in itertools.product([-15, 0, 15], pitch_keys):
                a = np.radians(bx)
                b = np.radians(by)
                R = np.array(
                    [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
                ) @ np.array(
                    [[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]]
                )
                world = {
                    n: joint[(bx, by)][n] + (base[n] - rest[n]) @ R.T
                    if coupled
                    else base[n]
                    if bx == 0
                    else rest[n] + bdata[bx][n] + (base[n] - rest[n]) @ R.T
                    for n in names
                }
                projected = {
                    n: np.c_[project(world[n], cam), world[n][:, 1]].astype("<f4")
                    for n in names
                }
                for m in meshes:
                    f.write(
                        projected[m["source"]["object"]][
                            m["source"]["vertices"]
                        ].tobytes()
                    )
                poses.append(
                    dict(hp["parameters"], ParamBodyAngleX=bx, ParamBodyAngleY=by)
                )
    pose_axes = [
        "ParamAngleX",
        "ParamAngleY",
        "ParamAngleZ",
        "ParamBodyAngleX",
        "ParamBodyAngleY",
    ]
    pose_keys = [sorted({p[k] for p in poses}) for k in pose_axes]
    pose_lookup = {tuple(p[k] for k in pose_axes): i for i, p in enumerate(poses)}
    neighbors = []
    for p in poses:
        value = [p[k] for k in pose_axes]
        adjacent = []
        for axis, keys in enumerate(pose_keys):
            i = keys.index(value[axis])
            for j in [i - 1, i + 1]:
                if 0 <= j < len(keys):
                    key = value.copy()
                    key[axis] = keys[j]
                    adjacent.append(pose_lookup[tuple(key)])
        neighbors.append(adjacent)
    inputfile = folder / "occlusion_input.json"
    index_by_id = {m["id"]: i for i, m in enumerate(meshes)}
    visibility = ir.get("visibility", {})
    pupils = [
        m["id"]
        for m in meshes
        if m["source"]["object"] == manifest["semantics"]["objects"]["gaze"]
    ]
    whites = [m for m in visibility.get("eye_targets", []) if m not in pupils]
    skin = visibility.get("skin_surface_meshes", [])
    priority_edges = [
        [index_by_id[a], index_by_id[b]]
        for before, after in [(skin, whites), (skin, pupils), (whites, pupils)]
        for a in before
        for b in after
        if a in index_by_id and b in index_by_id and a != b
    ]
    jsonwrite(
        inputfile,
        {
            "meshes": triangles,
            "vertices": offset,
            "poses": len(poses),
            "width": 384,
            "height": 512,
            "data": str(datafile),
            "vertex_ranges": vertex_ranges,
            "neighbors": neighbors,
            "priority_edges": priority_edges,
        },
    )
    subprocess.run(
        [
            str(ROOT / "rust/target/release/occlusion_stable.exe"),
            str(inputfile),
            str(folder / "occlusion_orders.f32"),
            str(folder / "occlusion_diagnostics.json"),
        ],
        check=True,
    )
    orders = np.fromfile(folder / "occlusion_orders.f32", "<f4").reshape(
        len(poses), len(meshes)
    )
    baselines = np.fromfile(
        folder / "occlusion_orders.f32.baseline.f32", "<f4"
    ).reshape(len(poses), len(meshes))
    lookup = {
        tuple(
            p[k]
            for k in [
                "ParamAngleX",
                "ParamAngleY",
                "ParamAngleZ",
                "ParamBodyAngleX",
                "ParamBodyAngleY",
            ]
        ): i
        for i, p in enumerate(poses)
    }
    for mi, m in enumerate(meshes):
        old_ref = None
        new_ref = None
        for cell in m["cells"]:
            p = {
                a["parameter"]: a["keys"][c]
                for a, c in zip(m["axes"], cell["coordinate"])
            }
            key = tuple(
                p.get(k, 0)
                for k in [
                    "ParamAngleX",
                    "ParamAngleY",
                    "ParamAngleZ",
                    "ParamBodyAngleX",
                    "ParamBodyAngleY",
                ]
            )
            pose = lookup[key]
            measured = cell.setdefault("depth_order", cell["order"])
            residual = measured - baselines[pose, mi]
            # Eye coverage comes from native geometric masks. Gaze depth must not
            # cross the deliberately fixed sclera/pupil layer gap: that otherwise
            # makes a whole iris disappear when looking sideways.
            if m["id"] in visibility.get("eye_targets", []):
                residual = 0
            new = float(orders[pose, mi] + residual)
            # Shapes have modest local depth changes; retain those residuals relative to the same pose's neutral.
            if all(
                p.get(a, 0) == 0
                for a in [
                    "ParamAngleX",
                    "ParamAngleY",
                    "ParamAngleZ",
                    "ParamBodyAngleX",
                    "ParamBodyAngleZ",
                ]
            ) and all(
                p.get(s, irparam["default"]) == irparam["default"]
                for s, irparam in [(q["id"], q) for q in ir["parameters"]]
                if s in p
            ):
                old_ref = measured
                new_ref = new
            cell["order"] = new
        if old_ref is not None:
            for b in m["blend_shapes"]:
                measured_orders = b.setdefault(
                    "depth_orders", b.get("orders", []).copy()
                )
                b["orders"] = [o + (new_ref - old_ref) for o in measured_orders]
    diagnostics = json.loads((folder / "occlusion_diagnostics.json").read_text())
    diagnostics["seconds"] = time.perf_counter() - start
    diagnostics["notes"] = (
        "384x512 coverage constraints, shared canonical reference, pose-local depth priority, closest-order projection and parameter-lattice smoothing. Non-eye expression depth residuals retained; mask-protected eye layering remains invariant. Remaining non-orderable patches are recorded for adaptive refinement."
    )
    ir["diagnostics"]["occlusion_optimization"] = diagnostics
    from .depth_masks import stabilize_linework_order
    stabilize_linework_order(ir)
    jsonwrite(path, ir)
    jsonwrite(folder / "statistics.json", ir["diagnostics"])
    proposals = []
    for mesh, score in zip(meshes, diagnostics["conflict_meshes"]):
        if score >= 300 and len(mesh["source"]["triangles"]) > 12:
            proposals.append(
                {
                    "mesh": mesh["id"],
                    "score": score,
                    "object": mesh["source"]["object"],
                    "triangles": mesh["source"]["triangles"],
                }
            )
    jsonwrite(
        folder / "refinement_proposal.json",
        sorted(proposals, key=lambda p: -p["score"]),
    )
    print(
        json.dumps(
            {k: v for k, v in diagnostics.items() if k != "conflict_meshes"}, indent=2
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rig-dir", type=Path, default=ROOT / "artifacts/rig")
    args = parser.parse_args()
    optimize(args.rig_dir)
