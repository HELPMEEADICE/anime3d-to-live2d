"""Run only in a BACKGROUND Blender process on the independent source snapshot.
Exports evaluated ORIGINAL triangle/UV correspondence, never alpha-derived meshes.
"""

import argparse, itertools, json, math, sys, time
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "artifacts/samples"))
    p.add_argument("--references-only", action="store_true")
    p.add_argument("--case-file")
    p.add_argument("--reference-dir")
    p.add_argument("--case-output")
    p.add_argument("--eye-masks", action="store_true")
    p.add_argument("--surface-labels", help="Original triangle RGB class IDs for an independent Blender depth pass")
    return p.parse_args(
        sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    )


def rotation(x=0.0, y=0.0, z=0.0):
    # x = yaw around world Z; y = pitch around world X; z = roll around front axis Y.
    return (
        Matrix.Rotation(math.radians(z), 4, "Y")
        @ Matrix.Rotation(math.radians(x), 4, "Z")
        @ Matrix.Rotation(math.radians(y), 4, "X")
    )


def main():
    args = arguments()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "config/convert.json").read_text())
    sem = json.loads((ROOT / "config/semantics.json").read_text())
    arm = bpy.data.objects[sem["armature"]]
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    gaze_name = sem["objects"]["gaze"]
    for o in meshes:
        # Outline geometry changes vertex correspondence; explicitly excluded and reported.
        for m in list(o.modifiers):
            if m.type != "ARMATURE":
                o.modifiers.remove(m)
    rest_basis = {p.name: p.matrix_basis.copy() for p in arm.pose.bones}
    shapes = sem["shape_parameters"]
    for pid, s in shapes.items():
        assert s["object"] in bpy.data.objects, s
        assert s["shape"] in bpy.data.objects[s["object"]].data.shape_keys.key_blocks, s

    def apply(p):
        for b in arm.pose.bones:
            b.matrix_basis = rest_basis[b.name]
        for o in meshes:
            if o.data.shape_keys:
                for k in o.data.shape_keys.key_blocks[1:]:
                    k.value = 0
        for semantic, prefix in [("body", "ParamBodyAngle"), ("head", "ParamAngle")]:
            b = arm.pose.bones[sem["bones"][semantic]]
            q = b.bone.matrix_local.to_3x3().to_4x4()
            b.matrix_basis = (
                q.inverted()
                @ rotation(*(p.get(prefix + i, 0) for i in "XYZ"))
                @ q
                @ rest_basis[b.name]
            )
        for semantic in ["eye_left", "eye_right"]:
            b = arm.pose.bones[sem["bones"][semantic]]
            q = b.bone.matrix_local.to_3x3().to_4x4()
            b.matrix_basis = (
                q.inverted()
                @ rotation(
                    p.get("ParamEyeBallX", 0) * 12, p.get("ParamEyeBallY", 0) * 10, 0
                )
                @ q
                @ rest_basis[b.name]
            )
        for pid, s in shapes.items():
            v = p.get(pid, s["default"])
            v = 1 - v if s.get("invert") else v
            bpy.data.objects[s["object"]].data.shape_keys.key_blocks[
                s["shape"]
            ].value = v
        # Procedural breath is declared separately from source controls.
        breath = p.get("ParamBreath", 0)
        if breath:
            b = arm.pose.bones[sem["bones"]["body"]]
            b.scale.x *= 1 + 0.012 * breath
            b.scale.y *= 1 + 0.006 * breath
        bpy.context.view_layer.update()

    def vertices(o):
        ev = o.evaluated_get(bpy.context.evaluated_depsgraph_get())
        m = ev.to_mesh()
        v = np.empty(len(m.vertices) * 3, dtype=np.float32)
        m.vertices.foreach_get("co", v)
        v = v.reshape(-1, 3)
        mat = np.array(ev.matrix_world, dtype=np.float32)
        v = v @ mat[:3, :3].T + mat[:3, 3]
        ev.to_mesh_clear()
        return v

    apply({})
    manifest = {
        "version": 1,
        "config": cfg,
        "semantics": sem,
        "objects": [],
        "head_poses": [],
        "warnings": [
            "MToon outline geometry excluded from correspondence and albedo references",
            "Body Y and smile-eye blend residuals are neutral-head approximations; Body X is jointly fitted",
            "ParamBreath is procedural; not a source blendshape",
            "No hair dynamics or physics3 binding is implemented",
        ],
    }
    for o in meshes:
        m = o.data
        m.calc_loop_triangles()
        assert m.uv_layers.active, f"UV missing: {o.name}"
        tri = np.array([list(t.vertices) for t in m.loop_triangles], np.int32)
        uv = np.array(
            [
                [list(m.uv_layers.active.data[i].uv) for i in t.loops]
                for t in m.loop_triangles
            ],
            np.float32,
        )
        mat_indices = np.array([t.material_index for t in m.loop_triangles], np.int32)
        weights = [
            [(o.vertex_groups[g.group].name, g.weight) for g in v.groups]
            for v in m.vertices
        ]
        textures = []
        for slot in o.material_slots:
            mat = slot.material
            candidates = [
                n
                for n in mat.node_tree.nodes
                if n.type == "TEX_IMAGE" and n.image and n.image.size[0] > 0
            ]
            # Match the image used by the base color group, not the localized display name.
            primary = next(
                (n.image for n in candidates if "BaseColorTexture" in n.name), None
            )
            if primary is None:
                primary = next(
                    (
                        n.image
                        for n in candidates
                        if not any(s in n.image.name for s in ["_Sdw", "_OLS"])
                    ),
                    None,
                )
            assert primary is not None, f"Cannot infer albedo {mat.name}"
            fn = out / (primary.name + ".png")
            primary.save_render(str(fn))
            textures.append(fn.name)
        v = vertices(o)
        np.savez_compressed(
            out / (o.name + "_source.npz"),
            positions=v,
            triangles=tri,
            uv=uv,
            material_indices=mat_indices,
            normals=np.array([list(v.normal) for v in m.vertices], np.float32),
        )
        (out / (o.name + "_weights.json")).write_text(
            json.dumps(weights), encoding="utf8"
        )
        manifest["objects"].append(
            {
                "name": o.name,
                "vertices": len(v),
                "triangles": len(tri),
                "textures": textures,
                "shape_keys": [k.name for k in m.shape_keys.key_blocks]
                if m.shape_keys
                else [],
                "groups": [g.name for g in o.vertex_groups],
            }
        )
    # Export all bone bind/rest data and morph deltas, including controls not selected for the rig.
    (out / "skeleton.json").write_text(
        json.dumps(
            [
                {
                    "name": b.name,
                    "parent": b.parent.name if b.parent else None,
                    "matrix_local": [list(r) for r in b.matrix_local],
                }
                for b in arm.data.bones
            ]
        ),
        encoding="utf8",
    )
    for o in meshes:
        if o.data.shape_keys:
            base = np.array(
                [list(v.co) for v in o.data.shape_keys.key_blocks[0].data], np.float32
            )
            np.savez_compressed(
                out / (o.name + "_morphs.npz"),
                **{
                    k.name: np.array([list(v.co) for v in k.data], np.float32) - base
                    for k in o.data.shape_keys.key_blocks[1:]
                },
            )
    # Camera is also recorded as an exact matrix; exporter and regression use identical canvas.
    scene = bpy.context.scene
    cd = bpy.data.cameras.new("ConverterCamera")
    camera = bpy.data.objects.new("ConverterCamera", cd)
    scene.collection.objects.link(camera)
    cc = Vector(cfg["camera"]["center"])
    camera.location = cc + Vector((0, -3, 0))
    camera.rotation_euler = (cc - camera.location).to_track_quat("-Z", "Y").to_euler()
    cd.type = cfg["camera"]["type"]
    cd.ortho_scale = cfg["camera"]["ortho_scale"]
    scene.camera = camera
    scene.render.resolution_x, scene.render.resolution_y = cfg["canvas"]
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.engine = "BLENDER_EEVEE"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    # Albedo references provide a controlled common appearance, not a claim to preserve MToon shading.
    for o in meshes:
        for slot in o.material_slots:
            old = slot.material
            name = old.name + "_ConverterAlbedo"
            mat = bpy.data.materials.get(name)
            if mat is None:
                mat = bpy.data.materials.new(name)
                mat.use_nodes = True
                nt = mat.node_tree
                nt.nodes.clear()
                image = next(
                    n.image
                    for n in old.node_tree.nodes
                    if n.type == "TEX_IMAGE"
                    and n.image
                    and "BaseColorTexture" in n.name
                )
                tex = nt.nodes.new("ShaderNodeTexImage")
                tex.image = image
                emit = nt.nodes.new("ShaderNodeEmission")
                transparent = nt.nodes.new("ShaderNodeBsdfTransparent")
                mix = nt.nodes.new("ShaderNodeMixShader")
                output = nt.nodes.new("ShaderNodeOutputMaterial")
                nt.links.new(tex.outputs["Color"], emit.inputs["Color"])
                nt.links.new(tex.outputs["Alpha"], mix.inputs[0])
                nt.links.new(transparent.outputs[0], mix.inputs[1])
                nt.links.new(emit.outputs[0], mix.inputs[2])
                nt.links.new(mix.outputs[0], output.inputs["Surface"])
                mat.use_backface_culling = True
            slot.material = mat
    bpy.context.view_layer.update()
    manifest["camera"] = {
        "matrix_world": [list(r) for r in camera.matrix_world],
        "projection": [
            list(r)
            for r in camera.calc_matrix_camera(
                bpy.context.evaluated_depsgraph_get(),
                x=cfg["canvas"][0],
                y=cfg["canvas"][1],
            )
        ],
        "canvas": cfg["canvas"],
    }
    if not args.references_only:
        start = time.perf_counter()
        for i, (z, y, x) in enumerate(
            itertools.product([-30, 0, 30], cfg["head_y_keys"], cfg["head_x_keys"])
        ):
            pose = {"ParamAngleX": x, "ParamAngleY": y, "ParamAngleZ": z}
            apply(pose)
            data = {o.name: vertices(o) for o in meshes}
            for pid, s in shapes.items():
                apply(dict(pose, **{pid: 1 - s["default"]}))
                o = bpy.data.objects[s["object"]]
                data[o.name + "__" + pid] = vertices(o) - data[o.name]
            for gy, gx in itertools.product([-1, 0, 1], repeat=2):
                apply(dict(pose, ParamEyeBallX=gx, ParamEyeBallY=gy))
                data[gaze_name + "__gaze_" + str(gx) + "_" + str(gy)] = vertices(
                    bpy.data.objects[gaze_name]
                )
            np.savez_compressed(out / f"head_{i:03}.npz", **data)
            manifest["head_poses"].append(
                {"file": f"head_{i:03}.npz", "parameters": pose}
            )
            if i % 10 == 0:
                print(
                    "SAMPLE",
                    i,
                    pose,
                    "elapsed",
                    round(time.perf_counter() - start, 2),
                    flush=True,
                )
        apply({})
        neutral = {o.name: vertices(o) for o in meshes}
        for pid in ["ParamBodyAngleX", "ParamBodyAngleY", "ParamBodyAngleZ"]:
            for v in [-15, 15]:
                apply({pid: v})
                np.savez_compressed(
                    out / (pid + "_" + str(v) + ".npz"),
                    **{o.name: vertices(o) - neutral[o.name] for o in meshes},
                )
        for bx, by in itertools.product([-15, 0, 15], repeat=2):
            apply({"ParamBodyAngleX": bx, "ParamBodyAngleY": by})
            np.savez_compressed(
                out / f"body_{bx}_{by}.npz", **{o.name: vertices(o) for o in meshes}
            )
        manifest["sampling_seconds"] = time.perf_counter() - start
        (out / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf8"
        )
    else:
        # Preserve sampling inventory on a references-only run.
        assert (out / "manifest.json").exists()
    cases = (
        json.loads(Path(args.case_file).read_text())
        if args.case_file
        else default_cases()
    )
    ref = Path(args.reference_dir).resolve() if args.reference_dir else out.parent / "reference"
    ref.mkdir(exist_ok=True)
    material_modes = []
    for mat in {s.material for o in meshes for s in o.material_slots}:
        emit = next(n for n in mat.node_tree.nodes if n.type == "EMISSION")
        tex = next(n for n in mat.node_tree.nodes if n.type == "TEX_IMAGE")
        is_eye = mat in [s.material for s in bpy.data.objects[gaze_name].material_slots]
        material_modes.append((mat, emit, tex, is_eye))

    def material_mode(eye_mask):
        for mat, emit, tex, is_eye in material_modes:
            for link in list(emit.inputs["Color"].links):
                mat.node_tree.links.remove(link)
            if eye_mask:
                emit.inputs["Color"].default_value = (
                    (1, 1, 1, 1) if is_eye else (0, 0, 0, 1)
                )
            else:
                mat.node_tree.links.new(tex.outputs["Color"], emit.inputs["Color"])

    surface_state = []
    if args.surface_labels:
        label_map = json.loads(Path(args.surface_labels).read_text())
        colors = [(0, 0, 0, 1), (1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
        for obj in meshes:
            original_slots = list(obj.data.materials)
            original_indices = [p.material_index for p in obj.data.polygons]
            surface_state.append((obj, original_slots, original_indices))
            diagnostic_slots = {}
            ids = label_map.get(obj.name, [0] * len(obj.data.loop_triangles))
            for tri, label in zip(obj.data.loop_triangles, ids):
                old_index = original_indices[tri.polygon_index]
                key = (old_index, label)
                if key not in diagnostic_slots:
                    mat = original_slots[old_index].copy()
                    mat.name += f"_Surface_{label}"
                    emit = next(n for n in mat.node_tree.nodes if n.type == "EMISSION")
                    for link in list(emit.inputs["Color"].links):
                        mat.node_tree.links.remove(link)
                    emit.inputs["Color"].default_value = colors[label]
                    diagnostic_slots[key] = len(obj.data.materials)
                    obj.data.materials.append(mat)
                obj.data.polygons[tri.polygon_index].material_index = diagnostic_slots[key]

    def surface_mode(enabled):
        for obj, slots, indices in surface_state:
            # Slots stay installed; per-polygon indices are swapped without touching alpha nodes.
            obj.data.polygons.foreach_set("material_index", obj["surface_indices"] if enabled else indices)

    for obj, slots, indices in surface_state:
        obj["surface_indices"] = [p.material_index for p in obj.data.polygons]
    surface_mode(False)

    for case in cases:
        apply(case["parameters"])
        for o in meshes:
            v = vertices(o)
            np.save(ref / (case["name"] + "__" + o.name + ".npy"), v)
        material_mode(False)
        scene.render.filepath = str(ref / (case["name"] + ".png"))
        bpy.ops.render.render(write_still=True)
        if args.eye_masks:
            material_mode(True)
            scene.render.filepath = str(ref / (case["name"] + "_eyes.png"))
            bpy.ops.render.render(write_still=True)
        if args.surface_labels:
            surface_mode(True)
            scene.render.filepath = str(ref / (case["name"] + "_surfaces.png"))
            bpy.ops.render.render(write_still=True)
            surface_mode(False)
        print("RENDER", case["name"], flush=True)
    case_output = (
        Path(args.case_output) if args.case_output else out.parent / "cases.json"
    )
    case_output.write_text(json.dumps(cases, indent=2), encoding="utf8")
    material_mode(False)
    apply({})
    if not args.references_only:
        bpy.ops.wm.save_as_mainfile(filepath=str(out.parent / "POPY_conversion.blend"))


def default_cases():
    p = [
        ("front", {}),
        ("left", {"ParamAngleX": -45}),
        ("right", {"ParamAngleX": 45}),
        ("up", {"ParamAngleY": 30}),
        ("down", {"ParamAngleY": -30}),
        ("roll", {"ParamAngleZ": 30}),
        ("blink_left", {"ParamEyeLOpen": 0}),
        ("blink_right", {"ParamEyeROpen": 0}),
        ("blink", {"ParamEyeLOpen": 0, "ParamEyeROpen": 0}),
        ("mouth", {"ParamMouthOpenY": 1}),
        ("turn_mouth", {"ParamAngleX": 30, "ParamMouthOpenY": 1}),
        ("turn_blink", {"ParamAngleX": -30, "ParamEyeLOpen": 0, "ParamEyeROpen": 0}),
        ("head_body", {"ParamAngleX": 30, "ParamBodyAngleX": 15}),
        (
            "extreme",
            {
                "ParamAngleX": 45,
                "ParamAngleY": -30,
                "ParamAngleZ": 30,
                "ParamMouthOpenY": 1,
                "ParamEyeLOpen": 0,
                "ParamEyeBallX": 1,
            },
        ),
        (
            "heldout",
            {
                "ParamAngleX": 22.5,
                "ParamAngleY": 7.5,
                "ParamAngleZ": 15,
                "ParamMouthOpenY": 0.6,
                "ParamEyeLOpen": 0.25,
                "ParamEyeBallX": 0.5,
            },
        ),
        ("brows", {"ParamBrowLForm": 1, "ParamBrowRY": 1}),
        ("body_y", {"ParamBodyAngleY": 15}),
        ("body_z", {"ParamBodyAngleZ": 15}),
        ("breath", {"ParamBreath": 1}),
    ]
    p += [
        (
            "heldout_negative",
            {
                "ParamAngleX": -22.5,
                "ParamAngleY": -7.5,
                "ParamAngleZ": -15,
                "ParamMouthOpenY": 0.3,
                "ParamEyeROpen": 0.4,
            },
        ),
        (
            "heldout_head_body",
            {
                "ParamAngleX": 22.5,
                "ParamAngleY": 7.5,
                "ParamAngleZ": -7.5,
                "ParamBodyAngleX": 7.5,
                "ParamMouthOpenY": 0.4,
            },
        ),
        ("pitch_mouth", {"ParamAngleY": -30, "ParamMouthOpenY": 1}),
        ("gaze_follow", {"ParamAngleX": 45, "ParamEyeBallX": 1, "ParamEyeBallY": 0.5}),
        ("mouth_form", {"ParamMouthForm": 1}),
        ("wink_smile", {"ParamEyeLSmile": 1, "ParamEyeROpen": 0}),
        ("tilt_breath", {"ParamAngleZ": 15, "ParamBreath": 1}),
        (
            "head_body_pitch",
            {"ParamAngleX": 30, "ParamBodyAngleY": 15, "ParamMouthOpenY": 0.5},
        ),
        (
            "all_extreme",
            {
                "ParamAngleX": 45,
                "ParamAngleY": -30,
                "ParamAngleZ": 30,
                "ParamBodyAngleX": 15,
                "ParamBodyAngleY": -15,
                "ParamBodyAngleZ": 15,
                "ParamEyeLOpen": 0,
                "ParamEyeROpen": 0,
                "ParamMouthOpenY": 1,
                "ParamMouthForm": 1,
                "ParamBrowLForm": 1,
            },
        ),
    ]
    return [{"name": n, "parameters": v} for n, v in p]


if __name__ == "__main__":
    main()
