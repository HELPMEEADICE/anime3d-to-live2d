"""Independent Blender pose references; never saves over the source project."""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from popy.kinematics import parameter_basis


def apply_pose(values, cfg, sem, arm, objects, update=True):
    basis = parameter_basis(cfg, values)
    for b in arm.pose.bones:
        b.matrix_basis = Matrix(basis.get(b.name, np.eye(4)).tolist())
    for semantic, prefix in [("body", "ParamBodyAngle"), ("head", "ParamAngle")]:
        bone = arm.pose.bones[sem["bones"][semantic]]
        q = bone.bone.matrix_local.to_3x3().to_4x4()
        x, y, z = [math.radians(values.get(prefix + a, 0)) for a in "XYZ"]
        bone.matrix_basis = (
            q.inverted()
            @ Matrix.Rotation(z, 4, "Y")
            @ Matrix.Rotation(x, 4, "Z")
            @ Matrix.Rotation(y, 4, "X")
            @ q
        )
    for semantic in ["eye_left", "eye_right"]:
        bone = arm.pose.bones[sem["bones"][semantic]]
        q = bone.bone.matrix_local.to_3x3().to_4x4()
        bone.matrix_basis = (
            q.inverted()
            @ Matrix.Rotation(math.radians(values.get("ParamEyeBallX", 0) * 12), 4, "Z")
            @ Matrix.Rotation(math.radians(values.get("ParamEyeBallY", 0) * 10), 4, "X")
            @ q
        )
    for o in objects:
        if o.data.shape_keys:
            for k in o.data.shape_keys.key_blocks[1:]:
                k.value = 0
    for pid, d in sem["shape_parameters"].items():
        value = values.get(pid, d["default"])
        bpy.data.objects[d["object"]].data.shape_keys.key_blocks[d["shape"]].value = (
            1 - value if d.get("invert") else value
        )
    breath = values.get("ParamBreath", 0)
    body = arm.pose.bones[sem["bones"]["body"]]
    body.scale.x *= 1 + 0.012 * breath
    body.scale.y *= 1 + 0.006 * breath
    if update:
        bpy.context.view_layer.update()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "config/fullbody.json"))
    p.add_argument("--cases", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--geometry-only", action="store_true")
    args = p.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    cfg = json.loads(Path(args.config).read_text())
    sem = json.loads((ROOT / "config/semantics.json").read_text())
    arm = bpy.data.objects[sem["armature"]]
    objects = [
        o
        for o in bpy.context.scene.objects
        if o.type == "MESH" and not o.name.startswith("POPY_Clearance_")
    ]
    for o in objects:
        for mod in list(o.modifiers):
            if mod.type != "ARMATURE" and not mod.name.startswith(
                ("POPY_Clearance_", "POPY_CollisionResolution")
            ):
                o.modifiers.remove(mod)
    if cfg.get("preserve_hair_clearance", True):
        from blender.fullbody_hair import install

        install()
    rest = {b.name: b.matrix_basis.copy() for b in arm.pose.bones}
    assert np.allclose(arm.matrix_world, np.eye(4), atol=1e-6), (
        "World-space source binding requires an identity armature transform"
    )
    assert all(np.max(np.abs(np.array(v) - np.eye(4))) < 1e-5 for v in rest.values()), (
        "Nonidentity source pose requires rebinding"
    )
    assert not any(b.constraints for b in arm.pose.bones), (
        "Constrained source must be baked first"
    )
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    camera_data = bpy.data.cameras.new("POPY_FullBody_Reference")
    camera = bpy.data.objects.new("POPY_FullBody_Reference", camera_data)
    scene.collection.objects.link(camera)
    target = Vector(cfg["camera"]["center"])
    camera.location = target + Vector((0, -3, 0))
    camera.rotation_euler = (
        (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    )
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = cfg["camera"]["ortho_scale"]
    scene.camera = camera
    scene.render.resolution_x, scene.render.resolution_y = cfg["canvas"]
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = 32
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    for o in objects:
        for slot in o.material_slots:
            old = slot.material
            name = old.name + "_FullBodyAlbedo"
            mat = bpy.data.materials.get(name)
            if mat is None:
                image = next(
                    n.image
                    for n in old.node_tree.nodes
                    if n.type == "TEX_IMAGE"
                    and n.image
                    and "BaseColorTexture" in n.name
                )
                mat = bpy.data.materials.new(name)
                mat.use_nodes = True
                n = mat.node_tree.nodes
                n.clear()
                tex = n.new("ShaderNodeTexImage")
                tex.image = image
                emit = n.new("ShaderNodeEmission")
                transparent = n.new("ShaderNodeBsdfTransparent")
                mix = n.new("ShaderNodeMixShader")
                output = n.new("ShaderNodeOutputMaterial")
                links = mat.node_tree.links
                links.new(tex.outputs["Color"], emit.inputs["Color"])
                links.new(tex.outputs["Alpha"], mix.inputs[0])
                links.new(transparent.outputs[0], mix.inputs[1])
                links.new(emit.outputs[0], mix.inputs[2])
                links.new(mix.outputs[0], output.inputs["Surface"])
                mat.use_backface_culling = old.use_backface_culling
            slot.material = mat
    cam = {
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
    # matrix_world is evaluated after location/rotation assignment.
    bpy.context.view_layer.update()
    cam["matrix_world"] = [list(r) for r in camera.matrix_world]
    (out / "camera.json").write_text(json.dumps(cam))
    for case in json.loads(Path(args.cases).read_text()):
        values = case["parameters"]
        apply_pose(values, cfg, sem, arm, objects)
        for o in objects:
            ev = o.evaluated_get(bpy.context.evaluated_depsgraph_get())
            mesh = ev.to_mesh()
            v = np.empty(len(mesh.vertices) * 3, np.float32)
            mesh.vertices.foreach_get("co", v)
            m = np.array(ev.matrix_world)
            v = v.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
            np.save(out / (case["name"] + "__" + o.name + ".npy"), v)
            ev.to_mesh_clear()
        if not args.geometry_only:
            scene.render.filepath = str(out / (case["name"] + ".png"))
            bpy.ops.render.render(write_still=True)
        print("FULLBODY_REFERENCE", case["name"], flush=True)
    if not args.geometry_only:
        bpy.ops.wm.save_as_mainfile(filepath=str(out.parent / "POPY_fullbody.blend"))


if __name__ == "__main__":
    main()
