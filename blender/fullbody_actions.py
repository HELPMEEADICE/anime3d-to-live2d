"""Bake reusable source-rig actions and facial NLA strips into a separate blend."""

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blender.fullbody import apply_pose
from popy.animation import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "delivery/POPY_fullbody.blend"))
    args = parser.parse_args(
        sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    )
    cfg = json.loads((ROOT / "config/fullbody.json").read_text())
    sem = json.loads((ROOT / "config/semantics.json").read_text())
    params = json.loads((ROOT / "artifacts/fullbody/parameters.json").read_text())
    model = ROOT / "output/POPY_fullbody"
    manifest = json.loads((model / "POPY.model3.json").read_text())["FileReferences"]
    arm = bpy.data.objects[sem["armature"]]
    meshes = [
        o
        for o in bpy.context.scene.objects
        if o.type == "MESH" and not o.name.startswith("POPY_Clearance_")
    ]
    if cfg.get("preserve_hair_clearance", True):
        from blender.fullbody_hair import install

        install()
    driven = sorted(
        {d["bone"] for c in cfg["controls"] for d in c["drives"]}
        | set(sem["bones"].values())
    )
    data = {"Armature": arm}
    for d in sem["shape_parameters"].values():
        data[d["object"]] = bpy.data.objects[d["object"]].data.shape_keys
    for target in data.values():
        target.animation_data_clear()
    for bone in driven:
        arm.pose.bones[bone].rotation_mode = "QUATERNION"
    clips = []
    for name, refs in manifest["Motions"].items():
        motion = json.loads((model / refs[0]["File"]).read_text())
        clips.append((name, motion, None, motion["Meta"]["Duration"]))
    for entry in manifest["Expressions"]:
        expression = json.loads((model / entry["File"]).read_text())
        clips.append(("expression_" + entry["Name"], None, expression, 1))
    timeline = []
    cursor = 1
    strip_records = []
    scene = bpy.context.scene
    scene.render.fps = 30
    for name, motion, expression, duration in clips:
        actions = {}
        for key, target in data.items():
            action = bpy.data.actions.new("POPY_" + name + "__" + key)
            target.animation_data_create().action = action
            actions[key] = action
        count = round(duration * 30)
        # Native joint/shape channels are sampled at 30 fps and remain editable.
        times = range(count + 1) if motion else [0, count]
        for i in times:
            values = evaluate(params, motion, i / 30, expression)
            apply_pose(values, cfg, sem, arm, meshes, update=False)
            frame = i + 1
            for bone_name in driven:
                bone = arm.pose.bones[bone_name]
                bone.keyframe_insert(
                    data_path="rotation_quaternion", frame=frame, group=bone_name
                )
                bone.keyframe_insert(data_path="scale", frame=frame, group=bone_name)
            for definition in sem["shape_parameters"].values():
                keys = bpy.data.objects[definition["object"]].data.shape_keys
                keys.key_blocks[definition["shape"]].keyframe_insert(
                    data_path="value", frame=frame, group="Facial shapes"
                )
        for key, target in data.items():
            slot = target.animation_data.action_slot
            strip_records.append((key, actions[key], slot, cursor, count))
            target.animation_data.action = None
            actions[key].use_fake_user = True
        timeline.append(
            {
                "name": name,
                "start_frame": cursor,
                "end_frame": cursor + count,
                "actions": {k: a.name for k, a in actions.items()},
            }
        )
        scene.timeline_markers.new(name, frame=cursor)
        cursor += count + 30
        print("BAKED_ACTION", name, count + 1, flush=True)
    tracks = {
        key: target.animation_data_create().nla_tracks.new()
        for key, target in data.items()
    }
    for key, track in tracks.items():
        track.name = "POPY motion and expression preview"
    for key, action, slot, start, count in strip_records:
        strip = tracks[key].strips.new(action.name, start, action)
        strip.action_slot = slot
        strip.extrapolation = "NOTHING"
        strip.blend_type = "REPLACE"
        strip.action_frame_start = 1
        strip.action_frame_end = count + 1
    camera_data = bpy.data.cameras.new("POPY_FullBody_Camera")
    camera = bpy.data.objects.new("POPY_FullBody_Camera", camera_data)
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
    scene.frame_start = 1
    scene.frame_end = cursor - 30
    scene.frame_set(1)
    bpy.context.view_layer.update()
    arm["POPY_animation_note"] = (
        "22 authored joint motions + 16 expression poses; use timeline markers/NLA. Source shape channels and bone curves are editable. Live2D parameters are defined in config/fullbody.json."
    )
    output = Path(args.out).resolve()
    output.parent.mkdir(exist_ok=True, parents=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    (ROOT / "artifacts/fullbody/blender_timeline.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "driven_bones": driven,
                "clips": timeline,
                "targets": list(data),
                "blend": output.name,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
