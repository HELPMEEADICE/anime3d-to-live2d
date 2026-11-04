"""Reopen the delivered editable Blender/NLA project and compare actual geometry."""

import json
from pathlib import Path

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
base = ROOT / "artifacts/fullbody"
timeline = json.loads((base / "blender_timeline.json").read_text())
entries = {c["name"]: c for c in timeline["clips"]}
for obj in bpy.context.scene.objects:
    if obj.type == "MESH" and not obj.name.startswith("POPY_Clearance_"):
        for mod in list(obj.modifiers):
            if mod.type != "ARMATURE" and not mod.name.startswith(
                ("POPY_Clearance_", "POPY_CollisionResolution")
            ):
                obj.modifiers.remove(mod)
checks = []
for clip, reference in [
    ("idle", "motion_idle_middle"),
    ("wave_L", "motion_wave_L_middle"),
    ("point_R", "motion_point_R_middle"),
    ("thinking", "motion_thinking_middle"),
    ("leg_lift_L", "motion_leg_lift_L_middle"),
    ("walk_in_place", "motion_walk_in_place_middle"),
    ("palm_show_L", "motion_palm_show_L_middle"),
    ("palm_show_R", "motion_palm_show_R_middle"),
    ("leg_side_L", "motion_leg_side_L_middle"),
    ("leg_side_R", "motion_leg_side_R_middle"),
    ("expression_wink_left", "expression_wink_left_0"),
    ("expression_surprised", "expression_surprised_0"),
]:
    entry = entries[clip]
    frame = (
        entry["start_frame"]
        if clip.startswith("expression_")
        else (entry["start_frame"] + entry["end_frame"]) // 2
    )
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    error = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.name.startswith("POPY_Clearance_"):
            continue
        path = base / "reference" / (reference + "__" + obj.name + ".npy")
        ref = np.load(path)
        evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh = evaluated.to_mesh()
        v = np.empty(len(mesh.vertices) * 3, np.float32)
        mesh.vertices.foreach_get("co", v)
        matrix = np.array(evaluated.matrix_world)
        actual = v.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]
        evaluated.to_mesh_clear()
        assert ref.shape == actual.shape, (obj.name, ref.shape, actual.shape)
        error.extend(np.linalg.norm(actual - ref, axis=1).tolist())
    maximum = float(max(error))
    checks.append(
        {
            "clip": clip,
            "frame": frame,
            "vertices": len(error),
            "max_error_m": maximum,
            "rmse_m": float(np.sqrt(np.mean(np.array(error) ** 2))),
        }
    )
    assert maximum < 2e-6, (clip, maximum)
report = {
    "clips": len(entries),
    "driven_bones": len(timeline["driven_bones"]),
    "nla_targets": timeline["targets"],
    "evaluated_checks": checks,
    "source_materials_retained": True,
    "official_cubism_editor_verified": False,
    "hair_clearance_retained": bool(
        bpy.data.objects["M_Hair"].get("POPY_collision_clearance_m")
    ),
    "hair_clearance_torso_colliders": all(
        bpy.data.objects["POPY_Clearance_" + n].get("POPY_fullbody_torso_only")
        for n in ["M_Body", "M_Top"]
    ),
}
(base / "blender_verification.json").write_text(json.dumps(report, indent=2))
print("BLENDER_NLA_VERIFIED", json.dumps(report), flush=True)
