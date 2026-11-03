"""Retain torso/neck hair clearance without a raised arm inflating its hull."""

import bmesh
import bpy

from blender.hair_collision import PREFIX
from blender.hair_collision import install as install_clearance


def install():
    hair = install_clearance()
    arm = bpy.data.objects["Armature"]
    arm_bones = set()
    for name in ["Arm_Left", "Arm_Right"]:
        bone = arm.data.bones[name]
        arm_bones.add(name)
        arm_bones.update(b.name for b in bone.children_recursive)
    removed = {}
    for name in ["M_Body", "M_Top"]:
        obj = bpy.data.objects[PREFIX + name]
        if obj.get("POPY_fullbody_torso_only"):
            continue
        selected = []
        for vertex in obj.data.vertices:
            weight = sum(
                g.weight
                for g in vertex.groups
                if obj.vertex_groups[g.group].name in arm_bones
            )
            if weight > 0.1:
                selected.append(vertex.index)
        mesh = bmesh.new()
        mesh.from_mesh(obj.data)
        mesh.verts.ensure_lookup_table()
        bmesh.ops.delete(mesh, geom=[mesh.verts[i] for i in selected], context="VERTS")
        mesh.to_mesh(obj.data)
        mesh.free()
        obj.data.update()
        obj["POPY_fullbody_torso_only"] = True
        removed[name] = len(selected)
    hair["POPY_fullbody_clearance_note"] = (
        "Torso/neck clearance retained. Arm-influenced vertices are removed only from private collision helpers to prevent raised arms inflating the global convex hull. Direct hair/hand contact is not simulated."
    )
    bpy.context.view_layer.update()
    return removed
