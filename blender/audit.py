"""Read-only, reusable audit. Run with Blender --background FILE --python audit.py."""

import bpy, json, sys, os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
target = root / "artifacts/scene_audit.json"
if "--" in sys.argv:
    target = Path(sys.argv[sys.argv.index("--") + 1])
target.parent.mkdir(parents=True, exist_ok=True)
a = {
    "blender": bpy.app.version_string,
    "binary": bpy.app.binary_path,
    "python": sys.version,
    "file": bpy.data.filepath,
    "scene": bpy.context.scene.name,
    "engine": bpy.context.scene.render.engine,
    "frame": bpy.context.scene.frame_current,
    "units": bpy.context.scene.unit_settings.system,
    "unit_scale": bpy.context.scene.unit_settings.scale_length,
    "coordinate_system": "right-handed Z-up, camera chosen by semantic config",
    "objects": [],
    "images": [],
    "materials": [],
}
for o in bpy.context.scene.objects:
    d = {
        "name": o.name,
        "type": o.type,
        "parent": o.parent.name if o.parent else None,
        "collections": [c.name for c in o.users_collection],
        "matrix_world": [list(r) for r in o.matrix_world],
        "animation": bool(o.animation_data),
    }
    if o.type == "MESH":
        m = o.data
        m.calc_loop_triangles()
        d.update(
            vertices=len(m.vertices),
            triangles=len(m.loop_triangles),
            uvs=[u.name for u in m.uv_layers],
            materials=[
                s.material.name if s.material else None for s in o.material_slots
            ],
            groups=[g.name for g in o.vertex_groups],
            shape_keys=[
                {
                    "name": k.name,
                    "value": k.value,
                    "min": k.slider_min,
                    "max": k.slider_max,
                }
                for k in m.shape_keys.key_blocks
            ]
            if m.shape_keys
            else [],
            modifiers=[
                {
                    "name": x.name,
                    "type": x.type,
                    "object": getattr(getattr(x, "object", None), "name", None),
                }
                for x in o.modifiers
            ],
            bounds=[
                list(o.matrix_world @ __import__("mathutils").Vector(v))
                for v in o.bound_box
            ],
        )
    if o.type == "ARMATURE":
        d["bones"] = [
            {
                "name": b.name,
                "parent": b.parent.name if b.parent else None,
                "head": list(b.head_local),
                "tail": list(b.tail_local),
                "deform": b.use_deform,
                "matrix_local": [list(r) for r in b.matrix_local],
            }
            for b in o.data.bones
        ]
        d["pose"] = [
            {
                "name": b.name,
                "matrix_basis": [list(r) for r in b.matrix_basis],
                "constraints": [c.type for c in b.constraints],
            }
            for b in o.pose.bones
        ]
    a["objects"].append(d)
for im in bpy.data.images:
    a["images"].append(
        {
            "name": im.name,
            "path": im.filepath,
            "size": list(im.size),
            "packed": bool(im.packed_file),
            "exists": os.path.exists(bpy.path.abspath(im.filepath)),
        }
    )
for mat in bpy.data.materials:
    a["materials"].append(
        {
            "name": mat.name,
            "use_nodes": mat.use_nodes,
            "nodes": [
                {
                    "type": n.type,
                    "name": n.name,
                    "image": getattr(getattr(n, "image", None), "name", None),
                }
                for n in mat.node_tree.nodes
            ]
            if mat.node_tree
            else [],
        }
    )
target.write_text(json.dumps(a, ensure_ascii=False, indent=2), encoding="utf8")
print("AUDIT", target, "meshes", len([o for o in a["objects"] if o["type"] == "MESH"]))
