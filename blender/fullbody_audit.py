"""Source-only articulated binding/material audit; no mutation or saving."""

import json
from pathlib import Path

import bpy

root = Path(__file__).resolve().parents[1]
arm = bpy.data.objects["Armature"]
objects = {
    o.name: {
        "vertices": len(o.data.vertices),
        "materials": [
            {"name": s.material.name, "culling": s.material.use_backface_culling}
            for s in o.material_slots
        ],
        "modifiers": [m.type for m in o.modifiers],
    }
    for o in bpy.context.scene.objects
    if o.type == "MESH"
}
report = {
    "blender": bpy.app.version_string,
    "armature_matrix_world": [list(r) for r in arm.matrix_world],
    "objects": objects,
    "source": Path(bpy.data.filepath).name,
    "bones": len(arm.data.bones),
}
out = root / "artifacts/fullbody/source_audit.json"
out.parent.mkdir(exist_ok=True, parents=True)
out.write_text(json.dumps(report, indent=2))
print("FULLBODY_SOURCE_AUDIT", json.dumps(report), flush=True)
