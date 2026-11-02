"""Render the original MToon materials, separately from controlled albedo comparisons."""

import bpy, json, math
from pathlib import Path
from mathutils import Vector, Matrix

root = Path(__file__).resolve().parents[1]
cfg = json.loads((root / "config/convert.json").read_text())
sem = json.loads((root / "config/semantics.json").read_text())
scene = bpy.context.scene
out = root / "artifacts/original_mtoon"
out.mkdir(exist_ok=True)
c = bpy.data.cameras.new("ReviewCamera")
camera = bpy.data.objects.new("ReviewCamera", c)
scene.collection.objects.link(camera)
center = Vector(cfg["camera"]["center"])
camera.location = center + Vector((0, -3, 0))
camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
c.type = "ORTHO"
c.ortho_scale = cfg["camera"]["ortho_scale"]
scene.camera = camera
scene.render.resolution_x = 1152
scene.render.resolution_y = 1536
scene.render.resolution_percentage = 100
scene.render.film_transparent = True
scene.render.engine = "BLENDER_EEVEE"
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.view_settings.view_transform = "Standard"
for name, location, power, size in [
    ("Key", (1, -3, 4), 400, 4),
    ("Fill", (-2, -1, 2), 150, 3),
]:
    data = bpy.data.lights.new("Review" + name, "AREA")
    light = bpy.data.objects.new("Review" + name, data)
    scene.collection.objects.link(light)
    light.location = location
    light.rotation_euler = (center - light.location).to_track_quat("-Z", "Y").to_euler()
    data.energy = power
    data.shape = "DISK"
    data.size = size
for name, yaw in [("front", 0), ("right", 45), ("left", -45)]:
    head = bpy.data.objects[sem["armature"]].pose.bones[sem["bones"]["head"]]
    q = head.bone.matrix_local.to_3x3().to_4x4()
    head.matrix_basis = q.inverted() @ Matrix.Rotation(math.radians(yaw), 4, "Z") @ q
    bpy.context.view_layer.update()
    scene.render.filepath = str(out / (name + ".png"))
    bpy.ops.render.render(write_still=True)
print("Original-material review renders:", out)
