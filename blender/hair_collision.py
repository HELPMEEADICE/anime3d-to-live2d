"""Non-destructive, armature-following hair/outfit clearance for the POPY copy.

Native modifiers and Geometry Nodes; no handlers or baked pose frames. Raw mesh,
UVs and rig remain editable. Simple evaluated subdivision improves contact
resolution before one-sided projection onto a closed outfit envelope.
"""

import json
from pathlib import Path

import bpy
import numpy as np
from mathutils.bvhtree import BVHTree

PREFIX = "POPY_Clearance_"
COLLIDERS = ["M_Body", "M_Top", "M_Bow", "M_Necktie"]


def install(clearance=None, passes=1):
    from popy.hair_semantics import policy, strand_sides

    config = policy()
    clearance = config["clearance_m"] if clearance is None else clearance
    hair = bpy.data.objects["M_Hair"]
    if not np.allclose(np.array(hair.matrix_world)[:3, 1], [0, -1, 0], atol=1e-5):
        raise ValueError(
            "Current projection policy expects the audited POPY hair local-Y orientation"
        )
    if any(m.name.startswith(PREFIX) for m in hair.modifiers):
        return hair
    collection = bpy.data.collections.new("POPY Hair Collision Helpers")
    bpy.context.scene.collection.children.link(collection)
    collection.hide_render = True
    front_group = hair.vertex_groups.new(name=PREFIX + "Front")
    back_group = hair.vertex_groups.new(name=PREFIX + "Back")
    hair.data.calc_loop_triangles()
    sides = strand_sides(
        np.array([list(hair.matrix_world @ v.co) for v in hair.data.vertices]),
        np.array([list(t.vertices) for t in hair.data.loop_triangles]),
        config["strand_classification_below_z"],
    )
    for vertex in hair.data.vertices:
        (front_group if sides[vertex.index] < 0 else back_group).add(
            [vertex.index], 1.0, "REPLACE"
        )
    targets = []
    for name in config["colliders"]:
        original = bpy.data.objects[name]
        target = original.copy()
        target.data = original.data.copy()
        target.name = PREFIX + name
        collection.objects.link(target)
        for modifier in list(target.modifiers):
            if modifier.type != "ARMATURE":
                target.modifiers.remove(modifier)
        target.hide_render = True
        target.display_type = "WIRE"
        target.hide_set(True)
        targets.append(target)
    # The outfit contains open panels and overlapping layers. Their individual
    # normals are not an inside/outside test. Build a closed, outward convex
    # envelope of evaluated colliders before applying Outside shrinkwrap.
    envelope = bpy.data.objects.new(
        PREFIX + "Envelope", bpy.data.meshes.new(PREFIX + "EnvelopeMesh")
    )
    collection.objects.link(envelope)
    envelope.hide_render = True
    envelope.hide_set(True)
    tree = bpy.data.node_groups.new(PREFIX + "ClosedEnvelope", "GeometryNodeTree")
    tree.interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )
    output = tree.nodes.new("NodeGroupOutput")
    output.location = (600, 0)
    join = tree.nodes.new("GeometryNodeJoinGeometry")
    join.location = (150, 0)
    hull = tree.nodes.new("GeometryNodeConvexHull")
    hull.location = (350, 0)
    for i, target in enumerate(targets):
        info = tree.nodes.new("GeometryNodeObjectInfo")
        info.location = (-200, -250 * i)
        info.label = target.name
        info.transform_space = "RELATIVE"
        info.inputs["Object"].default_value = target
        tree.links.new(info.outputs["Geometry"], join.inputs["Geometry"])
    tree.links.new(join.outputs["Geometry"], hull.inputs["Geometry"])
    tree.links.new(hull.outputs["Convex Hull"], output.inputs["Geometry"])
    envelope.modifiers.new(PREFIX + "EnvelopeNodes", "NODES").node_group = tree
    targets = [envelope, envelope]
    resolution = hair.modifiers.new("POPY_CollisionResolution", "SUBSURF")
    resolution.subdivision_type = "SIMPLE"
    resolution.levels = config["simple_subdivision"]
    resolution.render_levels = config["simple_subdivision"]
    first = (
        max(
            (i for i, m in enumerate(hair.modifiers) if m.type == "ARMATURE"),
            default=-1,
        )
        + 1
    )
    with bpy.context.temp_override(object=hair, active_object=hair):
        while list(hair.modifiers).index(resolution) > first:
            bpy.ops.object.modifier_move_up(modifier=resolution.name)
    for iteration in range(passes):
        for target_index, target in enumerate(targets):
            modifier = hair.modifiers.new(
                PREFIX + str(iteration) + "_" + target.name, "SHRINKWRAP"
            )
            # Check identifiers in the running Blender, rather than assuming a version.
            allowed = {
                v.identifier for v in modifier.bl_rna.properties["wrap_mode"].enum_items
            }
            if "OUTSIDE" not in allowed:
                raise RuntimeError(f"Outside shrinkwrap unavailable: {allowed}")
            modifier.wrap_method = "PROJECT"
            modifier.wrap_mode = "OUTSIDE"
            modifier.target = target
            modifier.offset = clearance
            modifier.vertex_group = (
                front_group.name if target_index == 0 else back_group.name
            )
            modifier.use_project_y = True
            modifier.use_positive_direction = target_index == 0
            modifier.use_negative_direction = target_index == 1
            modifier.cull_face = "FRONT"
            # Correct evaluated skinned vertices before an optional outline expansion.
            desired = first + 1 + iteration * len(targets) + target_index
            with bpy.context.temp_override(object=hair, active_object=hair):
                while list(hair.modifiers).index(modifier) > desired:
                    bpy.ops.object.modifier_move_up(modifier=modifier.name)
    hair["POPY_collision_clearance_m"] = clearance
    hair["POPY_collision_note"] = (
        "Connected source strands routed to one side of a closed outfit convex envelope. Simple subdivision before native Outside projection, 20mm clearance. No simulation or handlers. Conservative envelope; remaining grazing triangle contacts and large depth displacements require audit. Original source snapshot untouched."
    )
    bpy.context.view_layer.update()
    return hair


def geometry(obj):
    ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = ev.to_mesh()
    mesh.calc_loop_triangles()
    points = [ev.matrix_world @ v.co for v in mesh.vertices]
    triangles = [tuple(t.vertices) for t in mesh.loop_triangles]
    ev.to_mesh_clear()
    return points, triangles


def intersections(hair):
    points, triangles = geometry(hair)
    tree = BVHTree.FromPolygons(points, triangles, all_triangles=True, epsilon=0)
    result = {}
    for name in COLLIDERS:
        p, t = geometry(bpy.data.objects[PREFIX + name])
        other = BVHTree.FromPolygons(p, t, all_triangles=True, epsilon=0)
        hits = tree.overlap(other)
        selected = sorted({a for a, b in hits})
        ids = sorted({v for t in selected for v in triangles[t]})
        z = [points[v].z for v in ids]
        result[name] = {
            "triangle_pairs": len(hits),
            "hair_triangles": len(selected),
            "evaluated_hair_triangles": selected,
            "world_z_range": [min(z), max(z)] if z else [],
        }
    return result


def audit_pose(hair):
    modifiers = [m for m in hair.modifiers if m.name.startswith(PREFIX)]
    for m in modifiers:
        m.show_viewport = False
    bpy.context.view_layer.update()
    before_points = np.array(geometry(hair)[0])
    before = intersections(hair)
    for m in modifiers:
        m.show_viewport = True
    bpy.context.view_layer.update()
    after_points = np.array(geometry(hair)[0])
    after = intersections(hair)
    displacement = np.linalg.norm(after_points - before_points, axis=1)
    return {
        "before": before,
        "after": after,
        "max_displacement_m": float(displacement.max()),
        "moved_vertices": int((displacement > 1e-6).sum()),
    }


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    # This command writes only the dedicated output, never the input file.
    destination = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    if destination == Path(bpy.data.filepath).resolve():
        raise ValueError("Refusing to overwrite source .blend")
    install()
    bpy.ops.wm.save_as_mainfile(filepath=str(destination))
    print(
        json.dumps(
            {
                "saved": str(destination),
                "hair_vertices": len(bpy.data.objects["M_Hair"].data.vertices),
            }
        ),
        flush=True,
    )
