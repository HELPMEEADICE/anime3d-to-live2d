"""Factor repeated native inverted-mask unions without changing their geometry."""

from collections import Counter

import numpy as np
from PIL import Image


def factor_inherited_masks(
    ir,
    binary,
    minimum_sources=8,
    vertex_limit=60000,
    source_groups=None,
    prefix="FaceUnion",
):
    """One reusable opaque union replaces repeated legacy mask source lists.

    Native masks ignore source RGB/opacity. POPY's verified opaque texture
    pages permit exact concatenation with constant alpha; transparent models
    must retain UV/texture pages and are explicitly rejected here.
    """
    lookup = {m["id"]: m for m in ir["meshes"]}

    def group_for(mesh):
        if source_groups is not None:
            return tuple(sorted(source_groups.get(mesh["id"], [])))
        return tuple(sorted(lookup[mesh["face_partition_parent"]].get("masked_by", [])))

    groups = Counter()
    for m in ir["meshes"]:
        if m.get("mask_only") or not m.get("face_partition_parent"):
            continue
        key = group_for(m)
        if (
            len(key) >= minimum_sources
            and m.get("invert_mask")
            and set(key) <= set(m.get("masked_by", []))
        ):
            groups[key] += 1
    groups = {k: n for k, n in groups.items() if n > 1}
    if not groups:
        return {"unions": 0, "replaced_targets": 0}
    textures = {lookup[k]["texture"] for group in groups for k in group} | {0}
    for group in groups:
        if len({lookup[k].get("culling", True) for k in group}) != 1:
            raise ValueError("Mask union requires matching source culling flags")
    for index in textures:
        with Image.open(binary.parent / ir["textures"][index]) as image:
            if image.convert("RGBA").getchannel("A").getextrema() != (255, 255):
                raise ValueError(
                    "Packed mask unions require fully opaque source textures"
                )
    cursor = binary.stat().st_size // 4
    replacements, helpers, glues = {}, [], []
    with binary.open("ab") as stream:
        for gi, sources in enumerate(groups):
            chunks, chunk, count = [], [], 0
            for source in sources:
                size = len(lookup[source]["positions"]) // 2
                if size > vertex_limit:
                    raise ValueError("Mask source exceeds native union index budget")
                if count + size > vertex_limit:
                    chunks.append(chunk)
                    chunk, count = [], 0
                chunk.append(source)
                count += size
            if chunk:
                chunks.append(chunk)
            replacements[sources] = []
            for ci, chunk in enumerate(chunks):
                helper_id = f"Mask{prefix}_{gi}_{ci}"
                first = lookup[chunk[0]]
                positions, indices, provenance = [], [], []
                for source in chunk:
                    mesh = lookup[source]
                    offset = len(positions) // 2
                    size = len(mesh["positions"]) // 2
                    positions.extend(mesh["positions"])
                    indices.extend(offset + i for i in mesh["indices"])
                    glues.append(
                        {
                            "id": f"G_Depth_{prefix}_{len(glues)}",
                            "mesh_a": source,
                            "mesh_b": helper_id,
                            "pairs": [[i, offset + i, 0.0, 1.0] for i in range(size)],
                            "intensity": 1.0,
                        }
                    )
                    provenance.append(
                        {
                            "mesh": source,
                            "mask_vertex_start": offset,
                            "vertices": size,
                            "correspondence": "identity at original evaluated native corners; no resampling",
                        }
                    )
                size = len(positions) // 2
                helpers.append(
                    {
                        "id": helper_id,
                        "positions": positions,
                        "uvs": [0.5, 0.5] * size,
                        "indices": indices,
                        "texture": 0,
                        "mask_only": True,
                        "culling": first.get("culling", True),
                        "opacity": 0.0,
                        "masked_by": [],
                        "invert_mask": False,
                        "deformer": "head",
                        "axes": [],
                        "cells": [{"coordinate": [], "offset": cursor, "order": 0.0}],
                        "blend_shapes": [],
                        "source": {
                            "object": first["source"]["object"],
                            "vertices": [first["source"]["vertices"][0]] * size,
                            "triangles": [first["source"]["triangles"][0]]
                            * (len(indices) // 3),
                            "barycentric_sources": provenance,
                            "correspondence": "exact opaque native union of evaluated original triangles",
                        },
                    }
                )
                stream.write(np.zeros(size * 2, dtype="<f4").tobytes())
                cursor += size * 2
                replacements[sources].append(helper_id)
    replaced = 0
    for m in ir["meshes"]:
        if m.get("mask_only") or not m.get("face_partition_parent"):
            continue
        key = group_for(m)
        if key in replacements and set(key) <= set(m.get("masked_by", [])):
            m["masked_by"] = [x for x in m["masked_by"] if x not in key] + replacements[
                key
            ]
            replaced += 1
    ir["meshes"].extend(helpers)
    ir["glues"].extend(glues)
    return {
        "unions": len(helpers),
        "groups": len(groups),
        "replaced_targets": replaced,
        "sources_per_group": [len(k) for k in groups],
        "method": "Exact opaque alpha union with original winding and post-evaluation master corner Glue; <=60000 vertices per native ArtMesh",
    }


def pack_local_sources(ir, binary, vertex_limit=60000):
    """Pack local full foreground sources into each existing clipped helper.

    Shared large inherited unions stay shared. This trades some copied corners
    for far fewer mask draw calls while retaining exactly the native alpha OR.
    """
    lookup = {m["id"]: m for m in ir["meshes"]}
    targets = [
        m
        for m in ir["meshes"]
        if not m.get("mask_only") and m.get("face_depth_corrected")
    ]
    cursor = binary.stat().st_size // 4
    added, glues, replacements = [], [], 0
    total_vertices = 0
    for index in {m["texture"] for m in ir["meshes"]}:
        with Image.open(binary.parent / ir["textures"][index]) as image:
            if image.convert("RGBA").getchannel("A").getextrema() != (255, 255):
                raise ValueError(
                    "Packed mask unions require fully opaque source textures"
                )
    with binary.open("ab") as stream:
        for target in targets:
            if target.get("face_sources_packed"):
                continue
            sources = target.get("masked_by", [])
            existing = [
                x
                for x in sources
                if x.startswith(
                    ("MaskDepth_FaceHair_", "MaskDepth_HairFace_", "MaskDepth_Teeth_")
                )
            ]
            shared = [
                x for x in sources if x.startswith(("MaskFaceUnion_", "MaskMouthSkin_"))
            ]
            full = [x for x in sources if x not in existing and x not in shared]
            if not full:
                continue
            if len(existing) > 1:
                raise ValueError("Unexpected multiple local plane helpers")
            helper = lookup[existing[0]] if existing else None
            expected_culling = (
                helper.get("culling", True)
                if helper
                else lookup[full[0]].get("culling", True)
            )
            if any(lookup[x].get("culling", True) != expected_culling for x in full):
                raise ValueError(
                    "Local mask packing requires matching source culling flags"
                )
            packed_ids = existing[:]

            def fresh(
                target=target, packed_ids=packed_ids, expected_culling=expected_culling
            ):
                helper_id = f"MaskPacked_{target['id']}_{len(packed_ids)}"
                item = {
                    **target,
                    "id": helper_id,
                    "positions": [],
                    "uvs": [],
                    "indices": [],
                    "axes": [],
                    "cells": [],
                    "blend_shapes": [],
                    "mask_only": True,
                    "culling": expected_culling,
                    "opacity": 0.0,
                    "masked_by": [],
                    "invert_mask": False,
                    "source": {
                        "object": target["source"]["object"],
                        "vertices": [],
                        "triangles": [],
                        "barycentric_sources": [],
                        "correspondence": "Exact opaque native source alpha union",
                    },
                }
                packed_ids.append(helper_id)
                added.append(item)
                return item

            touched = []
            for source_id in full:
                source = lookup[source_id]
                size = len(source["positions"]) // 2
                if size > vertex_limit:
                    raise ValueError("Mask source exceeds native union index budget")
                if (
                    helper is None
                    or len(helper["positions"]) // 2 + size > vertex_limit
                ):
                    helper = fresh()
                if not any(h is helper for h in touched):
                    touched.append(helper)
                start = len(helper["positions"]) // 2
                helper["positions"].extend(source["positions"])
                helper["uvs"].extend([0.5, 0.5] * size)
                helper["indices"].extend(start + i for i in source["indices"])
                helper["source"]["vertices"].extend(
                    [target["source"]["vertices"][0]] * size
                )
                helper["source"]["triangles"].extend(
                    [target["source"]["triangles"][0]] * (len(source["indices"]) // 3)
                )
                helper["source"]["barycentric_sources"].append(
                    {
                        "mesh": source_id,
                        "mask_vertex_start": start,
                        "vertices": size,
                        "correspondence": "identity at original evaluated native corners",
                    }
                )
                glues.append(
                    {
                        "id": f"G_Depth_FullPack_{len(glues)}",
                        "mesh_a": source_id,
                        "mesh_b": helper["id"],
                        "pairs": [[i, start + i, 0.0, 1.0] for i in range(size)],
                        "intensity": 1.0,
                    }
                )
                total_vertices += size
            for item in touched:
                size = len(item["positions"]) // 2
                item["cells"] = [{"coordinate": [], "offset": cursor, "order": 0.0}]
                stream.write(np.zeros(size * 2, dtype="<f4").tobytes())
                cursor += size * 2
            target["masked_by"] = packed_ids + shared
            target["face_sources_packed"] = True
            replacements += 1
    ir["meshes"].extend(added)
    ir["glues"].extend(glues)
    return {
        "targets": replacements,
        "added_helpers": len(added),
        "copied_vertices": total_vertices,
        "glues": len(glues),
        "method": "Exact opaque source union folded into local clipped helpers; large repeated inherited unions remain shared",
    }
