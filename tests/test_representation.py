"""Lossless channel factoring and register relocation have explicit invariants."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from popy.glue_arena import assign_arenas, initial_overwrites
from popy.order_factor import factor, scalar_signature
from popy.point_graph import PointGraph
from popy.representation_review import resource_sha


def mesh(name, parent="body"):
    return {
        "id": name,
        "positions": [0.0] * 6,
        "axes": [{"parameter": "P", "keys": [0, 1]}],
        "cells": [
            {"coordinate": [i], "offset": 0, "order": 500.5 + i} for i in range(2)
        ],
        "blend_shapes": [],
        "deformer": parent,
    }


def copy_glue(a, b, count=3):
    return {
        "id": "copy_" + b,
        "mesh_a": a,
        "mesh_b": b,
        "pairs": [[i, i, 0.0, 1.0] for i in range(count)],
        "intensity": 1.0,
    }


class RepresentationTests(unittest.TestCase):
    def test_evidence_binds_nested_runtime_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            manifest = {
                "FileReferences": {
                    "Moc": "POPY.moc3",
                    "Textures": ["texture.png"],
                    "Motions": {"test": [{"File": "motion.motion3.json"}]},
                }
            }
            (base / "POPY.model3.json").write_text(json.dumps(manifest))
            (base / "texture.png").write_bytes(b"synthetic pixel fixture")
            (base / "motion.motion3.json").write_text("{}")
            old = resource_sha(base)
            (base / "motion.motion3.json").write_text('{"changed":true}')
            self.assertNotEqual(old, resource_sha(base))
            (base / "texture.png").unlink()
            with self.assertRaises(FileNotFoundError):
                resource_sha(base)

    def test_rotation_carriers_preserve_visual_ancestry_and_only_redirect_dead_vertices(
        self,
    ):
        a, b, c, d = [mesh(n) for n in "ABCD"]
        b["projection_warp"] = "NativeRoot"
        c["deformer"] = d["deformer"] = "head"
        c["axes"] = []
        c["cells"] = [{"coordinate": [], "offset": 0, "order": 17.0}]
        d["axes"] = []
        d["cells"] = [{"coordinate": [], "offset": 0, "order": 19.0}]
        h = {
            "axes": [{"parameter": "H", "keys": [-1, 1]}],
            "cells": [{"coordinate": [i]} for i in range(2)],
        }
        ir = {
            "meshes": [a, b, c, d],
            "glues": [copy_glue("A", n) for n in "BC"],
            "parts": [{"id": "P", "mesh_ids": list("ABCD")}],
            "body_warps": [
                {"id": "NativeRoot", "kind": "rotation", "root": True, "axes": []}
            ],
            "diagnostics": {},
            "head_rotation": copy.deepcopy(h),
            "head_warp": copy.deepcopy(h),
            "body_rotation": {"pivot": [0, 0]},
        }
        plan = factor(ir, None, rotation_carriers=True)
        self.assertEqual(plan["rotation_visual_carriers"], 1)
        self.assertEqual(plan["static_geometry_visual_redirects"], 1)
        self.assertEqual(c["projection_warp"], "D_ChannelVisual_HeadProjection")
        self.assertEqual(c["cells"][0]["order"], 17.0)
        self.assertEqual(d["deformer"], "head")
        self.assertEqual(ir["head_warp"], h)
        proxy = {n["id"]: n for n in ir["body_warps"]}
        self.assertEqual(proxy["D_ChannelVisual_HeadProjection"]["axes"], h["axes"])
        self.assertEqual(
            proxy["D_ChannelVisual_HeadProjection"]["parent"],
            "D_ChannelVisual_HeadRoll",
        )
        self.assertEqual(
            proxy["D_ChannelVisual_HeadRoll"]["parent"], "D_ChannelVisual_BodyRoll"
        )
        self.assertNotIn("parent", proxy["D_ChannelVisual_BodyRoll"])

    def test_first_read_blocks_dead_initial_classification(self):
        read = {**copy_glue("A", "B"), "intensity": 0.25}
        first = initial_overwrites([read, copy_glue("C", "B")])
        self.assertFalse(first[("B", 0)])
        first = initial_overwrites([{**copy_glue("A", "B"), "axes": []}])
        self.assertTrue(first[("B", 0)])
        self.assertFalse(first[("A", 0)])
        self.assertFalse(initial_overwrites([copy_glue("A", "A")])[("A", 0)])

    def test_channel_carrier_shares_only_same_axes_and_original_parent(self):
        meshes = [mesh(n, "head" if n == "D" else "body") for n in "ABCD"]
        ir = {
            "meshes": meshes,
            "glues": [copy_glue("A", n) for n in "BCD"],
            "parts": [{"id": "Part", "mesh_ids": list("ABCD")}],
            "body_warps": [],
            "diagnostics": {},
        }
        original = copy.deepcopy(ir["glues"])
        plan = factor(ir, None)
        self.assertEqual(plan["factored_meshes"], 3)
        self.assertEqual(plan["shared_visual_carriers"], 2)
        self.assertEqual(plan["shared_order_parts"], 1)
        self.assertEqual(ir["glues"], original)
        self.assertEqual(meshes[0]["axes"][0]["parameter"], "P")
        self.assertTrue(all(len(m["cells"]) == 1 for m in meshes[1:]))
        self.assertEqual(ir["order_groups"][0]["mesh_ids"], list("BCD"))
        self.assertEqual(ir["order_groups"][0]["org_parent"], "Part")

    def test_nonadjacent_identical_order_functions_do_not_coalesce_draw_groups(self):
        meshes = [mesh(n) for n in "ABCD"]
        ir = {
            "meshes": meshes,
            "glues": [copy_glue("A", n) for n in "BD"],
            "parts": [{"id": "P1", "mesh_ids": list("ABCD")}],
            "body_warps": [],
            "diagnostics": {},
        }
        plan = factor(ir, None)
        self.assertEqual(plan["shared_order_parts"], 1)
        self.assertEqual([g["mesh_ids"] for g in ir["order_groups"]], [["B"], ["D"]])
        self.assertNotEqual(*[g["group_id"] for g in ir["order_groups"]])

    def test_part_visibility_domains_and_blendshapes_are_preserved(self):
        meshes = [mesh(n) for n in "ABCD"]
        meshes[-1]["blend_shapes"] = [{"parameter": "Q"}]
        ir = {
            "meshes": meshes,
            "glues": [copy_glue("A", n) for n in "BCD"],
            "parts": [
                {"id": "P1", "mesh_ids": ["A", "B", "D"]},
                {"id": "P2", "mesh_ids": ["C"]},
            ],
            "body_warps": [],
            "diagnostics": {},
        }
        plan = factor(ir, None)
        self.assertEqual(plan["shared_order_parts"], 2)
        self.assertEqual(len(meshes[-1]["cells"]), 2)

    def test_scalar_grid_checks_coordinates_and_float_bits(self):
        a = mesh("A")
        b = copy.deepcopy(a)
        self.assertEqual(scalar_signature(a), scalar_signature(b))
        a["cells"][0]["order"] = 0.0
        b["cells"][0]["order"] = -0.0
        self.assertNotEqual(scalar_signature(a), scalar_signature(b))
        b["cells"][1]["coordinate"] = [0]
        with self.assertRaises(ValueError):
            scalar_signature(b)

    def test_bounded_arena_coloring_never_merges_glue_endpoints(self):
        meshes = [mesh(n) for n in "ABCD"]
        glues = [
            copy_glue("A", "B"),
            copy_glue("A", "C"),
            copy_glue("B", "C"),
            copy_glue("A", "D"),
        ]
        groups, mapping = assign_arenas(meshes, glues, capacity=6)
        self.assertTrue(all(g["vertices"] <= 6 for g in groups))
        self.assertNotEqual(mapping["B"]["arena"], mapping["C"]["arena"])
        self.assertNotIn("A", mapping)

    def test_real_self_interpolation_folding_is_explicitly_not_exact(self):
        exact = PointGraph()
        real = PointGraph(canonical=True)
        for g in [exact, real]:
            a = g.node(("input", 0, 0))
            n = g.constant(a, a, g.weight_bits(0.4))
            self.assertEqual(n == a, g.canonical)
        self.assertEqual(exact.folded_rounding, 0)
        self.assertEqual(real.folded_rounding, 1)


if __name__ == "__main__":
    unittest.main()
