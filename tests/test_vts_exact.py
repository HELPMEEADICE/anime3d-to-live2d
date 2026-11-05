import copy
import unittest

from popy.vts_exact import groups


def mesh(name):
    return {
        "id": name,
        "positions": [0.0] * 6,
        "axes": [],
        "texture": 0,
        "deformer": "body",
        "opacity": 1.0,
        "masked_by": [],
        "invert_mask": False,
        "culling": True,
        "cells": [{"coordinate": [], "order": 500.0, "offset": 0, "opacity": 1.0}],
        "blend_shapes": [],
    }


class ExactBatchTests(unittest.TestCase):
    def test_mask_sources_cannot_expand_via_a_batch(self):
        a, b, target = mesh("a"), mesh("b"), mesh("target")
        target["masked_by"] = ["a"]
        result = groups([a, b, target])
        self.assertEqual([len(g) for g in result], [1, 1, 1])

    def test_only_adjacent_equal_depth_programs_merge(self):
        a, b, c = mesh("a"), mesh("b"), mesh("c")
        b["cells"][0]["order"] = 501.0
        self.assertEqual(len(groups([a, b, c])), 3)
        # Different position-data offsets are allowed, but change no scalar or
        # interpolation program: each vertex keeps its own exact form data.
        b["cells"][0]["order"] = 500.0
        b["cells"][0]["offset"] = 123
        self.assertEqual(len(groups([a, b, c])), 1)

    def test_runtime_program_and_raster_state_differences_are_barriers(self):
        original = mesh("a")
        for field, value in [
            ("texture", 1),
            ("deformer", "head"),
            ("opacity", 0.5),
            ("culling", False),
            ("invert_mask", True),
            ("masked_by", ["third"]),
            ("axes", [{"parameter": "P", "keys": [0, 1]}]),
        ]:
            with self.subTest(field=field):
                other = {**copy.deepcopy(original), "id": "b", field: value}
                self.assertEqual(len(groups([original, other])), 2)

    def test_blend_neutral_pattern_is_part_of_the_contract(self):
        a, b = mesh("a"), mesh("b")
        a["blend_shapes"] = [
            {
                "parameter": "P",
                "keys": [0, 1],
                "neutral": 0,
                "offsets": [None, 17],
                "orders": [500, 500],
                "opacities": [1, 1],
            }
        ]
        b["blend_shapes"] = copy.deepcopy(a["blend_shapes"])
        b["blend_shapes"][0]["offsets"] = [None, 900]
        self.assertEqual(len(groups([a, b])), 1)
        b["blend_shapes"][0]["offsets"] = [900, None]
        self.assertEqual(len(groups([a, b])), 2)

    def test_vertex_cap_preserves_native_index_range(self):
        result = groups([mesh("a"), mesh("b"), mesh("c")], vertex_limit=6)
        self.assertEqual([len(g) for g in result], [2, 1])


if __name__ == "__main__":
    unittest.main()
