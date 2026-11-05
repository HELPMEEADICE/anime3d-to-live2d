import copy
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from popy.mask_proxy import opaque_runs, ordered_runs, require_opaque_sources


def mesh(texture=0, parent="head", vertices=3, nulls=None):
    return {
        "positions": [0.0] * (vertices * 2),
        "texture": texture,
        "culling": True,
        "deformer": parent,
        "axes": [],
        "cells": [{"coordinate": []}],
        "blend_shapes": []
        if nulls is None
        else [
            {
                "parameter": "Blink",
                "keys": [0, 1],
                "neutral": 1,
                "offsets": nulls,
                "orders": [0, 0],
                "opacities": [0, 0],
            }
        ],
    }


class MaskProxyTests(unittest.TestCase):
    def test_binary_union_can_cross_texture_and_painter_order(self):
        lookup = {"a": mesh(), "b": mesh(parent="body"), "c": mesh(texture=1)}
        self.assertEqual(ordered_runs(["a", "b", "c"], lookup), [["a"], ["b"], ["c"]])
        self.assertEqual(opaque_runs(["a", "b", "c"], lookup), [["a", "c"], ["b"]])

    def test_different_culling_or_blend_neutral_pattern_cannot_share_program(self):
        a = mesh(nulls=[8, None])
        b = copy.deepcopy(a)
        b["blend_shapes"][0]["offsets"] = [None, 8]
        c = copy.deepcopy(a)
        c["culling"] = False
        self.assertEqual(
            opaque_runs(["a", "b", "c"], {"a": a, "b": b, "c": c}),
            [["a"], ["b"], ["c"]],
        )

    def test_uint16_vertex_budget_retains_every_source(self):
        lookup = {str(i): mesh(vertices=4) for i in range(5)}
        self.assertEqual(
            opaque_runs(list(lookup), lookup, vertex_limit=8),
            [["0", "1"], ["2", "3"], ["4"]],
        )

    def test_single_translucent_texel_rejects_union_reordering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("RGBA", (3, 3), (10, 20, 30, 255))
            image.save(root / "opaque.png")
            require_opaque_sources(root, ["opaque.png"], [0])
            image.putpixel((1, 1), (10, 20, 30, 254))
            image.save(root / "translucent.png")
            with self.assertRaisesRegex(ValueError, "alpha = 255"):
                require_opaque_sources(root, ["translucent.png"], [0])


if __name__ == "__main__":
    unittest.main()
