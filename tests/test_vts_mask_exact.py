import unittest

from popy.vts_mask_exact import chains, invisible, trim_indices


def source(name):
    return {
        "id": name,
        "positions": [0.0] * 6,
        "texture": 0,
        "deformer": "head",
        "opacity": 0.0,
        "masked_by": [],
        "invert_mask": False,
        "culling": False,
        "axes": [],
        "cells": [{"coordinate": [], "offset": 0, "order": 17, "opacity": 0}],
        "blend_shapes": [],
    }


def target(name, clips):
    return {
        **source(name),
        "opacity": 1,
        "cells": [{"coordinate": [], "order": 500, "opacity": 1}],
        "masked_by": clips,
    }


class MaskExactTests(unittest.TestCase):
    def test_same_order_in_every_context_is_required(self):
        a, b = source("a"), source("b")
        first, second = target("t", ["a", "b"]), target("u", ["a", "b"])
        b["cells"][0]["order"] = 900  # zero-opacity mask sources do not use depth
        self.assertEqual(chains([a, b, first, second]), [["a", "b"]])
        second["masked_by"] = ["b", "a"]
        self.assertEqual(chains([a, b, first, second]), [])

    def test_foreign_source_between_members_blocks_a_merge(self):
        meshes = [source(n) for n in ["a", "b", "foreign"]]
        # foreign uses a different texture, and cannot be absorbed into the chain.
        meshes[-1]["texture"] = 1
        meshes += [target("t", ["a", "b"]), target("u", ["a", "foreign", "b"])]
        self.assertEqual(chains(meshes), [])

    def test_different_users_or_a_nonzero_blend_prevent_merging(self):
        a, b = source("a"), source("b")
        t = target("t", ["a", "b"])
        self.assertEqual(chains([a, b, t, target("u", ["b"])]), [])
        b["blend_shapes"] = [
            {
                "parameter": "P",
                "keys": [0, 1],
                "offsets": [None, 100],
                "neutral": 0,
                "orders": [0, 0],
                "opacities": [0, 1],
            }
        ]
        self.assertFalse(invisible(b))
        self.assertEqual(chains([a, b, t]), [])
        b["blend_shapes"][0].pop("opacities")
        self.assertFalse(invisible(b))

    def test_a_clipped_zero_opacity_source_is_not_batched(self):
        a, b, c = source("a"), source("b"), source("c")
        a["masked_by"] = ["c"]
        self.assertEqual(chains([a, b, c, target("t", ["a", "b"])]), [])

    def test_native_vertex_limit_splits_a_long_chain(self):
        meshes = [source(n) for n in ["a", "b", "c", "d"]]
        meshes.append(target("t", ["a", "b", "c", "d"]))
        self.assertEqual(chains(meshes, vertex_limit=6), [["a", "b"], ["c", "d"]])

    def test_only_repeated_indices_are_removed_and_order_is_preserved(self):
        indices = [0, 0, 2, 0, 1, 2, 2, 3, 2, 3, 2, 1]
        result, keep = trim_indices(indices)
        self.assertEqual(keep, [1, 3])
        self.assertEqual(result, [0, 1, 2, 3, 2, 1])
        self.assertEqual(trim_indices([0, 0, 0, 1, 2, 1]), ([0, 0, 0], [0]))
        self.assertEqual(trim_indices([]), ([], []))


if __name__ == "__main__":
    unittest.main()
