import unittest

import numpy as np

from popy.mask_context import audit_inventory, group_inventory
from popy.warp_fit import fit


class SpatialFitTests(unittest.TestCase):
    def test_vts_packing_overflows_instead_of_introducing_self_masks(self):
        inventory = [
            {"id": "M_Body_0", "clips": ["M_Body_1"], "invert_mask": True},
            {"id": "M_Body_1", "clips": ["M_Body_0"], "invert_mask": True},
        ]
        groups = group_inventory(
            inventory, {"M_Body_0": [0, 0, 0], "M_Body_1": [1, 0, 0]}, vts=True
        )
        self.assertEqual(len(groups), 2)
        for group in groups:
            union = {s for m in group for s in m["clips"]}
            self.assertFalse({m["id"] for m in group}.intersection(union))

    def test_recovers_bilinear_pose_fields(self):
        rng = np.random.default_rng(82)
        points = np.r_[np.array([[0, 0], [1, 0], [0, 1], [1, 1]]), rng.random((24, 2))]
        u, v = points.T
        b = np.column_stack([(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v])
        lattice = rng.normal(size=(21, 4, 2))
        poses = np.einsum("vk,pkd->pvd", b, lattice)
        _, _, error = fit(points, poses)
        self.assertLess(error.max(), 3e-7)

    def test_residual_is_not_silently_dropped(self):
        p = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])
        poses = p[None].copy()
        poses[0, 4, 1] += 1
        self.assertGreater(fit(p, poses)[2].max(), 0.5)
        self.assertIsNone(fit(np.zeros((6, 2)), np.zeros((1, 6, 2))))

    def test_normal_and_inverted_masks_are_separate(self):
        inventory = [
            {"id": "M_Hair_0", "clips": ["A"], "invert_mask": True},
            {"id": "M_Hair_1", "clips": ["B"], "invert_mask": False},
        ]
        self.assertEqual(len(group_inventory(inventory)), 2)
        self.assertEqual(audit_inventory(inventory, 1)["mask_contexts"], 2)
        self.assertFalse(audit_inventory(inventory, 1)["within_budget"])


if __name__ == "__main__":
    unittest.main()
