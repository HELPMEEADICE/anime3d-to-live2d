"""Asset-free tests for continuous pinned modes and physics wiring."""

import unittest

import numpy as np

from popy.secondary_physics import IDS, kind, modes, physics


class SecondaryPhysicsTests(unittest.TestCase):
    def test_roots_and_unrelated_surfaces_stay_fixed(self):
        p = np.array([[380, 120, 1], [384, 340, 2], [420, 560, 0]])
        np.testing.assert_array_equal(modes(p), 0)

    def test_tips_move_and_coincident_seams_agree(self):
        p = np.array([[440, 260, 1], [440, 260, 1], [460, 580, 2]])
        result = modes(p)
        np.testing.assert_array_equal(result[:, 0], result[:, 1])
        self.assertGreater(abs(result[0, 0, 0]), 5)
        self.assertGreater(abs(result[2, 2, 0]), 9)
        self.assertFalse(np.any(result[2:, :2]))
        self.assertFalse(np.any(result[:2, 2:]))

    def test_physics_has_no_cycles_or_missing_output(self):
        cfg = physics()
        self.assertEqual(cfg["Meta"]["TotalOutputCount"], len(IDS))
        outputs = [
            o["Destination"]["Id"] for s in cfg["PhysicsSettings"] for o in s["Output"]
        ]
        self.assertEqual(outputs, IDS)
        for s in cfg["PhysicsSettings"]:
            self.assertTrue(all(i["Source"]["Id"] not in IDS for i in s["Input"]))
            self.assertTrue(
                all(0 < o["VertexIndex"] < len(s["Vertices"]) for o in s["Output"])
            )

    def test_intersection_geometry_keeps_source_semantics(self):
        self.assertEqual(kind("MaskDepth_HairFace_M_Hair_239_F1"), 1)
        self.assertEqual(kind("MaskDepth_FaceHair_M_Face_9_F25"), 0)
        self.assertEqual(kind("Full_M_Skirts_12"), 2)
        self.assertEqual(kind("M_Headwear_1"), 0)


if __name__ == "__main__":
    unittest.main()
