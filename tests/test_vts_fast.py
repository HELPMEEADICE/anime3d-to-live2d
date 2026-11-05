import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from popy.vts_fast import keys_for, local, weld_mask_corners


class DistillationTests(unittest.TestCase):
    def test_local_coordinates_undo_nested_head_and_body_roll(self):
        pivot = {"head": np.array([400.0, 200.0]), "body": np.array([380.0, 500.0])}
        source = np.array([[390.0, 170.0], [440.0, 250.0]])

        def rotate(points, center, angle):
            a = np.radians(angle)
            matrix = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            return (points - center) @ matrix.T + center

        posed = rotate(rotate(source, pivot["head"], -23), pivot["body"], 13)
        recovered = local(
            posed, {"ParamBodyAngleZ": 13, "ParamAngleZ": -23}, True, pivot
        )
        np.testing.assert_allclose(recovered, source - pivot["head"], atol=1e-11)

    def test_nonzero_default_is_in_sampling_grid(self):
        keys = keys_for({"id": "ParamArmLRaise", "min": -85, "max": 80, "default": -65})
        self.assertIn(-65, keys)
        self.assertIn(0, keys)
        self.assertEqual(keys[0], -85)
        self.assertEqual(keys[-1], 80)

    def fixture(self, folder, mask=True, opaque=True):
        path = folder / "keyforms.f32"
        # Vertices 0/3 share the whole track. Vertex 4 shares the neutral
        # position but moves independently, and must NOT be welded to them.
        poses = np.array(
            [
                [[0, 0], [1, 0], [0, 1], [0, 0], [0, 0]],
                [[0, 0], [1, 0], [0, 1], [0, 0], [2, 1]],
            ],
            dtype="<f4",
        )
        path.write_bytes(poses.tobytes())
        Image.new("RGBA", (2, 2), (255, 255, 255, 255 if opaque else 100)).save(
            folder / "t.png"
        )
        mesh = {
            "id": "Mask_Test" if mask else "Full_Test",
            "positions": [0] * 10,
            "uvs": [0, 0, 1, 0, 0, 1, 1, 1, 0.5, 0.5],
            "indices": [0, 1, 2, 3, 4, 2],
            "cells": [{"offset": 0}, {"offset": 10}],
            "blend_shapes": [],
            "texture": 0,
            "source": {},
        }
        return {"textures": ["t.png"], "meshes": [mesh]}, path, poses

    def test_mask_weld_checks_motion_not_just_rest(self):
        with tempfile.TemporaryDirectory() as directory:
            ir, path, original = self.fixture(Path(directory))
            result = weld_mask_corners(ir, path)
            mesh = ir["meshes"][0]
            self.assertEqual(result["vertices_after"], 4)
            self.assertEqual(mesh["indices"][0], mesh["indices"][3])
            self.assertNotEqual(mesh["indices"][0], mesh["indices"][4])
            packed = np.fromfile(path, "<f4").reshape(2, 4, 2)
            for i in range(2):
                np.testing.assert_array_equal(
                    packed[i][mesh["indices"]], original[i][[0, 1, 2, 3, 4, 2]]
                )

    def test_visible_uv_seams_and_alpha_mask_seams_are_preserved(self):
        for mask, opaque in [(False, True), (True, False)]:
            with (
                self.subTest(mask=mask, opaque=opaque),
                tempfile.TemporaryDirectory() as directory,
            ):
                ir, path, original = self.fixture(Path(directory), mask, opaque)
                result = weld_mask_corners(ir, path)
                self.assertEqual(result["vertices_after"], 5)
                np.testing.assert_array_equal(
                    np.fromfile(path, "<f4").reshape(2, 5, 2), original
                )


if __name__ == "__main__":
    unittest.main()
