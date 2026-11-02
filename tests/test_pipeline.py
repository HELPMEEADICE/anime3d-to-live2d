"""Regression assertions against independently evaluated Blender ground truth."""

import json, unittest
from PIL import Image
import numpy as np
from popy.compile_ir import ROOT, project


class GeometryTests(unittest.TestCase):
    def test_projection_has_perspective_divide(self):
        cam = {
            "matrix_world": np.eye(4).tolist(),
            "projection": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
        }
        p = project(np.array([[1, 0, 2], [1, 0, 4]]), cam)
        self.assertAlmostEqual(float(p[0, 0]), 0.75)
        self.assertAlmostEqual(float(p[1, 0]), 0.625)


class PopyRuntimeRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((ROOT / "artifacts/report/metrics.json").read_text())
        cls.rows = {r["name"]: r for r in cls.report["cases"]}

    def test_real_model_and_parameter_inventory(self):
        moc = (ROOT / "output/POPY/POPY.moc3").read_bytes()
        self.assertEqual(moc[:4], b"MOC3")
        self.assertGreater(len(moc), 100000)
        runtime = json.loads(
            (ROOT / "artifacts/ayagami/runtime_report.json").read_text()
        )
        ids = {p["id"] for p in runtime["parameters"]}
        for p in [
            "ParamAngleX",
            "ParamAngleY",
            "ParamAngleZ",
            "ParamEyeLOpen",
            "ParamEyeROpen",
            "ParamMouthOpenY",
            "ParamMouthForm",
            "ParamBodyAngleX",
            "ParamBodyAngleY",
            "ParamBodyAngleZ",
        ]:
            self.assertIn(p, ids)
        self.assertGreater(runtime["artmeshes"], 50)

    def test_rendered_image_is_nonempty_and_parameters_change_it(self):
        images = {
            n: np.array(Image.open(ROOT / "artifacts/ayagami" / f"{n}.png"))
            for n in ["front", "left", "blink_left", "blink_right", "mouth"]
        }
        self.assertGreater(int((images["front"][:, :, 3] > 128).sum()), 50000)
        for n in ["left", "blink_left", "blink_right", "mouth"]:
            self.assertGreater(
                int(np.count_nonzero(images[n] != images["front"])), 1000, n
            )
        self.assertGreater(
            int(np.count_nonzero(images["blink_left"] != images["blink_right"])), 1000
        )

    def test_training_and_joint_expressions_match_actual_3d_vertices(self):
        for name in [
            "front",
            "left",
            "right",
            "up",
            "down",
            "roll",
            "blink_left",
            "blink_right",
            "blink",
            "mouth",
            "turn_mouth",
            "turn_blink",
            "extreme",
        ]:
            self.assertLess(self.rows[name]["geometry_rmse_px"], 0.15, name)
            self.assertGreater(self.rows[name]["silhouette_iou"], 0.99, name)

    def test_heldout_quality_is_explicitly_reported(self):
        for name in ["heldout", "heldout_negative", "heldout_head_body"]:
            row = self.rows[name]
            self.assertLess(row["geometry_rmse_px"], 3.0, name)
            self.assertGreater(row["silhouette_iou"], 0.98, name)
            self.assertIn("spurious_orientation_flips", row)

    def test_body_pitch_and_all_extremes_are_jointly_fitted(self):
        self.assertTrue(self.report["statistics"]["couple_body_pitch"])
        for name in ["head_body_pitch", "all_extreme"]:
            self.assertLess(self.rows[name]["geometry_rmse_px"], 0.2, name)
            self.assertGreater(self.rows[name]["silhouette_iou"], 0.99, name)

    def test_occlusion_visual_regression(self):
        for name in ["front", "blink", "head_body_pitch"]:
            self.assertLess(self.rows[name]["mae_foreground"], 0.015, name)
        self.assertLess(self.rows["all_extreme"]["mae_foreground"], 0.02)

    def test_runtime_loaded_every_mesh(self):
        runtime = json.loads(
            (ROOT / "artifacts/ayagami/runtime_report.json").read_text()
        )
        self.assertEqual(
            runtime["artmeshes"], self.report["statistics"]["artmesh_count"]
        )


if __name__ == "__main__":
    unittest.main()
