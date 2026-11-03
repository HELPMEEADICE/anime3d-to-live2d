"""Collar clipping invariants and measured native-runtime acceptance."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from popy.collar_clearance import apply_collar_clearance, convert
from popy.collar_regression import cases
from popy.compile_ir import ROOT
from popy.depth_masks import clip_barycentric_batch


class CollarGeometryTests(unittest.TestCase):
    def test_reported_pose_and_continuous_sequences(self):
        rows = cases()
        self.assertEqual(len(rows), 79)
        self.assertEqual(len({r["name"] for r in rows}), 79)
        self.assertEqual(
            rows[0]["parameters"],
            {"ParamAngleX": 15, "ParamAngleY": 11, "ParamBreath": 0.5},
        )
        self.assertEqual(sum("sequence" in r for r in rows), 42)

    def test_collar_clip_stays_on_foreground_source_triangle(self):
        distances = np.array([[[-1.0, 1.0, 1.0]], [[-0.1, 1.9, 1.9]]])
        bary = clip_barycentric_batch(distances, np.zeros((2, 1)))
        np.testing.assert_allclose(bary.sum(-1), 1)
        self.assertTrue(np.all(bary >= 0))
        self.assertLessEqual(
            float((bary * distances[..., None, :]).sum(-1).max()), 1e-12
        )
        # A moving plane changes edge intersections continuously.
        mid = clip_barycentric_batch(distances.mean(0, keepdims=True), np.zeros((1, 1)))
        np.testing.assert_allclose(mid, bary.mean(0, keepdims=True))

    def test_unverified_perspective_rejected_before_reading_assets(self):
        ir = {
            "diagnostics": {},
            "camera": {
                "projection": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
                "matrix_world": np.eye(4).tolist(),
            },
        }
        with self.assertRaisesRegex(ValueError, "orthographic"):
            apply_collar_clearance(ir, Path("missing"), Path("missing"))

    def test_cannot_overwrite_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaisesRegex(ValueError, "baseline"):
                convert(path / "POPY.rig.json", path)


class CollarRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "artifacts/neck_repair/report/metrics.json"
        if not path.exists():
            raise unittest.SkipTest("Run Blender/Ayagami collar regression first")
        cls.report = json.loads(path.read_text())

    def test_reported_pose_and_total_skin_leak_improve(self):
        r = self.report
        self.assertEqual(r["after"]["poses"], 79)
        self.assertLess(r["after"]["summary"]["leak"], r["before"]["summary"]["leak"])
        self.assertLess(
            sum(r["after"]["summary"][k] for k in ["leak", "missing"]),
            sum(r["before"]["summary"][k] for k in ["leak", "missing"]),
        )
        self.assertLess(
            r["after"]["cases"][0]["leak"], r["before"]["cases"][0]["leak"] * 0.25
        )
        self.assertLessEqual(
            r["after"]["summary"]["missing"],
            r["before"]["summary"]["missing"] * 1.02 + 1,
        )
        self.assertLess(
            r["after"]["temporal_false_change"], r["before"]["temporal_false_change"]
        )

    def test_eye_face_and_star_metrics_identical(self):
        f = self.report["features"]
        self.assertEqual(f["before"]["summary"], f["after"]["summary"])
        self.assertEqual(f["after"]["poses"], 77)

    def test_original_surface_tracks_and_head_masks_preserved(self):
        self.assertTrue(self.report["invariants"]["visible_geometry_and_order_equal"])
        self.assertTrue(self.report["invariants"]["non_neck_masks_equal"])
        self.assertEqual(self.report["legacy_runtime_poses"], 341)
        self.assertLess(self.report["native_glue"]["max_px"], 0.01)

    def test_report_matches_delivered_moc(self):
        self.assertEqual(
            self.report["moc_sha256"],
            hashlib.sha256(
                (ROOT / "output/POPY_neck_fixed/POPY.moc3").read_bytes()
            ).hexdigest(),
        )
