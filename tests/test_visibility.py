import hashlib
import json
import unittest

import numpy as np

from popy.compile_ir import ROOT
from popy.visibility import chord_scale, clip_triangle_barycentric


class VisibilityGeometryTests(unittest.TestCase):
    def test_clipped_surface_keeps_barycentric_source_correspondence(self):
        bounds = np.array([-1.0, 1.0, 1.0])
        polygon = np.array(clip_triangle_barycentric(bounds, 0))
        self.assertEqual(len(polygon), 3)
        np.testing.assert_allclose(polygon.sum(axis=1), 1)
        self.assertTrue(np.all(polygon >= 0))
        self.assertTrue(np.all(polygon @ bounds <= 1e-12))
        self.assertEqual(len(clip_triangle_barycentric(np.ones(3), 0)), 0)
        self.assertEqual(len(clip_triangle_barycentric(-np.ones(3), 0)), 3)

    def test_pitch_chord_is_exact_at_samples_and_corrects_midpoint(self):
        for value in [-15, 0, 15]:
            self.assertEqual(chord_scale(value), 1)
        self.assertAlmostEqual(chord_scale(7.5), np.cos(np.radians(7.5)))
        self.assertAlmostEqual(chord_scale(-7.5), chord_scale(7.5))


class VisibilityRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads(
            (ROOT / "artifacts/visibility/assessment/metrics.json").read_text()
        )

    def test_complete_sweep_and_welds(self):
        summary = self.result["summary"]
        self.assertEqual(summary["poses"], 165)
        self.assertEqual(summary["transitions"], 70)
        self.assertLess(summary["source_seam_max_px"], 0.0001)

    def test_closed_eyes_do_not_show_large_pupil_regions(self):
        rows = [
            r
            for r in self.result["cases"]
            if r["name"].startswith("eye_") and r["name"].endswith("_closed")
        ]
        self.assertEqual(len(rows), 15)
        for row in rows:
            self.assertLessEqual(row["eye_leak_pixels"], 8, row["name"])

    def test_overall_visibility_and_gap_budget(self):
        summary = self.result["summary"]
        # Absolute image budgets, not an assertion of perfect depth ordering.
        self.assertLess(summary["eye_leak_total"], 1500)
        self.assertLess(summary["eye_missing_total"], 6000)
        self.assertLess(summary["interior_crack_total"], 10000)

    def test_sidelooking_irises_and_closed_extreme_do_not_disappear_or_leak(self):
        rows = {r["name"]: r for r in self.result["cases"]}
        for name in ["gaze_side_-45_-1", "gaze_side_45_1", "gaze_follow"]:
            self.assertLess(rows[name]["eye_missing_pixels"], 30, name)
        self.assertEqual(rows["all_extreme"]["eye_leak_pixels"], 0)

    def test_measured_moc_matches_delivered_model(self):
        provenance = json.loads(
            (ROOT / "artifacts/visibility/provenance.json").read_text()
        )
        self.assertEqual(
            provenance["moc_sha256"],
            hashlib.sha256((ROOT / "output/POPY/POPY.moc3").read_bytes()).hexdigest(),
        )
        self.assertEqual(
            provenance["rig_sha256"],
            hashlib.sha256(
                (ROOT / "artifacts/rig/POPY.rig.json").read_bytes()
            ).hexdigest(),
        )

    def test_editor_roundtrip_retains_masks_and_welds(self):
        roundtrip = json.loads(
            (ROOT / "output/POPY_balanced/cmo_roundtrip.json").read_text()
        )
        rig=json.loads((ROOT/'artifacts/rig_balanced/POPY.rig.json').read_text())
        expected=sum(bool(m.get('masked_by')) for m in rig['meshes'])
        self.assertEqual(roundtrip["glues"], len(rig['glues']))
        self.assertEqual(roundtrip["masked_drawables"], expected)
        self.assertEqual(roundtrip["inverted_masks"], expected)
        self.assertEqual(roundtrip['animated_glues'],sum(bool(g.get('axes')) for g in rig['glues']))
        self.assertFalse(roundtrip["official_editor_verified"])


if __name__ == "__main__":
    unittest.main()
