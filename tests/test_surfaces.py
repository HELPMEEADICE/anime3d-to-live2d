import hashlib
import json
import unittest

import numpy as np

from popy.compile_ir import ROOT
from popy.depth_masks import (
    barycentric_glue_weights,
    clip_barycentric_batch,
    clipped_barycentric,
    deduplicate_barycentric,
)
from popy.field_compression import compress_glue_fields, reduce_grid


class SurfaceGeometryTests(unittest.TestCase):
    def test_scalar_axis_compression_retains_nontrivial_cross_parameter_function(self):
        # f(x,y,z)=x*y on a nonuniform grid, independent of z.
        axes = [
            {"parameter": "x", "keys": [-1, 0, 2]},
            {"parameter": "y", "keys": [0, 0.3, 1]},
            {"parameter": "z", "keys": [-5, 5]},
        ]
        values = np.array(
            [[[x * y, x * y] for y in axes[1]["keys"]] for x in axes[0]["keys"]]
        )
        ir = {
            "diagnostics": {},
            "glues": [{"axes": axes, "intensities": values.ravel().tolist()}],
        }
        report = compress_glue_fields(ir)
        g = ir["glues"][0]
        self.assertEqual([a["parameter"] for a in g["axes"]], ["x", "y"])
        grid = np.array(g["intensities"]).reshape(2, 2)
        # Independent bilinear evaluation at a point between all retained keys.
        actual = (
            np.array([1 - 0.7 / 3 - 1 / 3, 0.7 / 3 + 1 / 3])
            @ grid
            @ np.array([0.8, 0.2])
        )
        self.assertAlmostEqual(float(actual), 0.7 * 0.2)
        self.assertEqual(report["removed_axes"], 1)

    def test_subpixel_curve_reduction_has_measured_uniform_error(self):
        axes = [
            {"parameter": "x", "keys": [0, 0.5, 1]},
            {"parameter": "y", "keys": [0, 1]},
        ]
        values = np.array([[x + 1e-4 * y for y in [0, 1]] for x in [0, 0.5, 1]])
        kept, reduced, error, _, _ = reduce_grid(
            axes, values, {"x": 0, "y": 0}, effect_px=10, budget_px=0.002
        )
        self.assertEqual([a["parameter"] for a in kept], ["x"])
        samples = np.linspace(0, 1, 101)
        expected = samples + 1e-4 * (1 - samples)
        actual = np.interp(samples, kept[0]["keys"], reduced)
        self.assertLessEqual(float(np.max(np.abs(expected - actual))) * 10, 0.002)
        self.assertLessEqual(error, 0.002)

    def test_vectorized_clipping_matches_boundary_cases(self):
        rng = np.random.default_rng(17)
        values = np.r_[
            rng.normal(size=(1000, 3)), [[0, 0, 0], [0, 1, -1], [1, 1, 1], [-1, -1, -1]]
        ]
        actual = clip_barycentric_batch(values, np.zeros(len(values)))
        expected = np.array([clipped_barycentric(v, 0) for v in values])
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        np.testing.assert_allclose(actual.sum(-1), 1, atol=1e-12)
        self.assertTrue(np.all(actual >= 0))

    def test_chained_glue_keeps_points_in_source_triangle(self):
        rng = np.random.default_rng(11)
        points = rng.normal(size=(300, 3, 2))
        bary = rng.dirichlet([1, 1, 1], 300)
        a, b = barycentric_glue_weights(bary)
        actual = (points[:, 0] * (1 - a[:, None]) + points[:, 1] * a[:, None]) * (
            1 - b[:, None]
        ) + points[:, 2] * b[:, None]
        np.testing.assert_allclose(
            actual, np.einsum("ni,nij->nj", bary, points), atol=1e-12
        )

    def test_mask_corner_dedup_preserves_full_parameter_trajectory(self):
        master = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 2]])
        bary = np.array(
            [[[1, 0, 0], [1, 0, 0], [0.5, 0.5, 0]], [[1, 0, 0], [1, 0, 0], [0, 1, 0]]]
        )
        unique, weights, indices, _selected = deduplicate_barycentric(master, bary)
        self.assertEqual(len(unique), 2)
        points = np.array([[0.0, 0.0], [2, 0], [1, 1], [0, 2]])
        expected = np.einsum("pvi,vic->pvc", bary, points[master])
        actual = np.einsum("pvi,vic->pvc", weights, points[unique])[:, indices]
        np.testing.assert_allclose(expected, actual)


class SurfaceRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.now = json.loads(
            (ROOT / "artifacts/surfaces/assessment/metrics.json").read_text()
        )
        baseline = ROOT / "artifacts/feature_before/assessment/metrics.json"
        cls.before = json.loads(baseline.read_text()) if baseline.exists() else None

    def test_independent_continuous_expression_holdouts_are_present(self):
        self.assertEqual(self.now["poses"], 264)
        rows = [r for r in self.now["cases"] if r["name"].startswith("continuous_")]
        self.assertEqual(len(rows), 63)
        self.assertEqual(
            len(
                {
                    r["parameters"].get("ParamEyeLOpen")
                    for r in rows
                    if "blink" in r["name"]
                }
            ),
            21,
        )

    def test_brow_and_lash_false_coverage_and_missing_regions_improve(self):
        if self.before is None:
            self.skipTest("No independently captured pre-fix baseline")
        for k in ["brows", "lashes"]:
            self.assertLess(
                self.now["summary"][k]["leak_far"],
                self.before["summary"][k]["leak_far"],
                k,
            )
            self.assertLess(
                self.now["summary"][k]["missing"],
                self.before["summary"][k]["missing"] * 0.7,
                k,
            )

    def test_neck_leak_reduction_does_not_hide_large_valid_regions(self):
        if self.before is None:
            self.skipTest("No independently captured pre-fix baseline")
        self.assertLess(
            self.now["summary"]["neck_hair"]["leak"],
            self.before["summary"]["neck_hair"]["leak"] * 0.65,
        )
        self.assertLess(
            self.now["summary"]["neck_hair"]["missing"],
            self.before["summary"]["neck_hair"]["missing"] + 1000,
        )

    def test_native_glue_interop_and_source_welds(self):
        result = json.loads((ROOT / "artifacts/surfaces/native_glue.json").read_text())
        self.assertGreater(result["helpers"], 0)
        self.assertLess(result["max_px"], 0.0003)
        delta = ROOT / "artifacts/surfaces/scalar_compression_delta.json"
        if delta.exists():
            measured = json.loads(delta.read_text())
            self.assertEqual(measured["poses"], 264)
            self.assertLess(measured["helper_max_px"], 0.0503)
            self.assertLess(measured["visible_max_px"], 0.0003)

    def test_evaluated_model_matches_delivered_bytes(self):
        provenance = json.loads(
            (ROOT / "artifacts/surfaces/provenance.json").read_text()
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
        self.assertTrue(provenance["source_unchanged"])
