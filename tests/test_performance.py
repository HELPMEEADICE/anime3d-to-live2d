"""Compaction must preserve ordered floating-point programs, not just poses."""

import copy
import hashlib
import json
import unittest

import numpy as np

from popy.compile_ir import ROOT
from popy.performance_opt import compact_glues, factor_native_constants


def execute(glues, vertices, intensity=None):
    vertices = {k: v.copy() for k, v in vertices.items()}
    for g in glues:
        t = np.float32(g.get("intensity", 1) if not g.get("axes") else intensity)
        for i, j, wa, wb in g["pairs"]:
            a, b = vertices[g["mesh_a"]][i].copy(), vertices[g["mesh_b"]][j].copy()

            def lerp(a, b, t):
                return a * (np.float32(1) - t) + b * t

            vertices[g["mesh_a"]][i] = lerp(a, lerp(a, b, np.float32(wa)), t)
            vertices[g["mesh_b"]][j] = lerp(b, lerp(b, a, np.float32(wb)), t)
    return vertices


class CompactionTests(unittest.TestCase):
    def test_factorization_is_limited_to_native_constant_one_way_programs(self):
        native = {
            "id": "NativeG_0",
            "mesh_a": "A",
            "mesh_b": "B",
            "pairs": [[0, 0, 0, 1]],
            "intensity": 0.125,
        }
        face = {**native, "id": "G_Depth_0"}
        animated = {
            **native,
            "id": "NativeG_1",
            "axes": [{"parameter": "P", "keys": [0, 1]}],
            "intensities": [0, 1],
        }
        original = copy.deepcopy([native, face, animated])
        result, count = factor_native_constants(original)
        self.assertEqual(count, 1)
        self.assertEqual(result[0]["pairs"], [[0, 0, 0.0, 0.125]])
        self.assertEqual(result[0]["intensity"], 1)
        self.assertEqual(result[1:], original[1:])
        self.assertEqual(original, [native, face, animated])

    def test_source_drift_is_not_mislabeled_as_bit_exact(self):
        rng = np.random.default_rng(8042)
        vertices = {k: rng.uniform(-1e8, 1e8, (32, 2)).astype("f4") for k in ["A", "B"]}
        glues = [
            {
                "id": f"NativeG_{i}",
                "mesh_a": "A",
                "mesh_b": "B",
                "pairs": [[i % 32, i % 32, 0, 1]],
                "intensity": float(t),
            }
            for i, t in enumerate(rng.random(4000))
        ]
        factored, _ = factor_native_constants(glues)
        before, after = execute(glues, vertices), execute(factored, vertices)
        self.assertFalse(np.array_equal(before["A"], after["A"]))
        np.testing.assert_array_equal(after["A"], vertices["A"])

    def test_float_rounding_and_program_order_preserved(self):
        rng = np.random.default_rng(8042)
        vertices = {k: rng.uniform(-1e8, 1e8, (32, 2)).astype("f4") for k in ["A", "B"]}
        glues = [
            {
                "id": f"g{i}",
                "mesh_a": "A",
                "mesh_b": "B",
                "pairs": [[i % 32, (i * 7) % 32, 0, 1]],
                "intensity": float(t),
            }
            for i, t in enumerate(np.r_[0.0, 1.0, rng.random(4000)])
        ]
        snapshot = copy.deepcopy(glues)
        compacted, report = compact_glues(glues)
        self.assertEqual(report["constant_lerps_folded"], 0)
        self.assertEqual(report["input_pairs"], report["output_pairs"])
        self.assertEqual(glues, snapshot)
        for k, v in execute(glues, vertices).items():
            np.testing.assert_array_equal(
                v.view("u4"), execute(compacted, vertices)[k].view("u4")
            )

    def test_intervening_operation_is_a_barrier(self):
        glues = [
            {"id": "a", "mesh_a": "A", "mesh_b": "B", "pairs": [[0, 0, 0, 1]]},
            {"id": "b", "mesh_a": "B", "mesh_b": "A", "pairs": [[0, 0, 0, 1]]},
            {"id": "c", "mesh_a": "A", "mesh_b": "B", "pairs": [[0, 0, 0, 1]]},
        ]
        self.assertEqual(len(compact_glues(glues)[0]), 3)

    def test_independent_operations_batch_across_other_meshes(self):
        glues = [
            {"id": "a", "mesh_a": "A", "mesh_b": "B", "pairs": [[0, 0, 0, 1]]},
            {"id": "b", "mesh_a": "C", "mesh_b": "D", "pairs": [[0, 0, 0, 1]]},
            {"id": "c", "mesh_a": "A", "mesh_b": "B", "pairs": [[1, 1, 0, 1]]},
        ]
        result, _ = compact_glues(glues)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["pairs"], [[0, 0, 0, 1], [1, 1, 0, 1]])

    def test_read_before_write_hazard_preserved(self):
        glues = [
            {"id": "a", "mesh_a": "A", "mesh_b": "B", "pairs": [[0, 0, 0, 1]]},
            {"id": "b", "mesh_a": "C", "mesh_b": "A", "pairs": [[0, 0, 0, 1]]},
            {"id": "c", "mesh_a": "A", "mesh_b": "B", "pairs": [[0, 1, 0, 1]]},
        ]
        self.assertEqual(len(compact_glues(glues)[0]), 3)

    def test_distinct_animated_fields_are_not_approximated(self):
        axes = [{"parameter": "P", "keys": [-1, 0, 1]}]
        a = {
            "id": "a",
            "mesh_a": "A",
            "mesh_b": "B",
            "pairs": [[0, 0, 0, 1]],
            "axes": axes,
            "intensities": [0, 0.3, 1],
        }
        b = {**a, "id": "b", "intensities": [0, 0.30001, 1]}
        result, _ = compact_glues([a, b, b])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[1]["intensities"], b["intensities"])

    def test_noncanonical_pairs_and_out_of_domain_not_factored(self):
        for wa, wb, t in [(0.5, 0.5, 0.3), (0, 1, 1.5), (0, 0.3, 0.5)]:
            glue = {
                "id": "a",
                "mesh_a": "A",
                "mesh_b": "B",
                "pairs": [[0, 0, wa, wb]],
                "intensity": t,
            }
            result, r = compact_glues([glue])
            self.assertEqual(result, [glue])
            self.assertEqual(r["constant_lerps_folded"], 0)


class PerformanceAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "artifacts/perf_opt/report/metrics.json"
        if not path.exists():
            raise unittest.SkipTest(
                "Run real-MOC performance review with local POPY assets"
            )
        cls.report = json.loads(path.read_text())

    def test_every_proved_mode_matches_the_actual_model(self):
        for label, folder in [
            ("baseline", "POPY_fullbody"),
            ("exact", "POPY_optimized_exact"),
            ("precision", "POPY_fullbody_optimized"),
        ]:
            with (ROOT / "output" / folder / "POPY.moc3").open("rb") as stream:
                self.assertEqual(
                    hashlib.file_digest(stream, "sha256").hexdigest(),
                    self.report[label + "_moc_sha256"],
                )
        self.assertEqual(self.report["exact_evaluation"]["different_vertices"], 0)
        self.assertEqual(self.report["exact_evaluation"]["poses"], 1803)
        self.assertEqual(self.report["precision_evaluation"]["poses"], 1803)

    def test_precision_improves_blender_geometry_without_changing_head(self):
        r = self.report
        self.assertEqual(r["image_poses"], 362)
        self.assertEqual(r["geometry_poses"], 81)
        self.assertEqual(r["face_hair_headwear_delta_max_px"], 0)
        self.assertTrue(r["all_geometry_rmse_improved"])
        self.assertTrue(r["all_geometry_max_improved"])
        self.assertLess(r["after_geometry_max_px"], r["before_geometry_max_px"])
        self.assertTrue(all(c["exact_changed_rgba_channels"] == 0 for c in r["cases"]))


if __name__ == "__main__":
    unittest.main()
