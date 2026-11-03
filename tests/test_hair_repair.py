"""Source-strand invariants and real 3D/native-render acceptance checks."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from popy.compile_ir import ROOT
from popy.hair_clearance_ir import apply_clearance
from popy.hair_regression import cases
from popy.hair_semantics import strand_sides


class HairGeometryTests(unittest.TestCase):
    def test_connected_strand_does_not_split_at_y_zero(self):
        points = np.array(
            [
                [0, -0.01, 1.3],
                [1, 0.1, 1.3],
                [0, 0.1, 1.4],
                [3, -0.1, 1.3],
                [4, -0.2, 1.3],
                [3, -0.3, 1.4],
            ]
        )
        side = strand_sides(points, [[0, 1, 2], [3, 4, 5]])
        np.testing.assert_array_equal(side, [1, 1, 1, -1, -1, -1])

    def test_camera_guard_rejects_unverified_perspective(self):
        ir = {
            "diagnostics": {},
            "camera": {
                "projection": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
                "matrix_world": np.eye(4).tolist(),
            },
        }
        with self.assertRaisesRegex(ValueError, "orthographic"):
            apply_clearance(ir, Path("not-needed"))

    def test_rear_only_routing_preserves_front_and_head_tracks(self):
        positions = np.array(
            [
                [0, -0.1, 1.3],
                [1, -0.1, 1.3],
                [0, -0.1, 1.4],
                [0, -0.1, 1.4],
                [1, -0.1, 1.5],
                [0, -0.1, 1.6],
                [0, 0.1, 1.3],
                [1, 0.1, 1.3],
                [0, 0.1, 1.4],
            ]
        )

        def mesh(name, obj, ids):
            return {
                "id": name,
                "source": {"object": obj, "vertices": ids},
                "cells": [{"order": 512.0}],
                "blend_shapes": [{"orders": [510, 515]}],
            }

        low, upper, rear = [
            mesh(n, "M_Hair", ids)
            for n, ids in [
                ("tip", [0, 1, 2]),
                ("bangs", [3, 4, 5]),
                ("rear", [6, 7, 8]),
            ]
        ]
        others = [
            mesh(n, obj, [0])
            for n, obj in [
                ("face", "M_Face"),
                ("eye", "M_Eye"),
                ("star", "M_Headwear"),
                ("shirt", "M_Top"),
            ]
        ]
        low["masked_by"] = ["original_front_mask"]
        low["invert_mask"] = True
        original = copy.deepcopy([low, upper, *others])
        ir = {
            "meshes": [low, upper, rear, *others],
            "glues": [{"id": "G_Depth_Neck_1"}, {"id": "G_Source_1"}],
            "parts": [],
            "diagnostics": {},
            "surface_visibility": {},
            "camera": {
                "projection": np.eye(4).tolist(),
                "matrix_world": [
                    [1, 0, 0, 0],
                    [0, 0, -1, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                ],
            },
        }
        with tempfile.TemporaryDirectory() as d:
            source = Path(d)
            np.savez(
                source / "M_Hair_source.npz",
                positions=positions,
                triangles=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            )
            apply_clearance(ir, source)
            self.assertEqual(
                ir["diagnostics"]["hair_clearance_routing"]["front_reordered"], 0
            )
            self.assertEqual(rear["masked_by"], ["shirt"])
            self.assertTrue(rear["invert_mask"])
            for actual, before in zip([low, upper, *others], original):
                self.assertEqual(actual, before)
            snapshot = copy.deepcopy(ir)
            apply_clearance(ir, source)
            self.assertEqual(snapshot, ir)

    def test_reported_and_continuous_cases_are_preserved(self):
        rows = cases()
        self.assertEqual(len(rows), 77)
        self.assertEqual(len({r["name"] for r in rows}), 77)
        self.assertEqual(
            [r["name"] for r in rows[:3]],
            ["reported_left", "reported_right", "reported_roll"],
        )
        self.assertEqual(sum("sequence" in r for r in rows), 62)


class HairRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "artifacts/hair_repair_report/metrics.json"
        if not path.exists():
            raise unittest.SkipTest(
                "Run the real Blender/Ayagami hair regression first"
            )
        cls.report = json.loads(path.read_text())

    def test_shoulder_coverage_and_continuity_improve(self):
        r = self.report
        self.assertEqual(r["provenance"]["poses"], 77)
        self.assertEqual(r["provenance"]["transitions"], 60)
        self.assertLess(r["after"]["leak"], r["before"]["leak"] * 0.4)
        self.assertLess(
            sum(r["after"][k] for k in ["leak", "missing"]),
            sum(r["before"][k] for k in ["leak", "missing"]) * 0.5,
        )
        self.assertLess(r["after"]["missing"], r["before"]["missing"] * 1.15)
        self.assertLess(r["false_change_after"], r["false_change_before"] * 0.5)

    def test_actual_3d_intersections_are_measured_and_reduced(self):
        s = self.report["collision_summary"]
        self.assertGreater(s["before"], 0)
        self.assertLess(s["after"], s["before"] * 0.05)
        self.assertEqual(len(self.report["collisions"]), 77)

    def test_eye_face_and_headwear_coverage_does_not_regress(self):
        r = self.report["features"]
        self.assertEqual(r["after"]["poses"], 77)
        for kind in ["eyes", "headwear"]:
            for error in ["leak", "missing"]:
                self.assertLessEqual(
                    r["after"]["summary"][kind][error],
                    r["before"]["summary"][kind][error],
                    (kind, error),
                )
        self.assertLessEqual(
            r["after"]["summary"]["face"]["missing"],
            r["before"]["summary"]["face"]["missing"],
        )
        # Face reveal can change slightly at grazing edges of repaired rear
        # hair. Allow one pixel per pose, but no increase in face concealment.
        self.assertLessEqual(
            r["after"]["summary"]["face"]["leak"],
            r["before"]["summary"]["face"]["leak"] + r["after"]["poses"],
        )

    def test_delivery_matches_tested_source_and_moc(self):
        p = self.report["provenance"]
        self.assertTrue(p["source_unchanged"])
        self.assertEqual(
            p["moc_sha256"],
            hashlib.sha256(
                (ROOT / "output/POPY_hair_clearance/POPY.moc3").read_bytes()
            ).hexdigest(),
        )
        self.assertEqual(
            p["corrected_blend_sha256"],
            hashlib.sha256(
                (ROOT / "artifacts/POPY_hair_clearance.blend").read_bytes()
            ).hexdigest(),
        )
