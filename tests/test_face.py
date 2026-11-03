"""Facial surface geometry and actual pinned native-runtime acceptance."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from popy.compile_ir import ROOT
from popy.depth_masks import plane
from popy.face_clearance import (
    convert,
    order_tracked_surfaces,
    planar_groups,
    tracked_partition,
)
from popy.face_regression import cases
from popy.mask_union import factor_inherited_masks, pack_local_sources


class FaceGeometryTests(unittest.TestCase):
    def test_plane_depth_survives_rigid_head_body_rotation(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        triangle = np.array([[0, 1, 2]])
        n, t = plane(points, triangle)
        angle = 0.72
        r = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        shift = np.array([2.0, 4.0, 1.0])
        nn, tt = plane(points @ r.T + shift, triangle)
        sample = np.array([[0.2, 0.1, 0.2], [0.2, -0.1, 0.2]])
        np.testing.assert_allclose(
            sample @ n - t, (sample @ r.T + shift) @ nn - tt, atol=1e-12
        )

    def test_partition_preserves_all_source_faces_and_rejects_folds(self):
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        triangles = np.array([[0, 1, 2], [1, 0, 3]])
        groups = planar_groups(points, triangles)
        self.assertEqual(sorted(t for g in groups for t in g), [0, 1])
        self.assertEqual(len(groups), 2)

    def test_proxy_tracks_only_final_original_corners(self):
        mesh = {
            "id": "A",
            "source": {"object": "head", "vertices": [0, 1, 2, 3], "triangles": [5, 6]},
            "indices": [0, 1, 2, 1, 0, 3],
            "positions": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            "uvs": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            "cells": [{"offset": 0, "order": 42, "coordinate": []}],
            "axes": [],
            "blend_shapes": [],
            "masked_by": [],
            "opacity": 1.0,
        }
        ir = {"meshes": [mesh], "glues": []}
        rest = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "forms.f32"
            original = np.arange(8, dtype="<f4").tobytes()
            binary.write_bytes(original)
            tracked_partition(ir, [mesh], rest, binary, {}, {5})
            self.assertEqual(binary.read_bytes()[: len(original)], original)
        children = ir["meshes"][1:]
        self.assertEqual(len(children), 2)
        self.assertEqual(
            sorted(t for c in children for t in c["source"]["triangles"]), [5, 6]
        )
        self.assertTrue(mesh["mask_only"])
        self.assertEqual(mesh["indices"], [0, 1, 2, 1, 0, 3])
        self.assertEqual(len(ir["glues"]), 2)
        for g in ir["glues"]:
            self.assertTrue(all(p[2:] == [0.0, 1.0] for p in g["pairs"]))

    def test_independent_dense_cases_and_transitions(self):
        rows = cases()
        self.assertEqual(len(rows), 218)
        self.assertEqual(len({r["name"] for r in rows}), 218)
        self.assertEqual(sum("sequence" in r for r in rows), 63)
        self.assertEqual(rows[0]["parameters"]["ParamAngleX"], 45)
        self.assertEqual(rows[1]["parameters"]["ParamAngleX"], -11)

    def test_baseline_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaisesRegex(ValueError, "baseline"):
                convert(path / "POPY.rig.json", path)

    def test_subdivision_preserves_tie_order_against_headwear(self):
        ir = {
            "meshes": [
                {"id": "hair"},
                {"id": "star"},
                {"id": "hair_F0", "face_partition_parent": "hair"},
                {"id": "mask", "mask_only": True, "face_partition_parent": "hair"},
            ]
        }
        order_tracked_surfaces(ir)
        self.assertEqual(
            [m["id"] for m in ir["meshes"]], ["hair", "hair_F0", "star", "mask"]
        )

    def test_exact_opaque_mask_union_and_local_pack(self):
        import copy

        source = {
            "positions": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            "uvs": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            "indices": [0, 1, 2],
            "texture": 0,
            "source": {"object": "head", "vertices": [0, 1, 2], "triangles": [0]},
        }

        def rig():
            return {
                "textures": ["page.png"],
                "glues": [],
                "meshes": [
                    {**copy.deepcopy(source), "id": "A"},
                    {**copy.deepcopy(source), "id": "B"},
                    {
                        **copy.deepcopy(source),
                        "id": "parent",
                        "masked_by": ["A", "B"],
                        "invert_mask": True,
                    },
                    *[
                        {
                            **copy.deepcopy(source),
                            "id": n,
                            "face_partition_parent": "parent",
                            "face_depth_corrected": True,
                            "masked_by": ["A", "B"],
                            "invert_mask": True,
                        }
                        for n in ["C", "D"]
                    ],
                ],
            }

        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "forms.f32"
            binary.write_bytes(np.zeros(6, dtype="<f4").tobytes())
            Image.new("RGBA", (2, 2), (255, 255, 255, 255)).save(
                binary.parent / "page.png"
            )
            ir = rig()
            result = factor_inherited_masks(ir, binary, minimum_sources=2)
            self.assertEqual(result["unions"], 1)
            self.assertEqual(result["replaced_targets"], 2)
            self.assertEqual(ir["meshes"][-1]["indices"], [0, 1, 2, 3, 4, 5])
            self.assertTrue(
                all(p[2:] == [0.0, 1.0] for g in ir["glues"] for p in g["pairs"])
            )
            ir = rig()
            result = pack_local_sources(ir, binary)
            self.assertEqual(result["targets"], 2)
            self.assertEqual(result["copied_vertices"], 12)
            count = len(ir["meshes"])
            self.assertEqual(pack_local_sources(ir, binary)["targets"], 0)
            self.assertEqual(len(ir["meshes"]), count)
            Image.new("RGBA", (2, 2), (255, 255, 255, 128)).save(
                binary.parent / "page.png"
            )
            with self.assertRaisesRegex(ValueError, "opaque"):
                factor_inherited_masks(rig(), binary, minimum_sources=2)


class FaceRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "artifacts/face_repair/report/metrics.json"
        if not path.exists():
            raise unittest.SkipTest(
                "Run the independent Blender/native face regression first"
            )
        cls.result = json.loads(path.read_text())

    def test_actual_native_visibility_improves(self):
        a, b = [self.result[k]["summary"] for k in ["before", "after"]]
        self.assertLess(b["skin"]["leak"], a["skin"]["leak"])
        self.assertLess(b["hair"]["leak"], a["hair"]["leak"])
        self.assertLess(b["nose"]["missing"], a["nose"]["missing"] * 0.2)
        self.assertEqual(self.result["after"]["poses"], 218)

    def test_preserved_source_and_actual_glue(self):
        self.assertTrue(self.result["invariants"]["source_geometry_and_glue_preserved"])
        self.assertTrue(self.result["invariants"]["original_masks_preserved"])
        self.assertLess(self.result["native_glue"]["max_px"], 0.01)
        self.assertLess(self.result["proxy_error_px"], 0.001)
        self.assertTrue(self.result["exporter_baseline"]["byte_identical"])

    def test_report_matches_current_delivered_model(self):
        import hashlib

        path = ROOT / "output/POPY_face_fixed/POPY.moc3"
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(), self.result["moc_sha256"]
        )

    def test_reported_isolated_defects_are_removed(self):
        for spot in self.result["reported_spots"].values():
            self.assertGreater(spot["before"], 0)
            self.assertEqual(spot["after"], 0)

    def test_oral_source_id_error_reduces(self):
        a, b = [self.result["oral_ids"][k] for k in ["before", "after"]]
        self.assertEqual(b["poses"], 38)
        self.assertLess(b["summary"]["missing"], a["summary"]["missing"] * 0.1)
        self.assertLess(
            b["summary"]["missing"] + b["summary"]["leak"],
            (a["summary"]["missing"] + a["summary"]["leak"]) * 0.25,
        )

    def test_eye_headwear_and_collar_regressions(self):
        f = self.result["features"]
        for kind in ["eyes", "headwear"]:
            for metric in ["missing", "leak"]:
                self.assertEqual(
                    f["before"]["summary"][kind][metric],
                    f["after"]["summary"][kind][metric],
                )
        c = self.result["collar"]
        for metric in ["missing", "leak"]:
            self.assertLessEqual(
                c["after"]["summary"][metric], c["before"]["summary"][metric]
            )
