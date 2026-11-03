"""Independent kinematics/animation invariants plus actual POPY GPU regression."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from popy.animation import author_assets, curve_value, evaluate, motion_asset, segments
from popy.compile_ir import ROOT, project
from popy.fullbody_ir import reframe
from popy.kinematics import Skeleton, garment_order_bias, rotation
from popy.native_skinning import NativeGraph
from popy.source_simplify import endpoint_collapse


class BodyGeometryTests(unittest.TestCase):
    def test_shell_precedence_excludes_horizontal_inner_cap(self):
        source = {
            "positions": np.array(
                [[0, 0, 0.37], [1, 0, 0.37], [0, 1, 0.37], [0, 0, 0.2]]
            ),
            "triangles": np.array([[0, 1, 2], [0, 1, 3]]),
        }
        config = {
            "runtime_order_bias": {"Shoe": 80},
            "runtime_inner_caps": {
                "Shoe": {
                    "rest_min_z": 0.36,
                    "rest_max_z": 0.39,
                    "min_abs_normal_z": 0.9,
                    "bias": 0,
                }
            },
        }
        mesh = {"source": {"object": "Shoe", "triangles": [0]}}
        self.assertEqual(garment_order_bias(mesh, source, config), 0)
        mesh["source"]["triangles"] = [1]
        self.assertEqual(garment_order_bias(mesh, source, config), 80)

    def test_native_matrix_glue_is_convex_and_matches_3d_products(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "forms.f32"
            binary.write_bytes(b"")
            ir = {
                "canvas": [128, 128],
                "parameters": [{"id": "Angle", "min": -45, "max": 45, "default": 0}],
                "body_warps": [],
            }
            graph = NativeGraph(ir, binary, {})
            projection = np.array([[100, 5, -3, 40], [2, 80, 7, -20]], float)
            gain = float(2**20)
            root = graph.node(
                np.r_[projection.T * gain, -projection.T * gain], "Camera"
            )
            graph.gains[root] = gain
            translation = np.eye(4)
            translation[:3, 3] = [0.3, -0.2, 0.8]
            start = graph.right(root, translation, "Translate")
            target = graph.rotate(start, "Angle", "Y")
            graph.stream.close()
            state = {
                m["id"]: np.array(m["positions"]).reshape(-1, 2) for m in graph.meshes
            }
            for angle in [13.25, -22.75, 45, 0]:
                for glue in graph.glues:
                    values = glue.get("intensities", [glue.get("intensity", 1)])
                    self.assertGreaterEqual(min(values), 0)
                    self.assertLessEqual(max(values), 1)
                    intensity = (
                        np.interp(angle, glue["axes"][0]["keys"], values)
                        if glue.get("axes")
                        else values[0]
                    )
                    for a, b, wa, wb in glue["pairs"]:
                        self.assertEqual((wa, wb), (0, 1))
                        pa = state[glue["mesh_a"]][a]
                        pb = state[glue["mesh_b"]][b]
                        state[glue["mesh_b"]][b] = pb + (pa - pb) * intensity
                expected = projection @ translation @ rotation("Y", angle)
                actual = state[target][:4].T / graph.gains[target]
                np.testing.assert_allclose(actual, expected, atol=0.004)
                np.testing.assert_allclose(
                    state[target][4:], -state[target][:4], atol=1e-6
                )

    def test_endpoint_simplification_keeps_source_uv_and_pose_bounds(self):
        vertices = np.array(
            [[x * 0.0004, y * 0.0004, 0] for y in range(5) for x in range(5)]
        )
        tri = []
        for y in range(4):
            for x in range(4):
                a = y * 5 + x
                tri.extend([[a, a + 1, a + 5], [a + 1, a + 6, a + 5]])
        tri = np.array(tri)
        source = {"positions": vertices, "triangles": tri, "uv": vertices[tri, :2]}
        candidate = np.ones(len(vertices), bool)
        pinned = np.zeros(len(vertices), bool)
        pinned[0] = True
        result, valid, info = endpoint_collapse(
            source,
            candidate,
            np.array([vertices, vertices + [2, 3, 4]]),
            pinned,
            error_m=0.00065,
        )
        self.assertGreater(info["collapsed_vertices"], 0)
        self.assertEqual(result["vertex_remap"][0], 0)
        self.assertLessEqual(info["max_sampled_source_displacement_m"], 0.00065)
        self.assertTrue(all(len(set(t)) == 3 for t in result["triangles"][valid]))
        np.testing.assert_allclose(
            result["uv"][valid], vertices[result["triangles"][valid], :2]
        )
        _, _, rejected = endpoint_collapse(
            source,
            candidate,
            np.array([vertices, vertices * 100]),
            pinned,
            error_m=0.00065,
        )
        self.assertEqual(rejected["collapsed_vertices"], 0)

    def test_hierarchical_skinning_and_normalized_weights(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            child = np.eye(4)
            child[0, 3] = 1
            bones = [
                {"name": "Root", "parent": None, "matrix_local": np.eye(4).tolist()},
                {"name": "Child", "parent": "Root", "matrix_local": child.tolist()},
            ]
            (path / "skeleton.json").write_text(json.dumps(bones))
            (path / "Mesh_weights.json").write_text(
                json.dumps([[["Root", 2], ["Child", 2]], []])
            )
            np.savez(
                path / "Mesh_source.npz", positions=np.array([[1.5, 0, 0], [9, 8, 7]])
            )
            controls = [
                {
                    "id": "Parent",
                    "default": 0,
                    "drives": [{"bone": "Root", "axis": "Z"}],
                },
                {
                    "id": "Elbow",
                    "default": 0,
                    "drives": [{"bone": "Child", "axis": "Z"}],
                },
            ]
            s = Skeleton(path, {"controls": controls})
            a = s.evaluate("Mesh", {"Parent": 90, "Elbow": 90})
            np.testing.assert_allclose(a, [[-0.25, 1.25, 0], [9, 8, 7]], atol=1e-12)
            self.assertTrue(s.object("Mesh")[2][0].all())
            self.assertFalse(s.object("Mesh")[2][1].any())

    def test_nonexistent_bone_is_not_a_silent_control(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "skeleton.json").write_text(
                json.dumps(
                    [
                        {
                            "name": "Root",
                            "parent": None,
                            "matrix_local": np.eye(4).tolist(),
                        }
                    ]
                )
            )
            with self.assertRaisesRegex(ValueError, "Missing source bones"):
                Skeleton(
                    p,
                    {
                        "controls": [
                            {
                                "id": "Bad",
                                "default": 0,
                                "drives": [{"bone": "missing", "axis": "X"}],
                            }
                        ]
                    },
                )

    def test_rotation_is_rigid_and_axis_checked(self):
        for axis in "XYZ":
            r = rotation(axis, 37.5)[:3, :3]
            np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-15)
            self.assertAlmostEqual(np.linalg.det(r), 1)
        with self.assertRaises(ValueError):
            rotation("bad", 1)

    def test_camera_reframe_preserves_source_projection(self):
        size = np.array([768, 1024])
        scale = 0.68
        world = np.array(
            [[1, 0, 0, 0], [0, 0, -1, -3], [0, 1, 0, 1.31], [0, 0, 0, 1]], float
        )
        camera = {
            "matrix_world": world.tolist(),
            "projection": np.diag([2 / (scale * 0.75), 2 / scale, -0.002, 1]).tolist(),
            "canvas": size.tolist(),
        }
        ir = {
            "canvas": size.tolist(),
            "camera": camera,
            "body_rotation": {"pivot": [384, 751]},
            "root_warp": {"cells": []},
            "head_warp": {"cells": []},
            "body_warps": [],
            "head_rotation": {"cells": []},
            "meshes": [],
        }
        points = np.array([[0, 0, 0], [0.3, 0.1, 1.5], [-0.2, -0.1, 0.8]])
        before = project(points, camera) * size
        t = reframe(ir, {"camera": {"center": [0, 0, 0.82], "ortho_scale": 1.85}})
        expected = before * t["scale"] + t["translation"]
        np.testing.assert_allclose(
            project(points, ir["camera"]) * size, expected, atol=8e-5
        )

    def test_bezier_inverts_time_not_value(self):
        curve = {"Segments": [0, 0, 1, 0.1, 0, 0.2, 1, 1, 1]}
        # u=.5 has time .2375 and value .5; t/duration would be incorrect.
        self.assertAlmostEqual(curve_value(curve, 0.2375), 0.5, places=10)
        self.assertEqual(curve_value(curve, -1), 0)
        self.assertEqual(curve_value(curve, 2), 1)

    def test_linear_stepped_and_inverse_stepped(self):
        self.assertEqual(curve_value({"Segments": [0, 2, 0, 1, 4]}, 0.25), 2.5)
        self.assertEqual(curve_value({"Segments": [0, 2, 2, 1, 4]}, 0.25), 2)
        self.assertEqual(curve_value({"Segments": [0, 2, 3, 1, 4]}, 0.25), 4)
        for bad in [[0, 0, 1, 0, 0], [0, 0, 8, 1, 2], [0, 0, 0, 0, 2]]:
            with self.assertRaises(ValueError):
                list(segments({"Segments": bad}))

    def test_loop_expression_blends_clamp_and_unknown(self):
        p = [{"id": "A", "min": 0, "max": 1, "default": 0.2}]
        m = motion_asset({"A": [[0, 0], [1, 1]]}, 1, True)
        self.assertEqual(evaluate(p, m, 1)["A"], 0)
        exp = {"Parameters": [{"Id": "A", "Value": 0.8, "Blend": "Overwrite"}]}
        self.assertAlmostEqual(
            evaluate(p, expression=exp, expression_weight=0.5)["A"], 0.5
        )
        exp["Parameters"][0].update(Value=2, Blend="Multiply")
        self.assertAlmostEqual(evaluate(p, expression=exp)["A"], 0.4)
        exp["Parameters"][0].update(Value=2, Blend="Add")
        self.assertEqual(evaluate(p, expression=exp)["A"], 1)
        exp["Parameters"][0]["Id"] = "unknown"
        with self.assertRaises(ValueError):
            evaluate(p, expression=exp)

    def test_authored_assets_have_valid_cubism_counts_and_loop_ends(self):
        cfg = json.loads((ROOT / "config/fullbody.json").read_text())
        ids = {
            "ParamAngleX": 0,
            "ParamAngleY": 0,
            "ParamAngleZ": 0,
            "ParamBreath": 0,
            "ParamMouthForm": 0,
            "ParamMouthOpenY": 0,
            "ParamEyeLOpen": 1,
            "ParamEyeROpen": 1,
            "ParamEyeLSmile": 0,
            "ParamEyeRSmile": 0,
            "ParamEyeBallX": 0,
            "ParamEyeBallY": 0,
            "ParamBrowLY": 0,
            "ParamBrowRY": 0,
            "ParamBrowLForm": 0,
            "ParamBrowRForm": 0,
        }
        p = [
            {"id": k, "default": v, "min": -180, "max": 180} for k, v in ids.items()
        ] + cfg["controls"]
        with tempfile.TemporaryDirectory() as d:
            motions, expressions = author_assets(d, p)
            self.assertEqual((len(motions), len(expressions)), (15, 16))
            for refs in motions.values():
                m = json.loads((Path(d) / refs[0]["File"]).read_text())
                sg = pt = 0
                for c in m["Curves"]:
                    seq = list(segments(c))
                    sg += len(seq)
                    pt += 1 + sum(len(q) - 1 for _, q in seq)
                    self.assertEqual(seq[-1][1][-1][0], m["Meta"]["Duration"])
                    if m["Meta"]["Loop"]:
                        self.assertEqual(c["Segments"][1], seq[-1][1][-1][1])
                self.assertEqual(
                    (sg, pt),
                    (m["Meta"]["TotalSegmentCount"], m["Meta"]["TotalPointCount"]),
                )
                evaluate(p, m, m["Meta"]["Duration"] * 0.371)
            for e in expressions:
                evaluate(p, expression=json.loads((Path(d) / e["File"]).read_text()))


class FullBodyAssetTests(unittest.TestCase):
    def test_stocking_front_surface_covers_leg_skin(self):
        cases = {c["name"]: c for c in self.report["cases"]}
        for name in [
            "natural",
            "motion_leg_lift_L_middle",
            "ParamLegLSwing_max",
            "ParamLegRSwing_min",
            "stress_all",
        ]:
            c = cases[name]
            self.assertGreater(c["stocking_white_reference_pixels"], 300, name)
            self.assertGreater(c["stocking_white_recall_1px"], 0.93, name)

    def test_stocking_inner_cap_does_not_cover_leg_skin(self):
        cases = {c["name"]: c for c in self.report["cases"]}
        for name in ["natural", "motion_leg_lift_L_middle", "stress_all"]:
            c = cases[name]
            self.assertGreater(c["stocking_cap_skin_reference_pixels"], 20, name)
            self.assertGreater(c["stocking_cap_skin_recall_1px"], 0.94, name)

    @classmethod
    def setUpClass(cls):
        cls.base = ROOT / "artifacts/fullbody"
        if not (cls.base / "report/metrics.json").exists():
            raise unittest.SkipTest(
                "Run python -m popy.fullbody all with local POPY assets"
            )
        cls.report = json.loads((cls.base / "report/metrics.json").read_text())
        cls.model = ROOT / "output/POPY_fullbody"

    def test_native_model_parameters_and_real_asset_references(self):
        runtime = json.loads((self.base / "runtime/runtime_report.json").read_text())
        self.assertEqual(len(runtime["parameters"]), 37)
        defs = {p["id"] for p in runtime["parameters"]}
        config = json.loads((ROOT / "config/fullbody.json").read_text())
        self.assertTrue({c["id"] for c in config["controls"]} <= defs)
        with (self.model / "POPY.moc3").open("rb") as f:
            self.assertEqual(f.read(4), b"MOC3")
        manifest = json.loads((self.model / "POPY.model3.json").read_text())
        self.assertEqual(len(manifest["FileReferences"]["Motions"]), 15)
        self.assertEqual(len(manifest["FileReferences"]["Expressions"]), 16)
        for refs in manifest["FileReferences"]["Motions"].values():
            for ref in refs:
                m = json.loads((self.model / ref["File"]).read_text())
                self.assertTrue(all(c["Id"] in defs for c in m["Curves"]))
        for ref in manifest["FileReferences"]["Expressions"]:
            e = json.loads((self.model / ref["File"]).read_text())
            self.assertTrue(all(c["Id"] in defs for c in e["Parameters"]))

    def test_source_matches_blender_and_sparse_native_errors_are_bounded(self):
        probes = [r for r in self.report["cases"] if "articulated_rmse_px" in r]
        self.assertGreaterEqual(len(probes), 14)
        self.assertLess(max(r["skinning_max_error_m"] for r in probes), 1e-6)
        for r in probes:
            self.assertLess(r["articulated_rmse_px"], 0.1, r["name"])
            self.assertLess(r["articulated_max_px"], 1.0, r["name"])
        for name in ["natural", "source_T_pose"]:
            r = next(r for r in probes if r["name"] == name)
            self.assertLess(r["articulated_max_px"], 0.05)

    def test_hand_hinges_mirror_and_baked_blender_actions_match(self):
        cfg = json.loads((ROOT / "config/fullbody.json").read_text())
        skeleton = Skeleton(ROOT / "artifacts/samples", cfg)
        skin = skeleton.skin_matrices({"ParamHandLPose": 1, "ParamHandRPose": 1})
        mirror = np.diag([-1, 1, 1, 1])
        for name, index in skeleton.index.items():
            if name.endswith("_Left") and "Finger" in name:
                other = skeleton.index[name[:-5] + "_Right"]
                self.assertLess(
                    np.max(np.abs(mirror @ skin[index] @ mirror - skin[other])),
                    1e-3,
                    name,
                )
        actual = json.loads((self.base / "blender_verification.json").read_text())
        self.assertEqual(actual["clips"], 31)
        self.assertEqual(actual["driven_bones"], 46)
        self.assertEqual(len(actual["evaluated_checks"]), 8)
        self.assertLess(max(c["max_error_m"] for c in actual["evaluated_checks"]), 2e-6)

    def test_head_baseline_and_source_simplification_preserved(self):
        for c in self.report["head_baseline_preservation"]:
            self.assertGreater(c["unchanged_meshes"], 3000)
            self.assertLess(c["vertex_max_px"], 0.05)
        simplification = self.report["statistics"]["source_simplification"]["M_Body"]
        self.assertGreater(simplification["collapsed_vertices"], 0)
        self.assertLessEqual(
            simplification["max_sampled_source_displacement_m"], 0.0015
        )
        rig = json.loads((self.base / "rig/POPY.rig.json").read_text())
        inventory = {
            m["id"]: m
            for m in json.loads(
                (self.base / "runtime/model_inventory.json").read_text()
            )
        }
        for m in rig["meshes"]:
            if m.get("articulated"):
                self.assertEqual(
                    inventory[m["id"]]["culling"],
                    m["source"]["object"] not in {"M_Decorate", "M_Bracelet"},
                    m["id"],
                )
        operators = [g for g in rig["glues"] if g["id"].startswith("NativeG_")]
        self.assertGreater(len(operators), 1000)
        for g in operators:
            values = g.get("intensities", [g.get("intensity", 1)])
            self.assertTrue(min(values) >= 0 and max(values) <= 1, g["id"])
            self.assertTrue(all(p[2:] == [0, 1] for p in g["pairs"]), g["id"])
        packed = self.report["position_packing_verification"]
        self.assertLess(packed["packed_bytes"], packed["unpacked_bytes"] * 0.6)
        self.assertTrue(
            all(
                r["different_channels"] == 0 and r["all_vertices_bit_exact"]
                for r in packed["render_checks"]
            )
        )

    def test_every_limb_control_changes_actual_gpu_pixels(self):
        cfg = json.loads((ROOT / "config/fullbody.json").read_text())

        def image(name):
            return np.array(
                Image.open(self.base / "runtime" / (name + ".png")).convert("RGBA")
            )

        for c in cfg["controls"]:
            self.assertGreater(
                np.count_nonzero(image(c["id"] + "_min") != image(c["id"] + "_max")),
                5,
                c["id"],
            )

    def test_timeline_loops_and_references_are_complete(self):
        self.assertGreaterEqual(self.report["pose_count"], 190)
        for c in self.report["cases"]:
            self.assertTrue((self.base / "reference" / (c["name"] + ".png")).exists())
            self.assertTrue((self.base / "runtime" / (c["name"] + ".png")).exists())
        for name, last in [("wave_L", 40), ("walk_in_place", 20)]:
            a = np.array(
                Image.open(self.base / "runtime" / ("sequence_" + name + "_000.png")),
                float,
            )
            b = np.array(
                Image.open(
                    self.base
                    / "runtime"
                    / ("sequence_" + name + "_" + str(last).zfill(3) + ".png")
                ),
                float,
            )
            self.assertLess(np.abs(a - b).mean(), 0.01, name)

    def test_actual_model_hash_and_original_source_integrity(self):
        with (self.model / "POPY.moc3").open("rb") as f:
            self.assertEqual(
                hashlib.file_digest(f, "sha256").hexdigest(), self.report["moc_sha256"]
            )
        self.assertEqual(
            hashlib.sha256(
                (ROOT / "artifacts/POPY_source.blend").read_bytes()
            ).hexdigest(),
            "b4d94421c820d52db5d66da08302062328035a27af4fac378b818072faacbaed",
        )


if __name__ == "__main__":
    unittest.main()
