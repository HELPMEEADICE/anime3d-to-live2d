"""Emotion choreography invariants and current MOC/Blender regression coverage."""

import hashlib
import json
import tempfile
import unittest

import numpy as np

from popy.animation import author_assets, curve_value, evaluate, expression_presets
from popy.compile_ir import ROOT
from popy.expression_design import catalogue, emotion_motions, mirrored


def parameters():
    sem = json.loads((ROOT / "config/semantics.json").read_text())
    cfg = json.loads((ROOT / "config/fullbody.json").read_text())
    return (
        [
            {"id": pid, **{k: p[k] for k in ["min", "max", "default"]}}
            for pid, p in sem["shape_parameters"].items()
        ]
        + [
            {"id": "ParamAngle" + a, "min": -r, "max": r, "default": 0}
            for a, r in zip("XYZ", [45, 30, 30])
        ]
        + [
            {"id": "ParamBodyAngle" + a, "min": -15, "max": 15, "default": 0}
            for a in "XYZ"
        ]
        + [{"id": "ParamEyeBall" + a, "min": -1, "max": 1, "default": 0} for a in "XY"]
        + [{"id": "ParamBreath", "min": 0, "max": 1, "default": 0}]
        + cfg["controls"]
    )


class ExpressionDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        from pathlib import Path

        cls.path = Path(cls.tmp.name)
        cls.p = parameters()
        cls.motions, cls.expressions = author_assets(cls.path, cls.p)
        cls.emotes = {
            name: json.loads((cls.path / refs[0]["File"]).read_text())
            for name, refs in cls.motions.items()
            if name.startswith("emote_")
        }

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_catalogue_has_twelve_distinct_families_and_three_views(self):
        self.assertEqual(len(self.emotes), 36)
        self.assertEqual(len(self.expressions), 24)
        self.assertEqual(len(catalogue()["actions"]), 12)
        self.assertEqual(
            set(self.emotes), {n for a in catalogue()["actions"] for n in a["motions"]}
        )
        fingerprints = set()
        for name, motion in self.emotes.items():
            if name.endswith("_C"):
                value = evaluate(self.p, motion, motion["Meta"]["Duration"] / 2)
                fingerprints.add(tuple(value[k] for k in sorted(value)))
        self.assertEqual(len(fingerprints), 12)

    def test_raw_cubic_samples_respect_actual_domains_without_clamping(self):
        defs = {p["id"]: p for p in self.p}
        for name, m in self.emotes.items():
            self.assertEqual(len(m["Curves"]), 39)
            for c in m["Curves"]:
                p = defs[c["Id"]]
                for t in np.linspace(0, m["Meta"]["Duration"], 191):
                    v = curve_value(c, float(t))
                    self.assertGreaterEqual(v, p["min"] - 1e-9, (name, c["Id"], t))
                    self.assertLessEqual(v, p["max"] + 1e-9, (name, c["Id"], t))

    def test_emotes_return_every_channel_to_neutral_and_keep_legs_still(self):
        neutral = {p["id"]: p["default"] for p in self.p}
        for name, m in self.emotes.items():
            self.assertEqual(evaluate(self.p, m, 0), neutral, name)
            self.assertEqual(evaluate(self.p, m, m["Meta"]["Duration"]), neutral, name)
            for c in m["Curves"]:
                if any(s in c["Id"] for s in ["Leg", "Knee", "Ankle"]):
                    self.assertEqual(curve_value(c, 1.5), neutral[c["Id"]])

    def test_wink_selects_one_eye_and_variant_mirrors_asymmetry(self):
        for variant, side, other in [("C", "L", "R"), ("R", "R", "L")]:
            m = self.emotes["emote_wink_" + variant]
            v = evaluate(self.p, m, m["Meta"]["Duration"] / 2)
            self.assertEqual(v["ParamEye" + side + "Open"], 0)
            self.assertEqual(v["ParamEye" + other + "Open"], 1)
        l = self.emotes["emote_question_L"]
        r = self.emotes["emote_question_R"]
        a = evaluate(self.p, l, l["Meta"]["Duration"] / 2)
        b = evaluate(self.p, r, r["Meta"]["Duration"] / 2)
        for pid in [
            "ParamAngleX",
            "ParamAngleZ",
            "ParamEyeBallX",
            "ParamBrowLY",
            "ParamBrowRForm",
            "ParamElbowL",
        ]:
            k, v = mirrored(pid, a[pid])
            self.assertAlmostEqual(b[k], v, places=8)

    def test_emotions_are_not_static_presets_with_a_head_turn(self):
        for kind in ["laugh", "cry", "surprise", "thinking", "wink"]:
            m = self.emotes["emote_" + kind + "_C"]
            facial = [
                c
                for c in m["Curves"]
                if any(x in c["Id"] for x in ["Mouth", "Eye", "Brow"])
            ]
            snapshots = [
                tuple(
                    round(curve_value(c, m["Meta"]["Duration"] * t), 5) for c in facial
                )
                for t in [0, 0.2, 0.4, 0.5, 0.7, 0.85, 1]
            ]
            self.assertGreaterEqual(len(set(snapshots)), 5, kind)

    def test_unavailable_controls_fail_before_export(self):
        p = [v for v in self.p if v["id"] != "ParamForearmLTurn"]
        with self.assertRaises((ValueError, KeyError)):
            list(emotion_motions(p, expression_presets()))


class ExpressionAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "artifacts/expression_actions/metrics.json"
        if not path.exists():
            raise unittest.SkipTest(
                "Run expression action render review with local POPY assets"
            )
        cls.metrics = json.loads(path.read_text())

    def test_current_moc_is_tested_and_remains_byte_identical(self):
        with (ROOT / "output/POPY_fullbody/POPY.moc3").open("rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        self.assertEqual(actual, self.metrics["moc_sha256"])
        if self.metrics["previous_moc_sha256"] is not None:
            self.assertEqual(actual, self.metrics["previous_moc_sha256"])

    def test_all_actions_have_three_rendered_midpoints_and_blender_coverage(self):
        emotes = [
            r for r in self.metrics["cases"] if r["name"].startswith("motion_emote_")
        ]
        self.assertEqual(len(emotes), 108)
        self.assertEqual(self.metrics["blender_clips"], 82)
        self.assertEqual(self.metrics["blender_reopened_checks"], 20)
        for r in emotes:
            self.assertGreater(r["silhouette_iou"], 0.98, r["name"])
            self.assertLess(r["mae_foreground"], 0.03, r["name"])
