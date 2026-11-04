"""Opaque mask factorization must retain every source exactly once."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from popy.mask_batch import compile_plan, plan, validate_design


def inventory():
    sources = [
        {"id": name, "vertices": 3, "culling": True, "clips": []} for name in "ABCDEFG"
    ]
    return sources + [
        {"id": f"target_{i}", "vertices": 3, "culling": False, "clips": list(group)}
        for i, group in enumerate(["ABCD", "ABCE", "ABCF", "ABCG"])
    ]


class MaskBatchTests(unittest.TestCase):
    def test_shared_subexpressions_reuse_geometry_with_bounded_cost(self):
        source = inventory()
        snapshot = copy.deepcopy(source)
        design = plan(source, 15)
        self.assertEqual(design, plan(source, 15))
        self.assertEqual(design["added_vertices"], 15)
        self.assertEqual(design["input_mask_draws"], 16)
        self.assertEqual(design["output_mask_draws"], 8)
        self.assertEqual(design["helper_meshes"], 2)
        self.assertEqual(source, snapshot)
        validate_design(source, design)

    def test_zero_budget_and_empty_masks(self):
        for source in [inventory(), inventory()[:7]]:
            design = plan(source, 0)
            self.assertEqual(design["helper_meshes"], 0)
            self.assertEqual(design["input_mask_draws"], design["output_mask_draws"])

    def test_culling_is_not_combined_across_incompatible_triangles(self):
        source = inventory()
        source[0]["culling"] = False
        design = plan(source, 100)
        self.assertTrue(design["unions"])
        self.assertTrue(all("A" not in n["sources"] for n in design["unions"]))

    def test_missing_and_duplicate_membership_rejected(self):
        source = inventory()
        good = plan(source, 15)
        for replacement in [[], ["A", "B", "C", "D", "A"]]:
            bad = copy.deepcopy(good)
            bad["groups"][0]["after"] = replacement
            with self.assertRaisesRegex(ValueError, "lost or duplicated"):
                validate_design(source, bad)

    def test_cycles_and_stale_source_sizes_rejected(self):
        source = inventory()
        good = plan(source, 15)
        bad = copy.deepcopy(good)
        bad["unions"][0]["sources"][0] = bad["unions"][-1]["id"]
        with self.assertRaisesRegex(ValueError, "follow its operands"):
            validate_design(source, bad)
        source[0]["vertices"] += 1
        with self.assertRaisesRegex(ValueError, "vertex count"):
            validate_design(source, good)

    def test_ambiguous_input_rejected(self):
        source = inventory()
        source[-1]["clips"].append("A")
        with self.assertRaisesRegex(ValueError, "Duplicate mask"):
            plan(source)
        source = inventory()
        source[0]["id"] = "MaskBatch_0"
        with self.assertRaisesRegex(ValueError, "reserved"):
            plan(source)
        for budget, uses in [(-1, 3), (10, 1)]:
            with self.assertRaises(ValueError):
                plan(inventory(), budget, uses)
        for penalty in [-1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                plan(inventory(), 10, 3, penalty)
        source = inventory()
        source[-1]["clips"] = ["missing"]
        with self.assertRaisesRegex(ValueError, "Unknown mask source"):
            plan(source)

    def test_transparent_alpha_and_partial_visibility_domain_are_rejected(self):
        source = [{"id": "A", "vertices": 3, "culling": True, "clips": []}]
        design = plan(source, 0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGBA", (1, 1), (255, 255, 255, 128)).save(root / "texture.png")
            ir = {
                "meshes": [{"id": "A", "positions": [0] * 6, "axes": []}],
                "parameters": [],
                "textures": ["texture.png"],
            }
            path = root / "rig.json"
            path.write_text(json.dumps(ir))
            with self.assertRaisesRegex(ValueError, "fully opaque"):
                compile_plan(path, root / "separate", design)
            self.assertFalse((root / "separate").exists())
            ir["parameters"] = [{"id": "P", "min": -1, "max": 1}]
            ir["meshes"][0]["axes"] = [{"parameter": "P", "keys": [0, 1]}]
            path.write_text(json.dumps(ir))
            with self.assertRaisesRegex(ValueError, "constant source visibility"):
                compile_plan(path, root / "separate", design)

    def test_actual_export_native_and_pixel_evidence(self):
        from popy.mask_batch_review import BASE, fingerprints

        path = BASE / "report/metrics.json"
        if not path.exists():
            self.skipTest("Requires generated POPY assets and native/GPU regression")
        report = json.loads(path.read_text())

        self.assertEqual(report["inputs"], fingerprints())
        self.assertEqual(report["native"]["poses"], 1803)
        self.assertEqual(report["native"]["different_original_vertices"], 0)
        self.assertEqual(report["native"]["max_helper_copy_error_px"], 0)
        self.assertEqual(report["pixels"]["poses"], 362)
        self.assertEqual(report["pixels"]["different_channels"], 0)
        self.assertTrue(report["native"]["mask_union_triangle_multiplicities_verified"])
        self.assertEqual(report["model_stats"]["baseline"]["parameters"], 39)
        a, b = report["model_stats"]["baseline"], report["model_stats"]["candidate"]
        self.assertEqual(
            b["vertices"], a["vertices"] + report["mask_batch"]["added_vertices"]
        )
        for field in [
            "parameters",
            "motions",
            "expressions",
            "textures",
            "deformers",
            "max_deformer_depth",
        ]:
            self.assertEqual(a[field], b[field])


if __name__ == "__main__":
    unittest.main()
