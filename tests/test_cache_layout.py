import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from popy.cache_layout import (
    cache_order,
    first_use,
    overlap_paths,
    pack_positions,
    paired_permutations,
)
from popy.trajectory_pool import program


class LayoutTests(unittest.TestCase):
    def test_paths_share_exact_bits_and_reject_cycles(self):
        blocks = [b"A" * 8 + b"B" * 8, b"B" * 8 + b"A" * 8, b"B" * 8 + b"C" * 8]
        payload, positions, links = overlap_paths(
            blocks, [(8, 0, 1), (8, 1, 0), (8, 0, 2)]
        )
        self.assertEqual(links, 2)
        self.assertEqual(len(payload), 32)
        for i, block in enumerate(blocks):
            self.assertEqual(payload[positions[i] : positions[i] + len(block)], block)
        for overlap in [7, 24]:
            with self.subTest(overlap=overlap), self.assertRaises(ValueError):
                overlap_paths(blocks, [(overlap, 0, 1)])

    def test_first_use_keeps_unused_vertices(self):
        self.assertEqual(first_use([2, 0, 2, 4, 1, 0], 6), [2, 0, 4, 1, 3, 5])

    def test_cache_order_preserves_each_oriented_primitive(self):
        triangles = np.array(
            [[0, 1, 2], [5, 6, 7], [2, 1, 3], [7, 6, 8], [3, 1, 4], [9, 9, 9]]
        )
        indices, order = cache_order(triangles.ravel().tolist(), 10)
        self.assertEqual(sorted(order), list(range(len(triangles))))
        np.testing.assert_array_equal(
            np.array(indices).reshape(-1, 3), triangles[order]
        )

    def test_track_matching_distinguishes_signed_zero(self):
        a = np.array([[0.0, 3.0], [1.0, 2.0]], dtype="<f4")
        b = np.array([[-0.0, 3.0], [1.0, 2.0]], dtype="<f4")
        data = np.r_[a.ravel(), b.ravel()]

        def mesh(name, offset):
            return {
                "id": name,
                "positions": [0.0] * 4,
                "axes": [],
                "cells": [{"coordinate": [], "offset": offset}],
                "indices": [0, 1, 0],
                "deformer": "body",
                "blend_shapes": [],
            }

        meshes = [mesh("a", 0), mesh("b", 4)]
        orders, pairs = paired_permutations(meshes, data, program)
        self.assertEqual(pairs[0]["shared"], 1)
        self.assertEqual(a[orders["a"]][-1:].tobytes(), b[orders["b"]][:1].tobytes())

    def test_real_section_rewrite_borrows_subranges_without_changing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            counts = np.zeros(64, dtype="<u4")
            counts[9], counts[10] = 2, 10
            sections = {
                0: counts.tobytes(),
                70: struct.pack("<2I", 0, 6),
                71: np.array([1, 2, 3, 4, 5, 6, 5, 6, 7, 8], dtype="<f4").tobytes(),
            }
            raw = bytearray(1984)
            raw[:8] = b"MOC3\x05\0\0\0"
            for section, payload in sections.items():
                raw.extend(bytes(-len(raw) % 64))
                struct.pack_into("<I", raw, 64 + 4 * section, len(raw))
                raw.extend(payload)
            source, target = folder / "a.moc3", folder / "b.moc3"
            source.write_bytes(raw)
            report = pack_positions(
                source,
                target,
                {
                    "meshes": [
                        {"id": "a", "vertices": 3, "forms": [0]},
                        {"id": "b", "vertices": 2, "forms": [1]},
                    ]
                },
                [{"a": "a", "b": "b", "shared": 1}],
                folder / "report.json",
            )
            result = target.read_bytes()
            offsets = np.frombuffer(result, "<u4", 152, 64)
            refs = np.frombuffer(result, "<u4", 2, int(offsets[70]))
            for i, expected in enumerate([[1, 2, 3, 4, 5, 6], [5, 6, 7, 8]]):
                actual = np.frombuffer(
                    result, "<f4", len(expected), int(offsets[71]) + 4 * int(refs[i])
                )
                np.testing.assert_array_equal(actual, expected)
            self.assertEqual(report["pooled_position_bytes"], 32)
            self.assertEqual(
                json.loads((folder / "report.json").read_text())["path_links"], 1
            )


if __name__ == "__main__":
    unittest.main()
