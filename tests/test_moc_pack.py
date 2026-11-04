"""Synthetic section fixtures test exact storage transformations, not compatibility."""

import hashlib
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from popy.moc_pack import pack


def fixture():
    counts = [0] * 64
    for i, n in {2: 1, 3: 1, 4: 1, 7: 1, 8: 1, 9: 2, 10: 32, 23: 2, 24: 2}.items():
        counts[i] = n
    sections = {
        0: struct.pack("<64I", *counts),
        60: struct.pack("<I", 0),
        70: struct.pack("<2I", 0, 16),
    }
    # Noncanonical NaN payloads and signed zero must survive bit for bit.
    block = [0x80000000, 0x7FC01234, 0x3F800000, 0x7FA05678, 0, 0x40000000] + [0] * 10
    sections[71] = struct.pack("<32I", *(block * 2))
    for s in range(108, 114):
        sections[s] = struct.pack("<2I", *([0x3F800000] * 2 if s < 111 else [0] * 2))
    for s in [105, 106, 107, 137, 138, 139, 140]:
        sections[s] = struct.pack("<I", 1)
    for s in [141, 142]:
        sections[s] = struct.pack("<2I", 0, 1)
    raw = bytearray(1984)
    raw[:8] = b"MOC3" + struct.pack("<I", 5)
    for s in sorted(sections):
        raw.extend(b"\0" * (-len(raw) % 64))
        struct.pack_into("<I", raw, 64 + s * 4, len(raw))
        raw.extend(sections[s])
    raw.extend(b"\0" * (-len(raw) % 64))
    return raw, block


def read(path, section, count):
    b = path.read_bytes()
    start = struct.unpack_from("<I", b, 64 + 4 * section)[0]
    return np.frombuffer(b, "<u4", count, start).tolist()


class MocPackingTests(unittest.TestCase):
    def test_raw_position_and_color_pooling_preserves_ieee_bits(self):
        with tempfile.TemporaryDirectory() as t:
            a, b = Path(t) / "a.moc3", Path(t) / "b.moc3"
            raw, block = fixture()
            a.write_bytes(raw)
            r = pack(a, b)
            self.assertEqual(r["unique_position_blocks"], 1)
            self.assertEqual(r["unique_color_rows"], 1)
            self.assertEqual(read(b, 71, 16), block)
            self.assertEqual(read(b, 70, 2), [0, 0])
            self.assertEqual(read(b, 141, 2), [0, 0])
            self.assertEqual(read(b, 0, 64)[10], 16)
            self.assertEqual(a.read_bytes(), raw)

    def test_live_extent_trimming_includes_all_form_owners(self):
        with tempfile.TemporaryDirectory() as t:
            a, b = Path(t) / "a.moc3", Path(t) / "b.moc3"
            raw, block = fixture()
            a.write_bytes(raw)
            extents = {
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "warp": [6],
                "artmesh": [4, 6],
            }
            r = pack(a, b, extents=extents)
            self.assertEqual(r["new_position_floats"], 6)
            self.assertEqual(read(b, 71, 6), block[:6])
            self.assertEqual(read(b, 70, 2), [0, 0])
            self.assertEqual(read(b, 0, 64)[10], 6)
            bad = {**extents, "artmesh": [6]}
            with self.assertRaisesRegex(ValueError, "coverage"):
                pack(a, b, extents=bad)
            bad = {**extents, "source_sha256": "wrong"}
            with self.assertRaisesRegex(ValueError, "another input"):
                pack(a, b, extents=bad)

    def test_unknown_layout_or_nonzero_modified_section_trailer_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            a, b = Path(t) / "a.moc3", Path(t) / "b.moc3"
            raw, _ = fixture()
            a.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "Separate"):
                pack(a, a)
            raw[4] = 4
            a.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "v5"):
                pack(a, b)
            raw[4] = 5
            raw[800] = 1
            a.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "reserved"):
                pack(a, b)
            raw[800] = 0
            start = struct.unpack_from("<I", raw, 64 + 60 * 4)[0]
            raw[start + 4] = 1
            a.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "trailer"):
                pack(a, b)


if __name__ == "__main__":
    unittest.main()
