"""Lossless immutable-table interning for this project's validated MOC3 v5 files.

Reindexes existing position blocks and paired multiply/screen color rows. Every
IEEE-754 bit is retained; no coordinates, curves or textures are quantized. All
other sections are copied, with their file offsets relocated and zero padding.
Layout is checked against the pinned public runtime; this is not a general MOC
writer and rejects unknown versions/reserved header contents.
"""

import argparse
import hashlib
import json
import struct
import subprocess
from pathlib import Path

import numpy as np

from .compile_ir import jsonwrite


def runtime_extents(source):
    """Bind public-runtime extent discovery to this exact input file."""
    from .compile_ir import ROOT

    source = Path(source)
    before_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    descriptor = source.with_suffix(".extents.json")
    subprocess.run(
        [
            str(ROOT / "rust/target/release/position_extents.exe"),
            str(source.parent / "POPY.model3.json"),
            str(descriptor),
        ],
        check=True,
    )
    result = json.loads(descriptor.read_text())
    if Path(result["moc"]).resolve() != source.resolve():
        raise ValueError("Extent manifest refers to another MOC")
    if (
        result["moc_bytes"] != source.stat().st_size
        or hashlib.sha256(source.read_bytes()).hexdigest() != before_hash
    ):
        raise ValueError("Input changed during extent discovery")
    result["source_sha256"] = before_hash
    descriptor.write_text(json.dumps(result))
    return result


def pack(source, target, report_path=None, extents=None):
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve():
        raise ValueError("Separate packed output required")
    raw = source.read_bytes()
    if raw[:4] != b"MOC3" or struct.unpack_from("<I", raw, 4)[0] != 5:
        raise ValueError("Only audited little-endian MOC3 v5 is supported")
    offsets = np.frombuffer(raw, "<u4", count=152, offset=64).copy()
    starts = sorted({int(v) for v in offsets if v})
    if not starts or starts[0] < 64 + 4 * len(offsets):
        raise ValueError("Invalid section header")
    if any(raw[64 + 4 * len(offsets) : starts[0]]):
        raise ValueError("Unknown reserved header contents")
    if starts[-1] > len(raw) or any(v % 4 for v in starts):
        raise ValueError("Invalid section offsets")
    bounds = dict(zip(starts, starts[1:] + [len(raw)]))
    counts = np.frombuffer(raw, "<u4", count=64, offset=int(offsets[0])).copy()
    changes = {}

    def array(section, count):
        pos = int(offsets[section])
        if count == 0:
            return np.empty(0, dtype="<u4")
        if not pos or pos + 4 * count > bounds[pos]:
            raise ValueError(f"Invalid section extent {section}")
        return np.frombuffer(raw, "<u4", count=int(count), offset=pos)

    def replace(section, count, payload):
        if not count:
            return
        pos = int(offsets[section])
        if any(raw[pos + 4 * int(count) : bounds[pos]]):
            raise ValueError(f"Unknown nonzero section trailer {section}")
        if pos in changes:
            raise ValueError("Aliased nonempty sections need separate analysis")
        changes[pos] = payload

    old_position_count = int(counts[10])
    position_offsets = [array(60, counts[7]), array(70, counts[9])]
    lengths = None
    if extents is not None:
        if extents["source_sha256"] != hashlib.sha256(raw).hexdigest():
            raise ValueError("Position extents belong to another input")
        lengths = {}
        for refs, name in zip(position_offsets, ["warp", "artmesh"]):
            sizes = extents[name]
            if len(sizes) != len(refs):
                raise ValueError("Incomplete position extent coverage")
            for begin, size in zip(refs, sizes):
                begin, size = int(begin), int(size)
                if (
                    size <= 0
                    or size % 2
                    or begin % 2
                    or begin + size > old_position_count
                ):
                    raise ValueError("Invalid live position extent")
                lengths[begin] = max(lengths.get(begin, 0), size)
    unique_starts = np.unique(np.concatenate(position_offsets))
    old_blocks = len(unique_starts)
    pool, packed_positions, mapped = {}, bytearray(), []
    if old_position_count:
        if (
            not len(unique_starts)
            or unique_starts[0] != 0
            or unique_starts[-1] >= old_position_count
            or np.any(unique_starts % (2 if lengths is not None else 16))
        ):
            raise ValueError("Unsupported position-block layout")
        position_base = int(offsets[71])
        ends = np.r_[unique_starts[1:], old_position_count]
        for begin, end in zip(unique_starts, ends):
            if lengths is not None:
                # All users of this start, including BlendShape rows, were
                # visited. Their maximum extent is sufficient, even when two
                # existing position blocks overlap in a previously packed MOC.
                end = int(begin) + lengths[int(begin)]
            block = raw[position_base + 4 * int(begin) : position_base + 4 * int(end)]
            offset = pool.get(block)
            if offset is None:
                offset = len(packed_positions) // 4
                pool[block] = offset
                packed_positions.extend(block)
            mapped.append(offset)
        mapped = np.asarray(mapped, dtype="<u4")
        for section, refs in zip([60, 70], position_offsets):
            updated = mapped[np.searchsorted(unique_starts, refs)]
            replace(section, len(refs), updated.tobytes())
        replace(71, old_position_count, bytes(packed_positions))
        counts[10] = len(packed_positions) // 4

    old_colors = int(counts[23])
    if old_colors != counts[24]:
        raise ValueError("Separate color pools not yet supported")
    color_unique = 0
    if old_colors:
        rows = np.column_stack([array(i, old_colors) for i in range(108, 114)])
        unique, remap = np.unique(rows, axis=0, return_inverse=True)
        color_unique = len(unique)
        for section in range(108, 114):
            replace(
                section, old_colors, unique[:, section - 108].astype("<u4").tobytes()
            )
        for section, count in [
            (105, counts[2]),
            (106, counts[3]),
            (107, counts[4]),
            (137, counts[7]),
            (138, counts[7]),
            (139, counts[8]),
            (140, counts[8]),
            (141, counts[9]),
            (142, counts[9]),
        ]:
            refs = array(section, count)
            if np.any(refs >= old_colors):
                raise ValueError("Color reference outside pool")
            replace(section, count, remap[refs].astype("<u4").tobytes())
        counts[23] = counts[24] = color_unique
    replace(0, 64, counts.astype("<u4").tobytes())

    # Preserve every unmodified section, including unknown-but-addressed data.
    # Stronger alignment is legal; the forward reader requires only zero padding.
    result = bytearray(raw[: starts[0]])
    relocated = {}
    for begin in starts:
        result.extend(b"\0" * (-len(result) % 64))
        relocated[begin] = len(result)
        result.extend(changes.get(begin, raw[begin : bounds[begin]]))
    result.extend(b"\0" * (-len(result) % 64))
    for i, before in enumerate(offsets):
        struct.pack_into(
            "<I", result, 64 + 4 * i, relocated[int(before)] if before else 0
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(result)
    report = {
        "input_bytes": len(raw),
        "output_bytes": len(result),
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "output_sha256": hashlib.sha256(result).hexdigest(),
        "old_position_blocks": old_blocks,
        "unique_position_blocks": len(pool),
        "old_position_floats": old_position_count,
        "new_position_floats": int(counts[10]),
        "old_color_rows": old_colors,
        "unique_color_rows": color_unique,
        "float_values_quantized": 0,
        "logical_keyforms_removed": 0,
        "unused_position_padding_trimmed": lengths is not None,
        "method": "Exact live-position-block and raw-IEEE color-row interning; optional public-runtime extent discovery trims unused position padding; unchanged sections copied and header offsets relocated",
    }
    if report_path:
        jsonwrite(report_path, report)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("target", type=Path)
    p.add_argument("--report", type=Path)
    p.add_argument("--trim-padding", action="store_true")
    a = p.parse_args()
    extents = runtime_extents(a.source) if a.trim_padding else None
    print(pack(a.source, a.target, a.report, extents))


if __name__ == "__main__":
    main()
