"""Compact referenced float forms after geometric subdivision and mask generation."""

from pathlib import Path

import numpy as np


def compact_forms(ir, binary):
    from .field_compression import compress_glue_fields

    compress_glue_fields(ir, binary)
    binary = Path(binary)
    data = np.fromfile(binary, "<f4")
    packed = bytearray()
    offsets = {}

    def move(offset, size):
        if offset is None:
            return None
        if offset < 0 or offset + size > len(data):
            raise ValueError("Keyform range outside binary")
        key = (offset, size)
        if key not in offsets:
            offsets[key] = len(packed) // 4
            packed.extend(data[offset : offset + size].tobytes())
        return offsets[key]

    for mesh in ir["meshes"]:
        size = len(mesh["positions"])
        for c in mesh["cells"]:
            c["offset"] = move(c["offset"], size)
        for b in mesh["blend_shapes"]:
            b["offsets"] = [move(o, size) for o in b["offsets"]]
    temporary = binary.with_suffix(binary.suffix + ".tmp")
    temporary.write_bytes(packed)
    temporary.replace(binary)
    return len(packed)
