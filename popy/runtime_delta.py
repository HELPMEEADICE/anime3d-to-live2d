"""Measure actual native vertex changes between two evaluations of the same rig."""

import argparse
import json
from pathlib import Path

import numpy as np

from .compile_ir import ROOT, jsonwrite


def compare(before, after, rig, destination):
    ir = json.loads(Path(rig).read_text())
    helpers = {m["id"] for m in ir["meshes"] if m.get("mask_only")}
    scale = ir["canvas"][0] / 2
    results = []
    for case in json.loads((ROOT / "artifacts/surface_cases.json").read_text()):
        name = case["name"]
        a = {
            m["id"]: np.asarray(m["vertices"])
            for m in json.loads((Path(before) / (name + ".geometry.json")).read_text())
        }
        b = {
            m["id"]: np.asarray(m["vertices"])
            for m in json.loads((Path(after) / (name + ".geometry.json")).read_text())
        }
        assert a.keys() == b.keys(), (
            "This comparison requires identical mesh identities"
        )
        delta = {
            k: float(np.max(np.linalg.norm(a[k] - b[k], axis=1))) * scale for k in a
        }
        results.append(
            {
                "name": name,
                "visible_max_px": max(v for k, v in delta.items() if k not in helpers),
                "helper_max_px": max(v for k, v in delta.items() if k in helpers),
            }
        )
    result = {
        "method": "Actual Ayagami native vertices, same mesh identities and canvas; saved preceding evaluation versus final model",
        "poses": len(results),
        "visible_max_px": max(r["visible_max_px"] for r in results),
        "helper_max_px": max(r["helper_max_px"] for r in results),
        "cases": results,
    }
    jsonwrite(Path(destination), result)
    print({k: v for k, v in result.items() if k != "cases"})
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("before", type=Path)
    p.add_argument("after", type=Path)
    p.add_argument("--rig", type=Path, default=ROOT / "artifacts/rig/POPY.rig.json")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts/surfaces/scalar_compression_delta.json",
    )
    a = p.parse_args()
    compare(a.before, a.after, a.rig, a.out)
