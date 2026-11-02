"""Refine measured non-orderable patches; never change source topology or artwork."""

import argparse
import hashlib
import json

import numpy as np

from .compile_ir import ROOT, jsonwrite


def fingerprint(source):
    return hashlib.sha256(
        source["positions"].tobytes() + source["triangles"].tobytes()
    ).hexdigest()


def refine(folder=ROOT / "artifacts/rig", max_patches=24):
    proposals = json.loads((folder / "refinement_proposal.json").read_text())
    path = ROOT / "config/patch_refinement.json"
    result = (
        json.loads(path.read_text()) if path.exists() else {"version": 1, "objects": {}}
    )
    changes = 0
    for proposal in proposals[:max_patches]:
        name = proposal["object"]
        source = np.load(ROOT / "artifacts/samples" / f"{name}_source.npz")
        digest = fingerprint(source)
        entry = result["objects"].setdefault(
            name, {"source_sha256": digest, "triangles": [], "limit": 8}
        )
        if entry["source_sha256"] != digest:
            entry.update(source_sha256=digest, triangles=[])
        old = set(entry["triangles"])
        new = old | set(proposal["triangles"])
        changes += len(new - old)
        entry["triangles"] = sorted(new)
    result["method"] = (
        "Measured mutual coverage/cycle conflicts; selected source triangles use an eight-triangle budget"
    )
    jsonwrite(path, result)
    print("Adaptive source triangles added:", changes)
    return changes


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max-patches", type=int, default=24)
    a = p.parse_args()
    refine(max_patches=a.max_patches)
