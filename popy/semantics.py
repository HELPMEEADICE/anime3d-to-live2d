"""Conservative candidate ranking; writes suggestions, never silently binds ambiguous controls."""

import argparse, json, re
from pathlib import Path
from .compile_ir import ROOT


def suggest(audit):
    arms = [o for o in audit["objects"] if o["type"] == "ARMATURE"]
    meshes = [o for o in audit["objects"] if o["type"] == "MESH"]
    result = {
        "armatures": [o["name"] for o in arms],
        "bones": {},
        "objects": {},
        "shapes": {},
        "policy": "Candidates need explicit override where confidence is low. Lexical matches are corroborated by hierarchy, vertex groups and spatial height; they are not universal semantics.",
    }
    patterns = {
        "head": r"head|頭|头",
        "neck": r"neck|首|颈",
        "body": r"chest|spine2|upper.?chest|胸",
        "eye_left": r"eye.*left|left.*eye|左.*目",
        "eye_right": r"eye.*right|right.*eye|右.*目",
    }
    for role, pattern in patterns.items():
        candidates = []
        for arm in arms:
            for b in arm["bones"]:
                if not re.search(pattern, b["name"], re.I):
                    continue
                influenced = [o["name"] for o in meshes if b["name"] in o["groups"]]
                candidates.append(
                    {
                        "name": b["name"],
                        "armature": arm["name"],
                        "parent": b["parent"],
                        "head": b["head"],
                        "influenced_objects": influenced,
                        "evidence": [
                            "name",
                            "bone hierarchy",
                            "mesh vertex-group membership",
                        ],
                        "confidence": 0.9 if influenced else 0.55,
                    }
                )
        result["bones"][role] = sorted(candidates, key=lambda x: -x["confidence"])
    for o in meshes:
        shape_names = [s["name"] for s in o.get("shape_keys", [])]
        role = "unclassified"
        if any(re.search("mouth|口", n, re.I) for n in shape_names):
            role = "face/mouth/eyelids"
        elif any(re.search("brow|眉", n, re.I) for n in shape_names):
            role = "brows"
        elif any(re.search("eye|目", g, re.I) for g in o["groups"]):
            role = "eyes"
        elif any(re.search("hair|髪|发", g, re.I) for g in o["groups"]):
            role = "hair"
        elif re.search("body|体", o["name"], re.I):
            role = "body"
        result["objects"][o["name"]] = {
            "suggested_role": role,
            "shape_count": len(shape_names),
            "groups": o["groups"],
            "bounds": o.get("bounds"),
            "confidence": 0.85 if role != "unclassified" else 0.2,
        }
        for s in shape_names:
            if s.lower() == "basis":
                continue
            result["shapes"][o["name"] + "/" + s] = {
                "object": o["name"],
                "shape": s,
                "side": "left"
                if re.search(r"(_L$|left|左)", s, re.I)
                else "right"
                if re.search(r"(_R$|right|右)", s, re.I)
                else "both/unknown",
                "function": "close"
                if re.search("close|閉|闭", s, re.I)
                else "open"
                if re.search(r"mouth_A$|open|開|开", s, re.I)
                else "custom",
                "evidence": "actual shape-key inventory; no inferred missing blendshape",
            }
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("audit", nargs="?", default=str(ROOT / "artifacts/scene_audit.json"))
    p.add_argument("--out", default=str(ROOT / "artifacts/semantic_candidates.json"))
    a = p.parse_args()
    Path(a.out).write_text(
        json.dumps(
            suggest(json.loads(Path(a.audit).read_text(encoding="utf8"))),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf8",
    )
