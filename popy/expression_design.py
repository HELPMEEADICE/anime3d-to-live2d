"""Independently authored POPY emotional performances; no Anon curve copying (MIT)."""

import json
import math
from itertools import pairwise
from pathlib import Path

DESIGN = Path(__file__).resolve().parents[1] / "config/expression_actions.json"


def load_design(path=DESIGN):
    design = json.loads(Path(path).read_text(encoding="utf8"))
    if design["version"] != 1:
        raise ValueError("Unsupported expression action design version")
    ids = [a["id"] for a in design["actions"]]
    if len(ids) != len(set(ids)) or any(not x.isidentifier() for x in ids):
        raise ValueError("Expression action IDs must be unique identifiers")
    return design


def extra_presets():
    return load_design()["presets"]


def catalogue():
    design = load_design()
    return {
        "version": 1,
        "origin": "POPY-authored timings/curves. Anon used for performance categories only.",
        "playback": "Play emote_* without a simultaneous fixed exp3 to retain facial animation.",
        "variants": design["yaw_variants"],
        "actions": [
            {
                "type": a["id"],
                "label": a["label"],
                "duration": a["duration"],
                "motions": [
                    "emote_" + a["id"] + "_" + v for v in design["yaw_variants"]
                ],
                "limitation": a.get(
                    "limitation", "Existing audited facial controls only."
                ),
            }
            for a in design["actions"]
        ],
    }


def mirrored(pid, value):
    """Character-side mirror, including asymmetric eyes/brows and wrist targets."""
    for stem, tail in [
        ("ParamEye", "Open"),
        ("ParamEye", "Smile"),
        ("ParamBrow", "Y"),
        ("ParamBrow", "Form"),
        ("ParamArm", "Raise"),
        ("ParamArm", "Depth"),
        ("ParamForearm", "Turn"),
        ("ParamHand", "Pose"),
        ("ParamElbow", ""),
        ("ParamWrist", ""),
    ]:
        for side, other in [("L", "R"), ("R", "L")]:
            if pid == stem + side + tail:
                return stem + other + tail, value
    if pid in [
        "ParamAngleX",
        "ParamAngleZ",
        "ParamBodyAngleX",
        "ParamBodyAngleZ",
        "ParamEyeBallX",
    ]:
        value = -value
    return pid, value


def emotion_motions(parameters, presets, path=DESIGN):
    """Yield C/L/R clips with delayed channels, accents, blinks and neutral ends."""
    design = load_design(path)
    definitions = {p["id"]: p for p in parameters}
    for spec in design["actions"]:
        duration, attack = spec["duration"], spec["attack"]
        if not (math.isfinite(duration) and duration > 0 and 0.12 <= attack <= 0.4):
            raise ValueError("Invalid emotion duration or attack")
        for variant, yaw in design["yaw_variants"].items():
            if variant not in ["C", "L", "R"] or not math.isfinite(yaw):
                raise ValueError("Invalid emotion direction")
            targets = dict(presets[spec["preset"]])
            targets.update(spec["gesture"])
            for prefix, key in [("ParamAngle", "head"), ("ParamBodyAngle", "body")]:
                targets.update({prefix + axis: v for axis, v in zip("XYZ", spec[key])})
            if variant == "R":
                targets = dict(mirrored(k, v) for k, v in targets.items())
            targets["ParamAngleX"] += yaw
            targets["ParamBodyAngleX"] += yaw * 0.13
            targets["ParamEyeBallX"] = targets.get("ParamEyeBallX", 0) + yaw / 100
            tracks = {}

            def put(pid, keys, tracks=tracks, duration=duration):
                if pid not in definitions:
                    raise ValueError("Unavailable emotion control: " + pid)
                p = definitions[pid]
                if any(
                    not math.isfinite(t)
                    or not math.isfinite(v)
                    or not p["min"] <= v <= p["max"]
                    for t, v in keys
                ):
                    raise ValueError("Out-of-range emotion key: " + pid)
                if (
                    keys[0][0] != 0
                    or keys[-1][0] != 1
                    or any(b[0] <= a[0] for a, b in pairwise(keys))
                ):
                    raise ValueError(
                        "Emotion tracks must cover increasing normalized time"
                    )
                tracks[pid] = [[duration * t, v] for t, v in keys]

            def envelope(pid, target, delay=0, put=put, attack=attack):
                base = definitions[pid]["default"]
                put(
                    pid,
                    [
                        [0, base],
                        [0.04 + delay, base],
                        [attack + delay, target],
                        [0.65 + delay * 0.3, target],
                        [0.94, base],
                        [1, base],
                    ],
                )

            for pid, target in targets.items():
                # Eye saccade precedes face; head follows, then torso, elbow, wrist.
                delay = (
                    0
                    if "EyeBall" in pid
                    else 0.025
                    if any(x in pid for x in ["Eye", "Brow", "Mouth"])
                    else 0.06
                    if pid.startswith("ParamAngle")
                    else 0.1
                    if "Body" in pid or "Arm" in pid
                    else 0.13
                    if "Elbow" in pid
                    else 0.16
                )
                envelope(pid, target, delay)
            put("ParamBreath", [[0, 0], [0.38, 0.65], [0.7, 0.4], [1, 0]])
            kind = spec["id"]
            if kind in ["laugh", "cry"]:
                # Several unequal pulses; no repeated identical still poses.
                for pid, delta in [
                    ("ParamMouthOpenY", 0.15 if kind == "laugh" else 0.07),
                    ("ParamAngleY", 1.8 if kind == "laugh" else 1.1),
                    ("ParamBodyAngleY", 0.6),
                ]:
                    base, target = definitions[pid]["default"], targets[pid]
                    put(
                        pid,
                        [
                            [0, base],
                            [0.1, base],
                            [0.32, target],
                            [0.39, target - delta],
                            [0.46, target + delta],
                            [0.54, target - delta * 0.8],
                            [0.62, target + delta * 0.6],
                            [0.73, target * 0.65],
                            [0.94, base],
                            [1, base],
                        ],
                    )
            if kind == "surprise":
                put(
                    "ParamMouthOpenY",
                    [
                        [0, 0],
                        [0.08, 0],
                        [0.2, 0.8],
                        [0.37, 0.6],
                        [0.56, 0.35],
                        [0.78, 0.08],
                        [0.94, 0],
                        [1, 0],
                    ],
                )
                put(
                    "ParamAngleY",
                    [
                        [0, 0],
                        [0.09, 0],
                        [0.27, -10],
                        [0.43, -4],
                        [0.58, -6],
                        [0.94, 0],
                        [1, 0],
                    ],
                )
            if kind in ["serious", "thinking"]:
                value = targets["ParamAngleY"]
                put(
                    "ParamAngleY",
                    [
                        [0, 0],
                        [0.1, 0],
                        [0.4, value],
                        [0.56, value],
                        [0.66, value + 4],
                        [0.76, value - 1.5],
                        [0.94, 0],
                        [1, 0],
                    ],
                )
            if kind == "wink":
                side = "R" if variant == "R" else "L"
                put(
                    "ParamEye" + side + "Open",
                    [[0, 1], [0.23, 1], [0.34, 0], [0.52, 0], [0.66, 1], [1, 1]],
                )
                envelope("ParamEye" + side + "Smile", 0.18, 0.025)
            elif kind not in ["cry", "laugh", "drowsy"]:
                # Blink after reading the emotion, rather than obscuring its attack.
                for side in ["L", "R"]:
                    pid = "ParamEye" + side + "Open"
                    target = targets.get(pid, 1)
                    put(
                        pid,
                        [
                            [0, 1],
                            [0.065, 1],
                            [attack + 0.025, target],
                            [0.69, target],
                            [0.715, min(0.04, target)],
                            [0.755, target],
                            [0.94, 1],
                            [1, 1],
                        ],
                    )
            # Validate untouched defaults too; all curves reset at both boundaries.
            yield "emote_" + kind + "_" + variant, duration, tracks
