"""Cubism motion3/expression3 evaluation and POPY-authored performance assets (MIT)."""

import itertools
import json
import math
from pathlib import Path


def segments(curve):
    data = curve["Segments"]
    if len(data) < 2:
        raise ValueError("Motion curve has no starting point")
    previous = (float(data[0]), float(data[1]))
    i = 2
    while i < len(data):
        kind = data[i]
        count = 6 if kind == 1 else 2
        if kind not in [0, 1, 2, 3] or i + count >= len(data):
            raise ValueError("Invalid or truncated motion segment")
        points = [previous] + [
            (float(data[j]), float(data[j + 1])) for j in range(i + 1, i + count + 1, 2)
        ]
        if points[-1][0] <= previous[0]:
            raise ValueError("Motion segment time must increase")
        yield int(kind), points
        previous = points[-1]
        i += count + 1


def curve_value(curve, time):
    def cubic(p, u):
        return (
            (1 - u) ** 3 * p[0]
            + 3 * (1 - u) ** 2 * u * p[1]
            + 3 * (1 - u) * u * u * p[2]
            + u**3 * p[3]
        )

    value = curve["Segments"][1]
    if time <= curve["Segments"][0]:
        return float(value)
    for kind, p in segments(curve):
        if time > p[-1][0]:
            value = p[-1][1]
            continue
        if kind == 2:
            return p[-1][1] if time == p[-1][0] else p[0][1]
        if kind == 3:
            return p[-1][1]
        if kind == 0:
            u = (time - p[0][0]) / (p[-1][0] - p[0][0])
            return p[0][1] * (1 - u) + p[-1][1] * u
        if not p[0][0] <= p[1][0] <= p[2][0] <= p[3][0]:
            raise ValueError("Non-monotone Bezier time is not supported")
        lo, hi = 0.0, 1.0
        for _ in range(40):
            u = (lo + hi) / 2
            if cubic([q[0] for q in p], u) < time:
                lo = u
            else:
                hi = u
        return cubic([q[1] for q in p], (lo + hi) / 2)
    return float(value)


def evaluate(parameters, motion=None, time=0, expression=None, expression_weight=1):
    definitions = {p["id"]: p for p in parameters}
    values = {k: p["default"] for k, p in definitions.items()}
    if motion is not None:
        duration = motion["Meta"]["Duration"]
        if motion["Meta"].get("Loop") and duration > 0:
            time %= duration
        else:
            time = min(max(time, 0), duration)
        for curve in motion["Curves"]:
            if curve["Target"] != "Parameter":
                raise ValueError("This offline evaluator accepts Parameter curves only")
            pid = curve["Id"]
            if pid not in definitions:
                raise ValueError("Unmapped animation parameter: " + pid)
            values[pid] = curve_value(curve, time)
    if expression:
        for p in expression["Parameters"]:
            pid = p["Id"]
            if pid not in definitions:
                raise ValueError("Unmapped expression parameter: " + pid)
            v, w = p["Value"], min(max(expression_weight, 0), 1)
            blend = p.get("Blend", "Add")
            if blend == "Add":
                values[pid] += v * w
            elif blend == "Multiply":
                values[pid] *= 1 + (v - 1) * w
            elif blend == "Overwrite":
                values[pid] = values[pid] * (1 - w) + v * w
            else:
                raise ValueError("Unknown expression blend: " + str(blend))
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError("Nonfinite animated parameter")
    return {
        k: min(max(v, definitions[k]["min"]), definitions[k]["max"])
        for k, v in values.items()
    }


def motion_asset(tracks, duration, loop=False):
    curves = []
    count = points = 0
    for pid, keys in sorted(tracks.items()):
        if keys[0][0] != 0 or keys[-1][0] != duration:
            raise ValueError("Every authored track must cover its full duration")
        data = list(keys[0])
        slopes = motion_tangents(keys, loop)
        for i, (a, b) in enumerate(itertools.pairwise(keys)):
            dt = (b[0] - a[0]) / 3
            # C1 monotone Hermite -> restricted Cubism Bezier. Intermediate
            # monotone keys retain velocity instead of stopping at every key.
            data += [
                1,
                a[0] + dt,
                a[1] + slopes[i] * dt,
                b[0] - dt,
                b[1] - slopes[i + 1] * dt,
                b[0],
                b[1],
            ]
            count += 1
            points += 3
        points += 1
        curves.append({"Target": "Parameter", "Id": pid, "Segments": data})
    return {
        "Version": 3,
        "Meta": {
            "Duration": duration,
            "Fps": 30,
            "Loop": loop,
            "AreBeziersRestricted": True,
            "CurveCount": len(curves),
            "TotalSegmentCount": count,
            "TotalPointCount": points,
            "UserDataCount": 0,
            "TotalUserDataSize": 0,
            "FadeInTime": 0.3,
            "FadeOutTime": 0.3,
        },
        "Curves": curves,
    }


def motion_tangents(keys, loop=False):
    """Weighted harmonic tangents: no overshoot, shared speed at joins/seams."""
    if len(keys) < 2:
        raise ValueError("A motion track needs at least two keys")
    h = [b[0] - a[0] for a, b in itertools.pairwise(keys)]
    if any(v <= 0 for v in h):
        raise ValueError("Motion key times must increase")
    d = [(b[1] - a[1]) / t for (a, b), t in zip(itertools.pairwise(keys), h)]

    def blend(a, b, left, right):
        if a * b <= 0:
            return 0.0
        w1, w2 = 2 * right + left, right + 2 * left
        return (w1 + w2) / (w1 / a + w2 / b)

    slopes = (
        [0.0]
        + [blend(d[i - 1], d[i], h[i - 1], h[i]) for i in range(1, len(keys) - 1)]
        + [0.0]
    )
    if loop and keys[0][1] == keys[-1][1]:
        slopes[0] = slopes[-1] = blend(d[-1], d[0], h[-1], h[0])
    return slopes


def expression_presets():
    return {
        "neutral": {},
        "smile": {"ParamMouthForm": 1, "ParamEyeLSmile": 0.4, "ParamEyeRSmile": 0.4},
        "laugh": {
            "ParamMouthForm": 1,
            "ParamMouthOpenY": 0.6,
            "ParamEyeLSmile": 0.8,
            "ParamEyeRSmile": 0.8,
        },
        "angry": {
            "ParamBrowLForm": 1,
            "ParamBrowRForm": 1,
            "ParamEyeLOpen": 0.75,
            "ParamEyeROpen": 0.75,
        },
        "serious": {
            "ParamBrowLForm": 0.6,
            "ParamBrowRForm": 0.6,
            "ParamEyeLOpen": 0.9,
            "ParamEyeROpen": 0.9,
        },
        "surprised": {"ParamBrowLY": 1, "ParamBrowRY": 1, "ParamMouthOpenY": 0.7},
        "thinking": {
            "ParamBrowLY": 0.55,
            "ParamBrowRForm": 0.35,
            "ParamEyeBallX": 0.5,
            "ParamEyeBallY": 0.25,
            "ParamEyeLOpen": 0.8,
            "ParamEyeROpen": 0.8,
        },
        "uncertain": {
            "ParamBrowLY": 0.65,
            "ParamBrowRY": 0.25,
            "ParamEyeLOpen": 0.65,
            "ParamEyeROpen": 0.65,
        },
        "wink_left": {"ParamEyeLOpen": 0, "ParamMouthForm": 0.8},
        "wink_right": {"ParamEyeROpen": 0, "ParamMouthForm": 0.8},
        "eyes_closed": {"ParamEyeLOpen": 0, "ParamEyeROpen": 0},
        "joy_closed": {
            "ParamEyeLOpen": 0.2,
            "ParamEyeROpen": 0.2,
            "ParamEyeLSmile": 1,
            "ParamEyeRSmile": 1,
            "ParamMouthForm": 1,
        },
        "sleepy": {"ParamEyeLOpen": 0.3, "ParamEyeROpen": 0.3, "ParamMouthOpenY": 0.15},
        "look_left": {"ParamEyeBallX": -1},
        "look_right": {"ParamEyeBallX": 1},
        "look_up": {"ParamEyeBallY": 1},
    }


def author_assets(directory, parameters):
    directory = Path(directory)
    for folder in ["motions", "expressions"]:
        (directory / folder).mkdir(parents=True, exist_ok=True)
    defaults = {p["id"]: p["default"] for p in parameters}
    facial = [p for p in defaults if any(s in p for s in ["Eye", "Brow", "Mouth"])]
    expressions = []
    for name, values in expression_presets().items():
        asset = {
            "Type": "Live2D Expression",
            "FadeInTime": 0.3,
            "FadeOutTime": 0.3,
            "Parameters": [
                {"Id": p, "Value": values.get(p, defaults[p]), "Blend": "Overwrite"}
                for p in facial
            ],
        }
        path = "expressions/" + name + ".exp3.json"
        (directory / path).write_text(json.dumps(asset, indent=2))
        expressions.append({"Name": name, "File": path})
    motions = {}

    def save(name, duration, tracks, loop=False):
        # Explicit reset curves make offline playback deterministic across actions.
        full = {p: [[0, v], [duration, v]] for p, v in defaults.items()}
        full.update(tracks)
        # Preserve the reusable motion codec independently of POPY choreography.
        from .motion_design import refine_tracks

        full = refine_tracks(name, duration, full, loop)
        asset = motion_asset(full, duration, loop)
        path = "motions/" + name + ".motion3.json"
        (directory / path).write_text(json.dumps(asset, indent=2))
        motions[name] = [{"File": path, "FadeInTime": 0.3, "FadeOutTime": 0.3}]

    save(
        "idle",
        4,
        {
            "ParamBreath": [[0, 0], [2, 1], [4, 0]],
            "ParamAngleZ": [[0, -2], [2, 2], [4, -2]],
        },
        True,
    )
    for side in ["L", "R"]:
        sign = 1 if side == "L" else -1
        save(
            "wave_" + side,
            4,
            {
                "ParamArm" + side + "Raise": [
                    [0, -65],
                    [0.8, -20],
                    [1.6, 0],
                    [3, 0],
                    [4, -65],
                ],
                "ParamElbow" + side: [[0, 0], [1, 90], [3, 90], [4, 0]],
                "ParamWrist" + side: [
                    [0, 0],
                    [1, 0],
                    [1.5, 30],
                    [2, -30],
                    [2.5, 30],
                    [3, 0],
                    [4, 0],
                ],
                "ParamMouthForm": [[0, 0], [1, 1], [3, 1], [4, 0]],
                "ParamAngleZ": [[0, 0], [1, sign * 8], [3, sign * 8], [4, 0]],
            },
        )
        save(
            "point_" + side,
            4,
            {
                "ParamArm" + side + "Raise": [[0, -65], [1, -10], [3, -10], [4, -65]],
                "ParamElbow" + side: [[0, 0], [1, 30], [3, 30], [4, 0]],
                "ParamHand" + side + "Pose": [[0, 0], [1, -1], [3, -1], [4, 0]],
                "ParamAngleX": [[0, 0], [1, sign * 15], [3, sign * 15], [4, 0]],
            },
        )
        save(
            "fist_" + side,
            3,
            {
                "ParamHand" + side + "Pose": [[0, 0], [1, 1], [2, 1], [3, 0]],
                "ParamElbow" + side: [[0, 0], [1, 90], [2, 90], [3, 0]],
            },
        )
        save(
            "leg_lift_" + side,
            3,
            {
                "ParamLeg" + side + "Swing": [[0, 0], [1, 25], [2, 25], [3, 0]],
                "ParamKnee" + side: [[0, 0], [1, 30], [2, 30], [3, 0]],
                "ParamAnkle" + side: [[0, 0], [1, -10], [2, -10], [3, 0]],
            },
        )
    save(
        "stretch",
        4,
        {
            **{
                "ParamArm" + s + "Raise": [[0, -65], [1.5, 50], [2.5, 50], [4, -65]]
                for s in ["L", "R"]
            },
            **{
                "ParamElbow" + s: [[0, 0], [1.5, 30], [2.5, 30], [4, 0]]
                for s in ["L", "R"]
            },
            "ParamMouthForm": [[0, 0], [1.5, 1], [2.5, 1], [4, 0]],
        },
    )
    save(
        "thinking",
        4,
        {
            "ParamElbowL": [[0, 0], [1, 120], [3, 120], [4, 0]],
            "ParamArmLRaise": [[0, -65], [1, -45], [3, -45], [4, -65]],
            "ParamHandLPose": [[0, 0], [1, -1], [3, -1], [4, 0]],
            "ParamEyeBallX": [[0, 0], [1, 0.5], [3, 0.5], [4, 0]],
            "ParamBrowLY": [[0, 0], [1, 0.6], [3, 0.6], [4, 0]],
        },
    )
    save(
        "surprised",
        3,
        {
            **{
                "ParamBrow" + s + "Y": [[0, 0], [0.6, 1], [2, 1], [3, 0]]
                for s in ["L", "R"]
            },
            "ParamMouthOpenY": [[0, 0], [0.6, 0.7], [2, 0.7], [3, 0]],
            **{
                "ParamElbow" + s: [[0, 0], [0.6, 60], [2, 60], [3, 0]]
                for s in ["L", "R"]
            },
        },
    )
    save(
        "nod",
        3,
        {"ParamAngleY": [[0, 0], [0.6, 15], [1.2, -15], [1.8, 15], [2.4, -15], [3, 0]]},
    )
    save(
        "denial",
        3,
        {"ParamAngleX": [[0, 0], [0.6, -25], [1.2, 25], [1.8, -25], [2.4, 25], [3, 0]]},
    )
    save(
        "walk_in_place",
        2,
        {
            **{
                "ParamLeg" + s + "Swing": [
                    [0, 20 * sign],
                    [0.5, 0],
                    [1, -20 * sign],
                    [1.5, 0],
                    [2, 20 * sign],
                ]
                for s, sign in [("L", 1), ("R", -1)]
            },
            "ParamKneeL": [[0, 0], [0.5, 30], [1, 0], [2, 0]],
            "ParamKneeR": [[0, 0], [1, 0], [1.5, 30], [2, 0]],
        },
        True,
    )
    for side in ["L", "R"]:
        for action in ["palm_show", "leg_side", "side_step"]:
            save(action + "_" + side, 4, {})
    save("side_sway", 4, {}, True)
    return motions, expressions
