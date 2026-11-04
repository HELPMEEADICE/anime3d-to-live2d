"""POPY choreography in real joint parameters: lead, overlap and settling (MIT)."""

import math


def refine_tracks(name, duration, tracks, loop):
    tracks = dict(tracks)

    def put(pid, keys):
        if pid not in tracks:
            raise ValueError("Choreography requires control: " + pid)
        if keys[0][0] != 0 or keys[-1][0] != duration:
            raise ValueError("Choreography must cover the clip")
        tracks[pid] = keys

    def cycle(pid, base, amplitude, phase=0, periods=1, steps=16):
        put(
            pid,
            [
                [
                    duration * i / steps,
                    base
                    + amplitude * math.sin(2 * math.pi * periods * i / steps + phase),
                ]
                for i in range(steps + 1)
            ],
        )
        tracks[pid][-1][1] = tracks[pid][0][1]

    def pose(pid, target, lead=1, release=None, anticipation=None):
        base = tracks[pid][0][1]
        release = duration - 1 if release is None else release
        keys = [[0, base]]
        if anticipation is not None:
            keys.append([0.18, anticipation])
        keys.extend(
            [
                [lead * 0.7, base + 0.72 * (target - base)],
                [lead, target],
                [release, target],
                [duration, base],
            ]
        )
        put(pid, keys)

    if loop:
        cycle("ParamBreath", 0.5, 0.48, -math.pi / 2)
    else:
        put("ParamBreath", [[0, 0.25], [duration * 0.42, 0.8], [duration, 0.25]])

    if name == "idle":
        cycle("ParamBodyAngleZ", 0, 1.4)
        cycle("ParamAngleZ", 0, 2.1, 0.3)
        cycle("ParamAngleY", 0, 1.2, -0.5)
        for side, phase in [("L", 0), ("R", 0.55)]:
            cycle("ParamArm" + side + "Raise", -65, 1.8, phase)
            cycle("ParamElbow" + side, 4, 2, phase + 0.5)
            cycle("ParamWrist" + side, 0, 2, phase + 0.9)
            cycle("ParamForearm" + side + "Turn", 8, 3, phase)
        for side in ["L", "R"]:
            put(
                "ParamEye" + side + "Open",
                [[0, 1], [1.55, 1], [1.64, 0.05], [1.77, 1], [duration, 1]],
            )

    side = name[-1] if name.endswith(("_L", "_R")) else None
    if side:
        other = "R" if side == "L" else "L"
        sign = 1 if side == "L" else -1
        arm, elbow = "ParamArm" + side, "ParamElbow" + side
        wrist, turn = "ParamWrist" + side, "ParamForearm" + side + "Turn"
        hand, leg = "ParamHand" + side + "Pose", "ParamLeg" + side
        if name.startswith("wave_"):
            put(
                arm + "Raise",
                [
                    [0, -65],
                    [0.18, -69],
                    [0.62, -36],
                    [1.03, -8],
                    [1.35, -5],
                    [2.8, -7],
                    [3.2, -10],
                    [3.65, -49],
                    [4, -65],
                ],
            )
            put(arm + "Depth", [[0, 0], [0.4, 4], [1.15, 12], [2.8, 9], [4, 0]])
            put(
                elbow,
                [
                    [0, 2],
                    [0.25, 8],
                    [0.7, 64],
                    [1.15, 101],
                    [2, 96],
                    [2.85, 102],
                    [3.3, 83],
                    [3.7, 20],
                    [4, 2],
                ],
            )
            put(
                turn,
                [
                    [0, 0],
                    [0.32, 10],
                    [0.9, 80],
                    [1.25, 90],
                    [2, 83],
                    [2.8, 89],
                    [3.3, 70],
                    [4, 0],
                ],
            )
            put(
                wrist,
                [
                    [0, 0],
                    [0.7, -3],
                    [1.14, 0],
                    [1.43, 24],
                    [1.8, -25],
                    [2.17, 23],
                    [2.54, -21],
                    [2.91, 15],
                    [3.2, -5],
                    [3.55, 2],
                    [4, 0],
                ],
            )
            put(hand, [[0, 0], [0.65, 0.05], [1.2, 0], [2.9, 0], [3.65, 0.08], [4, 0]])
            put(
                "ParamBodyAngleZ",
                [
                    [0, 0],
                    [0.35, -sign],
                    [1, 2.3 * sign],
                    [2.8, 2 * sign],
                    [3.6, -0.7 * sign],
                    [4, 0],
                ],
            )
            put(
                "ParamAngleZ",
                [
                    [0, 0],
                    [0.55, 2 * sign],
                    [1.35, 6 * sign],
                    [2.8, 5 * sign],
                    [3.65, -sign],
                    [4, 0],
                ],
            )
            put("ParamAngleX", [[0, 0], [1.15, -3 * sign], [2.8, 2 * sign], [4, 0]])
            pose("ParamElbow" + other, 7, 0.9, 3)
        elif name.startswith("palm_show_"):
            pose(arm + "Raise", -23, 1.25, 2.9, -68)
            pose(arm + "Depth", 14, 1.35, 2.9)
            pose(elbow, 94, 1.4, 2.85)
            pose(turn, 90, 1.6, 2.8)
            pose(wrist, -6, 1.75, 2.75)
            pose("ParamAngleZ", 4 * sign, 1.6, 2.8)
        elif name.startswith("point_"):
            pose(arm + "Raise", -13, 1.05, 2.85, -69)
            pose(arm + "Depth", 13, 1.2, 2.8)
            pose(elbow, 28, 1.3, 2.8)
            pose(turn, 63, 1.35, 2.75)
            pose(hand, -1, 1.42, 2.7)
            pose("ParamBodyAngleX", sign * 3, 0.85, 2.9)
            pose("ParamAngleX", sign * 13, 0.72, 2.9)
            pose("ParamEyeBallX", sign * 0.3, 0.55, 3)
        elif name.startswith("fist_"):
            pose(arm + "Raise", -40, 0.85, 1.9)
            pose(elbow, 94, 1, 1.9)
            pose(turn, 72, 1.08, 1.9)
            pose(hand, 1, 1.2, 1.8)
            pose(wrist, -7, 1.3, 1.85)
            pose("ParamAngleZ", sign * 3, 0.9, 2)
        elif name.startswith("leg_lift_"):
            pose(leg + "Swing", 23, 1.05, 1.85)
            pose(leg + "Spread", 9, 1.12, 1.8)
            pose("ParamKnee" + side, 37, 1.2, 1.75)
            pose("ParamAnkle" + side, -9, 1.3, 1.7)
            pose("ParamBodyAngleZ", -3 * sign, 0.8, 1.95)
            pose("ParamArm" + other + "Raise", -48, 0.9, 1.9)
            pose(turn, 20, 1.1, 1.9)
        elif name.startswith(("leg_side_", "side_step_")):
            spread = 25 if name.startswith("leg_side_") else 22
            pose(leg + "Spread", spread, 1.45, 2.65, -2)
            pose(leg + "Swing", 4, 1.3, 2.7)
            pose("ParamKnee" + side, 12, 1.18, 2.75)
            pose("ParamAnkle" + side, -6, 1.65, 2.6)
            pose("ParamBodyAngleZ", -sign * 3, 0.8, 2.9)
            pose("ParamAngleZ", sign * 2, 1.05, 2.8)
            pose("ParamArm" + other + "Raise", -47, 1.2, 2.8)
            pose("ParamElbow" + other, 9, 1.3, 2.75)
            if name.startswith("side_step_"):
                pose("ParamLeg" + other + "Spread", 8, 1.65, 2.5)
                pose("ParamKnee" + other, 6, 1.55, 2.6)

    if name == "walk_in_place":
        for side, phase, sign in [("L", math.pi / 2, 1), ("R", -math.pi / 2, -1)]:
            cycle("ParamLeg" + side + "Swing", 0, 18, phase)
            cycle("ParamArm" + side + "Raise", -60, 5.5, phase + math.pi)
            cycle("ParamArm" + side + "Depth", 0, 10, phase + math.pi)
            cycle("ParamForearm" + side + "Turn", 12, 4, phase + 0.3)
            cycle("ParamElbow" + side, 9, 4, phase + 0.6)
            cycle("ParamWrist" + side, 0, 3, phase + 0.9)
            cycle("ParamLeg" + side + "Spread", 2, 1.5, phase + 0.4)
            # One swing-phase knee lift, low flexion through support.
            times = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
            knee = [3, 10, 32, 18, 4, 3, 2, 2, 3]
            ankle = [4, -3, -13, -5, 7, 5, 2, 0, 4]
            if side == "R":
                knee = knee[4:8] + knee[:5]
                ankle = ankle[4:8] + ankle[:5]
            put("ParamKnee" + side, list(map(list, zip(times, knee))))
            put("ParamAnkle" + side, list(map(list, zip(times, ankle))))
        cycle("ParamBodyAngleX", 0, 2.2)
        cycle("ParamBodyAngleZ", 0, 1.8, 0.15)
        cycle("ParamAngleZ", 0, 1.1, 0.15 + math.pi)
        cycle("ParamAngleY", 0, 0.8, 0, 2)
    elif name == "side_sway":
        cycle("ParamLegLSpread", 2, 7)
        cycle("ParamLegRSpread", 2, 7, math.pi)
        cycle("ParamKneeL", 4, 2, math.pi / 2)
        cycle("ParamKneeR", 4, 2, -math.pi / 2)
        cycle("ParamBodyAngleZ", 0, 3, 0.3)
        cycle("ParamAngleZ", 0, 2, 0.65 + math.pi)
        cycle("ParamArmLRaise", -63, 3, 0.5)
        cycle("ParamArmRRaise", -63, 3, 0.5 + math.pi)
    elif name == "thinking":
        pose("ParamForearmLTurn", 55, 1.25, 2.8)
        pose("ParamArmLDepth", 12, 1.1, 2.9)
        pose("ParamAngleZ", -6, 0.85, 2.9)
        pose("ParamBodyAngleZ", 2, 1.05, 2.8)
    elif name == "surprised":
        for side in ["L", "R"]:
            pose("ParamForearm" + side + "Turn", 75, 0.75, 1.9)
            pose("ParamArm" + side + "Raise", -42, 0.65, 2)
        pose("ParamAngleY", -5, 0.48, 2.1)
    elif name == "stretch":
        for side, delay in [("L", 0), ("R", 0.13)]:
            pose("ParamForearm" + side + "Turn", 45, 1.5 + delay, 2.45)
            pose("ParamArm" + side + "Raise", 46, 1.45 + delay, 2.5, -68)
            pose("ParamElbow" + side, 25, 1.65 + delay, 2.4)
        pose("ParamBodyAngleY", -3, 1.3, 2.6)
        pose("ParamAngleY", -6, 1.7, 2.5)
    return tracks
