"""Static and temporally adjacent test poses, including gaze and eye closures at side angles."""

import json

from .compile_ir import ROOT


def cases():
    result = json.loads((ROOT / "artifacts/cases.json").read_text())
    for x in [-45, -30, 0, 30, 45]:
        for y in [-30, 0, 30]:
            for state, left, right in [
                ("open", 1, 1),
                ("closed", 0, 0),
                ("left", 0, 1),
                ("right", 1, 0),
            ]:
                result.append(
                    {
                        "name": f"eye_{x}_{y}_{state}",
                        "parameters": {
                            "ParamAngleX": x,
                            "ParamAngleY": y,
                            "ParamEyeLOpen": left,
                            "ParamEyeROpen": right,
                        },
                    }
                )
    for i in range(31):
        x = -45 + i * 3
        result.append(
            {
                "name": f"sweep_yaw_{i:03}",
                "sequence": "yaw",
                "frame": i,
                "parameters": {
                    "ParamAngleX": x,
                    "ParamAngleY": -10,
                    "ParamEyeLOpen": 0.35,
                    "ParamEyeROpen": 0.7,
                    "ParamEyeBallX": 0.5,
                    "ParamMouthOpenY": 0.4,
                },
            }
        )
    for i in range(21):
        y = -30 + i * 3
        result.append(
            {
                "name": f"sweep_pitch_{i:03}",
                "sequence": "pitch",
                "frame": i,
                "parameters": {
                    "ParamAngleX": 35,
                    "ParamAngleY": y,
                    "ParamEyeLOpen": 0,
                    "ParamEyeROpen": 0,
                    "ParamEyeBallX": 1,
                },
            }
        )
    for i in range(21):
        z = -30 + i * 3
        result.append(
            {
                "name": f"sweep_roll_{i:03}",
                "sequence": "roll",
                "frame": i,
                "parameters": {
                    "ParamAngleX": 25,
                    "ParamAngleY": 10,
                    "ParamAngleZ": z,
                    "ParamBodyAngleX": 7.5,
                    "ParamBodyAngleY": -7.5,
                    "ParamMouthOpenY": 0.4,
                },
            }
        )
    for x in [-45, 45]:
        for gaze in [-1, 1]:
            result.append(
                {
                    "name": f"gaze_side_{x}_{gaze}",
                    "parameters": {
                        "ParamAngleX": x,
                        "ParamEyeBallX": gaze,
                        "ParamEyeBallY": 0.5,
                    },
                }
            )
    return result


if __name__ == "__main__":
    destination = ROOT / "artifacts/visibility_cases.json"
    destination.write_text(json.dumps(cases(), indent=2))
    print(destination, len(cases()), "poses")
