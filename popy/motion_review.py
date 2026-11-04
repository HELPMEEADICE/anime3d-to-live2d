"""Measured motion/palm/lateral-leg diagnostics for the delivered MOC corpus."""

import hashlib
import html
import json
import shutil
from itertools import pairwise

import numpy as np
from PIL import Image, ImageDraw, ImageSequence

from .animation import evaluate, segments
from .cli import run
from .compile_ir import ROOT, jsonwrite, project
from .face_repair import RENDERER
from .kinematics import Skeleton

BASE = ROOT / "artifacts/fullbody"
BEFORE = ROOT / "artifacts/motion_refine/before"


def palm(skeleton, side, values, camera):
    """Anatomical palm triangle, defined by audited wrist/finger-base landmarks."""
    suffix = "Left" if side == "L" else "Right"
    names = ["Hand_", "IndexFinger1_", "PinkyFinger1_"]
    rest = np.array([skeleton.bind[skeleton.index[n + suffix], :3, 3] for n in names])
    matrix = skeleton.skin_matrices(values)[skeleton.index["Hand_" + suffix]]
    posed = (np.c_[rest, np.ones(3)] @ matrix.T)[:, :3]
    normal = np.cross(posed[1] - posed[0], posed[2] - posed[0])
    normal *= -1 if side == "L" else 1
    normal /= np.linalg.norm(normal)
    q = project(posed, camera) * [768, 1024]
    a, b = q[1] - q[0], q[2] - q[0]
    return {
        "front_alignment": float(-normal[1]),
        "projected_area_px2": float(abs(a[0] * b[1] - a[1] * b[0]) / 2),
        "landmarks_px": q.tolist(),
    }


def join_velocities(motion):
    result = []
    for curve in motion["Curves"]:
        seq = list(segments(curve))
        for (_, left), (_, right) in pairwise(seq):
            # All authored profiles use restricted Bezier with monotonic time.
            if len(left) != 4 or len(right) != 4:
                continue
            before = (left[3][1] - left[2][1]) / (left[3][0] - left[2][0])
            after = (right[1][1] - right[0][1]) / (right[1][0] - right[0][0])
            monotone = (left[3][1] - left[0][1]) * (right[3][1] - right[0][1]) > 0
            result.append(
                {
                    "parameter": curve["Id"],
                    "time": left[3][0],
                    "left_velocity": before,
                    "right_velocity": after,
                    "monotone_through_key": monotone,
                }
            )
    return result


def review():
    cfg = json.loads((ROOT / "config/fullbody.json").read_text())
    skeleton = Skeleton(ROOT / "artifacts/samples", cfg)
    params = json.loads((BASE / "parameters.json").read_text())
    camera = json.loads((BASE / "reference/camera.json").read_text())
    result = {
        "palms": [],
        "lateral_legs": [],
        "motion_curves": [],
        "limitations": [
            "Palm metrics use source anatomical landmarks, not a claim of finger contact or palm-detail texture quality.",
            "Lateral steps are joint choreography: no root locomotion, floor-contact IK or cloth collision.",
            "C1 continuity removes key-by-key stops; it does not guarantee C2 acceleration continuity or subjective naturalness.",
        ],
    }
    for side in ["L", "R"]:
        current = json.loads(
            (
                ROOT
                / "output/POPY_fullbody/motions"
                / ("wave_" + side + ".motion3.json")
            ).read_text()
        )
        old = json.loads(
            (
                BEFORE / "POPY_fullbody/motions" / ("wave_" + side + ".motion3.json")
            ).read_text()
        )
        for t in [1.25, 1.6, 2, 2.4, 2.8]:
            a = palm(skeleton, side, evaluate(params, old, t), camera)
            b = palm(skeleton, side, evaluate(params, current, t), camera)
            result["palms"].append({"side": side, "time": t, "before": a, "after": b})
        i = skeleton.index["Foot_" + ("Left" if side == "L" else "Right")]
        rest = np.r_[skeleton.bind[i, :3, 3], 1]
        for angle in [-25, 0, 25, 35]:
            value = {"ParamLeg" + side + "Spread": angle}
            point = (skeleton.skin_matrices(value)[i] @ rest)[:3]
            result["lateral_legs"].append(
                {
                    "side": side,
                    "angle": angle,
                    "foot_dx_m": float(point[0] - rest[0]),
                    "foot_dx_px": float(
                        (
                            project(point[None], camera)
                            - project(rest[None, :3], camera)
                        )[0, 0]
                        * 768
                    ),
                }
            )
    for path in sorted((ROOT / "output/POPY_fullbody/motions").glob("*.motion3.json")):
        motion = json.loads(path.read_text())
        joins = join_velocities(motion)
        result["motion_curves"].append(
            {
                "name": path.stem,
                "maximum_velocity_join_error": max(
                    (abs(j["left_velocity"] - j["right_velocity"]) for j in joins),
                    default=0,
                ),
                "monotone_keys_with_nonzero_velocity": sum(
                    j["monotone_through_key"] and abs(j["left_velocity"]) > 1e-8
                    for j in joins
                ),
                "joins": joins,
            }
        )
    metrics = json.loads((BASE / "report/metrics.json").read_text())
    with (ROOT / "output/POPY_fullbody/POPY.moc3").open("rb") as stream:
        actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if metrics["moc_sha256"] != actual_hash:
        raise ValueError(
            "Motion review requires regression metrics for the current MOC"
        )
    result["moc_sha256"] = metrics["moc_sha256"]
    names = [
        "palm_edge_L",
        "palm_front_L",
        "palm_edge_R",
        "palm_front_R",
        "motion_leg_side_L_middle",
        "motion_leg_side_R_middle",
        "motion_side_step_L_middle",
        "motion_side_sway_quarter",
    ]
    grid = Image.new("RGB", (384 * 4, 540 * 2), "#303030")
    for i, name in enumerate(names):
        im = (
            Image.open(BASE / "runtime" / (name + ".png"))
            .convert("RGBA")
            .resize((384, 512))
        )
        grid.paste(im, ((i % 4) * 384, (i // 4) * 540), im)
        ImageDraw.Draw(grid).text(
            ((i % 4) * 384 + 5, (i // 4) * 540 + 515), name, fill="white"
        )
    out = ROOT / "artifacts/motion_refine"
    grid.save(out / "overview.jpg", quality=95)
    detail = Image.new("RGB", (320 * 4, 382), "#303030")
    for column, (side, view) in enumerate(
        [(s, v) for s in ["L", "R"] for v in ["edge", "front"]]
    ):
        values = {
            "ParamArm" + side + "Raise": -5,
            "ParamElbow" + side: 100,
            "ParamForearm" + side + "Turn": 90 if view == "front" else 0,
        }
        q = np.array(palm(skeleton, side, values, camera)["landmarks_px"])
        center = q.mean(0) + [0, -16]
        box = (
            int(center[0] - 48),
            int(center[1] - 56),
            int(center[0] + 48),
            int(center[1] + 56),
        )
        im = Image.open(
            BASE / "runtime" / ("palm_" + view + "_" + side + ".png")
        ).convert("RGBA")
        crop = im.crop(box).resize((288, 336))
        detail.paste(crop, (column * 320 + 16, 30), crop)
        ImageDraw.Draw(detail).text(
            (column * 320 + 10, 8),
            side + " " + view + " / actual MOC crop x3",
            fill="white",
        )
    detail.save(out / "palms_detail.png")
    old_preview, new_preview = BEFORE / "wave_L.webp", BASE / "wave_L.webp"
    if not old_preview.exists() or not new_preview.exists():
        raise FileNotFoundError(
            "Render wave_L at 15 fps and retain the matching v1 preview before generating a comparison"
        )
    if old_preview.exists() and new_preview.exists():
        old_frames = [
            f.convert("RGBA").copy()
            for f in ImageSequence.Iterator(Image.open(old_preview))
        ]
        new_frames = [
            f.convert("RGBA").copy()
            for f in ImageSequence.Iterator(Image.open(new_preview))
        ]
        if len(old_frames) != len(new_frames):
            raise ValueError(
                "Comparison previews must use matching frame rates/durations"
            )
        frames = []
        for a, b in zip(old_frames, new_frames):
            panel = Image.new("RGB", (1536, 1060), "#303030")
            panel.paste(a, (0, 36), a)
            panel.paste(b, (768, 36), b)
            draw = ImageDraw.Draw(panel)
            draw.text(
                (12, 10), "BEFORE / edge-on hand, synchronized stops", fill="white"
            )
            draw.text(
                (780, 10),
                "AFTER / palm turn, overlapping C1 joint tracks",
                fill="white",
            )
            frames.append(panel)
        frames[0].save(
            out / "wave_comparison.webp",
            save_all=True,
            append_images=frames[1:],
            duration=67,
            loop=0,
            quality=85,
        )
    if (out / "metrics.json").exists():
        previous = json.loads((out / "metrics.json").read_text())
        if (
            previous.get("moc_sha256") == result["moc_sha256"]
            and "matched_runtime_timing" in previous
        ):
            result["matched_runtime_timing"] = previous["matched_runtime_timing"]
    jsonwrite(out / "metrics.json", result)
    jsonwrite(ROOT / "benchmarks/popy/fullbody/motion_refinement.json", result)
    summary = {
        "palms": result["palms"],
        "lateral_legs": result["lateral_legs"],
        "limitations": result["limitations"],
        "moc_sha256": result["moc_sha256"],
    }
    (out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>POPY motion refinement</title>'
        "<style>body{background:#222;color:#eee;font:16px system-ui;max-width:1536px;margin:auto;padding:24px}img{max-width:100%}pre{white-space:pre-wrap}a{color:#9df}</style>"
        "<h1>POPY: palm turn / lateral legs / continuous motion</h1>"
        "<p>Actual MOC wave before / after, both sampled at 15 fps. Previews are not the internal model representation.</p>"
        '<img src="wave_comparison.webp"><h2>Actual MOC palm crops, magnified</h2>'
        '<img src="palms_detail.png"><h2>Lateral controls and actions</h2><img src="overview.jpg">'
        '<p><a href="metrics.json">All motion-curve joins and source landmark measurements</a></p><pre>'
        + html.escape(json.dumps(summary, indent=2))
        + "</pre>",
        encoding="utf8",
    )
    report = BASE / "report/motion_refinement"
    report.mkdir(exist_ok=True)
    for name in [
        "index.html",
        "metrics.json",
        "overview.jpg",
        "palms_detail.png",
        "wave_comparison.webp",
    ]:
        if (out / name).exists():
            shutil.copy2(out / name, report / name)
    index = BASE / "report/index.html"
    link = '<p><a href="motion_refinement/index.html">Palm / lateral-leg / before-after motion review</a></p>'
    text = index.read_text()
    if link not in text:
        index.write_text(
            text.replace(
                "<h1>POPY full body / motions / expressions</h1>",
                "<h1>POPY full body / motions / expressions</h1>" + link,
            ),
            encoding="utf8",
        )
    print(
        "MOTION_REVIEW",
        len(result["palms"]),
        "palm samples",
        len(result["lateral_legs"]),
        "lateral samples",
    )
    return result


def matched_timing():
    """Compare versions using exactly the same old parameter sequence."""
    parameters = json.loads((BEFORE / "parameters.json").read_text())
    motion = json.loads(
        (BEFORE / "POPY_fullbody/motions/wave_L.motion3.json").read_text()
    )
    expression = json.loads(
        (BEFORE / "POPY_fullbody/expressions/smile.exp3.json").read_text()
    )
    cases = [
        {
            "name": f"same_{i:03}",
            "parameters": evaluate(parameters, motion, i / 15, expression),
        }
        for i in range(60)
    ]
    out = ROOT / "artifacts/motion_refine"
    path = out / "matched_cases.json"
    jsonwrite(path, cases)
    results = {}
    for label, model in [
        ("before", BEFORE / "POPY_fullbody/POPY.model3.json"),
        ("after", ROOT / "output/POPY_fullbody/POPY.model3.json"),
    ]:
        target = out / ("matched_" + label)
        run("motion_matched_" + label, [RENDERER, model, path, target, "--images-only"])
        report = json.loads((target / "runtime_report.json").read_text())
        times = [r["pose_to_gpu_complete_ms"] for r in report["cases"][5:]]
        results[label] = {
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.quantile(times, 0.95)),
            "measured_frames": len(times),
        }
    results["note"] = (
        "Same v1 37-parameter wave+smile sequence on both models; first five frames excluded. This isolates version cost for shared poses, not the cost of new palm-turn choreography. Desktop was not isolated."
    )
    for file in [
        out / "metrics.json",
        ROOT / "benchmarks/popy/fullbody/motion_refinement.json",
        BASE / "report/motion_refinement/metrics.json",
    ]:
        data = json.loads(file.read_text())
        data["matched_runtime_timing"] = results
        jsonwrite(file, data)
    print("MATCHED_TIMING", json.dumps(results), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.benchmark:
        matched_timing()
    else:
        review()
