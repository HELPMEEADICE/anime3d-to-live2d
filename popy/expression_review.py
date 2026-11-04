"""Render actual MOC emotional clips and compare Blender face regions (MIT)."""

import argparse
import hashlib
import html
import json
import re
import shutil

import numpy as np
from PIL import Image, ImageDraw

from .animation import evaluate, segments
from .cli import run
from .compare import composite, ssim
from .compile_ir import ROOT, jsonwrite, project
from .expression_design import catalogue, load_design
from .face_repair import RENDERER

BASE = ROOT / "artifacts/fullbody"
MODEL = ROOT / "output/POPY_fullbody"
OUT = ROOT / "artifacts/expression_actions"


def motion_fingerprint():
    digest = hashlib.sha256()
    for spec in load_design()["actions"]:
        name = "emote_" + spec["id"] + "_C.motion3.json"
        digest.update(name.encode("utf8"))
        digest.update((MODEL / "motions" / name).read_bytes())
    return digest.hexdigest()


def audit_anon():
    """Read-only semantic inventory; never return/import reference animation curves."""
    reference = ROOT.parent / "anon"
    if not reference.exists():
        return {"available": False, "note": "Optional local semantic reference absent."}
    result = {"available": True, "motions": 0, "expressions": 0, "categories": {}}
    for folder, suffix in [("motions", "motion3"), ("expressions", "exp3")]:
        paths = sorted((reference / folder).glob("*." + suffix + ".json"))
        result[folder] = len(paths)
        for path in paths:
            family = path.stem.split(".")[0]
            if folder == "motions":
                family = family.rsplit("_", 1)[0]
            family = re.sub(r"\d+$", "", family)
            result["categories"][family] = result["categories"].get(family, 0) + 1
    selected = {}
    for kind in [
        "smile01",
        "smile02",
        "angry01",
        "sad01",
        "cry01",
        "wink01",
        "surprised01",
        "surprised02",
        "thinking01",
        "question01",
        "serious01",
    ]:
        path = reference / "motions" / ("mtn_" + kind + "_C.motion3.json")
        motion = json.loads(path.read_text(encoding="utf-8-sig"))
        moving = []
        for c in motion["Curves"]:
            values = [p[1] for _, points in segments(c) for p in points]
            if max(values) - min(values) > 0.01:
                moving.append(c["Id"])
        selected[kind] = {
            "duration": motion["Meta"]["Duration"],
            "moving_controls": moving,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    result["selected_semantics"] = selected
    result["reuse"] = (
        "Performance categories only; no textures, models, curves or expression values copied."
    )
    return result


def previews(fps=12):
    if not 0 < fps <= 60:
        raise ValueError("Preview fps must be in (0,60]")
    params = json.loads((BASE / "parameters.json").read_text())
    frames, clips = [], {}
    for spec in load_design()["actions"]:
        name = "emote_" + spec["id"] + "_C"
        motion = json.loads((MODEL / "motions" / (name + ".motion3.json")).read_text())
        clip = [
            {
                "name": name + "_" + str(i).zfill(4),
                "time": float(t),
                "parameters": evaluate(params, motion, float(t)),
            }
            for i, t in enumerate(np.arange(0, motion["Meta"]["Duration"], 1 / fps))
        ]
        frames.extend(clip)
        clips[name] = clip
    jsonwrite(OUT / "preview_cases.json", frames)
    run(
        "expression_action_previews",
        [
            RENDERER,
            MODEL / "POPY.model3.json",
            OUT / "preview_cases.json",
            OUT / "frames",
            "--images-only",
        ],
    )
    for name, clip in clips.items():
        images = []
        for case in clip:
            image = Image.open(OUT / "frames" / (case["name"] + ".png")).convert("RGBA")
            # Magnify existing full-body pixels; no camera/render change or inpainting.
            crop = image.crop((235, 65, 533, 310)).resize(
                (596, 490), Image.Resampling.LANCZOS
            )
            panel = Image.new("RGB", (596, 514), "#303039")
            panel.paste(crop, (0, 24), crop)
            ImageDraw.Draw(panel).text(
                (8, 6), name + f"  {case['time']:.2f}s", fill="white"
            )
            images.append(panel)
        images[0].save(
            OUT / (name + ".webp"),
            save_all=True,
            append_images=images[1:],
            duration=round(1000 / fps),
            loop=0,
            quality=88,
        )
    with (MODEL / "POPY.moc3").open("rb") as f:
        moc_sha = hashlib.file_digest(f, "sha256").hexdigest()
    result = {
        "fps": fps,
        "frames": len(frames),
        "clips": len(clips),
        "motion_sha256": motion_fingerprint(),
        "moc_sha256": moc_sha,
    }
    jsonwrite(OUT / "previews.json", result)
    return result


def review(create_previews=False, fps=12):
    OUT.mkdir(exist_ok=True, parents=True)
    summary = json.loads((BASE / "report/metrics.json").read_text())
    with (MODEL / "POPY.moc3").open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    manifest = json.loads((MODEL / "POPY.model3.json").read_text())["FileReferences"]
    expected = {n for a in catalogue()["actions"] for n in a["motions"]}
    if (
        summary["moc_sha256"] != actual
        or summary["motions"] != len(manifest["Motions"])
        or not expected <= manifest["Motions"].keys()
    ):
        raise ValueError(
            "Run fullbody assets/reference/blender/validate/report on the current model first"
        )
    result = {
        "moc_sha256": actual,
        "parameters": summary["parameters"],
        "motions": summary["motions"],
        "expressions": summary["expressions"],
        "new_emotion_motions": len(expected),
        "blender_clips": summary["blender_verification"]["clips"],
        "blender_reopened_checks": len(
            summary["blender_verification"]["evaluated_checks"]
        ),
        "reference": audit_anon(),
        "catalogue": catalogue(),
        "cases": [],
    }
    previous = OUT / "before/artifacts/fullbody/report/metrics.json"
    result["previous_moc_sha256"] = (
        json.loads(previous.read_text())["moc_sha256"] if previous.exists() else None
    )
    preserved = []
    for folder, ext in [("motions", "motion3"), ("expressions", "exp3")]:
        for before in sorted((OUT / "before" / folder).glob("*." + ext + ".json")):
            now = MODEL / folder / before.name
            if before.read_bytes() != now.read_bytes():
                raise AssertionError(
                    "Existing performance asset changed: " + before.name
                )
            preserved.append(before.name)
    result["byte_preserved_previous_assets"] = preserved
    camera = json.loads((BASE / "reference/camera.json").read_text())
    geometry_cases = []
    for m in summary["cases"]:
        if not (
            m["name"].startswith("motion_emote_")
            or any(
                m["name"].startswith("expression_" + name + "_")
                for name in load_design()["presets"]
            )
        ):
            continue
        name = m["name"]
        a_image = Image.open(BASE / "reference" / (name + ".png")).convert("RGBA")
        b_image = Image.open(BASE / "runtime" / (name + ".png")).convert("RGBA")
        q = project(np.load(BASE / "reference" / (name + "__M_Face.npy")), camera) * [
            768,
            1024,
        ]
        lo = np.maximum(np.floor(q.min(0)).astype(int) - 12, 0)
        hi = np.minimum(np.ceil(q.max(0)).astype(int) + 12, [768, 1024])
        box = tuple(int(v) for v in [*lo, *hi])
        a = np.array(a_image.crop(box), dtype=float) / 255
        b = np.array(b_image.crop(box), dtype=float) / 255
        ac, bc = composite(a), composite(b)
        mask = (a[..., 3] > 0.5) | (b[..., 3] > 0.5)
        error = np.abs(ac - bc).mean(2)
        row = {
            k: v
            for k, v in m.items()
            if k
            in [
                "name",
                "parameters",
                "silhouette_iou",
                "mae_foreground",
                "articulated_max_px",
            ]
        }
        row.update(
            face_box=box,
            face_mae=float(error[mask].mean()),
            face_ssim=ssim(ac, bc, mask),
        )
        result["cases"].append(row)
        if name.endswith("_middle") and name.startswith("motion_emote_"):
            geometry_cases.append(name)
        heat = np.stack(
            [np.clip(error * 4, 0, 1), np.clip(error, 0, 1), error * 0], axis=2
        )
        panel = Image.new("RGB", (900, 340), "#303039")
        for i, image in enumerate(
            [
                Image.fromarray(np.uint8(ac * 255)),
                Image.fromarray(np.uint8(bc * 255)),
                Image.fromarray(np.uint8(heat * 255)),
            ]
        ):
            image.thumbnail((300, 300), Image.Resampling.LANCZOS)
            panel.paste(image, (300 * i + (300 - image.width) // 2, 32))
        ImageDraw.Draw(panel).text(
            (8, 8),
            name + f" | Blender / Ayagami / error x4 | face MAE {row['face_mae']:.4f}",
            fill="white",
        )
        panel.save(OUT / (name + ".jpg"), quality=92)
    # Full source head vertices include occluded surfaces and intentional repair offsets.
    ir = json.loads((BASE / "rig/POPY.rig.json").read_text())
    meshes = [
        m
        for m in ir["meshes"]
        if m.get("source", {}).get("object") in ["M_Face", "M_Brow", "M_Eye"]
        and not m.get("mask_only")
    ]
    del ir
    geo = []
    for name in geometry_cases:
        native = {
            m["id"]: m
            for m in json.loads(
                (BASE / "geometry" / (name + ".geometry.json")).read_text()
            )
        }
        src = {
            obj: project(
                np.load(BASE / "reference" / (name + "__" + obj + ".npy")), camera
            )
            * [768, 1024]
            for obj in ["M_Face", "M_Eye", "M_Brow"]
        }
        distances = []
        for m in meshes:
            pred = np.array(native[m["id"]]["vertices"]) * 384 + [384, 512]
            ref = src[m["source"]["object"]][m["source"]["vertices"]]
            distances.extend(np.linalg.norm(pred - ref, axis=1).tolist())
        d = np.array(distances)
        geo.append(
            {
                "name": name,
                "vertex_samples": len(d),
                "rmse_px": float(np.sqrt(np.mean(d * d))),
                "p95_px": float(np.percentile(d, 95)),
                "max_px": float(d.max()),
            }
        )
    result["head_source_geometry"] = geo
    result["limitations"] = [
        "No tears/blush/hearts/sad-mouth or sad-brow controls; see per-action catalogue limitations.",
        "Face metrics use the projected source-face bounding region, including any hair within it; no feature segmentation.",
        "Head geometry includes hidden vertices and intentional previous repairs; it is not a visible-surface-only metric.",
        "12fps WebP files are offline previews, not measured realtime playback; MOC cost remains high.",
        "Independent finite pose validation only; no cloth/finger-contact solver or official Cubism Core/Editor validation.",
    ]
    result["worst_face_cases"] = sorted(
        result["cases"], key=lambda m: m["face_mae"], reverse=True
    )[:10]
    if create_previews:
        result["previews"] = previews(fps)
    elif (OUT / "previews.json").exists():
        result["previews"] = json.loads((OUT / "previews.json").read_text())
        if (
            result["previews"]["motion_sha256"] != motion_fingerprint()
            or result["previews"]["moc_sha256"] != actual
        ):
            raise ValueError(
                "Preview belongs to another motion/model revision; regenerate with --previews"
            )
    grid = Image.new("RGB", (320 * 4, 300 * 3), "#303039")
    for i, spec in enumerate(load_design()["actions"]):
        name = "motion_emote_" + spec["id"] + "_C_middle"
        im = Image.open(BASE / "runtime" / (name + ".png")).convert("RGBA")
        crop = im.crop((278, 85, 490, 270)).resize((298, 265), Image.Resampling.LANCZOS)
        x, y = (i % 4) * 320, (i // 4) * 300
        grid.paste(crop, (x + 11, y + 28), crop)
        ImageDraw.Draw(grid).text(
            (x + 10, y + 8), "emote_" + spec["id"] + "_C", fill="white"
        )
    grid.save(OUT / "overview.jpg", quality=94)
    jsonwrite(OUT / "metrics.json", result)
    jsonwrite(ROOT / "benchmarks/popy/fullbody/expression_actions.json", result)
    rows = []
    for a in catalogue()["actions"]:
        center = a["motions"][0]
        preview = OUT / (center + ".webp")
        rows.append(
            "<section><h2>"
            + html.escape(a["label"])
            + "</h2><p>"
            + html.escape(a["limitation"])
            + "</p>"
            + (
                f'<img loading="lazy" src="{center}.webp" width="478">'
                if preview.exists()
                else ""
            )
            + "<p>"
            + " / ".join(html.escape(n) for n in a["motions"])
            + "</p>"
            + "".join(
                f"<details><summary>{n}: Blender / Ayagami / heatmap</summary>"
                f'<img loading="lazy" src="motion_{n}_middle.jpg"></details>'
                for n in a["motions"]
            )
            + "</section>"
        )
    (OUT / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>POPY expression actions</title><style>body{background:#25252e;color:#eee;font:16px system-ui;max-width:1280px;margin:auto;padding:24px}img{max-width:100%}section{border-top:1px solid #666;padding:16px 0}pre{white-space:pre-wrap}a{color:#adf}</style>'
        "<h1>POPY 表情动作 / Emotional performances</h1><p>12 types × C/L/R = 36 new continuous motions. Play without a fixed expression override. C: front; L/R: ±24° head-turn variants. Actual MOC offline renders; no Anon assets copied.</p>"
        '<img src="overview.jpg"><p><a href="metrics.json">Measurements, source audit and limitations</a></p>'
        + "".join(rows),
        encoding="utf8",
    )
    dest = BASE / "report/expression_actions"
    dest.mkdir(exist_ok=True)
    for path in OUT.iterdir():
        if path.is_file() and (
            path.suffix in [".jpg", ".webp", ".html"] or path.name == "metrics.json"
        ):
            shutil.copy2(path, dest / path.name)
    index = BASE / "report/index.html"
    text = index.read_text(encoding="utf8")
    link = '<p><a href="expression_actions/index.html">New emotional motions / 表情动作预览及对照</a></p>'
    if link not in text:
        index.write_text(text.replace("</h1>", "</h1>" + link, 1), encoding="utf8")
    print(
        "EXPRESSION_REVIEW",
        json.dumps(
            {
                "cases": len(result["cases"]),
                "head_geometry_probes": len(geo),
                "worst_face_mae": result["worst_face_cases"][0]["face_mae"],
                "moc_sha256": actual,
                "preserved_previous_assets": len(preserved),
                "previews": result.get("previews"),
            }
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previews", action="store_true")
    parser.add_argument("--fps", type=float, default=12)
    args = parser.parse_args()
    review(args.previews, args.fps)


if __name__ == "__main__":
    main()
