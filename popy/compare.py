"""SPDX-License-Identifier: MIT. Image AND original-source geometry regression metrics."""

import argparse, json, html
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, binary_erosion, distance_transform_edt
from .compile_ir import ROOT, project


def ssim(a, b, mask):
    a = np.mean(a, axis=2)
    b = np.mean(b, axis=2)
    ma = gaussian_filter(a, 1.5)
    mb = gaussian_filter(b, 1.5)
    va = gaussian_filter(a * a, 1.5) - ma * ma
    vb = gaussian_filter(b * b, 1.5) - mb * mb
    cov = gaussian_filter(a * b, 1.5) - ma * mb
    s = ((2 * ma * mb + 0.01**2) * (2 * cov + 0.03**2)) / (
        (ma * ma + mb * mb + 0.01**2) * (va + vb + 0.03**2)
    )
    return float(np.mean(s[mask])) if mask.any() else 0.0


def composite(a, bg=0.7):
    return a[:, :, :3] * a[:, :, 3:4] + bg * (1 - a[:, :, 3:4])


def compare(root=ROOT, write=True, rig_dir=None, runtime_dir=None, output_dir=None):
    root = Path(root)
    rig_dir = Path(rig_dir or root / "artifacts/rig")
    runtime_dir = Path(runtime_dir or root / "artifacts/ayagami")
    ir = json.loads((rig_dir / "POPY.rig.json").read_text())
    cases = json.loads((root / "artifacts/cases.json").read_text())
    out = Path(output_dir or root / "artifacts/report")
    out.mkdir(exist_ok=True)
    results = []
    w, h = ir["canvas"]
    ppu = w / 2
    stats = json.loads((rig_dir / "statistics.json").read_text())
    for case in cases:
        name = case["name"]
        a = (
            np.array(
                Image.open(root / "artifacts/reference" / f"{name}.png").convert(
                    "RGBA"
                ),
                dtype=float,
            )
            / 255
        )
        b = (
            np.array(
                Image.open(runtime_dir / f"{name}.png").convert("RGBA"), dtype=float
            )
            / 255
        )
        if a.shape != b.shape:
            raise ValueError("Canvas mismatch")
        ar = a[:, :, 3] > 0.5
        br = b[:, :, 3] > 0.5
        union = ar | br
        ac = composite(a)
        bc = composite(b)
        diff = np.mean(np.abs(ac - bc), axis=2)
        boundary_a = ar & ~binary_erosion(ar)
        boundary_b = br & ~binary_erosion(br)
        distances = np.r_[
            distance_transform_edt(~boundary_b)[boundary_a],
            distance_transform_edt(~boundary_a)[boundary_b],
        ]
        g = {
            m["id"]: m
            for m in json.loads((runtime_dir / f"{name}.geometry.json").read_text())
        }
        cache = {}
        errors = []
        in_canvas_errors = []
        area_errors = 0
        triangle_count = 0
        for mesh in ir["meshes"]:
            if mesh.get("mask_only"):
                continue
            source = mesh["source"]
            obj = source["object"]
            if obj not in cache:
                cache[obj] = project(
                    np.load(
                        root / "artifacts/reference" / (name + "__" + obj + ".npy")
                    ),
                    ir["camera"],
                ) * [w, h]
            ref = cache[obj][source["vertices"]]
            pred = np.array(g[mesh["id"]]["vertices"]) * ppu + [w / 2, h / 2]
            d = np.linalg.norm(pred - ref, axis=1)
            errors.extend(d.tolist())
            visible = (
                (ref[:, 0] >= 0) & (ref[:, 0] < w) & (ref[:, 1] >= 0) & (ref[:, 1] < h)
            )
            in_canvas_errors.extend(d[visible].tolist())
            t = np.array(mesh["indices"]).reshape(-1, 3)

            def area(v):
                a = v[t[:, 1]] - v[t[:, 0]]
                b = v[t[:, 2]] - v[t[:, 0]]
                return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]

            ra = area(ref)
            pa = area(pred)
            area_errors += int(np.sum((ra * pa < 0) & (np.abs(ra) > 0.2)))
            triangle_count += len(t)
        e = np.array(errors)
        evis = np.array(in_canvas_errors)
        result = {
            "name": name,
            "parameters": case["parameters"],
            "mae_full": float(diff.mean()),
            "mae_foreground": float(diff[union].mean()) if union.any() else 1.0,
            "ssim_foreground": ssim(ac, bc, union),
            "silhouette_iou": float((ar & br).sum() / max(union.sum(), 1)),
            "contour_symmetric_mean_px": float(distances.mean())
            if len(distances)
            else 999.0,
            "geometry_rmse_px": float(np.sqrt(np.mean(e * e))),
            "geometry_p95_px": float(np.percentile(e, 95)),
            "geometry_max_px": float(e.max()),
            "geometry_in_canvas_rmse_px": float(np.sqrt(np.mean(evis * evis))),
            "spurious_orientation_flips": area_errors,
            "triangles": triangle_count,
        }
        results.append(result)
        heat = np.stack(
            [np.clip(diff * 4, 0, 1), np.clip(diff * 1.4, 0, 1), np.zeros_like(diff)], 2
        )
        heat = Image.fromarray(np.uint8(heat * 255))
        heat.save(out / (name + "_heat.png"))
        pair = Image.new("RGB", (w * 3, h + 36), "#20242c")
        d = ImageDraw.Draw(pair)
        pair.paste(Image.fromarray(np.uint8(ac * 255)), (0, 36))
        pair.paste(Image.fromarray(np.uint8(bc * 255)), (w, 36))
        pair.paste(heat, (w * 2, 36))
        d.text((10, 10), name + " : Blender", fill="white")
        d.text(
            (w + 10, 10), f"Ayagami  IoU {result['silhouette_iou']:.3f}", fill="white"
        )
        d.text(
            (2 * w + 10, 10),
            f"Error x4 / geometry RMSE {result['geometry_rmse_px']:.2f}px",
            fill="white",
        )
        pair.save(out / (name + "_compare.png"))
    worst = sorted(results, key=lambda r: r["mae_foreground"], reverse=True)
    report = {
        "source": "evaluated Blender 3D surface vertices and albedo reference renders",
        "runtime": "actual generated moc3 evaluated and rendered by pinned Ayagami",
        "metrics_note": "Geometry includes hidden source vertices and UV-seam duplicates; no image alignment optimization. SSIM is grayscale Gaussian local SSIM averaged over union foreground. Not a perceptual quality certification.",
        "statistics": stats,
        "cases": results,
        "worst_by_foreground_mae": [x["name"] for x in worst],
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    tiles = []
    for r in results:
        im = Image.open(out / (r["name"] + "_compare.png"))
        im.thumbnail((720, 340))
        tiles.append(im)
    sheet = Image.new("RGB", (1440, 354 * ((len(tiles) + 1) // 2)), "#242830")
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i % 2) * 720, (i // 2) * 354))
    sheet.save(out / "contact_sheet.jpg", quality=88)
    rows = "".join(
        "<tr><td>"
        + html.escape(r["name"])
        + "</td>"
        + "".join(
            f"<td>{r[k]:.4f}</td>"
            for k in [
                "mae_foreground",
                "ssim_foreground",
                "silhouette_iou",
                "geometry_rmse_px",
                "geometry_max_px",
            ]
        )
        + "</tr>"
        for r in results
    )
    images = "".join(
        f'<h3>{html.escape(r["name"])}</h3><img loading="lazy" src="{r["name"]}_compare.png">'
        for r in results
    )
    (out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>POPY regression</title><style>body{font:16px system-ui;background:#151820;color:#eee;margin:30px}img{max-width:100%}td,th{padding:8px;border-bottom:1px solid #555}table{border-collapse:collapse}</style><h1>POPY: Blender / Ayagami / error</h1><p>Controlled albedo appearance; MToon lighting and outlines are excluded. CMO3 has not been verified in Cubism Editor.</p><table><tr><th>Pose</th><th>FG MAE</th><th>SSIM</th><th>IoU</th><th>Geo RMSE px</th><th>Geo max px</th></tr>'
        + rows
        + "</table>"
        + images,
        encoding="utf8",
    )
    print(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in [
                        "name",
                        "mae_foreground",
                        "silhouette_iou",
                        "geometry_rmse_px",
                        "geometry_max_px",
                    ]
                }
                for r in results
            ],
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rig-dir", type=Path)
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--out", type=Path)
    a = parser.parse_args()
    compare(rig_dir=a.rig_dir, runtime_dir=a.runtime_dir, output_dir=a.out)
