"""Build a reviewable report from independent Blender and native runtime outputs."""

import html
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw

from .compile_ir import ROOT


def generate(current):
    current = Path(current).resolve()
    baseline = ROOT / "artifacts/feature_before/assessment/metrics.json"
    old = json.loads(baseline.read_text()) if baseline.exists() else None
    now = json.loads((current / "assessment/metrics.json").read_text())
    names = {"brows": "眉毛", "lashes": "睫毛", "neck_hair": "颈部头发"}
    rows = []
    md = [
        "# POPY 睫毛、眉毛与颈部头发遮挡回归",
        "",
        f"独立 Blender 深度诊断与实际 Ayagami GPU 渲染：{now['poses']} 个姿态，包含转头、俯仰、倾斜、单眼/双眼闭合、愤怒/抬眉与身体组合。",
        "",
        "| 部位 | 漏出 1px 前→后 | 误遮挡 1px 前→后 | 漏出 3px 前→后 |",
        "|---|---:|---:|---:|",
    ]
    for key, label in names.items():
        b = now["summary"][key]
        a = old["summary"][key] if old else None
        values = [
            f"{a[k]:,} → {b[k]:,}" if a else f"无基线 → {b[k]:,}"
            for k in ["leak", "missing", "leak_far"]
        ]
        rows.append(
            "<tr><td>"
            + label
            + "</td>"
            + "".join("<td>" + v + "</td>" for v in values)
            + "</tr>"
        )
        md.append(f"| {label} | {' | '.join(values)} |")
    notes = "这些指标比较表面可见区域，允许 1px 或 3px 边界误差；同时包含姿态插值导致的位置偏差，不能全部解释为穿模。不是所有连续参数组合均已验证。局部三维深度/排序仍属于 Cubism 二维表示的近似。"
    md += [
        "",
        notes,
        "",
        "原始 .blend 与独立源快照均未被覆盖。复杂参数化 Glue 裁剪方案已保留在实验产物中；默认交付只使用通过当前质量与性能回归的策略。",
    ]
    (current / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf8")

    def link(path):
        return html.escape(
            os.path.relpath(path, current).replace("\\", "/"), quote=True
        )

    preview = [
        "front",
        "all_extreme",
        "eye_45_-30_open",
        "head_body_pitch",
        "feature_-45_30_raised",
        "sweep_roll_017",
    ]
    sheet = Image.new("RGB", (1200, 6 * 330), "#252932")
    draw = ImageDraw.Draw(sheet)
    gallery = []
    for row, name in enumerate(preview):
        paths = [
            ROOT / "artifacts/surface_reference" / f"{name}.png",
            ROOT / "artifacts/feature_before/runtime" / f"{name}.png",
            current / "runtime" / f"{name}.png",
        ]
        for col, path in enumerate(paths):
            if not path.exists():
                draw.text(
                    (col * 400 + 8, row * 330 + 20),
                    "Independent baseline unavailable",
                    fill="white",
                )
                continue
            im = Image.open(path).convert("RGBA")
            bg = Image.new("RGBA", im.size, (155, 155, 155, 255))
            bg.alpha_composite(im)
            bg = bg.convert("RGB").crop((130, 50, 680, 600))
            bg.thumbnail((390, 310))
            sheet.paste(bg, (col * 400, row * 330 + 20))
            draw.text(
                (col * 400 + 8, row * 330 + 3),
                ["Blender", "Before", "After"][col] + " " + name,
                fill="white",
            )
        gallery.append(
            "<h3>"
            + html.escape(name)
            + '</h3><div class="images">'
            + "".join(
                f'<figure><img src="{link(p)}"><figcaption>{label}</figcaption></figure>'
                for p, label in zip(paths, ["Blender", "修复前", "修复后"])
                if p.exists()
            )
            + "</div>"
        )
    sheet.save(current / "before_after.jpg", quality=94)
    details = []
    for row in now["cases"]:
        n = row["name"]
        ref = ROOT / "artifacts/surface_reference" / f"{n}_surfaces.png"
        actual = current / "surfaces" / f"{n}.png"
        err = current / "assessment" / f"{n}_errors.png"
        images = [ref, actual] + ([err] if err.exists() else [])
        details.append(
            f'<details><summary>{html.escape(n)} · {html.escape(str({k: row[k] for k in names}))}</summary><div class="images">'
            + "".join(f'<img src="{link(p)}" loading="lazy">' for p in images)
            + "</div></details>"
        )
    content = (
        '<!doctype html><meta charset="utf-8"><title>POPY 表面遮挡验证</title><style>body{background:#1e222a;color:#eef2f8;font:16px system-ui;margin:32px}table{border-collapse:collapse}td,th{border:1px solid #536079;padding:10px} .images{display:flex;gap:12px}figure{margin:0}img{max-width:31vw;background:#999}summary{padding:9px;cursor:pointer}p{max-width:1100px;line-height:1.7}</style><h1>POPY 睫毛、眉毛与颈部头发</h1>'
        + f"<p>{now['poses']} 姿态，真实 Blender 深度参考与 Ayagami 原生模型渲染。新增 42 个连续眨眼、21 个连续眉毛表情样本没有用于选择修复区域。</p><table><tr><th>部位</th><th>漏出 1px 前→后</th><th>误遮挡 1px 前→后</th><th>漏出 3px 前→后</th></tr>"
        + "".join(rows)
        + "</table><p>"
        + notes
        + "</p>"
        + "".join(gallery)
        + "<h2>全部表面诊断</h2><p>红：眉毛；绿：睫毛；蓝：头发；误差图橙色：漏出，青色：误遮挡。每类使用独立纹理页，保留真实 alpha。</p>"
        + "".join(details)
    )
    (current / "index.html").write_text(content, encoding="utf8")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("current", type=Path)
    generate(p.parse_args().current)
