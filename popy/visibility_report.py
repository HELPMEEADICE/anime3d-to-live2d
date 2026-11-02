"""Factual before/after HTML and Markdown, using real renderer images and metrics."""

import argparse
import html
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw

from .compile_ir import ROOT


def generate(current):
    current = Path(current).resolve()
    before = ROOT / "artifacts/visibility_before"
    now = json.loads((current / "assessment/metrics.json").read_text())
    old = (
        json.loads((before / "assessment/metrics.json").read_text())
        if (before / "assessment/metrics.json").exists()
        else None
    )
    keys = [
        ("eye_leak_total", "眼部漏出像素"),
        ("eye_missing_total", "眼部误遮挡像素"),
        ("source_seam_max_px", "源接缝最大距离 px"),
        ("interior_crack_total", "内部透明缺口指标"),
        ("false_jump_total", "连续动作时域残差像素"),
    ]
    rows = []
    md = [
        "# POPY 遮挡、接缝与连续动作修复",
        "",
        f"实际 Ayagami GPU 渲染 {now['summary']['poses']} 个姿态、比较 {now['summary']['transitions']} 对相邻帧。Blender 参考图与眼部诊断均由真实深度渲染得到。眼部可见性比较允许 1 px 边界误差。",
        "",
        "| 指标 | 修复前 | 修复后 |",
        "|---|---:|---:|",
    ]
    for key, label in keys:
        a = str(old["summary"][key]) if old else "无基线"
        b = str(now["summary"][key])
        rows.append(f"<tr><td>{label}</td><td>{a}</td><td>{b}</td></tr>")
        md.append(f"| {label} | {a} | {b} |")
    notes = "透明缺口指标还包含超过 3 px 的轮廓偏差；时域残差也包含几何插值误差，不能直接等同于排序跳变数。零接缝距离只指可由同源位置、完整权重和全部 morph 证明等价的分块顶点，不证明没有其他遮挡洞。"
    md += [
        "",
        notes,
        "",
        "已实现：皮肤、口腔、睫毛与前发的前景包络原生反向遮罩、眼白/瞳孔稳定层次、同源 Glue 焊接、绝对父空间关键形、冲突区域细分、姿态深度约束与邻域排序平滑、身体俯仰弦缩补偿。",
        "",
        "仍有限制：Cubism 没有逐像素深度缓冲，交叉表面只能以分块与遮罩近似；未采样的多轴姿态、极端眼球运动及高阶表情组合仍需继续优化。没有声称所有连续角度的裂缝和错误覆盖均已消除。",
        "",
        "参阅 index.html 和 assessment/metrics.json；红色诊断为漏出，蓝色为误遮挡。",
    ]
    (current / "REPORT.md").write_text("\n".join(md), encoding="utf8")
    examples = [
        "front",
        "all_extreme",
        "eye_30_0_open",
        "gaze_side_-45_-1",
        "sweep_roll_020",
    ]
    contact = Image.new("RGB", (1152, 410 * len(examples)), "#b3b3b3")
    draw = ImageDraw.Draw(contact)
    for ri, name in enumerate(examples):
        for ci, (label, directory) in enumerate(
            [
                ("Blender", ROOT / "artifacts/visibility_reference"),
                ("Before", before / "runtime"),
                ("After", current / "runtime"),
            ]
        ):
            path = directory / f"{name}.png"
            if path.exists():
                image = Image.open(path).convert("RGBA")
                image.thumbnail((288, 384))
                x, y = ci * 384 + 48, ri * 410 + 24
                contact.paste(image, (x, y), image)
                draw.text(
                    (ci * 384 + 12, ri * 410 + 6), f"{label}: {name}", fill="black"
                )
    contact.save(current / "before_after.jpg", quality=93)
    gallery = []
    for r in sorted(
        now["cases"],
        key=lambda row: -(row["eye_leak_pixels"] + row["eye_missing_pixels"]),
    ):
        name = r["name"]
        ref = os.path.relpath(
            ROOT / "artifacts/visibility_reference" / f"{name}.png", current
        ).replace("\\", "/")
        gallery.append(
            f'<details><summary>{html.escape(name)} — leak {r["eye_leak_pixels"]}, missing {r["eye_missing_pixels"]}, alpha {r["interior_crack_pixels"]}</summary><img src="{ref}"><img src="runtime/{name}.png">'
            + (
                f'<img src="assessment/{name}_eye_errors.png">'
                if (current / "assessment" / f"{name}_eye_errors.png").exists()
                else ""
            )
            + "</details>"
        )
    (current / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>POPY visibility regression</title><style>body{font:16px system-ui;background:#20242c;color:#eee;margin:30px}td,th{padding:8px;border:1px solid #666}img{width:30%;background:#aaa}summary{padding:10px;cursor:pointer}.contact{width:min(100%,1152px)}</style><h1>POPY 遮挡与接缝实测</h1><table><tr><th>指标</th><th>修复前</th><th>修复后</th></tr>'
        + "".join(rows)
        + "</table><p>"
        + notes
        + '</p><img class="contact" src="before_after.jpg"><h2>全部姿态：Blender / Ayagami / 眼部诊断</h2>'
        + "".join(gallery),
        encoding="utf8",
    )
    print(current / "index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, default=ROOT / "artifacts/visibility")
    generate(parser.parse_args().current)
