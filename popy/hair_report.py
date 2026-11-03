"""Standalone local visual evidence and provenance for the shoulder-hair repair."""

import hashlib
import html
import json
import re
import shutil

from PIL import Image, ImageDraw

from .compile_ir import ROOT, jsonwrite


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate():
    out = ROOT / "artifacts/hair_repair_report"
    out.mkdir(exist_ok=True)
    before = json.loads(
        (ROOT / "artifacts/hair_clearance_assessment/before/metrics.json").read_text()
    )
    after = json.loads(
        (ROOT / "artifacts/hair_clearance_assessment/after/metrics.json").read_text()
    )
    collisions = []
    for row in after["cases"]:
        r = json.loads(
            (
                ROOT
                / "artifacts/hair_clearance_reference"
                / (row["name"] + "_collision.json")
            ).read_text()
        )
        collisions.append(
            {
                "name": row["name"],
                "parameters": row["parameters"],
                "before": {
                    k: {m: v[m] for m in ["triangle_pairs", "hair_triangles"]}
                    for k, v in r["before"].items()
                },
                "after": {
                    k: {m: v[m] for m in ["triangle_pairs", "hair_triangles"]}
                    for k, v in r["after"].items()
                },
                "max_displacement_m": r["max_displacement_m"],
                "moved_vertices": r["moved_vertices"],
            }
        )
    collision_summary = {
        t: sum(v["triangle_pairs"] for r in collisions for v in r[t].values())
        for t in ["before", "after"]
    }
    collision_summary["max_depth_displacement_m"] = max(
        r["max_displacement_m"] for r in collisions
    )
    model = ROOT / "output/POPY_hair_clearance"
    ir = ROOT / "artifacts/hair_clearance_rig/POPY.rig.json"
    stats = json.loads((ir.parent / "statistics.json").read_text())
    runtime = json.loads(
        (ROOT / "artifacts/hair_clearance_runtime/runtime_report.json").read_text()
    )
    linework = json.loads(
        (
            ROOT / "artifacts/hair_clearance_expression_assessment/metrics.json"
        ).read_text()
    )
    features = {
        k: json.loads(
            (
                ROOT / "artifacts/hair_feature_assessment" / k / "metrics.json"
            ).read_text()
        )
        for k in ["before", "after"]
    }
    source_record = ROOT / "artifacts/hair_input.json"
    input_record = (
        json.loads(source_record.read_text())
        if source_record.exists()
        else {
            "source": str(ROOT / "artifacts/POPY_source.blend"),
            "sha256_before": "b4d94421c820d52db5d66da08302062328035a27af4fac378b818072faacbaed",
        }
    )
    from pathlib import Path

    current_source_hash = digest(Path(input_record["source"]))
    provenance = {
        "source_sha256": current_source_hash,
        "corrected_blend_sha256": digest(ROOT / "artifacts/POPY_hair_clearance.blend"),
        "moc_sha256": digest(model / "POPY.moc3"),
        "rig_sha256": digest(ir),
        "cases_sha256": digest(ROOT / "artifacts/hair_cases.json"),
        "policy_sha256": digest(ROOT / "config/hair_clearance.json"),
        "source_unchanged": current_source_hash == input_record["sha256_before"],
        "poses": after["poses"],
        "expression_poses": linework["poses"],
        "transitions": len(after["transitions"]),
        "official_editor_verified": False,
    }
    result = {
        "before": before["summary"],
        "after": after["summary"],
        "false_change_before": before["false_change_total"],
        "false_change_after": after["false_change_total"],
        "collision_summary": collision_summary,
        "collisions": collisions,
        "provenance": provenance,
        "runtime": runtime,
        "model": {
            k: stats[k]
            for k in [
                "artmesh_count",
                "vertices",
                "triangles",
                "glues",
                "animated_glues",
                "masked_drawables",
            ]
        },
        "linework": {k: linework["summary"][k] for k in ["brows", "lashes"]},
        "features": features,
        "moc_bytes": (model / "POPY.moc3").stat().st_size,
    }
    glue_check = ROOT / "artifacts/hair_clearance_native_glue.json"
    if glue_check.exists():
        result["native_glue"] = json.loads(glue_check.read_text())
    performance = json.loads((ROOT / "artifacts/performance.json").read_text())
    stage_names = {
        "quality_export": ["hair_release_quality", "hair_quality_export"],
        "quality_77": ["hair_release_native77", "hair_runtime"],
        "quality_264": ["hair_release_native264", "hair_runtime_expression"],
        "editor_export": ["hair_release_editor_export_final", "hair_editor_export"],
        "editor_readback": ["hair_release_editor_readback", "hair_editor_readback"],
        "editor_77": ["hair_release_editor_native77", "hair_editor_runtime"],
    }
    result["performance"] = {
        name: {k: v for k, v in matches[-1].items() if k != "command"}
        for name, choices in stage_names.items()
        for matches in [
            [r for r in performance if r["stage"] in choices and r["exit_code"] == 0]
        ]
        if matches
    }
    python_log = ROOT / "artifacts/logs/hair_python_tests.log"
    rust_log = ROOT / "artifacts/logs/hair_rust_tests.log"
    if python_log.exists() and rust_log.exists():
        py_text, rs_text = python_log.read_text(), rust_log.read_text()
        result["tests"] = {
            "python": int(re.findall(r"Ran (\d+) tests", py_text)[-1]),
            "python_passed": py_text.rstrip().endswith("OK"),
            "rust": sum(
                map(int, re.findall(r"test result: ok\. (\d+) passed", rs_text))
            ),
            "rust_passed": "FAILED" not in rs_text,
        }
    balanced = ROOT / "output/POPY_hair_clearance_balanced"
    if (balanced / "cmo_roundtrip.json").exists():
        result["editor"] = {
            "moc_sha256": digest(balanced / "POPY.moc3"),
            "cmo_sha256": digest(balanced / "POPY.cmo3"),
            "own_codec": json.loads((balanced / "cmo_roundtrip.json").read_text()),
        }
    balanced_assessment = ROOT / "artifacts/hair_balanced_assessment/metrics.json"
    if balanced_assessment.exists():
        result["balanced_visibility"] = json.loads(balanced_assessment.read_text())[
            "summary"
        ]
    jsonwrite(out / "metrics.json", result)
    jsonwrite(out / "before.json", before)
    jsonwrite(out / "after.json", after)
    # Publishable numerical summaries intentionally exclude source topology or rendered assets.
    public = ROOT / "benchmarks/popy/hair_repair"
    public.mkdir(exist_ok=True)
    public_result = dict(result)
    public_result["runtime"] = {
        k: v for k, v in runtime.items() if k not in ["model", "command"]
    }
    jsonwrite(public / "metrics.json", public_result)
    jsonwrite(public / "before.json", before)
    jsonwrite(public / "after.json", after)
    paths = {
        "source": ROOT / "artifacts/hair_reference",
        "blender": ROOT / "artifacts/hair_clearance_reference",
        "before": ROOT / "artifacts/hair_before/runtime",
        "after": ROOT / "artifacts/hair_clearance_runtime",
    }
    for name, directory in paths.items():
        (out / name).mkdir(exist_ok=True)
        for row in after["cases"]:
            shutil.copy2(
                directory / (row["name"] + ".png"), out / name / (row["name"] + ".png")
            )
    mosaic = Image.new("RGB", (4 * 540, 3 * 300), (28, 30, 35))
    draw = ImageDraw.Draw(mosaic)
    for y, row in enumerate(after["cases"][:3]):
        for x, name in enumerate(paths):
            im = Image.open(out / name / (row["name"] + ".png")).convert("RGBA")
            bg = Image.new("RGBA", im.size, (35, 38, 44, 255))
            bg.alpha_composite(im)
            crop = bg.convert("RGB").crop((140, 390, 680, 660))
            mosaic.paste(crop, (x * 540, y * 300 + 30))
            draw.text(
                (x * 540 + 8, y * 300 + 8), row["name"] + " / " + name, fill="white"
            )
    mosaic.save(out / "comparison.jpg", quality=94)
    worst = sorted(after["cases"], key=lambda r: -(r["leak"] + r["missing"]))[:8]
    rows = "".join(
        f"<tr><td>{html.escape(r['name'])}</td><td>{r['leak']}</td><td>{r['missing']}</td></tr>"
        for r in worst
    )
    options = "".join(
        f'<option value="{html.escape(r["name"])}">{html.escape(r["name"])}</option>'
        for r in after["cases"]
    )
    markup = f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>POPY Hair Repair</title>
<style>body{{background:#171a20;color:#e5e7eb;font:16px system-ui;margin:30px;max-width:1600px}}a{{color:#8ccaff}}table{{border-collapse:collapse}}td,th{{padding:8px;border:1px solid #555}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:6px}}img{{width:100%;background:#292d34}}select{{font-size:18px}}pre{{white-space:pre-wrap}}</style>
<h1>POPY 肩颈头发修复：Blender + Live2D</h1><p>77 个姿态、60 对连续帧；另评估 264 个表情组合。原始 .blend 保持不变。</p>
<p>上方三张用户截图对应 reported_left / reported_right / reported_roll。下列列顺序为原 Blender、修正 Blender、旧 Live2D、新 Live2D。</p>
<img src="comparison.jpg"><h2>逐姿态对照</h2><select id="pose">{options}</select><div class="grid">{"".join(f'<div><p>{name}</p><img id="{name}" src="{name}/reported_left.png"></div>' for name in paths)}</div>
<h2>眼睛、脸部和发饰专项检查</h2><p>同一组 77 姿态的独立 RGB 深度标记。仅修正后侧头发与衣服的遮挡关系；前侧头发原有遮罩、额前网格、眼睛、脸部与发饰的排序轨道全部保留。</p><pre>{html.escape(json.dumps({k: features[k]["summary"] for k in features}, ensure_ascii=False, indent=2))}</pre>
<h2>指标</h2><pre>{html.escape(json.dumps({k: result[k] for k in ["before", "after", "false_change_before", "false_change_after", "collision_summary", "model", "linework", "moc_bytes"]}, ensure_ascii=False, indent=2))}</pre>
<h2>限制与未解决姿态</h2><p>仍有极端头身组合误差。3D 闭合包络与 20mm 间隙偏保守，最大深度移动约 {collision_summary["max_depth_displacement_m"]:.3f}m，尚不适合声称精细物理仿真；未解决头发自碰撞。相交数为相同求值细分拓扑下的 BVH 三角形对计数，非穿透体积。图像指标含投影插值误差。二维路由针对已审计的正面正交相机；不适用于任意相机。CMO3 仅可做自身库读回验证，未验证官方 Editor。</p>
<table><tr><th>最差姿态</th><th>错误露出 / px</th><th>错误遮掉 / px</th></tr>{rows}</table><p><a href="metrics.json">数值与文件哈希</a> · <a href="before.json">旧版逐姿态</a> · <a href="after.json">新版逐姿态</a></p>
<script>document.getElementById('pose').onchange=e=>{{for(const k of {json.dumps(list(paths))})document.getElementById(k).src=k+'/'+e.target.value+'.png';}};</script></html>"""
    (out / "index.html").write_text(markup, encoding="utf8")
    print(
        json.dumps({k: result[k] for k in ["before", "after", "collision_summary"]}),
        flush=True,
    )
    return result


if __name__ == "__main__":
    generate()
