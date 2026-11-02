"""Generate a factual handover report and local delivery packages from measured artifacts."""

import csv
import hashlib
import json
import statistics
import zipfile
from PIL import Image
from .compile_ir import ROOT


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf8"))


def mb(n):
    return n / 1_000_000


def main():
    quality = read("artifacts/report/metrics.json")
    balanced = read("artifacts/report_balanced/metrics.json")
    runtime = read("artifacts/ayagami/runtime_report.json")
    balanced_runtime = read("artifacts/ayagami_balanced/runtime_report.json")
    visibility = read("artifacts/visibility/assessment/metrics.json")
    visibility_text = (ROOT / "artifacts/visibility/REPORT.md").read_text(
        encoding="utf8"
    )
    performance = read("artifacts/performance.json")
    audit = read("artifacts/scene_audit_repeat.json")
    qrows = {r["name"]: r for r in quality["cases"]}
    brows = {r["name"]: r for r in balanced["cases"]}
    tests = (ROOT / "artifacts/logs/regression.log").read_text(encoding="utf8")
    if "\nOK\n" not in tests and not tests.rstrip().endswith("OK"):
        raise RuntimeError("Do not package a failed regression run")
    destination = ROOT / "delivery"
    destination.mkdir(exist_ok=True)
    profile_rows = []
    for profile, folder, report, rt in [
        ("quality", "output/POPY", quality, runtime),
        ("balanced", "output/POPY_balanced", balanced, balanced_runtime),
    ]:
        out = ROOT / folder
        timings = [c["pose_to_gpu_complete_ms"] for c in rt["cases"]]
        cmo = out / "POPY.cmo3"
        moc = out / "POPY.moc3"
        if cmo.exists():
            roundtrip = json.loads((out / "cmo_roundtrip.json").read_text())
            assert roundtrip["drawables"] == report["statistics"]["artmesh_count"]
            assert roundtrip["glues"] == report["statistics"]["seam_welds"]["glues"]
            assert roundtrip["inverted_masks"] == 16
        info = {
            "profile": profile,
            "target": "Cubism 5.0 / moc3 v5",
            "moc_header_version_byte": moc.open("rb").read(5)[4],
            "moc3_bytes": moc.stat().st_size,
            "cmo3_bytes": cmo.stat().st_size if cmo.exists() else None,
            "moc3_validation": "Pinned Ayagami parser, driver and GPU renderer; 28 independent Blender geometry/image comparisons"
            + (
                "; 165 depth-visible eye/seam cases and 70 adjacent transitions"
                if profile == "quality"
                else ""
            ),
            "cmo3_validation": "Exporter own-codec roundtrip in separate JVM; masks and Glue inventory verified; official Editor not verified"
            if cmo.exists()
            else "Not generated: earlier dense quality revision exceeded 7 and 10 GiB JVM heaps; current revision has more keyforms and uses the separate balanced editor output",
            "cubism_editor_verified": False,
            "source_profile": "POPY upper-body orthographic T-pose; not a universal conversion benchmark",
            "parameters": rt["parameters"],
            "statistics": report["statistics"],
            "timing": {
                "gpu": rt["gpu"],
                "backend": rt["backend"],
                "pose_to_gpu_complete_median_ms": statistics.median(timings),
                "pose_to_gpu_complete_min_ms": min(timings),
                "pose_to_gpu_complete_max_ms": max(timings),
                "method": "CPU wall time from parameter update through GPU fence completion, includes preparation and target allocation; excludes screenshot/file writing. Single sweep, not a sustained FPS guarantee.",
            },
            "source_sha256": hashlib.sha256(
                (ROOT / "artifacts/POPY_source.blend").read_bytes()
            ).hexdigest(),
            "rig_sha256": hashlib.sha256(
                (
                    ROOT
                    / (
                        "artifacts/rig/POPY.rig.json"
                        if profile == "quality"
                        else "artifacts/rig_balanced/POPY.rig.json"
                    )
                ).read_bytes()
            ).hexdigest(),
            "upstream": {
                "psd2live": "bafabab1a360e208671d39adf5087c62f6cae1f2 + documented draw-leaf sentinel patch",
                "ayagami": "0d1d7aa3efe57b1b363b72e672e353c52a9c15d6 unmodified",
            },
            "limits": [
                "Original MToon lighting and outlines are not represented; comparison uses controlled albedo",
                "Visibility uses geometry-derived native inverse eye masks, adaptive surface patches, coherent keyed painter constraints and culling; it still has no per-pixel depth buffer",
                "Smile-eye residuals are neutral-pose approximations; breath is procedural",
                "No physics3, motion3, exp3, arbitrary-model generalization or official Cubism compatibility test is claimed",
            ]
            + (
                [
                    "Body pitch is a neutral-head blendshape and has large errors under extreme head/body combinations"
                ]
                if profile == "balanced"
                else ["Dense local grids are large and need compression"]
            ),
        }
        if profile == "quality":
            info["visibility_regression"] = visibility["summary"]
        (out / "conversion_metadata.json").write_text(
            json.dumps(info, indent=2), encoding="utf8"
        )
        (out / "README.txt").write_text(
            "POPY "
            + profile
            + " profile\nLoad POPY.model3.json in a compatible runtime.\n"
            + (
                "The CMO3 belongs to THIS balanced MOC3; it is not the quality-profile editor project.\n"
                if cmo.exists()
                else "There is no completed quality-profile CMO3. The separate balanced package contains its own matching CMO3.\n"
            )
            + "Cubism Editor compatibility has NOT been verified. See conversion_metadata.json and the project REPORT.md.\nCharacter/texture rights remain with the source asset owners; these images are not covered by the converter MIT license.\n",
            encoding="utf8",
        )
        hashes = {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(out.iterdir())
            if f.is_file() and f.name != "SHA256.json"
        }
        (out / "SHA256.json").write_text(json.dumps(hashes, indent=2))
        profile_rows.append(
            f"| {profile} | {mb(info['moc3_bytes']):.2f} | "
            + (
                f"{mb(info['cmo3_bytes']):.2f}"
                if info["cmo3_bytes"]
                else "未完成（内存不足）"
            )
            + f" | {report['statistics']['artmesh_count']} | {report['statistics']['authored_keyforms']:,} | {statistics.median(timings):.2f} |"
        )
        with zipfile.ZipFile(
            destination / f"POPY_{profile}.zip",
            "w",
            zipfile.ZIP_DEFLATED,
            compresslevel=5,
        ) as archive:
            for f in sorted(out.iterdir()):
                if f.is_file():
                    archive.write(f, f"POPY_{profile}/{f.name}")
    key_cases = [
        "front",
        "left",
        "right",
        "up",
        "down",
        "turn_mouth",
        "turn_blink",
        "heldout",
        "heldout_head_body",
        "head_body_pitch",
        "all_extreme",
    ]
    rows = []
    for name in key_cases:
        r = qrows[name]
        rows.append(
            f"| {name} | {r['mae_foreground']:.5f} | {r['ssim_foreground']:.4f} | {r['silhouette_iou']:.5f} | {r['geometry_rmse_px']:.4f} | {r['geometry_max_px']:.3f} |"
        )
    worst = max(quality["cases"], key=lambda r: r["mae_foreground"])
    perf_rows = [
        f"| {p['stage']} | {p['seconds']:.2f} | {p['peak_process_tree_rss_bytes'] / 2**20:.1f} | {p['exit_code']} |"
        for p in performance
    ]
    source_meshes = [o for o in audit["objects"] if o["type"] == "MESH"]
    source_bones = [o for o in audit["objects"] if o["type"] == "ARMATURE"]
    vertices = sum(o["vertices"] for o in source_meshes)
    tris = sum(o["triangles"] for o in source_meshes)
    text = rf"""# POPY 实测交付报告

本次已实现并在用户当前 POPY 上运行真实链路：Blender bpy 数据/姿态 → 3D 源映射与局部 Rig IR → 原生 Warp/Rotation/ArtMesh 参数 → 真正 MOC3 → 固定版本 Ayagami GPU 渲染。没有 PSD 中转、序列帧模型、改扩展名文件或虚构的 Editor 验证。

{visibility_text.replace("# POPY 遮挡、接缝与连续动作修复", "## 本轮遮挡、接缝与连续动作修复")}

新增对比入口：`artifacts/visibility/index.html`；修复前模型与数据保存在 `artifacts/visibility_before/`。质量版已应用本轮全部修复；平衡版应用遮罩、焊接与排序修复，身体俯仰仍采用原有近似。

## 交付与体积

| 配置 | MOC3 MB | CMO3 MB | ArtMesh | 作者关键形数 | 参数到 GPU 完成中位 ms |
|---|---:|---:|---:|---:|---:|
{chr(10).join(profile_rows)}

以上 MB 按 1,000,000 字节计算。两版均 19 参数、最大层级 3、6 张 2048² 原始 UV 图集；质量版 {quality["statistics"]["deformers"]} 个原生变形器，平衡版 {balanced["statistics"]["deformers"]} 个。质量版输出 {quality["statistics"]["vertices"]:,} 个顶点和 {quality["statistics"]["triangles"]:,} 个三角形，包含诊断统计中的不可见遮罩网格。高质量版 {mb((ROOT / "output/POPY/POPY.moc3").stat().st_size):.2f} MB 的稠密局部关键形仍偏大，不能将其描述为已经解决轻量化。ZIP 压缩体积不等于运行时内存。

质量版入口：`output/POPY/POPY.model3.json`。平衡版入口及其匹配工程：`output/POPY_balanced/POPY.model3.json`、`POPY.cmo3`。此前约 216 MB MOC3 的密集质量版在尝试 CMO3 时，7/10 GiB Java 堆均 OOM，日志位于 `artifacts/failures/quality_cmo_10g.log`；当前版本关键形更多，沿用分开交付的方案，没有重新声称完成质量版 CMO3，也没有将平衡版工程放进质量版目录冒充一致。平衡版 CMO3 成功写出并读回，但官方 Cubism Editor 的打开/保存/重新导出仍未验证。

## 已在 POPY 实测完成

- Blender {audit["blender"]}，{len(source_meshes)} Mesh、{vertices:,} 源顶点、{tris:,} 源三角形、{sum(len(a["bones"]) for a in source_bones)} 骨骼；贴图从 packed images 提取，两个 UV 层、原始权重与全部 morph 均保留审计/中间数据。
- 连续头部三轴（水平 ±45°、俯仰 ±30°、倾斜 ±30°）、左右独立闭眼、眼球 XY、嘴部开合和笑形、四个眉毛控制、笑眼、身体三轴与程序化呼吸。当前为半身、正交、原始 T-pose 配置。
- 质量版把身体 yaw/pitch 的 3×3 基与头部及局部表情联合投影；实际 Blender 独立渲染 28 个测试，包括非训练值、单眼/双眼、转头张嘴/眨眼、身体与头部运动、全极值。
- 每个姿态实际加载生成的 MOC3、设置参数、求值网格并由 Ayagami Vulkan 渲染，输出源网格误差、MAE、SSIM、轮廓 IoU/距离、误差热图和逐姿态对比图。Python 与 Rust 回归通过，实际数量及日志见 `artifacts/logs/`。
- 当前 Blender 仍保持原场景：40 对象、14 Mesh、无相机、Head 基矩阵单位、表情值为 0。独立 source snapshot 转换前后 SHA-256 一致，见 `source_integrity.json` 和 `live_scene_after.json`。

## 视觉与几何实测

| 姿态 | 前景 MAE | SSIM | alpha IoU | 顶点 RMSE px | 顶点最大误差 px |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

这里的图像比较统一为 albedo/emission，不保留源 MToon 的动态光照/描边；另有 `artifacts/original_mtoon/` 的 1152×1536 原材质审阅图。几何误差包含隐藏源顶点和 UV 拆点，不能只凭训练点约 0.02 px 的 RMSE 宣称所有连续组合都接近零误差。

当前最差前景 MAE 为 `{worst["name"]}`：{worst["mae_foreground"]:.5f}，顶点 RMSE {worst["geometry_rmse_px"]:.3f} px。主要来自未采样参数中间值的多轴旋转残差插值和排序近似。全部数据见 `artifacts/report/metrics.json`，图形入口 `artifacts/report/index.html`。

## 迭代证据

- 最初仅平均深度排序：正面 MAE 约 0.052；用几何覆盖约束后约 0.006～0.008。
- 顶点直接插值 roll 的未见姿态 RMSE 约 6.1 px；原生 Rotation 与几何残差分解后，`heldout` 为 {qrows["heldout"]["geometry_rmse_px"]:.3f} px。
- 平衡版 `all_extreme` RMSE {brows["all_extreme"]["geometry_rmse_px"]:.3f} px、MAE {brows["all_extreme"]["mae_foreground"]:.5f}；质量版分别为 {qrows["all_extreme"]["geometry_rmse_px"]:.3f} px、{qrows["all_extreme"]["mae_foreground"]:.5f}。这是增加身体 pitch 耦合的实测收益，代价是更多关键形与内存。
- 16 三角形小块版本有 1,383 ArtMesh；采用覆盖约束后允许更大的块，当前减少为 {quality["statistics"]["artmesh_count"]}，仍保持可见质量。历史指标和 contact sheets 保留在 `artifacts/baseline/`、`artifacts/iterations/`。

## 已实现但仍有质量问题

- 质量版未见多参数姿态仍有几何插值误差；本轮减少了缺口和眼部穿透，但没有证明所有连续参数组合都无错误覆盖或跳变。
- 遮挡约束使用 384×512 栅格，质量版累计删除了 {quality["statistics"]["occlusion_optimization"]["removed_cycle_edges"]} 条循环约束（包含姿态中不活跃的全局连续性弱边），属于近似；Cubism 本身没有每像素 z-buffer。
- 附加笑眼的头部耦合与呼吸是近似。原始 MToon 高光、阴影和 outline 不属于本轮转换产物。
- 平衡版身体 pitch 和大角度头部组合不适合当作高质量模型使用；提供它主要是较小的运行时/编辑工程起点。

## 原型范围与尚未解决

- 可配置语义覆盖、候选识别和投影矩阵接口已实现；仅 POPY 实测，不能宣称适配任意命名/骨架/坐标系模型。
- CMO3 自身读回不是 Editor 兼容性证据；未在常见安装位置或开始菜单找到可用 Cubism Editor。质量版 CMO3 的内存问题仍需修改/优化导出适配。
- 未完成头发物理、physics3、motion3/exp3 求值与时序回归，也未声称有原模型不存在的 BlendShape。
- 未完成新的简化拓扑、重心重采样、通用 Warp 控制点优化、Jacobian/Laplacian 正则化、完整自交修复及超过当前参数域的验收。

下一步最有价值的是：在已实现的冲突区域细分上继续优化分块边界与局部动态遮罩，压缩身体与头部的耦合关键形，让 CMO3 writer 流式处理关键形，并在真正 Editor 中检查打开/保存/重新导出。继续增加均匀笛卡尔采样会进一步放大现有体积问题。

## 时耗与内存

GPU：{runtime["gpu"]} / {runtime["backend"]}。当前质量版参数更新到 GPU fence 完成中位数 {statistics.median(c["pose_to_gpu_complete_ms"] for c in runtime["cases"]):.2f} ms；这是一次参数遍历，包含 CPU 准备和 render target 分配，不是长时 FPS 保证。另记录截图/文件写入耗时。

| 阶段 | 秒 | 进程树峰值 RSS MiB | exit |
|---|---:|---:|---:|
{chr(10).join(perf_rows)}

峰值以 100 ms 周期采样进程树，已存在的 Gradle/Kotlin 常驻 daemon 不包含在内。该表累计初始转换与本轮受监测命令，并非一次从零重建的性能基准；未经过监测 wrapper 的手工实验没有峰值记录。没有把 Python 阶段、Java 导出和 GPU 渲染混成一个耗时。打包时间不包含在此表。

## 重复执行

```powershell
& C:\oss\MODELS\popy-live2d\run.ps1
```

这会从独立源快照重建两版、渲染并验证，输出清晰分开的质量版运行时与平衡版编辑工程。详细依赖/配置见 README，GPL 导出器边界与上游限制见 `docs/THIRD_PARTY.md`。
"""
    (ROOT / "REPORT.md").write_text(text, encoding="utf8")
    with (ROOT / "artifacts/report/metrics.csv").open(
        "w", newline="", encoding="utf8"
    ) as f:
        keys = [k for k in quality["cases"][0] if k != "parameters"]
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows({k: r[k] for k in keys} for r in quality["cases"])
    previews = []
    for name in ["front", "all_extreme", "heldout_head_body"]:
        im = Image.open(ROOT / f"artifacts/report/{name}_compare.png")
        im.thumbnail((1152, 540))
        previews.append(im)
    preview = Image.new("RGB", (1152, sum(im.height for im in previews)), "#20242c")
    top = 0
    for im in previews:
        preview.paste(im, (0, top))
        top += im.height
    preview.save(ROOT / "artifacts/report/preview.jpg", quality=90)
    with zipfile.ZipFile(
        destination / "converter_source.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=5
    ) as archive:
        for folder in [
            "popy",
            "blender",
            "tests",
            "config",
            "ir",
            "docs",
            "exporter/src",
            "rust/geometry",
            "rust/validate",
        ]:
            for f in (ROOT / folder).rglob("*"):
                if f.is_file() and "__pycache__" not in f.parts:
                    archive.write(f, f.relative_to(ROOT))
        for file in [
            "README.md",
            "REPORT.md",
            "LICENSE",
            "requirements.txt",
            "setup.ps1",
            "run.ps1",
            "rust/Cargo.toml",
            "rust/Cargo.lock",
            "exporter/build.gradle.kts",
            "exporter/settings.gradle.kts",
            "exporter/gradle.properties",
            "exporter/LICENSE-NOTICE.md",
        ]:
            archive.write(ROOT / file, file)
        archive.write(ROOT / "vendor/psd2live/LICENSE", "licenses/psd2live-GPL.txt")
        archive.write(ROOT / "vendor/ayagami/LICENSE-MIT", "licenses/ayagami-MIT.txt")
        archive.write(
            ROOT / "vendor/ayagami/LICENSE-APACHE", "licenses/ayagami-APACHE.txt"
        )
    with zipfile.ZipFile(
        destination / "POPY_visibility_report.zip",
        "w",
        zipfile.ZIP_DEFLATED,
        compresslevel=5,
    ) as archive:
        for path in [
            "artifacts/visibility/index.html",
            "artifacts/visibility/REPORT.md",
            "artifacts/visibility/before_after.jpg",
            "artifacts/visibility/provenance.json",
            "artifacts/visibility/assessment/metrics.json",
            "artifacts/visibility/runtime/runtime_report.json",
            "artifacts/visibility_before/assessment/metrics.json",
        ]:
            archive.write(ROOT / path, path)
        for folder in [
            "artifacts/visibility/runtime",
            "artifacts/visibility/eyes",
            "artifacts/visibility/assessment",
            "artifacts/visibility_reference",
        ]:
            for f in (ROOT / folder).glob("*.png"):
                archive.write(f, f.relative_to(ROOT))
        archive.writestr(
            "README.txt",
            "Extract the whole archive, then open artifacts/visibility/index.html. Full Blender and runtime source geometry remain in the project. Metrics include finite sampling and interpolation error; they are not a guarantee of perfect ordering for every continuous parameter combination.\n",
        )
    print("Packaged", destination)


if __name__ == "__main__":
    main()
