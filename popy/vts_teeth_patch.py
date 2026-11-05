"""Restore the complete original tooth clipping program in the VTS package.

This is a deterministic repair, not a rendered equivalence certification.
Run: python -m popy.vts_teeth_patch
"""

import json
import shutil
import zipfile

import numpy as np

from .compile_ir import ROOT, jsonwrite
from .mask_context import rewrite


def read_clips(path, inventory):
    raw = path.read_bytes()
    if raw[:8] != b"MOC3\x05\0\0\0":
        raise ValueError("Only the audited MOC5 layout is supported")
    offsets = np.frombuffer(raw, "<u4", 152, 64)
    counts = np.frombuffer(raw, "<u4", 64, int(offsets[0]))
    if int(counts[4]) != len(inventory):
        raise ValueError("Mesh inventory mismatch")
    ids = [
        raw[int(offsets[33]) + i * 64 : int(offsets[33]) + (i + 1) * 64]
        .split(b"\0")[0]
        .decode()
        for i in range(len(inventory))
    ]
    if ids != [m["id"] for m in inventory]:
        raise ValueError("Mesh identity mismatch")
    starts = np.frombuffer(raw, "<u4", len(ids), int(offsets[47]))
    sizes = np.frombuffer(raw, "<u4", len(ids), int(offsets[48]))
    refs = np.frombuffer(raw, "<u4", int(counts[17]), int(offsets[80]))
    return {
        name: [ids[int(j)] for j in refs[int(start) : int(start + size)]]
        for name, start, size in zip(ids, starts, sizes)
    }


def main():
    original = ROOT / "output/POPY_maskbatched/POPY.moc3"
    previous = ROOT / "delivery/POPY_VTS_48mask"
    target = ROOT / "delivery/POPY_VTS_TeethFixed"
    archive = target.with_suffix(".zip")
    inventory = json.loads(
        (
            ROOT / "artifacts/representation_final/candidate/model_inventory.json"
        ).read_text()
    )
    clips = read_clips(previous / "POPY.moc3", inventory)
    before = {tuple(sorted(c)) for c in clips.values() if c}
    teeth = [
        m
        for m in inventory
        if any(s.startswith("MaskDepth_Teeth_") for s in m["clips"])
    ]
    if len(teeth) != 55 or not all(
        m["invert_mask"] and "MaskMouthSkin_0_0" in m["clips"] for m in teeth
    ):
        raise ValueError("Unexpected POPY tooth inventory; refusing partial repair")
    # Restore entire original contexts, including any peer targets. Keeping their
    # original membership also preserves the mask renderer's target bounds.
    tooth_contexts = {tuple(sorted(m["clips"])) for m in teeth}
    changed = []
    for mesh in inventory:
        if tuple(sorted(mesh["clips"])) in tooth_contexts:
            if clips[mesh["id"]] != mesh["clips"]:
                changed.append(mesh["id"])
            clips[mesh["id"]] = mesh["clips"]
    target.mkdir(parents=True, exist_ok=True)
    for path in previous.iterdir():
        if path.name in {"POPY.moc3", "README.txt", "tested_poses.json", "preview.png"}:
            continue
        if path.is_dir():
            shutil.copytree(path, target / path.name, dirs_exist_ok=True)
        else:
            shutil.copy2(path, target / path.name)
    rewrite(
        original,
        target / "POPY.moc3",
        inventory,
        ROOT / "artifacts/vts_teeth_patch/rewrite.json",
        design=[],
        clip_overrides=clips,
    )
    after = {tuple(sorted(c)) for c in clips.values() if c}
    report = {
        "repair": "Restore all original per-surface tooth depth and mouth-skin inverse masks",
        "before_contexts": len(before),
        "after_contexts": len(after),
        "tooth_targets": len(teeth),
        "restored_targets": changed,
        "restored_contexts": len(after - before),
        "geometry_textures_keyforms_glue_unchanged": True,
        "vts_mask_setting": "256 masks, 512x512 pixels per mask or higher",
        "render_regression_run": False,
        "vts_app_tested": False,
        "other_occlusion_rules": "Identical to previous VTS delivery",
    }
    jsonwrite(target / "repair_manifest.json", report)
    (target / "README.txt").write_text(
        "POPY VTS 牙齿遮挡修复版\n\n"
        "安装：将整个 POPY_VTS_TeethFixed 文件夹放入\n"
        "VTube Studio_Data/StreamingAssets/Live2DModels/，加载 POPY.model3.json。\n"
        "首次加载选择 Auto-Setup。请替换旧版本或切换到本文件夹中的模型。\n\n"
        "重要：本版使用 80 组遮罩，超过 VTS 默认的 64 组。\n"
        "加载前在 VTS 的 Live2D Mask Config 中选择 256 masks，\n"
        "每组分辨率至少 512x512；已有更高分辨率时请保留。\n"
        "官方说明：https://github.com/DenchiSoft/VTubeStudio/wiki/Live2D-Items\n\n"
        "修复：恢复全部 55 个牙齿分片的原始局部三维深度遮罩与口部皮肤遮罩，\n"
        "补回旧版省掉的 32 组遮罩，防止牙齿从嘴唇/面部等前景透出。\n"
        "保留原几何、纹理、关键形、Glue、动作和表情；其他遮挡规则沿用旧版。\n\n"
        "按要求修复后直接交付，未新增渲染回归或 VTS 实机加载测试。\n"
        "不能将此包视为已完成全角度验证；其他区域仍有旧版的有限采样限制。\n"
        "文件体积和 Glue 运算量未在此补丁中优化。无新增 CMO3。\n",
        encoding="utf-8-sig",
    )
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
    ) as z:
        for path in sorted(target.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(target.parent))
    print(
        json.dumps(
            {
                "archive": str(archive),
                "bytes": archive.stat().st_size,
                "mask_contexts": len(after),
                "restored_targets": len(changed),
            }
        )
    )


if __name__ == "__main__":
    main()
