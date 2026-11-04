# 分层 Warp 拟合与遮罩预算实验

本轮是实际导出、运行和比较后的研究结果，**没有修复默认模型的遮罩兼容性，也没有把失败候选作为交付版**。原 MOC、源工程与默认配置未修改。当前 Blender MCP 只读连接到默认 Cube 场景，因此使用已有 POPY 三维提取数据和参考图。

## 报错原因

从真实运行时 inventory 统计，默认模型有 **1,490 个不同的非空遮罩源集合**、1,980 个被裁剪对象，单个源列表最多 398 项。上轮公共表达式优化减少的是重复源绘制量，没有减少组合数。这是两个不同指标，不能用绘制量下降声称修复加载器的 mask budget。

[官方遮罩说明](https://docs.live2d.com/en/cubism-editor-manual/clipping-mask/) 列出的默认数量包括 Native/Web 的 36/model、Unity 的 64/scene；实际宿主可有其他配置。[VTube Studio 文档](https://github.com/DenchiSoft/VTubeStudio/wiki/Live2D-Items) 描述默认总计 64 和可调整配置。当前 1,490 组甚至超过 256，因此不能建议简单调到 256 就解决问题。Native 的高精度逐次遮罩模式可绕开数量限制，但由宿主决定是否暴露，且不是本项目的压缩/性能修复。

## 两个真实遮罩候选：均未通过画质验收

用户已明确优先兼容 **VTube Studio**。后续按默认合计 64 组预算设计；其他已加载模型也占用预算。没有发现运行中的 VTS，默认 Steam 库也未找到安装，因此未进行 VTS 程序内加载验证。

新增 VTS 分组使用原三维空间中心，并将目标/源依赖作为不能共组的冲突约束：任何新组都不能把组内目标引入自身 mask，且不得删除原遮挡源。超过容量时允许显式溢出，不能通过删除源伪造达标。实际候选 `output/POPY_vts_safe_trial/` 为 **53 组**、最大源列表 597，所有原始源保留、新增自遮罩为零。但首批 5 姿态合计仍有 **6,795 个像素**不同，最大差 255，**没有通过画质验收，也不是 VTS 兼容性交付**。数值见 `benchmarks/popy/fullbody/vts_mask_trial.json`。

自遮罩是较早语义合并的真实缺陷：面部粗并集把 237 个目标也作为了源。禁止这种自引用改善了候选，但额外源仍可覆盖其他目标；单纯图着色和静态空间分区不足以解决动态可见性。删除组内源的试验也失败，VTS 模式已经禁止该选项。

`popy.mask_context` 核对 MOC 的 ID 和每个原始 clips 列表后，仅重写引用池，保留全部原 Glue、坐标、拓扑、材质及参数。它拒绝覆盖源文件，输出只作为实验。

| 方案 | 遮罩组合 | 首批 5 姿态错误像素总数 | 最大通道差 |
|---|---:|---:|---:|
| 按语义对象合并 | 5 | 10,607 | 255 |
| 按原始 3D 顶点中心空间聚类 | 32 | 9,642 | 255 |

两份 MOC 均实际由 Ayagami 加载和渲染，分别位于 `output/POPY_mask_context_trial/`、`output/POPY_mask_spatial_trial/`，**请勿用作质量模型**。法向/反向遮罩分开；空间聚类由 POPY 源顶点坐标驱动，固定随机种子。减少数量成功并不代表遮挡正确：目标局部专用的裁剪片段合并后会遮掉其他目标；按静态空间邻近分组仍不能保证在转头/表情中一致。

## 真正接入导出的局部 Warp 拟合

`popy.warp_fit` 对原始取样几何建立空间双线性基，求所有存储姿态的最小二乘控制点，计算每个原顶点残差，不从 PNG 轮廓猜形。保留不能通过门槛的原网格，跳过已被 Glue 覆盖初始值或包含独立 BlendShape 的对象。本轮 488 个网格中，只有 5 个 Headwear 局部片通过数值代理门槛，说明单一粗格点不适合直接覆盖面部和复杂三维曲率。

通过的五个局部格点挂到原投影父节点下，ArtMesh 使用归一化局部坐标，排序网格分离至 Part。所有参数轴和存储姿态保留，原 Glue 程序不变。实际生成 `output/POPY_spatial_warp_trial/POPY.moc3`：ArtMesh 主关键形 1,438,609→1,433,889，新增 5 个局部 Warp。此原型没有把骨架的 360,242 个 Glue 全部替换掉，没有声称解决整体运行时成本。

362 姿态原生对照共比较 248,683,502 个顶点，最大变化 **0.0007422921 px**；外观浮点值、有效深度和绘制序列零差异，但顶点不再逐位相同。严格检查器按设计返回非零并保留报告。362 张 RGBA 对比累计 **97 个像素**不同，最大通道差 133。即使几何差小于千分之一像素，边缘跨过采样位置仍能改变光栅值，因此这个模型只能标为拟合原型，不能套用旧版“像素无损”的结论。

数值 admission 中的 `conservative_screen_proxy_px` 只是带安全倍率的筛选代理，**不是父级 Jacobian 或完整组合误差的证明**。父 Warp 与子 Warp 的组合不普遍可交换，必须导出后再测，不能仅凭局部拟合残差验收。

## 复现

```powershell
# 只审计，超过预算返回非零；不会生成降质模型
python -m popy.mask_context --audit-only --limit 36
python -m popy.mask_context --audit-only --vts
# 实验性遮罩合并，默认输出 trial 路径
python -m popy.mask_context
python -m popy.mask_context --spatial-rig artifacts/representation_final/rig/POPY.rig.json --out output/POPY_mask_spatial_trial/POPY.moc3
# 保留源并禁止自遮罩的 VTS 预算候选，仍须画质验收
python -m popy.mask_context --spatial-rig artifacts/representation_final/rig/POPY.rig.json --out output/POPY_vts_safe_trial/POPY.moc3 --vts
# 拟合并生成独立 IR；未通过门槛的网格原样保留
python -m popy.warp_fit --emit artifacts/deformer_fit/rig
# 已安装的独立导出桥；输出仅作为试验，不覆盖默认模型
$env:JAVA_HOME='C:\lib\java'
$env:JAVA_OPTS='-Xmx20g -XX:+UseG1GC -XX:+ExitOnOutOfMemoryError'
exporter/build/install/popy-cubism-exporter/bin/popy-cubism-exporter.bat artifacts/deformer_fit/rig/POPY.rig.json output/POPY_spatial_warp_trial --no-cmo --deduplicate-positions
```

使用既有独立 GPL 导出器将该 IR 导出，再用原生 `hierarchy_equivalence` 和 `popy-validate` 检查。原始结果在 `artifacts/deformer_fit/`；不含角色几何的公开数值为 `benchmarks/popy/fullbody/spatial_fit_masks.json`。

下一项应实现参数连续的可见表面裁切/局部遮挡拓扑分区，使大量目标专用 mask 成为可见几何或少量共享语义 mask；再用分层局部投影基和可控残差表达头部及骨架。现在不能把低误差局部拟合、mask 数量通过或 Ayagami 加载成功，单独当作完成了整个转换架构。
