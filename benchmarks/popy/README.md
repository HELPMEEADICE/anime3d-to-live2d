# POPY 数值实验记录

这些 JSON 来自本地已完成的 Blender / Ayagami 实测，记录于 2026-11-03（Asia/Shanghai）。公开记录不包含角色图像、纹理、模型或可重建网格的顶点数据。

- `geometry_quality.json`、`geometry_balanced.json`：28 个独立 Blender 姿态的图像/几何指标、模型统计。
- `visibility_before.json`、`visibility_after.json`：165 姿态、70 对相邻帧，眼部可见性、同源接缝和时域残差。
- `runtime_quality.json`、`runtime_balanced.json`：实际 GPU、参数清单与运行耗时。
- `cmo_roundtrip.json`：平衡版 CMO3 在独立 JVM 内的自身格式库读回清单；并非官方 Cubism Editor 验证。
- `provenance.json`：本地模型/IR/参考用例的 SHA-256；这里只发布指纹，不发布角色二进制。
- `versions.json`：实测上游固定版本和环境。
- `performance.json`：已监测阶段的耗时/RSS 数值；删去本机命令及绝对路径，累计记录不等于一次全新转换基准。
- `surface_before.json`、`surface_after.json`：264 姿态的眉毛、睫毛与颈部头发可见性，包含 63 个未用于区域选择的连续表情样本。
- `surface_provenance.json`、`surface_native_glue.json`、`surface_scalar_compression.json`：最终模型指纹、原生 Glue 互操作检查、264 姿态的标量曲线精简前后原生顶点变化。
- `surface_runtime.json`、`validation.json`：本轮实际渲染耗时、测试数量与编辑工程内存条件。

眼部漏出总像素从 13,978 降为 467；同源接缝最大距离从 0.02343 px 降为约 0.000092 px。眼部指标允许 1 px 边界误差。透明缺口指标还包含超过 3 px 的轮廓偏差；时域残差还包含几何插值误差，不能直接作为离散排序跳变次数。

本轮表面修复以此前眼部修复版为基线：睫毛 3px 容差漏出 433→93，眉毛 37→13；颈部头发 1px 容差漏出 54,609→26,133，但误遮挡 5,821→6,050。原有缺口指标由上一版本 3,705 增至 3,817，时域残差由 206,598 降至 201,187；没有声称每项指标均改善。质量版规模增至 1283 个 ArtMesh、约 16.9 万顶点、约 9 万 Glue，当前运行开销仍偏高。新平衡版 CMO3 约 203 MB，独立自身格式库读回成功，未验证官方 Editor。

这些是有限采样的历史实测记录，不能证明任意连续姿态、其他资产或官方 Cubism 兼容性。公开测试只能重跑不含资产的几何单元测试；完整回归需要合法取得匹配的 POPY 资产并重新生成中间数据。
