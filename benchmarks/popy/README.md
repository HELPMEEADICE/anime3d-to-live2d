# POPY 数值实验记录

这些 JSON 来自本地已完成的 Blender / Ayagami 实测，记录于 2026-11-03（Asia/Shanghai）。公开记录不包含角色图像、纹理、模型或可重建网格的顶点数据。

后续面部修复记录更新于 2026-11-04；各子目录保留当时的基线与实测范围。

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
# 后续肩颈头发修复

`hair_repair/` 保存 2026-11-03 的 77 姿态、60 连续帧对、Blender 实际网格相交和眼睛/脸部/发饰专项检查。旧版和新版均对齐到修正后的同一份 Blender 参考；因此不能和本目录上一阶段的颈部小 ROI 直接作比值。新报告明确记录误遮挡增加，以及保守三维避碰包络的大幅深度位移限制，见 [方法说明](../../docs/HAIR_CLEARANCE.md)。角色图像、模型与原始几何数据不在公开仓库。
# 后续脖子与衣领修复

`collar_repair/metrics.json` 保存截图角度和完整 79 姿态的同一参考前后比较、77 姿态眼/脸/发饰检查、原生 Glue 验证及模型哈希；`performance.json` 记录实际成功阶段的时间和进程树 RSS。脖子错误露出 5,113→350，误遮挡 573→580，不能解释为每个极端姿态均无穿透。详见 [方法与限制](../../docs/COLLAR_CLEARANCE.md)。本目录不含角色资产或渲染图。

# 后续面部穿透遮盖修复

`face_repair/metrics.json` 记录 218 组 Blender / 原生 MOC 图像对照、60 对连续帧、38 组口腔 ID 对照（主集合子集）、77 组眼/脸/发饰及 79 组衣领回归。几何筛查的 2,835 组不另算图像验收。截图三处错误像素 4/8/7→0；皮肤错误露出 11,651→9,618，头发 12,965→5,775，鼻线缺失 1,456→84。牙齿总错误 1,411→67，但漏出 16→65；头发超过 3px 的缺失也有退步。完整局限和高模型/运行成本见 [方法说明](../../docs/FACE_CLEARANCE.md)。`performance.json` 保留已测阶段，失败原型与最终路径分开标注，删除本机命令和绝对路径。模型和图像仍只保留本地。
## 全身扩展记录

新增 `fullbody/` 保存参考 Anon 后的 POPY 四肢、动作和表情数值记录。模型、纹理、源网格、动画 Blender 工程和渲染图仍只保留在本地；操作流程见 [全身扩展](../../docs/FULLBODY.md)。具体指标及尚存限制以 `fullbody/metrics.json` 为准。

动作细化版的 `fullbody/motion_refinement.json` 另记录翻掌前后源标志点测量、髋关节侧摆的足部位移、曲线接缝速度及相同参数序列的运行时计时。该阶段主回归为 238 姿态 / 45 组实际 MOC 几何探针，完整说明见 [动作细化](../../docs/MOTION_REFINEMENT.md)。首版与新版主报告的姿态集合不同，不能直接把两组中位耗时解释为版本性能变化。

表情动作版的 `fullbody/expression_actions.json` 保存新增 12 类 × 三朝向动作的局部面部指标、36 组头部几何探针、原有文件字节保留检查、Blender NLA 复开检查和 Anon 只读语义审计。本轮主回归扩展为 362 姿态 / 81 组实际 MOC 几何探针。预览由真实 MOC 离线求值生成，不代表实时帧率；说明见 [表情动作](../../docs/EXPRESSION_ACTIONS.md)。
