# POPY 实测交付报告

## 无损通道与存储压缩（2026-11-04）

默认包为 `delivery/POPY_fullbody_channelpacked.zip`：MOC3 563,838,720→318,960,192 字节（−43.43%），ArtMesh 主关键形 5,439,788→1,438,609，四进程峰值 RSS 中位 1,972.97→1,751.75 MiB（约 −11.21%）。以首次读写分析去掉无用初始几何、分离 scalar Part 排序通道、共享浮点外观载体，再按位共享位置/颜色表和裁剪无用填充；不量化、不减面、不降低纹理/遮罩分辨率或控制范围。

另保留 `delivery/POPY_fullbody_factored.zip` 旋转载体候选，MOC3 为 319,789,056 字节。它用 63 个零缩放 Rotation 和三个外观祖先代理替代无用 Warp，并重定向 1,422 个静态辅助网格。四模型同进程对照的纯 CPU 平均改善约 5.86%；双模型同设备交替的整帧平均/P95 改善约 2.68% / 2.97%。但独立进程八轮整帧平均退步约 7.23%，所以未替换默认版。默认存储版独立进程平均略有改善而 P95 退步；不能称作稳定提速。**360,242 个 Glue、3,213,851 个 Glue 关键形仍在，卡顿及约 5 MB 目标未解决。**

两版各 1,803 姿态、1,238,608,713 个原始顶点的位模式及外观/有效排序比较为零差异，362 组 DX12 RGBA 均逐像素一致；保持全部原网格、UV、六张纹理、39 参数、58 动作、24 表情。116 项 Python / 7 项 Rust 测试、ZIP CRC、MOC 指纹和各 90 个资源引用验证通过。完整原始计时、保留的失败方案、Anon 统计、GitHub 研究及复现见 [优化说明](docs/REPRESENTATION_OPTIMIZATION.md) 和 `benchmarks/popy/fullbody/representation*.json`。原 `.blend`、旧交付及未修改 Ayagami 依赖保留；没有新 CMO3 或官方 Editor/Core 兼容性结论。

## 无损遮罩公共表达式优化（2026-11-04）

在上一轮严格无损模型上，将不透明遮罩的重复三角形并集分解为共享辅助 ArtMesh。固定网格成本纳入评分后，新增 425 个辅助网格和 49,995 个顶点；全部遮罩集合理论源绘制总数 82,894→19,532（不是逐帧实际 draw call 计数）。原始网格、三角形、纹理、39 参数、58 动作、24 表情不删减，MOC 体积增加约 0.30%。真实 MOC 的 1,803 姿态、1,148,467,728 个原始顶点比较全部逐位一致，复制误差为 0；Vulkan 和 DX12 各 362 个姿态的前后 RGBA 均完全一致。

Vulkan 八轮同序列交替计时：整帧中位数 195.74→164.20 ms（约 −16.11%），平均数 281.96→271.53 ms（约 −3.70%），P95 340.50→327.14 ms（约 −3.92%）。最大值仍达到数秒，因此**没有解决所有卡顿，也没有达到稳定实时性能**。只按顶点成本合并的首版虽然降低中位数，平均数与 P95 退步，失败记录保留；不能把理论绘制数量改善等同于帧率改善。

新版运行时包为 `delivery/POPY_fullbody_maskbatched.zip`；方法、后端差异、复现和限制见 [遮罩批处理](docs/MASK_BATCHING.md)。固定版本 Ayagami 保持未修改；官方 Cubism Core/Editor 和浏览器 WebGPU 性能仍未验证。完整数值记录为 `benchmarks/popy/fullbody/mask_batch.json`、`mask_batch_vulkan.json` 和 `mask_batch_initial.json`。

另在 Windows DX12 上重做前后各 362 个图像及八轮计时：中位数 256.28→212.49 ms，平均数 255.74→211.68 ms，P95 279.95→225.54 ms，分别下降约 17.08% / 17.23% / 19.43%。本次 DX12 未见数秒长帧，但常态耗时仍较高，不代表浏览器性能。104 项 Python / 7 项 Rust 测试通过，ZIP CRC、MOC 指纹及 90 个资源引用均通过校验。

## 运算图性能优化（2026-11-04）

保留全部现有拓扑、纹理、39 参数、58 动作和 24 表情，加入两种独立导出：严格无损的逐顶点依赖图批处理，以及对三维骨架常量 Glue 做代数因子转移的精度版。无损版 1,803 姿态/362 渲染逐位一致，CPU 中位开销约降 4%；精度版 Glue 减少 36.82%，CPU 中位数约降 22.28%，81 组身体/四肢最大重投影误差 0.035868→0.003569 px。精度版完整帧中位数下降 7.59%，但本次平均数和 P95 上升，GPU 长帧没有解决。保留原版和两种交付，优先严格无损包可保证原画面不变。完整数据、算法和限制见 [性能说明](docs/PERFORMANCE_OPTIMIZATION.md) 与 `benchmarks/popy/fullbody/performance_optimization.json`；96 项 Python 和 7 项 Rust 测试通过。

## 表情动作扩展（2026-11-04）

在动作细化版基础上增加 36 段 `emote_*`：12 类情绪各有 C/L/R 版本，合计 58 motion3、24 exp3、39 参数。眉眼/视线/嘴形与头、身体、手臂分时编排。仅研究本地 Anon 的表演类别，不复制资源、曲线或表情参数值。完整说明和能力限制见 [EXPRESSION_ACTIONS.md](docs/EXPRESSION_ACTIONS.md)，实际验证见 `benchmarks/popy/fullbody/expression_actions.json`。新增动作复用已验证的 MOC，不修改几何绑定；难过、抽泣和害羞仍受缺少苦眉、下撇嘴、泪水和脸红通道限制。

## 四肢动作细化（上一阶段）

本轮将全身版扩展为 39 个连续参数、22 段 motion3 和 16 个表情。独立前臂翻掌使挥手时掌面朝向镜头；腿部侧摆从 0～15° 扩展到 −25～35°，新增左右侧抬腿、侧步和左右摇摆。肩、肘、翻掌、腕摆分阶段衔接，单调 Hermite / Cubism Bezier 共享切线，避免每个关键点都停顿。实现、测量与边界见 [动作细化文档](docs/MOTION_REFINEMENT.md)。下文首版统计作为历史记录保留；最新数值以 `benchmarks/popy/fullbody/` 为准。

## 首版全身、四肢动作与表情

参考本地 Anon 的左右独立控制和动作/表情组织，基于 POPY 实际骨架实现 18 个新增控制：肩抬落/前后摆、肘、腕、张手/指向/握拳、髋前后/外展、膝和踝。保留原 19 个参数，总计 37 个连续参数；附带 15 个独立编写的 motion3 和 16 个 exp3。未复制 Anon 的角色素材、绑定或动作曲线。使用 `delivery/POPY_fullbody.zip` 和可编辑的 `delivery/POPY_fullbody.blend`，完整说明见 [全身扩展](docs/FULLBODY.md)。

四肢密集关键形原型在组合姿态下仍有约 14.29 px 的最大关节误差，最终改用原生 Glue 实现三维骨骼投影矩阵乘积和真实权重蒙皮。运算强度均在 0～1，运行时只需设置模型参数，没有外部顶点回调。最终 207 组 Blender / Ayagami 图像及 25 组几何探针中，保留源顶点的最大投影误差约 0.03288 px；独立重开 Blender 工程后，8 个检查点与参考几何最大差为 0 m。有限采样不能证明全部连续组合无误差。

模型含 5,446 个 ArtMesh、636,952 个顶点、745,166 个三角形及 6,381,022 个 ArtMesh 关键形。位置块无损共享使 MOC 从 1,471,098,368 降到 607,229,120 字节；5 组姿态图像和全部输出顶点在优化前后逐位一致。文件仍然很大，固定硬件上的参数求值及 GPU 完成耗时、导出约 20 GiB 进程树内存峰值见 `benchmarks/popy/fullbody/`，不能当作轻量商业模型。

原有面部修复作为几何基线保留，Blender 的头发/躯干清障也保留；抬臂不再扩张私有碰撞凸包。补充了白袜覆盖腿部及袜口内侧封口的专项回归。服装仍由源蒙皮驱动，平均深度排序、剔除和服装偏置近似处理交叉表面；没有布料碰撞、手/头发接触模拟或足底锁定。16 个表情预设使用已有审计控制，没有 Anon 专有的泪滴、爱心眼等新素材。本轮无新 CMO3，也未验证官方 Cubism Core/Editor。

## 最新面部修复

本轮最新交付为 `output/POPY_face_fixed` / `delivery/POPY_face_fixed.zip`。实现与限制见 [面部修复文档](docs/FACE_CLEARANCE.md)，完整公开数值见 `benchmarks/popy/face_repair/metrics.json`，本地逐姿态图像见 `artifacts/face_repair/report/index.html`。下文衣领、肩颈头发和更早转换结果作为历史基线保留。当前仍有极端复合姿态残差，未宣称全角度零穿模，也未更新或验证编辑器工程。

## 最新脖子与衣领修复

请使用 `delivery/POPY_neck_fixed.zip`。截图姿态错误露出 314→2，79 姿态合计 5,113→350；误遮挡 573→580。眼/脸/发饰指标保持原值，详见 [本轮报告](docs/COLLAR_CLEARANCE.md)。以下肩颈头发版及原始转换统计作为历史基线保留。

## 上一轮肩颈头发修复

2026-11-03 的后续修复同时提供 Blender 独立副本与原生 Live2D 后侧遮挡约束。77 姿态、60 相邻帧和 264 表情组合实测结果及已知退步见 [头发修复报告](docs/HAIR_CLEARANCE.md)，本地可视报告在 `artifacts/hair_repair_report/index.html`。新增文件以 `POPY_hair_fixed` 命名，前侧头发及全部绘制顺序轨道保留；下文为前一阶段的基线统计。

本次已实现并在用户当前 POPY 上运行真实链路：Blender bpy 数据/姿态 → 3D 源映射与局部 Rig IR → 原生 Warp/Rotation/ArtMesh 参数 → 真正 MOC3 → 固定版本 Ayagami GPU 渲染。没有 PSD 中转、序列帧模型、改扩展名文件或虚构的 Editor 验证。

## 本轮遮挡、接缝与连续动作修复

实际 Ayagami GPU 渲染 165 个姿态、比较 70 对相邻帧。Blender 参考图与眼部诊断均由真实深度渲染得到。眼部可见性比较允许 1 px 边界误差。

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| 眼部漏出像素 | 13978 | 467 |
| 眼部误遮挡像素 | 5574 | 1780 |
| 源接缝最大距离 px | 0.02343197008550215 | 9.226521776159533e-05 |
| 内部透明缺口指标 | 43566 | 3817 |
| 连续动作时域残差像素 | 230034 | 201187 |

透明缺口指标还包含超过 3 px 的轮廓偏差；时域残差也包含几何插值误差，不能直接等同于排序跳变数。零接缝距离只指可由同源位置、完整权重和全部 morph 证明等价的分块顶点，不证明没有其他遮挡洞。

已实现：皮肤、口腔、睫毛与前发的前景包络原生反向遮罩、眼白/瞳孔稳定层次、同源 Glue 焊接、绝对父空间关键形、冲突区域细分、姿态深度约束与邻域排序平滑、身体俯仰弦缩补偿。

仍有限制：Cubism 没有逐像素深度缓冲，交叉表面只能以分块与遮罩近似；未采样的多轴姿态、极端眼球运动及高阶表情组合仍需继续优化。没有声称所有连续角度的裂缝和错误覆盖均已消除。

参阅 index.html 和 assessment/metrics.json；红色诊断为漏出，蓝色为误遮挡。

## 本轮睫毛、眉毛与颈部头发修复

独立 Blender 深度诊断与实际 Ayagami GPU 渲染：264 个姿态，包含转头、俯仰、倾斜、单眼/双眼闭合、愤怒/抬眉与身体组合。

| 部位 | 漏出 1px 前→后 | 误遮挡 1px 前→后 | 漏出 3px 前→后 |
|---|---:|---:|---:|
| 眉毛 | 772 → 701 | 8,804 → 5,197 | 37 → 13 |
| 睫毛 | 2,383 → 2,032 | 22,607 → 4,753 | 433 → 93 |
| 颈部头发 | 54,609 → 26,133 | 5,821 → 6,050 | 45,595 → 20,703 |

这些指标比较表面可见区域，允许 1px 或 3px 边界误差；同时包含姿态插值导致的位置偏差，不能全部解释为穿模。不是所有连续参数组合均已验证。局部三维深度/排序仍属于 Cubism 二维表示的近似。

原始 .blend 与独立源快照均未被覆盖。复杂参数化 Glue 裁剪方案已保留在实验产物中；默认交付只使用通过当前质量与性能回归的策略。


表面遮挡的普通着色、RGB 深度诊断及全部样本在 `artifacts/surfaces/index.html`。原模型保存在 `artifacts/feature_before/`。新增 63 个连续控制样本未参与误差区域选择；局部支撑平面、身体旋转近似和未采样组合仍有残差。实现与成本取舍见 `docs/SURFACE_VISIBILITY.md`。

新增对比入口：`artifacts/visibility/index.html`；修复前模型与数据保存在 `artifacts/visibility_before/`。质量版已应用本轮全部修复；平衡版应用遮罩、焊接与排序修复，身体俯仰仍采用原有近似。

## 交付与体积

| 配置 | MOC3 MB | CMO3 MB | ArtMesh | 作者关键形数 | 参数到 GPU 完成中位 ms |
|---|---:|---:|---:|---:|---:|
| quality | 323.90 | 未完成（内存不足） | 1283 | 1,405,015 | 62.34 |
| balanced | 122.43 | 203.38 | 1283 | 468,445 | 69.85 |

以上 MB 按 1,000,000 字节计算。两版均 19 参数、最大层级 3、6 张 2048² 原始 UV 图集；质量版 8 个原生变形器，平衡版 4 个。质量版输出 168,764 个顶点和 146,173 个三角形，包含诊断统计中的不可见遮罩网格。高质量版 323.90 MB 的稠密局部关键形仍偏大，不能将其描述为已经解决轻量化。ZIP 压缩体积不等于运行时内存。

质量版入口：`output/POPY/POPY.model3.json`。平衡版入口及其匹配工程：`output/POPY_balanced/POPY.model3.json`、`POPY.cmo3`。此前约 216 MB MOC3 的密集质量版在尝试 CMO3 时，7/10 GiB Java 堆均 OOM，日志位于 `artifacts/failures/quality_cmo_10g.log`；当前版本关键形更多，沿用分开交付的方案，没有重新声称完成质量版 CMO3，也没有将平衡版工程放进质量版目录冒充一致。平衡版 CMO3 成功写出并读回，但官方 Cubism Editor 的打开/保存/重新导出仍未验证。

本轮平衡版 CMO3 在 16 GiB 最大堆下写入、24 GiB 下独立读回，16 GiB 读回仍会 OOM。默认 CLI 分别配置这两个阶段，内存较小的机器应使用 `export --no-cmo`。局部深度遮罩新增大量动画 Glue，当前质量版运行开销明显偏高，优先优化方向是减少裁剪辅助顶点和重复强度曲线，而不是再增加均匀姿态采样。

## 已在 POPY 实测完成

- Blender 5.2.2 LTS，14 Mesh、30,514 源顶点、51,288 源三角形、233 骨骼；贴图从 packed images 提取，两个 UV 层、原始权重与全部 morph 均保留审计/中间数据。
- 连续头部三轴（水平 ±45°、俯仰 ±30°、倾斜 ±30°）、左右独立闭眼、眼球 XY、嘴部开合和笑形、四个眉毛控制、笑眼、身体三轴与程序化呼吸。当前为半身、正交、原始 T-pose 配置。
- 质量版把身体 yaw/pitch 的 3×3 基与头部及局部表情联合投影；实际 Blender 独立渲染 28 个测试，包括非训练值、单眼/双眼、转头张嘴/眨眼、身体与头部运动、全极值。
- 每个姿态实际加载生成的 MOC3、设置参数、求值网格并由 Ayagami Vulkan 渲染，输出源网格误差、MAE、SSIM、轮廓 IoU/距离、误差热图和逐姿态对比图。Python 与 Rust 回归通过，实际数量及日志见 `artifacts/logs/`。
- 本轮 Blender MCP 未连接到运行实例，使用先前经 MCP 保存的独立快照，在后台副本上生成参考；没有声称再次检查了实时场景。独立 source snapshot 转换前后 SHA-256 一致，见 `source_integrity.json`。

## 视觉与几何实测

| 姿态 | 前景 MAE | SSIM | alpha IoU | 顶点 RMSE px | 顶点最大误差 px |
|---|---:|---:|---:|---:|---:|
| front | 0.00611 | 0.9753 | 0.99989 | 0.0001 | 0.000 |
| left | 0.00746 | 0.9704 | 0.99987 | 0.0001 | 0.000 |
| right | 0.00921 | 0.9645 | 0.99990 | 0.0001 | 0.000 |
| up | 0.00609 | 0.9738 | 0.99982 | 0.0001 | 0.000 |
| down | 0.00921 | 0.9672 | 0.99986 | 0.0001 | 0.000 |
| turn_mouth | 0.00678 | 0.9731 | 0.99988 | 0.0001 | 0.000 |
| turn_blink | 0.00661 | 0.9731 | 0.99992 | 0.0001 | 0.000 |
| heldout | 0.01478 | 0.9040 | 0.98779 | 1.2842 | 3.925 |
| heldout_head_body | 0.02084 | 0.8559 | 0.98568 | 2.1532 | 6.655 |
| head_body_pitch | 0.00669 | 0.9719 | 0.99986 | 0.0001 | 0.001 |
| all_extreme | 0.00779 | 0.9699 | 0.99983 | 0.0003 | 0.001 |

这里的图像比较统一为 albedo/emission，不保留源 MToon 的动态光照/描边；另有 `artifacts/original_mtoon/` 的 1152×1536 原材质审阅图。几何误差包含隐藏源顶点和 UV 拆点，不能只凭训练点约 0.02 px 的 RMSE 宣称所有连续组合都接近零误差。

当前最差前景 MAE 为 `heldout_head_body`：0.02084，顶点 RMSE 2.153 px。主要来自未采样参数中间值的多轴旋转残差插值和排序近似。全部数据见 `artifacts/report/metrics.json`，图形入口 `artifacts/report/index.html`。

## 迭代证据

- 最初仅平均深度排序：正面 MAE 约 0.052；用几何覆盖约束后约 0.006～0.008。
- 顶点直接插值 roll 的未见姿态 RMSE 约 6.1 px；原生 Rotation 与几何残差分解后，`heldout` 为 1.284 px。
- 平衡版 `all_extreme` RMSE 30.184 px、MAE 0.10156；质量版分别为 0.000 px、0.00779。这是增加身体 pitch 耦合的实测收益，代价是更多关键形与内存。
- 16 三角形小块版本有 1,383 ArtMesh；采用覆盖约束后允许更大的块，当前减少为 1283，仍保持可见质量。历史指标和 contact sheets 保留在 `artifacts/baseline/`、`artifacts/iterations/`。

## 已实现但仍有质量问题

- 质量版未见多参数姿态仍有几何插值误差；本轮减少了缺口和眼部穿透，但没有证明所有连续参数组合都无错误覆盖或跳变。
- 遮挡约束使用 384×512 栅格，质量版累计删除了 1549472 条循环约束（包含姿态中不活跃的全局连续性弱边），属于近似；Cubism 本身没有每像素 z-buffer。
- 附加笑眼的头部耦合与呼吸是近似。原始 MToon 高光、阴影和 outline 不属于本轮转换产物。
- 平衡版身体 pitch 和大角度头部组合不适合当作高质量模型使用；提供它主要是较小的运行时/编辑工程起点。

## 原型范围与尚未解决

- 可配置语义覆盖、候选识别和投影矩阵接口已实现；仅 POPY 实测，不能宣称适配任意命名/骨架/坐标系模型。
- CMO3 自身读回不是 Editor 兼容性证据；未在常见安装位置或开始菜单找到可用 Cubism Editor。质量版 CMO3 的内存问题仍需修改/优化导出适配。
- 未完成头发物理、physics3、motion3/exp3 求值与时序回归，也未声称有原模型不存在的 BlendShape。
- 未完成新的简化拓扑、重心重采样、通用 Warp 控制点优化、Jacobian/Laplacian 正则化、完整自交修复及超过当前参数域的验收。

下一步最有价值的是：在已实现的冲突区域细分上继续优化分块边界与局部动态遮罩，压缩身体与头部的耦合关键形，让 CMO3 writer 流式处理关键形，并在真正 Editor 中检查打开/保存/重新导出。继续增加均匀笛卡尔采样会进一步放大现有体积问题。

## 时耗与内存

GPU：NVIDIA GeForce RTX 5080 Laptop GPU / Vulkan。当前质量版参数更新到 GPU fence 完成中位数 62.34 ms；这是一次参数遍历，包含 CPU 准备和 render target 分配，不是长时 FPS 保证。另记录截图/文件写入耗时。

| 阶段 | 秒 | 进程树峰值 RSS MiB | exit |
|---|---:|---:|---:|
| audit | 7.59 | 806.5 | 0 |
| semantics | 0.48 | 30.1 | 0 |
| sampling | 55.64 | 1298.9 | 0 |
| geometry_ir | 109.25 | 462.4 | 0 |
| occlusion | 19.84 | 447.4 | 0 |
| export | 10.09 | 4262.5 | 0 |
| editor_geometry | 27.85 | 170.2 | 0 |
| editor_occlusion | 6.21 | 184.8 | 0 |
| editor_export | 52.40 | 7338.2 | 0 |
| editor_ayagami | 3.66 | 223.4 | 0 |
| editor_comparison | 22.93 | 353.8 | 0 |
| ayagami | 2.96 | 362.1 | 0 |
| comparison | 22.02 | 555.1 | 0 |
| rust_tests | 1.01 | 204.5 | 0 |
| regression | 0.33 | 237.5 | 0 |
| geometry_ir | 127.83 | 473.7 | 0 |
| occlusion | 20.46 | 489.9 | 0 |
| export | 10.37 | 3705.5 | 0 |
| editor_geometry | 54.31 | 178.7 | 0 |
| editor_occlusion | 6.98 | 166.8 | 0 |
| editor_export | 52.08 | 7486.6 | 0 |
| editor_ayagami | 4.42 | 230.2 | 0 |
| editor_comparison | 21.96 | 357.8 | 0 |
| ayagami | 2.34 | 399.2 | 0 |
| comparison | 22.18 | 569.6 | 0 |
| rust_tests | 0.45 | 47.6 | 0 |
| regression | 0.47 | 224.5 | 0 |
| delivery | 14.37 | 248.8 | 0 |
| visibility_runtime | 11.05 | 487.3 | 0 |
| visibility_eye_runtime | 6.34 | 398.3 | 0 |
| visibility_report | 0.55 | 48.4 | 0 |
| visibility_runtime | 10.77 | 399.2 | 0 |
| visibility_eye_runtime | 6.25 | 425.0 | 0 |
| visibility_report | 0.68 | 49.5 | 0 |
| ayagami | 2.45 | 434.5 | 0 |
| comparison | 25.26 | 629.8 | 0 |
| rust_tests | 0.33 | 45.7 | 0 |
| regression | 0.99 | 303.2 | 0 |
| delivery | 12.45 | 277.2 | 0 |
| delivery | 24.81 | 277.2 | 0 |
| feature_export | 10.61 | 5163.2 | 0 |
| feature_diagnostic_export | 10.38 | 4430.9 | 0 |
| feature_diagnostic_render | 5.27 | 891.2 | 0 |
| feature_runtime | 7.07 | 408.9 | 0 |
| feature_glue_export | 8.47 | 7281.8 | 0 |
| feature_glue_diagnostic_export | 8.82 | 7253.9 | 0 |
| feature_glue_diagnostic_render | 10.52 | 1050.4 | 0 |
| feature_glue_runtime | 18.88 | 508.3 | 0 |
| feature_glue_assessment | 14.79 | 103.5 | 0 |
| feature_neck_full_export | 8.44 | 7321.3 | 0 |
| feature_neck_full_render | 10.31 | 1064.8 | 0 |
| feature_neck_full_assessment | 14.46 | 103.1 | 0 |
| feature_planar_export | 17.62 | 7417.2 | 3 |
| feature_sparse_export | 11.13 | 9860.4 | 0 |
| feature_sparse_diagnostic_export | 11.16 | 9310.8 | 0 |
| feature_sparse_diagnostic_render | 29.09 | 827.4 | 0 |
| feature_sparse_runtime | 28.65 | 609.3 | 0 |
| feature_sparse_assessment | 14.10 | 104.3 | 0 |
| feature_light_occlusion | 138.14 | 1855.5 | 0 |
| feature_light_export | 8.09 | 5448.6 | 0 |
| feature_light_diagnostic_export | 7.32 | 5397.2 | 0 |
| feature_light_diagnostic_render | 3.71 | 1086.2 | 0 |
| feature_light_runtime | 4.56 | 430.3 | 0 |
| feature_light_assessment | 21.17 | 104.6 | 0 |
| feature_layered_export | 6.02 | 6200.5 | 0 |
| feature_layered_diagnostic_export | 5.66 | 6188.4 | 0 |
| feature_layered_diagnostic_render | 3.83 | 973.5 | 0 |
| feature_layered_runtime | 4.56 | 433.8 | 0 |
| feature_layered_assessment | 20.31 | 106.6 | 0 |
| feature_footprint_export | 8.82 | 6641.0 | 0 |
| feature_footprint_diagnostic_export | 7.43 | 6595.2 | 0 |
| feature_footprint_diagnostic_render | 6.99 | 977.9 | 0 |
| feature_footprint_runtime | 10.68 | 500.1 | 0 |
| feature_footprint_assessment | 19.86 | 105.1 | 0 |
| feature_neck_planar_export | 9.60 | 7294.8 | 0 |
| feature_neck_planar_diagnostic_export | 10.04 | 7078.2 | 0 |
| feature_neck_planar_diagnostic_render | 9.39 | 827.7 | 0 |
| feature_neck_planar_runtime | 14.49 | 534.6 | 0 |
| feature_neck_planar_assessment | 19.27 | 104.0 | 0 |
| surface_before_native | 6.64 | 405.3 | 0 |
| surface_before_diagnostic | 5.73 | 996.3 | 0 |
| surface_heldout_diagnostic | 12.50 | 925.3 | 0 |
| surface_before_assessment | 55.24 | 107.0 | 0 |
| geometry_ir | 251.48 | 1171.3 | 0 |
| surface_heldout_assessment | 51.14 | 106.9 | 0 |
| occlusion | 233.82 | 2280.1 | 0 |
| surface_final_export | 9.87 | 6855.6 | 0 |
| surface_final_native28 | 4.10 | 512.7 | 0 |
| surface_final_compare28 | 29.76 | 1242.4 | 0 |
| surface_diagnostic_export | 11.17 | 6590.4 | 0 |
| surface_native | 26.68 | 504.2 | 0 |
| surface_diagnostic | 15.43 | 1024.3 | 0 |
| surface_final_regression264 | 208.14 | 6670.9 | 0 |
| visibility_runtime | 16.39 | 500.1 | 0 |
| visibility_eye_runtime | 9.80 | 494.1 | 0 |
| visibility_report | 0.57 | 45.4 | 0 |
| surface_final_eyes165 | 146.70 | 1211.0 | 0 |
| surface_balanced_export | 794.91 | 7620.4 | 3 |
| surface_balanced_export_retry | 98.43 | 6268.4 | 4294967295 |
| surface_final2_export | 14.23 | 8917.2 | 0 |
| surface_final2_native28 | 5.97 | 545.6 | 0 |
| surface_final2_compare28 | 36.95 | 1475.9 | 0 |
| surface_diagnostic_export | 16.61 | 7718.4 | 0 |
| surface_native | 39.90 | 560.7 | 0 |
| surface_diagnostic | 25.00 | 1151.6 | 0 |
| surface_balanced_final_export | 277.02 | 10809.4 | 3 |
| surface_final2_regression264 | 238.99 | 7795.8 | 0 |
| visibility_runtime | 14.65 | 553.8 | 0 |
| visibility_eye_runtime | 10.87 | 534.0 | 0 |
| visibility_report | 0.43 | 49.3 | 0 |
| surface_final2_eyes165 | 152.99 | 1445.1 | 0 |
| surface_compressed_export | 0.12 | 5.0 | 1 |
| surface_compressed_export | 13.26 | 6969.7 | 0 |
| surface_diagnostic_export | 13.86 | 6807.6 | 0 |
| surface_native | 41.89 | 556.5 | 0 |
| surface_diagnostic | 26.83 | 1041.1 | 0 |
| visibility_runtime | 29.87 | 555.9 | 0 |
| visibility_eye_runtime | 19.75 | 538.2 | 0 |
| visibility_report | 2.28 | 50.9 | 0 |
| ayagami | 11.72 | 536.0 | 0 |
| comparison | 53.90 | 1469.7 | 0 |
| surface_balanced_compressed_export | 435.52 | 16708.5 | 0 |
| surface_balanced_compressed_roundtrip | 60.56 | 16799.4 | 3 |
| surface_balanced_native | 11.95 | 342.3 | 0 |
| surface_balanced_comparison | 49.22 | 1073.2 | 0 |
| surface_balanced_roundtrip_24g | 151.90 | 19985.5 | 0 |
| rust_tests | 2.79 | 47.8 | 0 |
| regression | 10.51 | 970.7 | 0 |
| surface_reader_build | 62.27 | 1388.7 | 0 |
| surface_consuming_read_16g | 66.65 | 16800.9 | 3 |
| surface_final_exporter_build | 6.83 | 123.4 | 0 |
| delivery | 57.12 | 2115.8 | 0 |

峰值以 100 ms 周期采样进程树，已存在的 Gradle/Kotlin 常驻 daemon 不包含在内。该表累计初始转换与本轮受监测命令，并非一次从零重建的性能基准；未经过监测 wrapper 的手工实验没有峰值记录。没有把 Python 阶段、Java 导出和 GPU 渲染混成一个耗时。打包时间不包含在此表。

## 重复执行

```powershell
& C:\oss\MODELS\popy-live2d\run.ps1
```

这会从独立源快照重建两版、渲染并验证，输出清晰分开的质量版运行时与平衡版编辑工程。详细依赖/配置见 README，GPL 导出器边界与上游限制见 `docs/THIRD_PARTY.md`。
# 本轮脖子/衣领修复

最新质量模型为 `output/POPY_neck_fixed`，打包为 `delivery/POPY_neck_fixed.zip`。79 姿态颈部错误露出 5,113→350，误遮挡 573→580；截图姿态 314→2、误遮挡均为 0。40 相邻帧额外变化 1,296→399。此前 341 姿态重新求值渲染，77 姿态眼/脸/发饰指标完全保持原值。42 项 Python、2 项 Rust 几何测试通过。详细方法、性能代价、复现命令和极值残差见 [COLLAR_CLEARANCE.md](docs/COLLAR_CLEARANCE.md)。本轮仅更新质量 MOC，旧 CMO3 未包含此修复。
