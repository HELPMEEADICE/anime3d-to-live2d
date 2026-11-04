# POPY：保留画质的关键形、通道及存储优化

默认交付为 **`delivery/POPY_fullbody_channelpacked.zip`**，加载 `POPY/POPY.model3.json`；本地入口为 `output/POPY_channelpacked/POPY.model3.json`。另保留 `delivery/POPY_fullbody_factored.zip` 旋转载体候选，入口位于 `output/POPY_factored/`。两个模型均通过真实 MOC 的 1,803 姿态逐位对照和 362 姿态 RGBA 对照。候选的整帧收益不稳定，没有替换默认包。

| 指标 | 上轮遮罩批处理基线 | 默认存储版 | 旋转载体候选 |
|---|---:|---:|---:|
| MOC3 字节 | 563,838,720 | 318,960,192 | 319,789,056 |
| ArtMesh 主关键形 | 5,439,788 | 1,438,609 | 1,438,609 |
| Glue | 360,242 | 360,242 | 360,242 |
| 顶点 / 三角形 | 686,971 / 804,842 | 相同 | 相同 |
| 参数 / 动作 / 表情 | 39 / 58 / 24 | 相同 | 相同 |
| DX12 峰值进程树 RSS 中位，MiB | 1,972.97 | 1,751.75 | 1,749.25（另一组八轮） |

默认版 MOC 减少 **43.43%**，RSS 减少约 **11.21%**；没有量化坐标、减面、缩小纹理/遮罩分辨率或削减参数范围。移出的 ArtMesh 几何网格仍需由 2,561,935 个 scalar Part 行表达排序，不能把 ArtMesh 行数下降 73.55% 说成全部类型关键形都下降这么多。六张 2048² 纹理、UV、所有 Glue 字段及运算顺序保留。

默认 MOC SHA-256：`f26af30cfb76248ba0dfd0a647ed930c3de42521c706c05337312ec3be4b8075`。候选：`eb0ca4e654790829fad8cdbf3b245f0edbec9c9c24b4261a40e97e016e88b88d`。

## 实际实现

默认版证明初始几何在首次读取前已被 Glue 覆盖，再把密集 ArtMesh 网格的排序通道分离到共享 Part；按父节点和参数轴共享浮点外观载体，保留透明度、颜色与失效依赖。排序组保留原位置和稳定顺序。MOC 后处理共享逐位相同的位置块及颜色行，通过公开运行时 API 遍历主关键形和 BlendShape，裁掉不会被读取的位置填充。完整推导见 [默认版算法及边界](REPRESENTATION_STORAGE.md)。

候选额外以 63 个零缩放 Rotation 替代无用 Warp，建立三个保留原外观浮点层级的代理，让 1,422 个初始值同样无用的静态辅助网格跳过头部 Warp；原始可见几何、遮罩及 Glue 程序不改。说明及未采用实验见 [旋转载体与数学实验](REPRESENTATION_ROTATION.md)。

每个版本均有 **1,238,608,713** 个原始顶点比较，顶点位模式、外观浮点值、有效深度、可见绘制顺序全为零差异；362 个全分辨率 RGBA 图像全为零差异。原有 Blender albedo 误差仍保留，没有声称解决毛发物理、衣物碰撞或 MToon 动态光照。

## 性能结果及未解决的问题

独立进程采用两轮 ABBA，每版本四个进程，每个 5 帧预热加 60 帧测量。默认存储版整帧平均 295.90→285.03 ms，但 P95 为 395.58→409.50 ms，没有稳定提速。旋转载体在另一组八轮中平均 269.53→289.01 ms、P95 367.98→376.82 ms，发生退步，故保持候选身份。

补做同进程逐姿态旋转/逆序对照：四版本各 240 次纯 CPU Driver 求值，基线/默认/仅身体旋转/完整旋转载体的平均为 99.383 / 99.860 / 96.485 / 93.563 ms，完整候选改善约 **5.86%**。同设备、同目标、交替姿态的两模型 DX12 测试各 240 次，基线/完整候选平均为 223.10 / 217.11 ms，P95 为 247.32 / 239.98 ms，约改善 **2.68% / 2.97%**。双模型驻留不同于普通单模型宿主，不能用相对诊断覆盖独立进程的退步记录。

桌面环境未隔离，轮次波动较大；尚未验证浏览器 WebGPU 或官方 Cubism Core。**360,242 个 Glue、3,213,851 个 Glue 关键形及约 201 MB 的唯一位置 float 数据仍在，卡顿未彻底解决，也未达到约 5 MB。** 本地 Anon 实际只有 5,705 个 ArtMesh 主关键形、6 个 Glue、9,190 个顶点，以 537 个分层 Deformer 组织绑定，MOC 为 4,633,024 字节。仅读取统计，没有复制资源或绑定。

尝试过的 arena 图着色虽减少 Glue，却增加输出拷贝并在初测变慢；不可变表达式 DAG 和实数自插值消除目前仅为审计，没有混入默认版。后续低秩拟合应回到真实三维骨骼局部投影基，保留遮挡边界和表情耦合残差，不能任意截断后称作无损。GitHub 研究、官方资料及未复用项目见候选说明的来源链接。

## 复现和记录

需要匹配的本地 POPY 遮罩批处理 IR、资产与参考数据。默认流程：

```powershell
python -m popy.representation_review all
```

旋转载体候选：

```powershell
$env:POPY_REPRESENTATION_CONFIG='config/representation_rotation.json'
python -m popy.representation_review all
Remove-Item Env:POPY_REPRESENTATION_CONFIG
```

阶段为 `build|rig|export|pack|validate|images|benchmark|paired|test|report|package`；`all` 隔离并监测每阶段。MOC、资源、配置、计划和用例绑定 SHA；验收不完整或陈旧时拒绝打包。数学优化器、打包器及外部验证程序是独立 MIT 代码，导出桥的层级/Part/Rotation 支持留在独立 GPL 进程。见 [许可边界](THIRD_PARTY.md)。

- 默认数值：`benchmarks/popy/fullbody/representation.json`；图像报告：`artifacts/representation_final/report/index.html`。
- 候选数值：`benchmarks/popy/fullbody/representation_rotation.json`；图像报告：`artifacts/representation_fast/report/index.html`。
- 实验与 Anon 统计：`benchmarks/popy/fullbody/representation_experiments.json`。
- 两个 ZIP 的 CRC、MOC 指纹及 90 个资源引用均实际验证。代码、文档和数值公开，角色模型、图片、纹理与 ZIP 留在本地。

没有新 CMO3；历史编辑工程不含本轮转换。Ayagami 验证不等于官方 Core 或 Editor 兼容性验证。
