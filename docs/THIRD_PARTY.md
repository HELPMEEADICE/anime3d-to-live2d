# 来源和边界

| 组件 | 固定版本 | 本项目使用方式 | 许可 |
|---|---|---|---|
| PSD2Live | `bafabab1a360e208671d39adf5087c62f6cae1f2` | 单独 Kotlin/JVM 导出进程，直接构建 PuppetModel | Umamo、Cubism target 为 GPL-3.0；format-model/format-compile 为 MIT |
| Ayagami | `0d1d7aa3efe57b1b363b72e672e353c52a9c15d6` | 外部 Rust 程序调用公开解析器、driver、wgpu renderer | MIT/Apache-2.0，见 vendor 内文件 |
| Blender | 5.2.2 LTS | 当前运行实例经 MCP 只读审计、copy 快照；后台副本提取渲染 | 上游 Blender 自身许可 |
| POPY | 用户当前资产 | 本地实验和生成物 | 没有为角色资产声明新的许可 |

参考：[PSD2Live](https://github.com/tsunehimatoi/psd2live)、[Ayagami](https://github.com/AyagamiDev/ayagami)。

公开仓库不包含 `vendor/`、上游构建产物或角色资产。`setup.ps1` 按上表固定提交获取依赖；GPL 导出适配器的完整许可正文随 `exporter/LICENSE` 提供。根目录 MIT 许可只适用于其范围中声明的原创代码和研究记录，不对 GPL 适配器、第三方依赖或角色素材重新授权。

PSD2Live 的 MIT 中立格式模块不等于 MIT 二进制导出器。本实现没有把 GPL 导出代码复制进 MIT 几何模块，`exporter/` 作为明确 GPL 的独立可执行程序保留完整可构建源码。分发它仍须遵守 GPL 及依赖条款；没有把进程隔离当作任意下游闭源组合的许可结论。

唯一上游工作副本补丁：`Moc3RenderOrderLowering.kt` 的 ArtMesh 叶节点 `groupIndex` 从 `0` 改成 `-1`。Ayagami 的公开加载器验证叶节点不得引用一个子绘制组，否则拒绝模型。另一个必要修复在本项目适配器中：显式填充 `renderRoot`，避免模型可求值但无绘制节点。补丁由 `popy/setup_exporter.py` 精确、幂等应用；未向任何上游提交。

Ayagami 仓库的 AI 贡献/研究限制已读。按用户明确要求，仅在此独立项目中将未修改的固定版本作为运行时依赖调用，不修改 Ayagami 源码，也不向其上游提交 AI 生成内容。本实现不把 Ayagami 的验证作为 CMO3 工程兼容性验证。
