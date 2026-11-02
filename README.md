# Anime3D → Live2D

Geometry-driven Blender/Python/Rust research converter with a separate GPL Cubism exporter and an Ayagami GPU validation harness.

**公开仓库包含代码、研究文档与数值测试记录。** POPY 源模型、纹理、角色渲染、`.blend/.moc3/.cmo3`、密集几何采样及本地交付 ZIP 不随仓库发布，其再分发权限尚未确认。下文的 `artifacts/`、`output/` 和 `delivery/` 是本地实验路径或运行后生成的位置，刚克隆仓库时并不存在。可直接阅读 [实测研究报告](REPORT.md)、[算法与格式边界](docs/RESEARCH.md)、[公开数值记录](benchmarks/popy/README.md) 和 [许可说明](docs/THIRD_PARTY.md)。

这是一个已在当前 POPY 上运行的几何驱动研发实现。它从 Blender 的真实网格、蒙皮姿态、Shape Keys 和 UV 构建参数模型，生成真正的 MOC3 与 CMO3，并调用 Ayagami 在无窗口 GPU 环境中渲染和回归。内部不经过 PSD，也不将姿势作为序列帧贴图。

**当前仍不是“任意模型一键得到商业级 Live2D”的成品。** 当前交付是 POPY 原始 T-pose 的半身配置，笑眼的姿态耦合、复杂遮挡、头发物理和模型成本仍有限制。质量版已联合拟合身体俯仰；完整参数极值经过实际回归。质量版 CMO3 在 7/10 GiB 堆下均遇到导出器内存不足，因此另交付与其自身 MOC3 匹配的平衡版 CMO3。它通过自身格式库重新读取，但没有在官方 Cubism Editor 中验证，不能宣称编辑器兼容。

## 当前交付位置

- `output/POPY/POPY.model3.json`：质量版运行时入口，旁边有真实 `.moc3/.cdi3.json` 与纹理，**没有冒充已完成的质量版 CMO3**。
- `output/POPY_balanced/POPY.model3.json`：平衡版入口，包含其匹配的 `.moc3/.cmo3`。身体俯仰独立时准确，和头部的大幅组合有明显误差。
- `artifacts/report/index.html`：逐姿态 Blender / Ayagami / 误差图、各项指标。
- `artifacts/visibility/index.html`：本轮眼部遮挡、接缝和连续动作修复的 165 姿态 / 70 相邻帧对比，保留修复前基线。
- `REPORT.md`：最终实测统计、阶段改进与未解决问题。
- `artifacts/rig/POPY.rig.json` 和 `keyforms.f32`：独立中间表示与真实关键形。
- `artifacts/samples/`：原始三角形、loop UV、骨骼、权重、全部 Shape Key 差分、105 组头部三轴投影来源。
- `artifacts/reference/`、`artifacts/ayagami/`：参考与实际运行时图像、运行时求值顶点。
- `artifacts/original_mtoon/`：保留原 MToon 材质的独立高分辨率审阅图。
- `artifacts/POPY_source.blend`：从当前 Blender 通过 MCP 保存的独立快照，转换不覆盖它。

## 重复运行

已实测环境为 Windows、Python 3.14、Rust、JDK 26、Blender 5.2.2 LTS，以及支持 Vulkan 的 GPU。主要依赖版本记录在锁文件和 `benchmarks/popy/versions.json`；Python 数值依赖目前使用 requirements.txt 中的版本范围。新环境执行：

```powershell
git clone https://github.com/HELPMEEADICE/anime3d-to-live2d.git
cd anime3d-to-live2d
.\setup.ps1
$env:JAVA_HOME = 'C:\path\to\jdk-26'
# 将有权使用的、与当前配置匹配的独立源快照放入此位置。
Copy-Item -LiteralPath 'C:\path\to\POPY.blend' -Destination artifacts\POPY_source.blend
.\run.ps1
```

也可用 `python -m popy.cli all --skip-build` 复用已构建程序。完整命令执行审计、后台采样与渲染、Rust 分块、IR 拟合、遮挡排序优化、Cubism 导出、Ayagami 实际渲染、图像/几何比较和回归测试。错误会停止流程，详细日志在 `artifacts/logs/`。

运行前需要匹配 `config/semantics.json`、骨架、对象与真实 Shape Keys。当前仅 POPY 实测，其他角色需要重新审计和适配，不能直接使用这份配置宣称通用转换成功。Blender 安装位置不同时：

```powershell
python -m popy.cli all --blender 'C:\path\to\blender.exe'
```

JDK 可通过 `JAVA_HOME` 设置，源快照默认在 `artifacts/POPY_source.blend`。重新安装依赖需要网络；`rust/Cargo.lock` 固定实际测试的 Rust 依赖。导出器需要 JDK 26，首次构建会下载 Gradle/Kotlin 依赖。

不需要角色素材的测试（先安装 Python 依赖并获取固定版本的上游依赖）：

```powershell
python -m unittest discover -s tests -p test_pipeline.py -k GeometryTests -v
python -m unittest discover -s tests -p test_visibility.py -k VisibilityGeometryTests -v
cargo test --manifest-path rust/Cargo.toml -p popy-geometry --locked
```

上述测试只检查几何算法，不代替实机图像验证。完整的 16 项 Python 回归依赖本地生成的模型、图像和报告；仓库不带这些角色资产，直接执行完整回归会提示缺少数据。

单独重跑：`python -m popy.cli sample|rig|export|validate|report|test`。`export --no-cmo` 只输出质量版 MOC3，适合调试。默认完整流程还会独立构建、导出并渲染验证平衡版工程；两版文件不会混放。`export --full-cmo` 可再次尝试完整质量版 CMO3，但本轮已实测其超过 10 GiB Java 堆，不作为默认可完成路径。

## 配置与实现

`config/semantics.json` 中的名称来自对当前场景的实际检查，不是对任意模型的命名假设。`popy/semantics.py` 结合骨骼层级、顶点组、空间范围和命名生成候选，低可信度不会偷偷生成控制。替换模型时必须检查候选和覆盖映射，包括 gaze 对象，没有验证跨模型通用性。

`config/convert.json` 定义画布、相机、采样域、网格分块预算。改变相机或采样键后应从 `sample` 重跑。支持投影矩阵的透视除法，但**实际质量验收仅使用当前正交相机**。

头部 X/Y 使用 15° 间隔，Z 的真实三维残差采样为 -30/0/30°；原生旋转变形器承担连续 Z 转动。质量版将身体水平转动/俯仰的 3×3 组合纳入联合关键形，平衡版只联合水平转动；眼睛、嘴和眉毛按三角形的实际形变支持域分组，仅在必要的局部参数上组合。左右眼独立、眼球 XY、嘴部开合/笑形、四个眉毛控制、笑眼、呼吸和身体三轴共 19 个参数。

遮挡使用真实几何深度约束：Rust 在 384×512 投影上建立覆盖关系，以共享排序参考与参数邻域平滑减少跳变，当前姿态的真实深度优先于全局弱约束。24 个高冲突区域按配置自动细分。同源位置、完整蒙皮权重和全部 morph 一致的分块边界用原生 Glue 焊接，关键形存绝对父空间坐标以避免浮点相消。

眼部采用皮肤、口腔、睫毛与前发真实表面的原生反向遮罩，按重心坐标裁掉确定在眼后方的表面；眼白、瞳孔维持稳定层次，眼球转动不再使眼白跨层覆盖瞳孔，口腔也可遮住投影到嘴部开口中的眼白。质量版另用原生曲线补偿身体俯仰中间值的弦缩。该方案仍不提供逐像素深度缓冲，交叉表面和未采样组合存在近似。

单独重跑新增回归：`python -m popy.cli visibility`。若源快照、参考姿态和渲染配置均未变化，可执行 `python -m popy.visibility_regression --reuse-reference`，复用已经独立渲染的 Blender 参考图。完整 `run.ps1` 会重新生成参考并验证，不默认沿用缓存。

本轮没有生成 physics3、motion3 或 exp3：没有用空壳文件暗示已完成头发动力学、动作和表情文件求值。静态表情参数可以直接控制，运行时验证不依赖这些附属文件。

研究、格式能力边界、许可与后续优化见 `docs/RESEARCH.md` 和 `docs/THIRD_PARTY.md`。
