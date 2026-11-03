# POPY 肩颈头发修复（2026-11-03）

这次针对用户截图中的三个姿态，修复后侧发梢穿过肩部、领口错误露出与断续覆盖。另交付带实际避碰修饰器的独立 Blender 工程。原始 `artifacts/POPY_source.blend` 未改动。

## 实测结果

相同的 77 个姿态包括截图对应的 3 个姿态、31 帧连续水平转动、31 帧连续倾斜和 12 个头身极值组合。图像比较使用修正后的 Blender 模型作为参考，旧版和新版 Live2D 都与**同一份**参考比较。ROI 从真实衣服顶点确定，覆盖肩颈与上胸，避免原来的小颈部 ROI 漏掉肩上的发梢。

| 指标 | 修复前 | 本次交付 |
|---|---:|---:|
| 77 姿态错误露出，1px 容差 | 71,761 | 11,977 |
| 77 姿态误遮挡，1px 容差 | 30,301 | 33,611 |
| 60 相邻帧额外变化像素 | 30,362 | 11,746 |
| 三张截图姿态的错误露出 | 6,025 | 73 |
| 三张截图姿态的误遮挡 | 335 | 396 |
| Blender 三角形相交对 | 114,383 | 2,505 |

**错误露出减少约 83%，误遮挡增加约 11%。** 这是保守后侧遮挡的质量取舍，不能解释为所有区域均改善。相邻帧指标包含运动插值误差，不能等同于纯排序跳变计数。相交统计在同一份求值细分拓扑上关闭/开启避碰修饰器比较，计的是 BVH 三角形对，不是穿透体积。

独立的眼睛、脸部、发饰 RGB 深度标记又覆盖了这 77 个姿态：眼睛和发饰的露出/误遮挡指标保持原值；脸部误遮挡 23,181 → 23,074，额外露出 5,829 → 5,840。另对 264 个表情/头身组合重新运行 Ayagami，眉毛和睫毛指标保持上一版原值。所有原有参数、三角形来源、顶点轨迹和绘制顺序轨道保持不变；本次没有把整片头发前置。

## 实现

- `popy/hair_semantics.py` 按原始三维连通关系划分发束，以源空间的前/后关系决定绕行方向。同一束不会在 Y=0 处被拆成两种碰撞路线。
- `blender/hair_collision.py` 在独立副本中复制身体、衣服、蝴蝶结和领带的碰撞几何，保留其蒙皮，使用 Geometry Nodes 建立随骨骼更新的闭合凸包。头发经两级 Simple 求值细分后，沿已审计的轴向投影到指定一侧，保留 20mm 间隙。原始可编辑网格、UV、骨架和材质保留；不依赖 Python 帧回调或烘焙动画帧。
- `popy/hair_clearance_ir.py` 仅把 214 个后侧低位 ArtMesh 约束在身体/服装的共享原生反向遮罩之后。95 个前侧低位 ArtMesh、原有前侧遮罩、眼睛、脸部及星星发饰全部保留。移除 4 个被替代的后侧曲面近似遮罩，保留 2 个前侧深度遮罩。
- `popy/hair_regression.py` 和 `popy/feature_regression.py` 使用真实 Blender 深度标记、独立诊断 MOC 与 Ayagami GPU 图像评估。`popy/hair_report.py` 输出独立 HTML、四列对照图、逐姿态指标和文件哈希。

原生修饰器的行为依据 [Blender Shrinkwrap 文档](https://docs.blender.org/manual/id/dev/modeling/modifiers/deform/shrinkwrap.html)；开放的单层衣服不适合作为可靠内外判定，因此最终使用闭合碰撞包络。当前算法及坐标系只在 POPY 上验证。

## 交付与重复运行

在已完成原始转换、安装过本仓库依赖的工作目录运行：

```powershell
python -m popy.hair_repair all
```

首次使用仍需先按 README 执行 `python -m popy.cli all` 生成审计、采样、原始 Rig 和基线模型。角色资产不随公开仓库分发。

平衡版是另一个独立模型，不具备质量版完整的头身联合拟合。它的 77 姿态肩颈指标为错误露出 126,216、误遮挡 123,645，明显差于质量版，尤其是复合身体俯仰。运行时请优先使用质量版；平衡版主要保留可继续编辑的工程成果，不能作为同等质量替代。

完整修复命令另存 Blender 工程，生成几何遮罩、质量版 MOC、平衡版 MOC/CMO，在 Ayagami 中渲染、读回 CMO、生成报告、运行测试及打包。可单独运行 `blender / rig / export / validate / editor / report / test / package` 阶段。`--no-editor` 可用于跳过 CMO 导出；编辑工程写入和读回仍分别按 16 / 24 GiB JVM 堆配置。没有使用官方 Cubism Editor 验证，CMO 自身库读回成功不能代替官方兼容性检查。

- `delivery/POPY_hair_fixed.zip`：本次质量版运行时资源。
- `delivery/POPY_hair_fixed_editable.zip`：本次平衡版 MOC 与匹配 CMO。
- `delivery/POPY_hair_fixed.blend`：保留原材质和骨架的独立 Blender 修复工程。
- `delivery/POPY_hair_repair_report.zip`：可独立打开的对比报告。
- `artifacts/hair_repair_report/index.html`：本地报告入口。
- `output/POPY_hair_clearance/POPY.model3.json`：最新质量版入口。

旧的 `output/POPY`、`output/POPY_balanced` 与旧交付 ZIP 作为基线保留。请使用带 `hair_fixed` 名称的新文件查看本次修复。

## 未解决的限制

仍有极端头身组合的遮挡与几何插值误差，尤其是水平转头 ±45°、低头 -30°、同时身体转动和倾斜。Blender 凸包偏保守，最大深度移动约 0.240m，后侧几何可能显得偏厚；仍有掠射相交，未解决头发自碰撞，也没有精细发束物理。下一步应使用保持发束长度的局部碰撞约束和 Laplacian/ARAP 优化，减少深度位移。

二维共享遮罩针对当前正面正交相机，与保守三维碰撞包络属于兼容近似，不是任意视角的精确三维深度系统。当前约 322 MB 的质量版 MOC 和较多原生 Glue 仍有运行成本。不能宣称所有连续角度均无裂缝或穿模。

公开提交只包含代码、文档和 `benchmarks/popy/hair_repair/` 数值记录。模型、纹理、图像、密集采样和 `.blend/.moc3/.cmo3` 保留本地；上游 Ayagami 未修改，也未向其上游提交内容。
