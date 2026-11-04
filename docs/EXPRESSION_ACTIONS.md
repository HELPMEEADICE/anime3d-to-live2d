# POPY 表情动作

在四肢细化版上新增 12 类、每类 C/L/R 三个朝向的 36 段 `motion3`。总计 39 个连续参数、58 段动作、24 个静态 `exp3`。这些是随时间变化的眉眼、眼神、嘴形、头身与手臂编排；模型仍连续求值原来的 3D 来源绑定，没有用预渲染帧替换模型。

## 动作目录

| 动作类型 | 文件名前缀 | 表演 |
| --- | --- | --- |
| 微笑 | `emote_smile` | 眼神与微笑先展开，轻微歪头、手臂跟随 |
| 开心笑 | `emote_laugh` | 眉眼笑意、不同幅度的嘴部及头身节奏 |
| 生气 | `emote_angry` | 皱眉、收眼睑、握拳与前倾 |
| 失落 | `emote_sad` | 低头、垂眼、手臂放松；没有原生苦眉或下撇嘴 |
| 抽泣姿态 | `emote_cry` | 低眼睑、嘴形与轻微抽动；没有泪水或湿眼特效 |
| 害羞 | `emote_shy` | 躲闪视线、低头微笑与手部收拢；没有脸红 |
| 惊讶 | `emote_surprise` | 快速张嘴、抬眉、后仰，再缓慢恢复 |
| 疑问 | `emote_question` | 非对称眉眼、歪头、翻掌 |
| 思考 | `emote_thinking` | 先看向一侧、头手跟随、末段轻点头 |
| 认真 | `emote_serious` | 收眉眼、轻微点头确认 |
| 俏皮眨眼 | `emote_wink` | 单眼闭合与笑意、歪头和抬手；R 版本换另一眼 |
| 困倦 | `emote_drowsy` | 逐渐垂下眼睑和头部，轻微张嘴 |

例如 `emote_laugh_C.motion3.json`。C 为正面，L/R 为 `ParamAngleX` +24°/−24° 的转头版本；配合小幅身体转动、提前的眼神变化，并镜像非对称的眉眼/手势。它们都从中性进入，最终每个通道回到中性。

新静态预设为 `smile_soft`、`shy`、`sad`、`cry`、`worried`、`skeptical`、`confident`、`drowsy`。保留原 16 个预设及 22 段四肢动作。可编辑配置见 `config/expression_actions.json`；独立编排器为 `popy/expression_design.py`。`output/POPY_fullbody/emotion_catalog.json` 随模型包交付动作名、中文说明、持续时间和能力限制。

## 播放和复现

重新加载 `delivery/POPY_fullbody.zip` 内的 `POPY.model3.json`，选择 `emote_*` 动作。动态表情已经写在 motion3 内，播放时不要另外叠加固定 `exp3`，否则静态 Overwrite 表情会覆盖眼睛/眉毛/嘴的动画。宿主仍需自行实现 motion3 播放；没有改动 Ayagami 上游。

```powershell
# 已有完整模型与本地源资产时：仅更新动作，不重建 MOC
python -m popy.fullbody assets
python -m popy.fullbody reference
python -m popy.fullbody blender
python -m popy.fullbody validate
python -m popy.fullbody report
python -m popy.fullbody expressions
python -m popy.fullbody test
python -m popy.fullbody package

# 单段动作的完整身体预览
python -m popy.fullbody frames --motion emote_shy_R --fps 15
```

`python -m popy.fullbody all` 也包括新增表情动作阶段。`expressions` 阶段产生 12 段正面离线预览，以及 36 个组合的 Blender/Ayagami/热力图。`python -m popy.expression_review --previews --fps 12` 可单独重建动作展示。离线 WebP 的帧率不代表实际运行速度。

独立 Blender 工程包含 58 段动作和 24 个表情，共 82 个 NLA 片段，按 30 fps 烘焙；通过时间轴标记定位。46 根控制骨骼与已有脸/眉 Shape Key 可继续编辑。源 `.blend` 不覆盖，后台实验也不修改用户实时场景。

## 参考边界与验证

只从本地 `../anon` 研究表情类别、左右版本组织和头/眼/手协作方式。源目录实际有 66 个 motion3、36 个 exp3；抽样检查了微笑、生气、难过、哭泣、惊讶、疑问、思考、认真和眨眼的动态控制类型。POPY 的数值、时序、手势及 Bézier 曲线独立编写，没有复制 Anon 资源或直接移植其模型专属控制。`popy/expression_review.py` 保存只读审计及来源摘要。

曲线继续使用不超出关键值的 C1 单调 Hermite/Bézier 插值。眼神先于脸部展开，头部晚于眼神，身体、肘、腕再错峰跟随；笑与抽泣使用不等幅脉冲，惊讶包含快速反应后回弹，认真/思考包含点头。附带 191 个时间样本/曲线的原始数值范围检查，避免仅靠输出钳制掩盖超范围曲线。

图像回归覆盖全部新增动作的 1/4、1/2、3/4 时刻，以及新静态表情的正面/转头。除全图轮廓/颜色指标，还单独比较投影面部包围区域的 MAE、SSIM 和 36 个中段的实际 MOC 头部顶点。面部框可能包含头发；头部几何误差包含被遮挡顶点、原有修复偏移，不能当成可见面部像素误差。数值报告为 `benchmarks/popy/fullbody/expression_actions.json`；本地展示入口为 `artifacts/expression_actions/index.html`，也加入完整交付报告。

难过/抽泣缺少独立苦眉、下撇嘴，害羞没有腮红，未增加泪水、爱心眼、脸部阴影或表情图层。动作只组合模型已有的真实控制。这次增加表现编排，没有解决所有接触碰撞、足底约束、动作间实时过渡或运行成本问题。没有新增 CMO3，没有官方 Cubism Editor/Core 验证。

## 本机实测结果

全项目 86 项 Python 测试通过。362 个主集合姿态、81 组实际 MOC 几何探针通过回归；其中新增表情相关图像 124 组（36 段动作各 3 个时刻，加 8 个静态表情各 2 个朝向），另测 36 个中段的头部源顶点。独立重开 Blender 的 20 个检查点每组 141,633 个顶点，最大差为 0 m。

| 指标 | 实测 |
| --- | --- |
| 主集合轮廓 IoU 最低值 | 0.992599 |
| 主集合平均前景 RGB MAE，0～1 | 0.012144 |
| 保留的四肢源顶点最大投影误差 | 0.035869 px |
| 表情专项面部区域平均 MAE | 0.022844 |
| 面部区域最差 MAE | 0.033934，左转惊讶 1/4 时刻 |
| 头部源顶点最大误差 | 1.067121 px，左转抽泣中段 |
| 头部源顶点最差 RMSE | 0.635048 px |
| 离线动态预览 | 12 段 / 564 帧，12 fps |

头部、面部与四肢误差使用不同定义，不能互相替代。最差面部区域包括前发边界与张嘴，差异可能包含旧头部插值、遮挡近似及两种渲染器的边缘采样；热力图可直接查看，尚未逐像素分离这些成因。下阶段值得优先改善头部/表情联合形变，并在审计剩余源形态键后补充苦眉/下撇嘴。

原有 22 段动作和 16 个表情共 38 个文件逐字节保留，MOC SHA256 仍为 `b48a2002f2bbf7b3e3a9c2ad049e6d3a282d1247bc956fdc54465292978bb538`，体积 565,880,768 字节。没有新增长期运行的绑定复杂度；动作与预设增加少量资源。当前主集合的参数求值至 GPU 完成中位耗时为 296.82 ms，模型仍然昂贵；不同阶段姿态集合不同，这个数值不能直接用于版本性能对比，也不能把 12 fps 的离线预览当作实时帧率。
