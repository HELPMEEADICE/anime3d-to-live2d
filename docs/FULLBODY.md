# POPY 全身、四肢动作与表情

本轮参考工作区 `../anon` 的控制组织，扩展已有面部修复版。新的入口是本地 `output/POPY_fullbody/POPY.model3.json`，模型包为 `delivery/POPY_fullbody.zip`；可编辑的三维动画工程为 `delivery/POPY_fullbody.blend`。旧 `POPY_face_fixed` 保留作为头部回归基线。

## 参考了什么

实际读取 Anon 的 model3、cdi3、66 个 motion3、36 个 exp3：其 cdi3 有 188 个参数，包含左右上臂、前臂、腕、袖子、层次和多种手势控制，没有独立的腿部动作参数。动作曲线中实际遇到 Linear、Bezier、InverseStepped。Anon 的左右前臂取值、手势切换和 POPY 的骨骼角度并不具有直接数值对应关系。

这里复用的是左右独立控制、动作命名语义及表情预设的组织方式。POPY 的关节角、手指驱动、网格关键形、动作时序和表情配方由本项目独立制作。没有复制 Anon 的 MOC、纹理、绑定、动作曲线或表情文件。审计见 `artifacts/fullbody/anon_audit.json`；公开测量记录保留数量、参数名称和文件摘要。

## 可操作的控制

原有 19 个头部、眼、嘴、眉、身体、呼吸参数继续保留。新增 20 个参数，总计 39 个。L/R 指角色解剖学左右；角色左手位于正面画面的右侧。

| 参数（L/R 各一项） | 范围 / 默认 | 来源与含义 |
|---|---|---|
| `ParamArmLRaise` / `ParamArmRRaise` | −80～50° / −65° | 上臂抬落；新模型默认自然垂臂 |
| `ParamArmLDepth` / `ParamArmRDepth` | −30～30° / 0° | 肩关节前后摆，正值向相机 |
| `ParamElbowL` / `ParamElbowR` | 0～120° / 0° | 肘弯曲 |
| `ParamWristL` / `ParamWristR` | −35～35° / 0° | 腕弯曲 |
| `ParamHandLPose` / `ParamHandRPose` | −1～1 / 0 | −1 指向、0 张手、1 握拳；实际手指骨骼连续弯曲 |
| `ParamForearmLTurn` / `ParamForearmRTurn` | −100～100° / 0° | 前臂绕轴翻掌；屈肘展示时 +90° 掌面朝向相机 |
| `ParamLegLSwing` / `ParamLegRSwing` | −25～25° / 0° | 髋关节前后摆 |
| `ParamLegLSpread` / `ParamLegRSpread` | −25～35° / 0° | 腿向内交叉或向外侧摆 |
| `ParamKneeL` / `ParamKneeR` | 0～60° / 0° | 屈膝 |
| `ParamAnkleL` / `ParamAnkleR` | −20～20° / 0° | 踝俯仰 |

具体骨骼、旋转轴、左右符号和采样键都在 `config/fullbody.json`，来自 POPY 的实际骨架。手指使用局部 X 轴屈曲；拇指基部的左右对掌符号单独校准。它不是不经审计即可套用其他角色的通用骨架映射。

## 动作和表情的使用

附带 58 个真实 motion3，原有 22 个四肢/基础动作：`idle`、左右 `wave` / `point` / `fist` / `leg_lift`、`stretch`、`thinking`、`surprised`、`nod`、`denial`、`walk_in_place`，以及左右 `palm_show` / `leg_side` / `side_step` 和 `side_sway`。另有 12 类、每类 C/L/R 三个版本的 36 个 `emote_*` 表情动作，详见 [表情动作说明](EXPRESSION_ACTIONS.md)。

附带 24 个 exp3，原有 16 个：`neutral`、`smile`、`laugh`、`angry`、`serious`、`surprised`、`thinking`、`uncertain`、左右 `wink`、`eyes_closed`、`joy_closed`、`sleepy`、左右看和上看。新增 `smile_soft`、`shy`、`sad`、`cry`、`worried`、`skeptical`、`confident`、`drowsy`。它们组合实际已有的眼、眉、嘴参数；没有声称已经实现 Anon 特有的泪滴、爱心眼、汗和表情阴影。

model3 中登记了所有文件。支持 motion3/exp3 的宿主可以读取这些资源；宿主是否实现动作播放、淡入淡出和表情混合，需要由其自身能力决定。这里没有改动 Ayagami 上游，其公开解析器、参数驱动器和 GPU 渲染器用于实际 MOC 求值。外部 Python 播放器负责曲线、循环、表情 Add/Multiply/Overwrite 和范围限制，再将每一时刻的参数交给 Ayagami 渲染。该离线播放器没有实现多个动作之间的宿主级交叉淡入淡出。

```powershell
python -m popy.fullbody frames --motion wave_L --expression smile --fps 15
python -m popy.fullbody frames --motion walk_in_place --fps 15
```

输出在 `artifacts/fullbody/wave_L.webp`、`walk_in_place.webp` 和各自帧目录。这些预览由真实 MOC 参数逐帧求值生成；模型内部保留网格、关键形、骨骼投影对应和连续参数。

Blender 工程中保留原 MToon 材质和描边修改器，烘焙 46 根实际控制骨骼及 Face/Brow Shape Key 通道，按 30 fps 制作 58 段动作和 24 段静态表情，共 82 个片段。打开 `delivery/POPY_fullbody.blend` 后，用时间轴标记/NLA 预览；每段之间留 1 秒空隙，曲线和原骨骼可以继续编辑。`artifacts/fullbody/blender_timeline.json` 记录具体帧范围与各数据块的 Action。独立重开工程后，20 个检查点与参考几何逐点比较，结果见 `blender_verification.json`。

## 几何实现

`popy/kinematics.py` 从真实源三角形、骨骼绑定矩阵和蒙皮权重求三维姿态，先与独立 Blender 求值核对。初始密集联合关键形在未采样姿态产生明显弦缩，最终默认改用 `popy/native_skinning.py` 的原生矩阵运算图：隐藏 ArtMesh 的八个顶点携带投影齐次骨骼矩阵的正负列向量，关节单参数 Glue 曲线组合父子矩阵，再按真实绑定逆矩阵和蒙皮权重生成各源顶点。可见网格通过单向 Glue 接收结果。设置普通 Live2D 参数即可连续求值，不需要应用程序计算三维骨骼或回调顶点。

矩阵列使用固定增益和正负双载体，使加减运算可以写成凸组合。所有新运算 Glue 的强度都在 0～1，每对权重均为 A=0、B=1；没有依赖负强度或负权重。[官方 Glue 说明](https://docs.live2d.com/en/cubism-editor-manual/glue/)描述了权重与兼容性控制。本轮仅在固定 Ayagami 实测，尚未验证官方 Core/Editor 对完整模型的支持。旋转曲线以 0.5° 采样，小范围控制以 0.005 采样，避免多关节几何的完整笛卡尔积。该方法目前要求正交投影。

Ayagami 对只有 Glue 依赖的参数可能跳过更新，因此每个运算参数另有一个恒定关键形的不可见依赖标记。每次运算显式重置目标载体，避免跨帧累积。矩阵节点、增益和源骨骼对应保存在 Rig IR 中。

原半身转换裁掉的身体、手、腿、鞋、裙子等表面从源拓扑恢复，Rust 原有连通分片器按材料和关节支持域重新分片。源材质均为双面；运行时对身体、上衣、裙子和鞋等实体表面启用背面剔除，减少缺少逐像素深度缓冲时背面错误覆盖正面的现象，薄发光饰品和手环保持双面。覆写记录在 `runtime_culling`，独立 Blender 参考仍按源材质渲染，差异计入指标。

绘制顺序保留姿态下的相机空间平均深度表，误差预算为 0.5 个绘制顺序单位；这不等于逐像素深度，也不是严格的图像误差界。几何由矩阵图连续求值，与这些粗网格的绘制顺序采样分开。

视觉复查发现白袜被腿部皮肤覆盖，最终对 `M_Shoes` 和 `M_Underpants` 分别增加 80 / 20 个顺序单位，记录在 `runtime_order_bias`。袜口内侧的两个水平封口分片通过静止高度和三角形法线识别，保持零偏置，防止灰色内面盖住小腿；阈值在 `runtime_inner_caps`。它是衣物覆盖关系的近似，不会改变头部层次。回归另外计算小腿参考白色区域与袜口皮肤的一像素容差召回率，防止此问题重现。

头部沿用已修复的网格、关键形、层级和 Glue。全身取景通过一致缩放父变形器的控制点、相对枢轴和根枢轴实现；原头部归一化几何与 Glue 权重保持。三个实际原模型/新模型对照检查原有网格变形，而非仅检查 JSON 相似。

源手指网格通过保留原端点的连通边折叠减少规模。UV 接缝、已覆盖的原网格边界、拓扑 link 条件和原三角形朝向受约束；每次折叠还在 15 个真实蒙皮姿态上检查所有被合并顶点的空间距离，最大允许 1.5 mm。保留顶点仍是原三维顶点，原顶点到保留顶点的映射写入 Rig 目录。这个有限姿态包络不构成所有未采样姿态的严格误差界。图像回归另外覆盖简化产生的轮廓与纹理差异。

回归中发现腿部稀疏采样的弦缩、握拳中间值和多参数混合误差。增加关键形仍不足以达到预算，因此引入上述矩阵图。报告保留密集原型与最终矩阵模型在相同源顶点上的比较，另有未采样腿部/手指及多参数极值组合。

导出器还有独立 GPL 补丁 `moc_position_dedup.patch`：对位模式完全相同的连续位置块共享位置索引，保留全部关键形、绘制顺序和标量数据，并同步修正 CountInfo 的位置值总数。优化由 `--deduplicate-positions` 显式启用，其他转换阶段的默认导出行为保持。`packing` 阶段将同一最终 IR 分别按原始/优化路径导出，并用 Ayagami 对照图像和所有可见、不可见顶点输出，禁止存在差异时打包。该补丁属于 `exporter/` GPL 进程，未混入 MIT 几何模块。

三维工程保留已有的头发/躯干清障修改器。摆动手臂的顶点仅从私有碰撞代理中排除，避免全身凸包随抬臂膨胀并整体推开头发；原角色网格不做此裁切。它仍是躯干和脖子附近的清障，直接手/头发接触没有新增模拟。

## 复现与验证

先具备当前工程的 `artifacts/POPY_source.blend`、`artifacts/samples/`、面部修复 Rig 与其 MOC 基线，以及已构建的 Rust/独立导出器。新克隆仓库需要先按根 README 完成既有转换及逐项修复流程；角色资产不在 GitHub 仓库中。

```powershell
$env:JAVA_HOME='C:\lib\java'
$env:JAVA_OPTS='-Xmx20g -XX:+UseG1GC -XX:+ExitOnOutOfMemoryError'
python -m popy.fullbody all
```

可单独执行 `build|audit|rig|ordering|export|assets|reference|blender|validate|packing|report|test|package`。`ordering` 可按配置更新已生成 IR 的服装层次偏置；修改后仍需导出和回归。每次 `export` 后运行 `assets`，重新登记 motion3/exp3。`blender` 阶段的重开验证需要已生成 `reference`，`report` 需要当前模型的 `packing` 结果。`all` 会精确应用并构建导出器补丁。Blender 路径可以使用 `--blender`，Anon 路径可以使用 `--reference-model` 修改。实际审计/采样均在独立后台进程中完成，不修改打开的用户工程，不覆盖原始快照。

报告位于 `artifacts/fullbody/report/index.html`。覆盖四肢最小/最大/未采样值、表情与转头、各动作多个时刻、连续挥手/原地步行、身体/头部/表情/四肢复合姿态，以及旧头部基线。指标包含轮廓 IoU、前景 MAE、局部灰度 SSIM、真实 MOC 顶点重投影误差和 Python/Blender 蒙皮差异。顶点指标基于保留的原顶点；被简化点另有源几何采样距离记录。公开结果在 `benchmarks/popy/fullbody/`。

无需角色素材的新增检查：

```powershell
python -m unittest discover -s tests -p test_fullbody.py -k BodyGeometryTests -v
```

## 仍有限制

当前动作细化版的规模和最终回归数值见 [翻掌、连贯动作与腿部侧摆](MOTION_REFINEMENT.md) 及 `benchmarks/popy/fullbody/metrics.json`。首版曾为 607,229,120 字节；新增旋转维度后调整排序采样，避免笛卡尔积膨胀。几何仍使用 0.5° 矩阵旋转字段。
- 绘制顺序关键形、原生 Glue 运算图及既有遮挡辅助网格使文件、导出内存和运行成本很高。这是可运行的研发版本，尚不具备轻量模型的成本。
- 衣服跟随真实源蒙皮，未新增布料碰撞求解；行走没有足底接触 IK、落脚锁定或整体位移，不能当作已完成的物理行走系统。
- 指向和握拳是程序化手指姿态，没有手指间接触约束；单参数矩阵曲线仍有插值及浮点误差，平均深度分片仍存在局部遮挡和排序近似。
- 原有极端头发/下颌残差、微小轮廓和透明边采样差异继续记录。有限姿态验证不能证明任意动作都不穿模。
- 头部沿用旧版形变和遮挡结构；矩阵图已覆盖躯干/四肢呼吸，尚未将整个头部与新增原始表情 Shape Keys 迁移到同一运算图。
- 本轮生成 MOC3、model3、cdi3、motion3、exp3 和独立 Blender 工程。没有新的 CMO3，没有官方 Cubism Editor/Core 兼容性验证；旧编辑工程不包含这些四肢功能。

后续优先减少仍然密集的绘制顺序表和既有头部辅助结构，再将三维表情差分接入矩阵图，并完善衣袖碰撞和足底约束。
