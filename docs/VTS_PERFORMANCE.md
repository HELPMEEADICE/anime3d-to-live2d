# POPY VTS 有损性能版

本版针对用户允许损失部分画质的新要求，保留牙齿修复版作为 teacher 和回退版本。它不是旧版无损优化的替代性证明，也不要求修改 VTube Studio 或 Ayagami。

## 实现

通过固定、未修改的 Ayagami 公共 Driver 对真实 MOC 离线求值，保留实际可见的 `M_*`、`Full_*` 和 80 组遮罩引用到的来源。删除其余原生矩阵/Glue 中间载体；原始 Blender 工程不修改。

先扫描 39 个参数的单轴响应，再按每个网格的响应强度选择局部主参数：头部 X/Y 加一个主要表情、手臂抬升/肘部/纵深或前臂转动、腿部摆动/张开/膝盖、躯干 X/Y。局部主参数保留笛卡尔张量；其余参数用分段线性独立残差近似，不再构造完整高维张量。共采样 1,138 个 teacher 姿态。

如果残差参数默认值不是零，以 `F(p)-F(0)` 为 Cubism 增量，并从主网格减去默认增量和，避免眼睛默认打开、手臂默认下垂造成重复叠加。头和身体的 roll 从样本中逆变换掉，交给两层 Rotation Deformer。

遮罩来源的重复三角形角点以**全部已拟合主关键形和残差关键形**轨迹去重，网格为 1/64 画布像素；仅对 alpha 全不透明的隐藏遮罩进行，避免静止时重合、运动后分离的点被误焊。可见网格保留 UV 接缝和原拓扑。脸部和眼睛纹理 2048，身体、服装、头发和裙子降为 1024。所有 80 组原遮罩引用保留，包含全部 55 个牙齿分片的遮罩。

独立 MIT 采样/拟合程序只输出中立 IR；GPL 导出器改动保留在 `exporter/`，复用 PSD2Live 生成真实 MOC。没有复制上游二进制写入器到 MIT 模块，也没有向 Ayagami 上游提交内容。

## 本机结果

| 指标 | 牙齿修复版 | 性能版 |
|---|---:|---:|
| MOC 字节数 | 563,375,104 | 108,735,616 |
| 主 ArtMesh Keyform（不含 BlendShape） | 5,439,788 | 129,677 |
| 含 BlendShape 的作者关键形/增量数 | — | 171,932 |
| Glue / Glue Keyform | 360,242 / 3,213,851 | 0 / 0 |
| 含隐藏遮罩的顶点 | 686,971 | 173,000 |
| 三角形 | 804,842 | 195,441 |
| ArtMesh | 5,875 | 3,998 |
| Deformer / 最大深度 | 10 / 3 | 2 / 2 |
| CPU Driver 中位数 | 62.566 ms | 1.596 ms |
| 参数设置至 GPU 完成中位数 | 101.535 ms | 29.095 ms |
| 同一 12 姿态渲染进程峰值 RSS | 1,273.59 MiB | 685.13 MiB |

RTX 5080 Laptop GPU，原生 DX12，未修改的 Ayagami `0d1d7aa3efe57b1b363b72e672e353c52a9c15d6`。CPU 和 GPU 各 4 轮、每轮 7 个计时姿态，预热 5 个姿态，模型交替执行且同时驻留。GPU 包含渲染提交/同步，不含 PNG 下载写盘。桌面未隔离；**这些不是 VTS 实机帧率或 VTS 内存数字**。原始 JSON、逐姿态画质指标放在交付包的 `performance_report.json` 和本地 `artifacts/vts_fast/`。

## 画质取舍与限制

静止正面轮廓 IoU 1.0、前景 SSIM 0.9993；转头同时眨眼 IoU 0.9901、SSIM 0.9502。挥手组合是本次最差姿态：IoU 0.8806、SSIM 0.4598，前景 MAE 0.0893。独立残差未跟随所有上游关节旋转，会造成腰部、手腕、衣袖等复合动作位置偏差；大角度身体和头部叠加也有误差。不是只降低纹理造成的变化，也不宣称画质无损。

仅检查 12 个实际渲染姿态，比较对象为已有 Live2D teacher，未新增 Blender Ground Truth 或完整回归；旧模型遗留问题可能继承。没有 VTube Studio 实机加载、Cubism Core 或 Editor 兼容性验收，也无新增 CMO3。加载需在 VTS `Live2D Mask Config` 设置 256 masks、每组至少 512×512。

## 重复运行

需要本机牙齿修复包、原冻结采样的相机/骨架、JDK、Rust 和固定依赖。角色资产不进入 GitHub：

```powershell
cargo build --release --locked --manifest-path rust/Cargo.toml -p popy-validate --bins
python -m popy.vts_fast all
```

可单独执行 `probe|scan|plan|rig|export|review|report|package`。输出 `delivery/POPY_VTS_Performance.zip`，原质量包不覆盖。后续最值得改进的是按骨段共享局部坐标变形器，使手腕/手型残差随上游关节旋转，再补少量头部与身体之间的交互残差；无需恢复几十万个 Glue。
