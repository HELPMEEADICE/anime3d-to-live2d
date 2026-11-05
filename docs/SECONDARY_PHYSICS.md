# POPY 头发与裙摆物理

本地交付：`delivery/POPY_VTS_Physics.zip`。解压整个文件夹至 VTube Studio 的 Live2DModels，加载 `POPY.model3.json`，开启物理。继续使用 256 masks、每组至少 512×512。不要把四个新增物理输出参数再绑定为追踪输入。前版包保留。

基线为 POPY_VTS_MaskBatch。新模型有 43 参数，四个新输出是 `ParamHairSway`、`ParamHairFlutter`、`ParamSkirtSway`、`ParamSkirtFlutter`。头部转向/倾斜与身体转动驱动头发；身体转动和腿部前后运动驱动裙摆。没有恒定风，静止后会停止摆动。

空间部分采用四个低秩附加变形模态：发根和腰线处位移及一阶导数为零，发梢、裙摆逐渐增加位移，眼周发丝降低幅度。依次追溯原始表面名称、teacher 遮罩并集叶节点、焊接后的顶点、各次合批和排列映射，给可见表面与隐藏遮罩复制相同的空间场。混合遮罩的焊接来源定位误差实测为 0 像素；不按当前代理的贴图名称猜测来源。新增 BlendShape 以主网格默认形为参考，顺序和透明度增量为零。没有重新拟合现有姿态，也没有降低纹理、网格精度。

时序部分是标准 physics3：两组双段摆链，60 Hz，阻尼、惯性和回弹。输出限于 −1～1。不是三维头发/布料求解器，没有身体碰撞、布料自碰撞或新暴露表面的自动深度求交；既有拟合误差仍存在，极端姿势与物理极值组合仍可能穿插。当前 Blender MCP 无法连接，本轮使用此前保存的 POPY 几何、姿态和材质数据，没有修改 Blender 工程。没有新增 CMO3。

实测交付内容：

- 2473 ArtMesh、185669 顶点、167318 三角形、2 个旋转变形器、零 Glue、80 个遮罩上下文，均与基线相同。
- 新增 2664 个非零附加关键形，原始 102602 主关键形不变。新增位置数据 964928 字节。
- MOC 88136256 → 89147136 字节；ZIP 约 41.2 MiB。
- 关闭物理的 362 姿态：全画布 RGBA 逐像素一致。纹理、已有动作和表情保留。
- 每个物理参数正负极值均驱动真实可见顶点；头发最大位移约 7.07 / 3.00 像素，裙摆约 10.49 / 4.23 像素，测试的顺序和透明度保持不变。
- 原生 Ayagami 公共 PhysicsEngine：1320 帧、22 秒，四个输出均有效且有界，停动后最后一秒残余最大值约 1.92e−16。121 帧带真实物理输出的图像及 GIF 预览保留在本地。
- 物理求解中位约 1.0 微秒，P95 1.1 微秒；不包含网格和 GPU。
- 同设备四轮交替测试，物理输出为零时完整帧中位 30.370 → 30.572 ms；P95 58.650 → 58.866 ms。桌面未隔离，不宣称加速。此项没有测量开启物理后的整帧成本；带物理图像的渲染包含读回，不能拿来直接比较。

全部是固定版本的原生 Ayagami DX12 测试，尚未在 VTS/Cubism Core/Editor 实机验证。Ayagami 源码保持未修改，未向其上游提交内容。物理格式参考 [Live2D 官方 physics3 规范](https://github.com/Live2D/CubismSpecs/blob/master/FileFormats/physics3.json.md) 和 [官方 SDK 物理说明](https://docs.live2d.com/en/cubism-sdk-manual/physics/)；没有复制官方实现源码。

复现（需要已有 POPY 本地资产和各级转换数据）：

```powershell
cargo build --release --locked --manifest-path rust/Cargo.toml -p popy-validate --bins
python -m popy.secondary_physics all
python -m unittest discover -s tests -p test_secondary_physics.py -v
```

实现位于 `popy/secondary_physics.py`，时序验收程序为 `rust/validate/src/bin/physics_trace.rs`。数值报告是 `benchmarks/popy/fullbody/secondary_physics.json`；角色模型、图片、预览和 ZIP 只保留本地。已有独立 GPL 导出器通过进程调用，新增空间场和时序测试代码采用项目 MIT 许可边界，不向独立转换模块混入上游 GPL 实现。
