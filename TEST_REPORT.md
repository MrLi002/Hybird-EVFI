# 当前数据读取改动验证

GoPro 与 BS-ERGB 已统一为 OF-AF BS-ERGB 式帧间文件布局和 tuple 接口；训练/评测入口完成适配。

- 环境：Python 3.12、PyTorch 2.5.1+cpu、NumPy 2.3.5，CUDA 不可用。
- 完整命令：`python -m pytest tests -q`；结果：**36 passed, 1 skipped**。
- 14 项数据测试覆盖：两种原始字段/极性/坐标、精确事件文件区间、通道打包、共同裁剪/翻转、裁剪后时间归一化、无外部时间戳依赖、逐 skip 枚举、字典适配、测试步长/目标/异常帧、空事件/单时间戳、文件数错误与时间重置检查、旧数据流程断点限制。
- CLI 测试覆盖 BS-ERGB 与 GoPro 训练、评测、原始事件单对推理，以及新流程连续训练/断点恢复结果一致。
- 模型兼容、梯度、Large+BASR 全模块的小算例测试通过。
- 跳过项为 CUDA/CuPy 数值与梯度对照；未运行真实数据集训练或 GPU 性能测试。
- 当前 JUnit 明细见 `docs/test_results.xml`。

---

以下为原版本交付时的历史验证记录；其中物理时间单位/边界测试不再描述当前数据读取协议。模型尺寸检查仍是原记录。

# 验证记录

本次源码交付前运行的结果，不代表真实数据集的科研指标。

## 环境

- Python 3.12.14
- PyTorch 2.5.1+cpu，CUDA 不可用
- NumPy 2.3.5、einops 0.8.0、scikit-image 0.25.2、Pillow 13.0.0

## 自动测试

`python -m pytest tests -q`：**27 passed, 1 skipped**。

跳过的是 CUDA/CuPy 与 PyTorch splatting 的数值和梯度对照，原因是没有 CUDA 设备。原始 JUnit 结果见 `docs/test_results.xml`。

已覆盖：

- 正确 warp 的恒等、平移、越界及光流尺度变换。
- 查询节点、节点之间与 legacy 插值；默认 128 bin、16 步、17 节点。
- 与固定提交的原 OF-AF 模块对照：173 个参数键一致、严格加载；本次 CPU 同后端手动比较最大输出差为 0，自动测试容差 1e-5。
- u=32 和 u=44+BASR 的多尺度输出、零初始化、特征编码调用次数、前向和反向。
- BASR 四种修复/支持组合、空 Key、MLP 偏置清零、完整模块 gate、无 Key 时有效结构监督区域为空。
- 可微 PyTorch splat 的恒等、零度量等价、碰撞情况下 metric 梯度，以及 float64 autograd gradcheck。
- 光度掩码按有效像素归一化、空掩码、区域 SSIM；前后向平滑分别引用对应网格图像。
- LPIPS 的冻结参数与预测梯度路径、normalize 参数：使用测试替身，不下载实际 VGG/AlexNet 权重。
- 合成 GoPro / BS-ERGB 原始 npz、时间单位转换、裁剪时间轴不变、空事件/单事件、目标采样与错误时间轴报错。
- 完整训练 CLI、epoch 保存/恢复：从 epoch0 恢复后的 epoch1 模型参数与连续训练逐项完全一致。
- 评测 CLI 的指标与预测保存；单对插帧 CLI 输出文件。
- Large+BASR+mask+flow_corr+splat_metric+光流及区域损失+激活检查点，完成 CPU 小算例训练，并确认 metric 和 flow_corr 输出层发生参数更新。

部分自动训练算例使用 32×32、8 bin / 2 步以缩短检查；默认 128-bin / 16 步另有独立检查。

## 原始目标尺寸前向检查

- RGB：`[1,3,256,256]`；事件：`[1,128,256,256]`。
- 16 步估流，Large `unit_dim=44`，mask、BASR、flow_corr、splat_metric 均开启。
- 目标时刻 t=0.5，预测 `[1,1,3,256,256]`。
- 三尺度 `[1,3,256,256]`、`[1,3,128,128]`、`[1,3,64,64]`，全部数值有限。
- 该检查仅为 CPU inference；没有运行该尺寸的训练反向。
- 原始记录见 `docs/full_resolution_smoke.json`，耗时不用于硬件性能比较。

## 尚未验证

没有真实 GoPro/BS-ERGB 数据训练或论文指标复现；没有 GPU 显存、CUDA AMP、CuPy kernel 编译与速度测量；没有下载官方预训练模型进行真实图像对比。项目附源码和运行说明，不含训练权重、数据集或虚构实验结果。

`bash -n train_model.sh` 与全项目 Python 语法编译通过。
