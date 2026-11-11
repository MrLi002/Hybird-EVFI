# Hybrid EVFI：OF-AF 结构的事件插帧实现

依据 `docs/design.md`（用户上传的修订版（2））实现。默认输入 `event[B,128,H,W]`，16 次估流更新，17 个时间节点。保留 OF-AF 的编码器、BiFEB、BiW 参数结构，接入 CBMNet 多尺度 Transformer 残差精修及修改后的 DSER++ BASR。

这是可训练源码，未附训练权重；小型 CPU 检查结果见 `TEST_REPORT.md`。没有在真实 GoPro / BS-ERGB 上重训，也没有保证论文指标。

## 1. 安装

推荐 Python 3.10/3.11，PyTorch 2.5.1；此次自动检查使用 Python 3.12 和 PyTorch 2.5.1+cpu。

```bash
conda create -n hybrid_evfi python=3.10 -y
conda activate hybrid_evfi
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# CUDA 12 环境使用原 OF-AF 的 CuPy splatting 时安装：
pip install -r requirements-cuda.txt
```

`--splat_backend auto` 在 CUDA 上优先使用未修改的原 CuPy kernel；缺少 CuPy 时提示并使用较慢的 PyTorch 后端。显式选择 `cupy` 时依赖或编译错误直接报出。CPU 始终可使用 `--splat_backend torch`。原 kernel 仍需要可用的 CUDA/NVRTC 运行环境。

LPIPS 使用 torchvision 的预训练 VGG/AlexNet，首次调用可能需要下载权重；离线环境请先准备 torchvision 与 LPIPS 权重缓存。`--w_lpips 0` 仅用于快速接口检查或明确的损失消融。

## 2. 目录与模块

| 文件 | 作用 |
| --- | --- |
| `train_model.py` / `train_model.sh` | 训练、分组学习率、epoch 断点续训；shell 脚本顺序执行 B0、P2 |
| `bsergb_eval.py` / `gopro_eval.py` | 流式评测、预测保存、PSNR/SSIM/LPIPS 与计时 |
| `interpolate.py` | 两张关键帧 + 全区间体素/原始事件，任意时刻插帧 |
| `models/encoder.py`、`BiFEB.py` | OF-AF 原编码和递推估流结构 |
| `models/query.py`、`blocks.py` | 升序时间查询、原版兼容开关、像素单位 warp |
| `models/BiW.py`、`softsplat.py`、`softsplat_cuda.py` | 双向变形、原内部 mask、诊断覆盖与二值 hole、两种 splatting 后端 |
| `models/cbm_modules.py`、`refine_head.py` | CBMNet 组件及三级残差解码器 |
| `models/event_adapter.py` | 全区间事件编码缓存与连续时间/上下文条件化 |
| `models/basr.py` | 分离 repair/support、空 Key 安全处理、完整模块残差门控 |
| `models/flow_corr.py`、`losses.py` | 后向相关修正、光度/平滑/区域 SSIM/LPIPS |
| `datasets/`、`util/event.py`、`util/voxelization.py` | 原始事件读取、明确时间边界、体素化与共同增强 |
| `tests/` | 数值、梯度、原 OF-AF 对照、数据和完整 CLI 测试 |

`unit_dim=32` 和 `44` 都可运行；需要 Large 时显式指定 `--unit_dim 44`。

## 3. 数据格式与时间单位

传入某个 split 的目录，例如 `/data/GOPRO/train`、`/data/BSERGB/1_TRAIN`。也支持直接传单场景目录。程序不会自动合并训练、验证或测试目录。

| 数据 | 场景下图像 | 场景下事件 | 事件数组字段 |
| --- | --- | --- | --- |
| GoPro | `imgs/*.png` | `events_clean_8x_v2/*.npz` | `x,y,t,p` |
| BS-ERGB | `images/*.png` | `events/*.npz` | `x,y,timestamp,polarity` |

BS-ERGB 和帧率事件布局约定第 j 个 npz 对应第 j、j+1 张图像之间的事件，沿用 OF-AF 的文件对应关系。GoPro 的高帧率事件文件需要 `timestamps_clean_8x_v2.txt`，按时间选择相关 npz，再按关键帧边界过滤。

每个场景可放 `images_timestamps.txt`，每行一个图像时间戳，与自然排序后的图像一一对应。也支持 `timestamps_images.txt`、图像目录下 `timestamps.txt`，或通过 `--frame_timestamps <场景内文件名>` 指定。时间戳格式为单列数值或“图像名 时间戳”。单位转换后，所有事件、图像和事件文件索引时间都须处在同一秒制时间轴。

| 参数 | 含义 |
| --- | --- |
| `--event_time_scale 1e-6` | 原始事件与事件文件索引时间为微秒时转为秒 |
| `--frame_time_scale 1e-6` | 图像时间戳文本为微秒时转为秒 |
| `--image_fps`、`--time_offset` | 已知帧率和同步原点时构造时间戳；GoPro 缺省按已知的 240 FPS、原点 0 处理 |
| `--legacy_event_time` | 明确选择原 OF-AF 的首末事件时间归一化；目标时刻按帧索引比计算 |
| `--event_coords ofaf` | BS-ERGB 使用原仓库的坐标换算；GoPro 不作该换算 |
| `--event_coords pixels` | npz 的 x/y 已是 RGB 对齐后的像素坐标 |
| `--polarity auto` | 自动识别 0/1 或 -1/+1；也可指定 `01` 或 `pm1` |

BS-ERGB 新时间模式缺少时间戳和已知帧率时会报错，需要补充实际时间信息，或明确开启 legacy 模式。仅更改 bin 数不能恢复物理时间边界。程序不会生成 v2e 事件；需要按已有流程生成并整理原始 npz。

训练返回具名数据字典，包含 im0/im1、targets、event、times，取代原脚本按 RGB 通道打包的内部格式；模型默认调用及 OF-AF 目录入口保持一致。每个样本 K 自动取 `min(num_targets, skip)`。混合 skip 推荐 `batch_size=1`；更大 batch 必须提供相同目标时刻，否则明确报错。

## 4. P1：训练 OF-AF 修正基线 B0

```bash
python train_model.py --dataset gopro --data_root /data/GOPRO/train \
  --refine resblock --voxel_bins 128 --nb_of_flow 16 \
  --skips 9 11 15 23 31 --num_targets 1 \
  --epoch 20 --lr 1e-4 --batch_size 1 --crop_size 256 --num_worker 8 \
  --checkpoint_bifeb --save_dir train/gopro_b0
```

先以 K=1 检查显存与收敛，再根据实际显存将 `--num_targets` 改为 4。默认训练使用 Charbonnier + 0.1×LPIPS；默认 `legacy_query/warp/event_time/lpips` 均关闭。

## 5. P2：替换为 CBMNet-Large 精修

```bash
python train_model.py --dataset gopro --data_root /data/GOPRO/train \
  --refine transformer --unit_dim 44 --voxel_bins 128 --nb_of_flow 16 \
  --skips 9 11 15 23 31 --num_targets 1 \
  --epoch 20 --lr 1e-4 --batch_size 1 --crop_size 256 --num_worker 8 \
  --weights_path train/gopro_b0/last.pth --only_weights \
  --checkpoint_bifeb --checkpoint_refine --save_dir train/gopro_hybrid_large
```

`--weights_path` 只迁移匹配参数，不加载优化器；匹配详情写入 `weight_loading.json`。从 B0 成功加载的参数默认使用 0.1 倍学习率，新增参数使用完整学习率。三个输出卷积零初始化，所以 P2 初始预测等于粗融合 fuse，不等于含旧精修残差的 B0 最终预测。

可选 `--cbm_weights <CBMNet权重>` 初始化匹配的精修内部层和编码器，自动过滤四个输入卷积、RGB 输出头及形状不匹配的事件首层。32/44 宽度必须与来源权重相符。此处不加载 DSER++ 整体权重。

## 6. P3/P4：独立开关

在对应上一阶段命令上逐项添加；每次更换模型结构使用 `--weights_path`，不使用 `--resume`。

| 阶段 | 参数 |
| --- | --- |
| 掩码精修 | `--mask_in_refine` |
| BASR | `--mask_in_refine --basr --basr_heads 4 2` |
| BASR + 区域结构监督 | 再添加 `--w_ps 1` |
| 光流辅助监督 | `--w_flow 0.1` |
| 相关残差修正 | 再添加 `--flow_corr` |
| 可学习前向投影度量 | 再添加 `--splat_metric`，且 `w_flow` 必须大于 0 |

默认保守地要求一个参考 patch 完整落在支持区域内。`--basr_support_ratio` 可用于支持比例阈值消融，默认 1.0。无 Key、无需修复或双侧都不可靠时整个 BASR 回退主干，MLP 偏置也不会造成空来源输出。动态位置编码避免原 8000-token 限制，query 分块减少单次注意力矩阵大小；这并不意味着全分辨率训练成本与 token 数无关。

只实现 `warped` Key，不开放未定义掩码坐标映射的 raw Key。完整 BASR 还会影响后续解码卷积；门控保护的是该模块输出位置，不承诺最终 RGB 图像在所有非 gate 位置逐像素不变。

## 7. BS-ERGB 微调、断点续训

假设场景中已经提供秒制图像时间戳，原事件为微秒，坐标仍为原 BS-ERGB 编码：

```bash
python train_model.py --dataset bsergb --data_root /data/BSERGB/1_TRAIN \
  --refine transformer --unit_dim 44 --skips 1 3 --num_targets 1 \
  --frame_timestamps images_timestamps.txt --event_time_scale 1e-6 \
  --weights_path train/gopro_hybrid_large/last.pth --only_weights \
  --old_lr_scale 1 --epoch 20 --lr 1e-5 --batch_size 1 --crop_size 256 \
  --checkpoint_bifeb --checkpoint_refine --save_dir train/bsergb_hybrid_large
```

上面的文件名和单位是输入示例，须对应实际数据。原 npz 已是像素坐标时改 `--event_coords pixels`。若只具备原仓库所需的图像与事件，可以显式加 `--legacy_event_time` 使用其归一化约定，并在评测时保持一致。

```bash
python train_model.py --resume train/bsergb_hybrid_large/last.pth
```

保存 `last.pth`、每轮 `epochN.pth`，可选独立验证集产生 `best.pth`。恢复包含模型、优化器、参数分组、scheduler、AMP scaler、epoch/global_step、Python/NumPy/PyTorch RNG 与 DataLoader generator。断点在 epoch 结束写入，不提供批次中途精确续接。

恢复时未显式传入的参数从 checkpoint 恢复；可以显式修改数据根目录或保存位置。模型结构不可在 resume 中更改。延长总 epoch 需要 `--reset_scheduler`，这是重建余弦计划的选择，会改变后续学习率轨迹。

`--val_root` 指定独立验证 split，`--val_scene_file` 可选场景列表。训练与验证场景路径重叠时拒绝运行。训练会保存 `args.json`、`data_manifest.json`、`log.jsonl`；验证保存的是浮点 PSNR，与评测脚本 8-bit PSNR 分开标记。

## 8. 评测与预测保存

```bash
python bsergb_eval.py --checkpoint train/bsergb_hybrid_large/last.pth \
  --test_root /data/BSERGB/1_TEST --skips 1 3 \
  --frame_timestamps images_timestamps.txt --event_time_scale 1e-6 \
  --save_pred --save_root eval/bsergb_hybrid_large
```

自训练 checkpoint 自动恢复模型配置；数据时间边界、单位、坐标和 LPIPS 模式仍需与对应实验明确一致。旧权重没有配置元数据时需传入必要模型开关。

```bash
python bsergb_eval.py --checkpoint /path/to/official_bsergb.pth \
  --test_root /data/BSERGB/1_TEST --refine resblock \
  --voxel_bins 128 --nb_of_flow 16 \
  --legacy_query --legacy_warp --legacy_event_time --legacy_lpips \
  --save_root eval/ofaf_legacy
```

输出逐帧 `per_frame.jsonl`、分 skip 的 `summary.json`、数据/参数清单；图像保存在 `save_root/skip/scene/起始帧编号/目标文件名`。PSNR/SSIM 使用 8-bit RGB，LPIPS 使用 AlexNet。此项目不包含 FloLPIPS，未将其他指标充作 FloLPIPS。GoPro 原目标选择可用 `--gopro_odd_targets`，默认评全部中间帧，并在结果中记录协议。评测 stride 默认沿用原仓库的 skip。

首次无 GPU/无 LPIPS 权重检查可加 `--device cpu --skip_lpips --max_intervals 1`；这属于子集调试，不能当作正式测试结果。GPU 计时包含 pair 编码与逐帧精修，排除 IO/保存/指标，并记录没有排除 warmup，不冒充论文表格的同硬件对比。

## 9. 单对关键帧插帧

```bash
python interpolate.py --checkpoint train/bsergb_hybrid_large/last.pth \
  --im0 /path/I0.png --im1 /path/I1.png --voxel /path/voxel_01.npy \
  --ratio 16 --output outputs/example
# 或指定 --times 0.1 0.5 0.9
```

voxel 为 `[128,H,W]`，使用训练时相同的时间归一化和极性约定。也可以传单个包含整个区间事件的 `--events interval.npz --tau0 ... --tau1 ...`，配合单位与坐标参数。输出逐目标流式保存，不把所有预测和辅助大张量一直留在 GPU。

## 10. 自检

```bash
python -m pytest -q
python scripts/create_toy_dataset.py --output toy_data --dataset bsergb
python train_model.py --dataset bsergb --data_root toy_data --event_coords pixels \
  --skips 1 --crop_size 32 --num_targets 1 --num_worker 0 --epoch 1 \
  --max_steps 1 --w_lpips 0 --device cpu --num_threads 2 --save_dir train/toy
```

测试数据只用于验证代码路径，不用于科研结论。单元检查还使用 8-bin/2 步缩小部分算例；单独测试默认 128-bin/16 步/17 节点和原模型兼容。

原始模块的固定提交、文件归属及使用范围见 `THIRD_PARTY_NOTICES.md`。实现上的明确补充与限制见 `docs/IMPLEMENTATION_NOTES.md`。

