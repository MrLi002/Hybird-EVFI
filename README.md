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
| `datasets/`、`util/event.py`、`util/voxelization.py` | OF-AF 式帧间事件索引、首末事件归一化、体素化与共同增强 |
| `tests/` | 数值、梯度、原 OF-AF 对照、数据和完整 CLI 测试 |

`unit_dim=32` 和 `44` 都可运行；需要 Large 时显式指定 `--unit_dim 44`。

## 3. 数据格式：统一采用 OF-AF 的 BS-ERGB 读取方式

传入训练或测试 split 目录，例如 `/data/GOPRO/train`、`/data/BSERGB/1_TRAIN`，也支持直接传单场景目录。每个场景统一使用 `images/*.png` 和 `events/*.npz`，文件名须补零并按时间排序。

| 数据 | 场景下图像 | 场景下事件 | 字段与固定解析规则 |
| --- | --- | --- | --- |
| GoPro | `images/*.png` | `events/*.npz` | `x,y,t,p`；像素坐标，p 已为 -1/+1 |
| BS-ERGB | `images/*.png` | `events/*.npz` | `x,y,timestamp,polarity`；沿用 OF-AF 原坐标换算与 0/1→-1/+1 |

第 j 个事件文件对应第 j、j+1 张图像之间的全部事件。N 张图像应配 N-1 个事件文件。GoPro 原高帧率事件文件需要提前按 RGB 间隔合并；只修改文件夹名称不能建立对应关系。各 npz 内的事件时间戳必须处于连续时间轴，不能每个文件从零重新计时。

读取端不搜索、不打开 `timestamps.txt`、`images_timestamps.txt` 或 `timestamps_clean_8x_v2.txt`。使用 `event_names[start:end]` 合并事件，先共同裁剪 RGB/事件，再按剩余事件的最小/最大时间归一化为体素；目标时刻按帧索引比计算。图像序列应等时间间隔采样。空事件返回零体素，单时间戳事件作有限值处理。

`GOPRO_Train` 和 `BSERGB_Train` 都使用 `(voxel_bins, data_root, scene, skip_frame)` 构造，保留 `get_img/get_events/random_crop/__getitem__`，返回 OF-AF 式 `(imgs, voxel, times)`：

- imgs：`[3*(skip+2),H,W]`，依次按通道打包端点与全部中间帧；
- voxel：`[voxel_bins,H,W]`，整个端点区间的事件；
- times：`[skip]`，值为 `1/(skip+1)...skip/(skip+1)`。

公开测试类采用 `(voxel_bins, data_root, scene, skip_frame)`，返回全图 tuple；`get_batch(start)` 返回加 batch 维度的 tuple。评测入口负责指标、保存和模型所需补边。

顶层 `RawVFIData` 仅将 tuple 转为现有 Hybrid 训练/评测需要的具名字典。训练按每个 skip、每个场景、每个起点逐项枚举，等价于 OF-AF 的逐 skip 拼接；不随机选择 skip。默认 GoPro 训练 skip=31、测试 skips=7/15，BS-ERGB 训练/测试 skips=1/3。

默认使用全部中间帧。`--num_targets K` 仍可显式启用训练目标子采样以降低显存，实际数量为 min(K,skip)；它与原版全目标训练不是同一采样设置。混合 skip/目标子采样使用 batch_size=1；更大 batch 必须具有相同目标时刻。

移除了数据集入口的 `legacy_event_time/frame_timestamps/image_fps/time_offset/frame_time_scale/event_time_scale/event_coords/polarity/scene_file/eval_stride/gopro_odd_targets/include_bad_frames` 开关。坐标、极性及评测协议按数据集固定处理；`legacy_query/legacy_warp/legacy_lpips` 是独立模型/指标开关，继续保留。

## 4. P1：训练 OF-AF 修正基线 B0

```bash
python train_model.py --dataset gopro --data_root /data/GOPRO/train \
  --refine resblock --voxel_bins 128 --nb_of_flow 16 \
  --skips 31 --num_targets 1 \
  --epoch 20 --lr 1e-4 --batch_size 1 --crop_size 256 --num_worker 8 \
  --checkpoint_bifeb --save_dir train/gopro_b0
```

该命令显式以 K=1 子采样以检查显存与收敛；删除 `--num_targets 1` 才是原 OF-AF 的全部目标训练。默认损失为 Charbonnier + 0.1×LPIPS；默认 `legacy_query/legacy_warp/legacy_lpips` 关闭，事件首末归一化固定启用。

## 5. P2：替换为 CBMNet-Large 精修

```bash
python train_model.py --dataset gopro --data_root /data/GOPRO/train \
  --refine transformer --unit_dim 44 --voxel_bins 128 --nb_of_flow 16 \
  --skips 31 --num_targets 1 \
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

场景使用原 BS-ERGB 事件字段与坐标编码，不需要额外图像时间戳文件：

```bash
python train_model.py --dataset bsergb --data_root /data/BSERGB/1_TRAIN \
  --refine transformer --unit_dim 44 --skips 1 3 --num_targets 1 \
  --weights_path train/gopro_hybrid_large/last.pth --only_weights \
  --old_lr_scale 1 --epoch 20 --lr 1e-5 --batch_size 1 --crop_size 256 \
  --checkpoint_bifeb --checkpoint_refine --save_dir train/bsergb_hybrid_large
```

删除 `--num_targets 1` 使用全部目标。模型可加载旧权重进行迁移；本次改动之前的时间边界读取流程所保存的断点，须通过 `--weights_path` 开始新训练，不能作为同一数据流程原样续训。新断点记录 `data_protocol=ofaf_frame_aligned_v1`。

```bash
python train_model.py --resume train/bsergb_hybrid_large/last.pth
```

保存 `last.pth`、每轮 `epochN.pth`，可选独立验证集产生 `best.pth`。恢复包含模型、优化器、参数分组、scheduler、AMP scaler、epoch/global_step、Python/NumPy/PyTorch RNG 与 DataLoader generator。断点在 epoch 结束写入，不提供批次中途精确续接。

恢复时未显式传入的参数从 checkpoint 恢复；可以显式修改数据根目录或保存位置。模型结构不可在 resume 中更改。延长总 epoch 需要 `--reset_scheduler`，这是重建余弦计划的选择，会改变后续学习率轨迹。

`--val_root` 指定独立验证 split。训练与验证场景路径重叠时拒绝运行。训练会保存 `args.json`、`data_manifest.json`、`log.jsonl`；验证保存的是浮点 PSNR，与评测脚本 8-bit PSNR 分开标记。

终端使用 `tqdm` 显示 `Train Epoch(0/20): ... sum_loss=..., learn_rate=...`，包含当前批次/总批次、耗时、预计剩余时间和速度。`sum_loss` 是本轮所有已完成 batch 的总损失累加，保留四位小数；`learn_rate` 显示最后一个优化器参数组的当前学习率，保留六位小数。`--log_every` 默认 1，控制进度条刷新所需的最少 batch 数；实际刷新也由 tqdm 的时间间隔限制。设置 `--max_steps` 时，总批次显示实际训练上限；`log.jsonl` 继续保存每轮平均损失和验证指标。

## 8. 评测与预测保存

```bash
python bsergb_eval.py --checkpoint train/bsergb_hybrid_large/last.pth \
  --test_root /data/BSERGB/1_TEST --skips 1 3 \
  --save_pred --save_root eval/bsergb_hybrid_large
```

自训练 checkpoint 自动恢复模型配置；数据采用第 3 节固定解析规则。旧权重没有配置元数据时需传入必要模型开关；LPIPS 模式需与对应实验一致。

```bash
python bsergb_eval.py --checkpoint /path/to/official_bsergb.pth \
  --test_root /data/BSERGB/1_TEST --refine resblock \
  --voxel_bins 128 --nb_of_flow 16 \
  --legacy_query --legacy_warp --legacy_lpips \
  --save_root eval/ofaf_legacy
```

输出逐帧 `per_frame.jsonl`、分 skip 的 `summary.json`、数据/参数清单；图像保存在 `save_root/skip/scene/起始帧编号/目标文件名`。PSNR/SSIM 使用 8-bit RGB，LPIPS 使用 AlexNet；此项目不包含 FloLPIPS。评测步长固定为 skip。BS-ERGB 评全部中间帧，并沿用原异常帧排除名单；GoPro 沿用原版中间帧零基索引 1、3、5……的选择规则。

首次无 GPU/无 LPIPS 权重检查可加 `--device cpu --skip_lpips --max_intervals 1`；这属于子集调试，不能当作正式测试结果。GPU 计时包含 pair 编码与逐帧精修，排除 IO/保存/指标，并记录没有排除 warmup，不冒充论文表格的同硬件对比。

## 9. 单对关键帧插帧

```bash
python interpolate.py --checkpoint train/bsergb_hybrid_large/last.pth \
  --im0 /path/I0.png --im1 /path/I1.png --voxel /path/voxel_01.npy \
  --ratio 16 --output outputs/example
# 或指定 --times 0.1 0.5 0.9
```

voxel 为 `[128,H,W]`，使用训练时相同的时间归一化和极性约定。也可以传单个包含整个区间事件的 `--events interval.npz --dataset gopro`（或 bsergb），按同一事件首末归一化与固定字段规则处理。输出逐目标流式保存，不把所有预测和辅助大张量一直留在 GPU。

## 10. 自检

```bash
python -m pytest -q
python scripts/create_toy_dataset.py --output toy_data --dataset bsergb
python train_model.py --dataset bsergb --data_root toy_data \
  --skips 1 --crop_size 32 --num_targets 1 --num_worker 0 --epoch 1 \
  --max_steps 1 --w_lpips 0 --device cpu --num_threads 2 --save_dir train/toy
```

测试数据只用于验证代码路径，不用于科研结论。单元检查还使用 8-bin/2 步缩小部分算例；单独测试默认 128-bin/16 步/17 节点和原模型兼容。

原始模块的固定提交、文件归属及使用范围见 `THIRD_PARTY_NOTICES.md`。实现上的明确补充与限制见 `docs/IMPLEMENTATION_NOTES.md`。
