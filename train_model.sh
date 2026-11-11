#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${GOPRO_ROOT:?Set GOPRO_ROOT to the GoPro training split directory}"

# P1: OF-AF motion/BiW with revised Query and warp.
python train_model.py --dataset gopro --data_root "$GOPRO_ROOT" \
  --refine resblock --voxel_bins 128 --nb_of_flow 16 \
  --skips 31 --num_targets 1 --epoch 20 --lr 1e-4 \
  --batch_size 1 --crop_size 256 --num_worker 8 --checkpoint_bifeb \
  --save_dir train/gopro_b0

# P2: Large CBMNet refinement + full-interval event adapter.
python train_model.py --dataset gopro --data_root "$GOPRO_ROOT" \
  --refine transformer --unit_dim 44 --voxel_bins 128 --nb_of_flow 16 \
  --skips 31 --num_targets 1 --epoch 20 --lr 1e-4 \
  --batch_size 1 --crop_size 256 --num_worker 8 --checkpoint_bifeb --checkpoint_refine \
  --weights_path train/gopro_b0/last.pth --only_weights \
  --save_dir train/gopro_hybrid_large
