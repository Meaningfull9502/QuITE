#!/bin/bash
# Reproduce QuITE++ forecasting results (paper Table 4).
#
# Hyperparameters:
#   - Paper §6.1 ("For QuITE++, we tune the hidden dimension, number of
#     layers, and number of attention heads via grid search over {32, 64},
#     {1, 2, 3}, and {1, 2, 4, 8}, respectively."). Defaults below are
#     middle-of-grid values; override via env vars.
#   - Paper Table C.1(a): per-dataset patch_size / stride / batch_size / lr.

GPU=${GPU:-0}
HID=${HID:-64}
NHEAD=${NHEAD:-4}
NLAYER=${NLAYER:-2}
SEEDS=${SEEDS:-"1 2 3 4 5"}

run() {
    python train_quite_plus.py \
        --hid_dim $HID --nhead $NHEAD --nlayer $NLAYER \
        --patience 50 --lr 1e-3 --gpu $GPU "$@"
}

for seed in $SEEDS; do
    # ---------- Human Activity ----------
    run --dataset activity --history 3000 --patch_size 750 --stride 750 --batch_size 32 --seed $seed
    run --dataset activity --history 2000 --patch_size 500 --stride 500 --batch_size 32 --seed $seed
    run --dataset activity --history 1000 --patch_size 250 --stride 250 --batch_size 32 --seed $seed

    # ---------- USHCN ----------
    run --dataset ushcn --history 24 --pred_window 1  --patch_size 1.5 --stride 1.5 --batch_size 128 --seed $seed
    run --dataset ushcn --history 24 --pred_window 6  --patch_size 1.5 --stride 1.5 --batch_size 128 --seed $seed
    run --dataset ushcn --history 24 --pred_window 12 --patch_size 1.5 --stride 1.5 --batch_size 128 --seed $seed

    # ---------- PhysioNet — Table C.1: history→patch 12→6, 24→6, 36→9 ----------
    run --dataset physionet --history 12 --patch_size 6 --stride 6 --batch_size 64 --seed $seed
    run --dataset physionet --history 24 --patch_size 6 --stride 6 --batch_size 64 --seed $seed
    run --dataset physionet --history 36 --patch_size 9 --stride 9 --batch_size 64 --seed $seed

    # ---------- MIMIC-III — Table C.1: history→(patch,bs) 12→(6,16), 24→(12,8), 36→(4.5,8) ----------
    run --dataset mimic --history 12 --patch_size 6   --stride 6   --batch_size 16 --seed $seed
    run --dataset mimic --history 24 --patch_size 12  --stride 12  --batch_size 8  --seed $seed
    run --dataset mimic --history 36 --patch_size 4.5 --stride 4.5 --batch_size 8  --seed $seed
done
