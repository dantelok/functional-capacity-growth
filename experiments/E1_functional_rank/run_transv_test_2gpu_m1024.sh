#!/usr/bin/env bash
# Empirical test: after the current multi-GPU run finishes, try seq128 m=1024 on
# 2 GPUs with QR. Expected to OOM at TSQR (~52.6 GB/GPU); we want to see the real
# error. If it somehow fits, the result lands in the real dir and we skip 3-GPU.
set -u
cd "$(dirname "$0")/../.."   # repo root
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE=checkpoints/exp1_shared_base/base_final.pt

echo "waiting for current multi-GPU run to finish...  $(date '+%H:%M:%S')"
while pgrep -f "multigpu_transversality.py" >/dev/null; do sleep 20; done

echo "===== TEST: seq128 m1024 on 2 GPUs (QR)  $(date '+%H:%M:%S') ====="
CUDA_VISIBLE_DEVICES=0,2 python3 -u experiments/E1_functional_rank/multigpu_transversality.py \
  --base-checkpoint "$BASE" --seq-len 128 --batch-size 2 --seed 42 --n-old-samples 1024 \
  --output-dir results/transversality_grid/seq128_m1024 \
  || echo "!!!! seq128 m1024 (2-GPU QR) FAILED -- see error above !!!!"
echo "===== TEST DONE  $(date '+%H:%M:%S') ====="
