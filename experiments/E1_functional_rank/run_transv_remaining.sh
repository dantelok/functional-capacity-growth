#!/usr/bin/env bash
# Orchestrate the remaining transversality grid, QR projector throughout.
#   Phase 1 (multi-GPU QR, both visible GPUs, D-sharded TSQR):
#       seq128 m=512, seq64 m=1024   (each ~52GB single-GPU -> needs 2 GPUs)
#   Phase 2 (single-GPU QR, parallel):
#       seq32 m up to 1024 on GPU0, seq64 m up to 512 on GPU2
# seq128 m=1024 needs 3 GPUs (QR ~105GB); run it separately (see docs/RUN_ORDER.md).
set -u
cd "$(dirname "$0")/../.."   # repo root
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE=checkpoints/exp1_shared_base/base_final.pt

mg() {  # multi-GPU QR run: $1=seq $2=m
  echo "===== [multi-GPU QR] seq$1 m$2  $(date '+%H:%M:%S') ====="
  CUDA_VISIBLE_DEVICES=0,2 python3 -u experiments/E1_functional_rank/multigpu_transversality.py \
    --base-checkpoint "$BASE" --seq-len "$1" --batch-size 2 --seed 42 --n-old-samples "$2" \
    --output-dir "results/transversality_grid/seq$1_m$2" \
    || echo "!!!! seq$1 m$2 (multi-GPU) FAILED !!!!"
}

mg 128 512
mg 64 1024

echo "===== [2] single-GPU QR seq32 (GPU0) + seq64 (GPU2)  $(date '+%H:%M:%S') ====="
bash experiments/E1_functional_rank/run_transv_grid.sh 0 32:64 32:128 32:256 32:512 32:1024 &
P0=$!
bash experiments/E1_functional_rank/run_transv_grid.sh 2 64:64 64:128 64:256 64:512 &
P2=$!
wait $P0 $P2
echo "===== ALL REMAINING DONE  $(date '+%H:%M:%S') ====="
