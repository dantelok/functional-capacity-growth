#!/usr/bin/env bash
# Wait for the single-GPU seq32/seq64 QR loops to finish, then run the two
# multi-GPU QR points that need both GPUs: seq128 m=512 and seq64 m=1024.
set -u
cd "$(dirname "$0")/../.."   # repo root
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE=checkpoints/exp1_shared_base/base_final.pt

echo "waiting for single-GPU loops to free the GPUs...  $(date '+%H:%M:%S')"
while pgrep -f "run_transv_grid.sh" >/dev/null || pgrep -f "diagnose_transversality.py" >/dev/null; do
  sleep 30
done
echo "GPUs free; starting multi-GPU QR runs  $(date '+%H:%M:%S')"

mg() {  # $1=seq $2=m
  echo "===== [multi-GPU QR] seq$1 m$2  $(date '+%H:%M:%S') ====="
  CUDA_VISIBLE_DEVICES=0,2 python3 -u experiments/E1_functional_rank/multigpu_transversality.py \
    --base-checkpoint "$BASE" --seq-len "$1" --batch-size 2 --seed 42 --n-old-samples "$2" \
    --output-dir "results/transversality_grid/seq$1_m$2" \
    || echo "!!!! seq$1 m$2 (multi-GPU) FAILED !!!!"
}

mg 128 512
mg 64 1024
echo "===== MULTI-GPU RUNS DONE  $(date '+%H:%M:%S') ====="
