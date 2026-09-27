#!/usr/bin/env bash
# After the in-flight multi-GPU run finishes, (re)run any missing multi-GPU QR
# points with the memory-fixed script (model freed before TSQR).
set -u
cd "$(dirname "$0")/../.."   # repo root
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE=checkpoints/exp1_shared_base/base_final.pt

echo "waiting for in-flight multi-GPU run to finish...  $(date '+%H:%M:%S')"
while pgrep -f "multigpu_transversality.py" >/dev/null; do sleep 20; done

mg() {  # $1=seq $2=m
  if [ -f "results/transversality_grid/seq$1_m$2/results.json" ]; then
    echo "seq$1 m$2 already done, skipping"; return
  fi
  echo "===== [multi-GPU QR retry] seq$1 m$2  $(date '+%H:%M:%S') ====="
  CUDA_VISIBLE_DEVICES=0,2 python3 -u experiments/E1_functional_rank/multigpu_transversality.py \
    --base-checkpoint "$BASE" --seq-len "$1" --batch-size 2 --seed 42 --n-old-samples "$2" \
    --output-dir "results/transversality_grid/seq$1_m$2" \
    || echo "!!!! seq$1 m$2 FAILED again !!!!"
}

mg 128 512
mg 64 1024
echo "===== RERUN DONE  $(date '+%H:%M:%S') ====="
