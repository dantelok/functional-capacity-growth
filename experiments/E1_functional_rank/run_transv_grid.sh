#!/usr/bin/env bash
# Single-GPU QR transversality sweep across "seq:m" pairs (reference projector,
# experiments/E1_functional_rank/diagnose_transversality.py). Batch 2, fp32, seed 42.
#   Usage: run_transv_grid.sh <cuda_device> seq:m seq:m ...
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
G="$1"; shift
BASE=checkpoints/exp1_shared_base/base_final.pt
for pair in "$@"; do
  s="${pair%:*}"; m="${pair#*:}"
  echo "==== seq=$s m=$m on GPU $G  $(date '+%H:%M:%S') ===="
  CUDA_VISIBLE_DEVICES="$G" python3 -u experiments/E1_functional_rank/diagnose_transversality.py \
    --base-checkpoint "$BASE" --growth-factor 4 --batch-size 2 --seq-len "$s" \
    --n-old-samples "$m" --seed 42 --device cuda \
    --output-dir "results/transversality_grid/seq${s}_m${m}" \
    || echo "!!!! seq=$s m=$m FAILED !!!!"
done
echo "==== GPU $G done  $(date '+%H:%M:%S') ===="
