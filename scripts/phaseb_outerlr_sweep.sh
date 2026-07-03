#!/bin/bash
# Phase B: tune the MuLoCo incumbent — outer-lr sweep at d6/1500, M4, H30, fp32 wire, seed 0.
# lr 8.0 (EMA-scale; = paper's 0.8 classic) already exists: probe_muloco_fp32.json val 3.9287.
# Gate-B question: does a tuned outer step close the 0.173 gap to the DP reference (3.7555)?
set -e
cd /home/jonas/git/nanochat
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src
export PYTHONUNBUFFERED=1
RUN="uv run --project /home/jonas/git/tct-models python"
R=/home/jonas/git/gns/results

LOG=$R/phaseb_outerlr_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "logging to $LOG"

for OLR in 4 12 16; do
  echo "=== muloco fp32 H30 outer_lr=$OLR"
  $RUN scripts/train_diloco.py --depth 6 --num-iterations 1500 --workers 4 \
    --preset muloco --h 30 --outer-lr $OLR --matrix-lr 0.02 --eval-every 100 \
    --out $R/phaseb_muloco_olr$OLR.json
done
echo "PHASE-B OUTER-LR SWEEP COMPLETE"
