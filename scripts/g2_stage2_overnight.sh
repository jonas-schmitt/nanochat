#!/bin/bash
# G2 Stage-2 overnight chain (2026-07-03):
#   1. d6->d8 trend gate on the Stage-1 knee + the WWD-stacked rank-3 genome (Rung-2 stacking test),
#      EMA-FAIR judging (every arm at its best eval protocol) — the decisive gate.
#   2. WWD x EMA confound seeds 1,2 (multi-seeds the s0 on-EMA edge +0.0118).
# Usage: bash scripts/g2_stage2_overnight.sh
set -e
cd /home/jonas/git/nanochat
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src
export PYTHONUNBUFFERED=1
RUN="uv run --project /home/jonas/git/tct-models python"
R=/home/jonas/git/gns/results

LOG=$R/g2_stage2_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "logging to $LOG"

echo "=== PHASE 1: Stage-2 trend gate (d6,d8 x seeds 0,1,2 x 1500 steps; knee0 + wwd-stacked knee1)"
$RUN -u scripts/eval_program_trend.py --depths 6,8 --seeds 0,1,2 --steps 1500 \
  --lr 0.02 --weight-decay 0.28 \
  --search-json $R/g2_stage2_input.json --n-knees 2 \
  --out $R/g2_trend_gate.json
echo "PHASE 1 DONE -> $R/g2_trend_gate.json"

echo "=== PHASE 2: WWD x EMA confound seeds 1,2"
for S in 1 2; do
  $RUN scripts/train_compare_precond.py --depth 6 --num-iterations 1500 \
    --arms muon,muon_wwd --matrix-lr-grid 0.02 --weight-decay 0.14 \
    --shampoo-recompute-every 100 --ema-eval-beta 0.999 --eval-every 100 \
    --seed $S --out $R/wwd_ema_confound_s$S.json
done
echo "PHASE 2 DONE -> $R/wwd_ema_confound_s{1,2}.json"
echo "OVERNIGHT CHAIN COMPLETE"
