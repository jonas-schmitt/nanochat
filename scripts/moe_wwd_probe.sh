#!/bin/bash
# A1 — WWD × MoE probe (ARCHITECTURE-TRANSFER TRACK, TODO.md 2026-07-03).
# Mechanism: WWD = regularizer vs data-unjustified directions; MoE experts see ~1/E of tokens
# (per-expert starvation) => WWD edge should GROW dense -> E=2 -> E=4 (watch for an interior
# peak — datarep s0 was non-monotone: +0.021 @4x but +0.003 @16x).
# Dense E=1 reference pair at wd0.14 already exists: wwd_ema_confound_s{0,1,2}.json.
# Stage 1 (default): seed 0, E in {2,4}, wd in {0.14, 0.28} (retune-lambda-per-arch guard).
# Stage 2 (after judging): SEEDS="1 2" at the chosen wd.
# Usage: bash scripts/moe_wwd_probe.sh   |   SEEDS="1 2" WDS="0.14" bash scripts/moe_wwd_probe.sh
set -e
cd /home/jonas/git/nanochat
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src
export PYTHONUNBUFFERED=1
RUN="uv run --project /home/jonas/git/tct-models python"
R=/home/jonas/git/gns/results

LOG=$R/moe_wwd_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "logging to $LOG"

for S in ${SEEDS:-0}; do
  for E in ${ES:-2 4}; do
    for WD in ${WDS:-0.14 0.28}; do
      echo "=== MoE E=$E wd=$WD seed=$S (muon vs muon_wwd, d6/1500, rc100, EMA-eval)"
      $RUN scripts/train_compare_precond.py --depth 6 --num-iterations 1500 \
        --arms muon,muon_wwd --matrix-lr-grid 0.02 --weight-decay $WD \
        --shampoo-recompute-every 100 --ema-eval-beta 0.999 --eval-every 100 \
        --moe-experts $E --seed $S \
        --out $R/moe_e${E}_wd${WD}_s${S}.json
    done
  done
done
echo "MOE WWD PROBE COMPLETE"
