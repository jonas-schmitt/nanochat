#!/bin/bash
# PRE-PARK QUEUE (TODO.md CURRENT FOCUS 2026-07-03) — the last GNS GPU spend before tct handoff.
#   P1: d8 muon λ-grid — λ-fairness confound on the depth-growth signal (was retuned at d6 only).
#   P2: d12 dense WWD pairs, 3 seeds — does the WWD edge keep growing at depth? (tier-deciding)
#       Uses the d8-λ-grid winner via WD_D12 (default 0.14 = current d8 protocol; override after P1).
#   P3: Phase-C probe re-runs at tuned outer-lr 4 (lr-8 verdicts invalidated).
# Launch AFTER moe_lambda_law.sh completes.
set -e
cd /home/jonas/git/nanochat
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src
export PYTHONUNBUFFERED=1
RUN="uv run --project /home/jonas/git/tct-models python"
R=/home/jonas/git/gns/results

LOG=$R/prepark_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "logging to $LOG"

echo "=== P1: d8 muon lambda-grid (fairness check for the depth-growth signal)"
for WD in 0.14 0.2 0.28; do
  $RUN scripts/train_compare_precond.py --depth 8 --num-iterations 1500 \
    --arms muon --matrix-lr-grid 0.02 --weight-decay $WD \
    --ema-eval-beta 0.999 --eval-every 100 --seed 0 \
    --out $R/d8_muon_wd${WD}_s0.json
done

echo "=== P2: d12 dense WWD pairs x 3 seeds (wd ${WD_D12:-0.14})"
for S in 0 1 2; do
  $RUN scripts/train_compare_precond.py --depth 12 --num-iterations 1500 \
    --arms muon,muon_wwd --matrix-lr-grid 0.02 --weight-decay ${WD_D12:-0.14} \
    --shampoo-recompute-every 100 --ema-eval-beta 0.999 --eval-every 100 --seed $S \
    --out $R/d12_wwd_wd${WD_D12:-0.14}_s$S.json
done

echo "=== P3: Phase-C probes at TUNED outer-lr 4 (lr-8 verdicts were invalidated)"
for T in polar whitened; do
  $RUN scripts/train_diloco.py --depth 6 --num-iterations 1500 --workers 4 \
    --preset muloco --h 30 --outer-lr 4 --outer-transform $T --matrix-lr 0.02 \
    --eval-every 100 --out $R/probe_outer_${T}_olr4.json
done
$RUN scripts/train_diloco.py --depth 6 --num-iterations 1500 --workers 4 \
  --preset muloco --h 30 --outer-lr 4 --outer-transform whitened --outer-whiten-guard \
  --matrix-lr 0.02 --eval-every 100 --out $R/probe_outer_whitened_guarded_olr4.json

echo "PRE-PARK QUEUE COMPLETE"
