#!/bin/bash
# OVERNIGHT MASTER (2026-07-04) — everything outstanding, highest potential first (~10.5 h).
# Phases are independent; a failure in one must not kill the rest (per-phase guards, no set -e).
cd /home/jonas/git/nanochat
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src
export PYTHONUNBUFFERED=1
RUN="uv run --project /home/jonas/git/tct-models python"
R=/home/jonas/git/gns/results

LOG=$R/overnight_master_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "logging to $LOG"

pair () {  # pair <depth> <experts> <wd> <seed> <out>
  $RUN scripts/train_compare_precond.py --depth $1 --num-iterations 1500 \
    --arms muon,muon_wwd --matrix-lr-grid 0.02 --weight-decay $3 \
    --shampoo-recompute-every 100 --ema-eval-beta 0.999 --eval-every 100 \
    --moe-experts $2 --seed $4 --out $R/$5 || echo "PHASE-FAIL: $5"
}

echo "=== P1: d8 muon lambda-grid (wd 0.2, 0.28; wd0.14 reference = moe_d8_e1_wd0.14_s0)"
for WD in 0.2 0.28; do
  $RUN scripts/train_compare_precond.py --depth 8 --num-iterations 1500 \
    --arms muon --matrix-lr-grid 0.02 --weight-decay $WD \
    --ema-eval-beta 0.999 --eval-every 100 --seed 0 \
    --out $R/d8_muon_wd${WD}_s0.json || echo "PHASE-FAIL: d8 lambda $WD"
done

echo "=== P2: d12 dense WWD trend, 3 seeds (TIER-DECIDING)"
for S in 0 1 2; do
  pair 12 1 0.14 $S d12_wwd_wd0.14_s$S.json
done

echo "=== P3: MoE E=2 seed-2 pairs (completes pre-registered prediction 3)"
pair 6 2 0.14 2 moe_e2_wd0.14_s2.json
pair 6 2 0.28 2 moe_e2_wd0.28_s2.json

echo "=== P4: lambda-law addendum (pre-registered predictions 4+5)"
pair 6 8 0.28 0 moe_e8_wd0.28_s0.json
pair 6 4 0.56 0 moe_e4_wd0.56_s0.json

echo "=== P5: Gate-B closure (outer-lr bracket + 2bit at tuned lr)"
for OLR in 2 6; do
  $RUN scripts/train_diloco.py --depth 6 --num-iterations 1500 --workers 4 \
    --preset muloco --h 30 --outer-lr $OLR --matrix-lr 0.02 --eval-every 100 \
    --out $R/phaseb_muloco_olr$OLR.json || echo "PHASE-FAIL: olr$OLR"
done
$RUN scripts/train_diloco.py --depth 6 --num-iterations 1500 --workers 4 \
  --preset muloco2bit --h 30 --outer-lr 4 --matrix-lr 0.02 --eval-every 100 \
  --out $R/probe_muloco_2bit_olr4.json || echo "PHASE-FAIL: 2bit olr4"

echo "=== P6: Phase-C probes at TUNED outer-lr 4 (lr-8 verdicts invalidated)"
for T in polar whitened; do
  $RUN scripts/train_diloco.py --depth 6 --num-iterations 1500 --workers 4 \
    --preset muloco --h 30 --outer-lr 4 --outer-transform $T --matrix-lr 0.02 \
    --eval-every 100 --out $R/probe_outer_${T}_olr4.json || echo "PHASE-FAIL: outer $T"
done
$RUN scripts/train_diloco.py --depth 6 --num-iterations 1500 --workers 4 \
  --preset muloco --h 30 --outer-lr 4 --outer-transform whitened --outer-whiten-guard \
  --matrix-lr 0.02 --eval-every 100 --out $R/probe_outer_whitened_guarded_olr4.json \
  || echo "PHASE-FAIL: whitened-guarded"

echo "OVERNIGHT MASTER COMPLETE"
