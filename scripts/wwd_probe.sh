#!/usr/bin/env bash
# Whitened weight decay (muon_wwd) — d6/1500 VAL screen. Whitened WD is a GENERALIZATION lever, so
# score on VAL (best_val) at 1500 steps (400-step noise ~0.06 swamps it). The real threat is the
# SCALAR-lambda control: whitened must beat RETUNED isotropic WD, not just default. Stage 1 = seed-0
# scout (muon @ wd{0.14,0.28,0.42} vs muon_wwd @ wd{0.14,0.28}); seeds 1,2 gated on it. Resumable.
set -u
cd /home/jonas/git/nanochat
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src PYTHONUNBUFFERED=1
PYBIN="uv run --project /home/jonas/git/tct-models python"
R=/home/jonas/git/gns/results
ts() { date '+%a %d. %b %H:%M:%S'; }
c="--depth 6 --aspect-ratio 64 --device-batch-size 16 --num-iterations 1500 --matrix-lr-grid 0.02 --eval-every 100 --seed 0"

echo "### WWD smoke (30 steps) $(ts) ###"
$PYBIN scripts/train_compare_precond.py --arms muon_wwd --wwd-strength 1.0 $c --num-iterations 30 \
  --eval-every 10 --out "$R/wwd_smoke2.json"

echo "### WWD d6/1500 scout (seed 0) $(ts) ###"
for WD in 0.14 0.28 0.42; do
  echo "--- muon wd=$WD  $(ts) ---"
  $PYBIN scripts/train_compare_precond.py --arms muon $c --weight-decay "$WD" \
    --out "$R/wwd_muon_wd${WD}_s0.json"
done
for WD in 0.14 0.28; do
  echo "--- muon_wwd s1.0 wd=$WD  $(ts) ---"
  $PYBIN scripts/train_compare_precond.py --arms muon_wwd --wwd-strength 1.0 $c --weight-decay "$WD" \
    --out "$R/wwd_wwd_wd${WD}_s0.json"
done
echo "### WWD scout DONE $(ts) ###"
