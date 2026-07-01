#!/usr/bin/env bash
# Whitened WD scale gate: does the CONFIRMED d6 win (paired +0.0125 ± 0.0021, t~6, 3/3 seeds vs
# retuned-isotropic control) HOLD/GROW at d8? Project's standard gate (a d6 win that shrinks at d8
# is a small-scale artifact). Paired muon vs muon_wwd at wd0.14, seeds 0,1,2, d8/1500. Resumable.
set -u
cd /home/jonas/git/nanochat
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src PYTHONUNBUFFERED=1
PYBIN="uv run --project /home/jonas/git/tct-models python"
R=/home/jonas/git/gns/results
ts() { date '+%a %d. %b %H:%M:%S'; }
c="--depth 8 --aspect-ratio 64 --device-batch-size 16 --num-iterations 1500 --matrix-lr-grid 0.02 --eval-every 100 --weight-decay 0.14"

echo "### WWD d8 scale gate $(ts) ###"
for S in 0 1 2; do
  echo "--- muon d8 wd0.14 seed=$S  $(ts) ---"
  $PYBIN scripts/train_compare_precond.py --arms muon $c --seed "$S" --out "$R/wwd_d8_muon_s${S}.json"
  echo "--- muon_wwd d8 s1.0 wd0.14 seed=$S  $(ts) ---"
  $PYBIN scripts/train_compare_precond.py --arms muon_wwd --wwd-strength 1.0 $c --seed "$S" \
    --out "$R/wwd_d8_wwd_s${S}.json"
done
echo "### WWD d8 gate DONE $(ts) ###"
