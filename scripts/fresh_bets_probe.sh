#!/usr/bin/env bash
# Fresh temporal bets — falsification. muon_stiefel (EMA of orthogonalized dirs) + muon_anderson (adaptive
# secant extrapolation) vs muon. MUST be at 1500 steps: d6/400 noise (~0.06) swamps small effects (io_split lesson).
# A 30-step smoke runs first to catch NaN/dispatch errors before the expensive runs. Persistent + resumable.
set -u
cd /home/jonas/git/nanochat
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src PYTHONUNBUFFERED=1
PYBIN="uv run --project /home/jonas/git/tct-models python"
R=/home/jonas/git/gns/results
ts() { date '+%a %d. %b %H:%M:%S'; }
c15="--depth 6 --aspect-ratio 64 --device-batch-size 16 --num-iterations 1500 --matrix-lr-grid 0.02 --eval-every 100"

echo "### FRESH-BET SMOKE (30 steps, catch NaN)  $(ts) ###"
$PYBIN scripts/train_compare_precond.py --arms muon_stiefel,muon_anderson \
  --depth 6 --aspect-ratio 64 --device-batch-size 16 --num-iterations 30 \
  --matrix-lr-grid 0.02 --eval-every 10 --seed 0 --out "$R/freshbet_smoke.json"

echo "### FRESH-BET d6/1500 screen (seeds 0,1)  $(ts) ###"
for S in 0 1; do
  for ARM in muon muon_stiefel muon_anderson; do
    echo "--- $ARM seed=$S  $(ts) ---"
    $PYBIN scripts/train_compare_precond.py --arms "$ARM" $c15 --seed "$S" \
      --out "$R/freshbet_${ARM}_s${S}.json"
  done
done
echo "### FRESH-BET DONE  $(ts) ###"
