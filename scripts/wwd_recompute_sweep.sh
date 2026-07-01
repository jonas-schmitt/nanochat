#!/usr/bin/env bash
# Rung 1 — iso-FLOP decider for WWD. Does the +0.0125 survive STALE factors (recompute 50/100 vs 10)?
# Decay is slow-moving => should tolerate it => wall drops toward plain muon (680s) while the win holds.
# muon_wwd FIRST (the priority), then ortho_shampoo+WWD (does the DESCENT also tolerate staleness?).
# Have already (recompute=10, d6/1500 s0 wd0.14): muon 4.1806/680s, muon_wwd 4.1683/842s,
#   ortho_shampoo+WWD 4.1648/834s @lr0.02.
set -u
cd /home/jonas/git/nanochat
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src PYTHONUNBUFFERED=1
PYBIN="uv run --project /home/jonas/git/tct-models python"
R=/home/jonas/git/gns/results
ts() { date '+%a %d. %b %H:%M:%S'; }
c="--depth 6 --aspect-ratio 64 --device-batch-size 16 --num-iterations 1500 --eval-every 100 --weight-decay 0.14 --seed 0 --matrix-lr-grid 0.02"

echo "### WWD recompute sweep — waiting for GPU $(ts) ###"
while pgrep -f "train_compare_precond.py" >/dev/null; do sleep 15; done
for RC in 50 100; do
  echo "--- muon_wwd recompute=$RC  $(ts) ---"
  $PYBIN scripts/train_compare_precond.py --arms muon_wwd --wwd-power 0.5 $c \
    --shampoo-recompute-every "$RC" --out "$R/wwd_rc${RC}_s0.json"
  echo "--- ortho_shampoo+WWD recompute=$RC  $(ts) ---"
  $PYBIN scripts/train_compare_precond.py --arms ortho_shampoo --wwd --wwd-power 0.5 $c \
    --shampoo-recompute-every "$RC" --out "$R/curvwwd_rc${RC}_s0.json"
done
echo "### WWD recompute sweep DONE $(ts) ###"
