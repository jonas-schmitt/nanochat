#!/usr/bin/env bash
# Re-test the CURVATURE angle under the new AMORTIZED-cost regime: WWD already maintains L,R, so
# ortho_shampoo-descent (L^-1/4 g R^-1/4, polar) layered WITH whitened decay is ~free vs muon_wwd
# (both pay the factor cost). Question (A): does curvature-descent add ANYTHING over WWD-alone now?
# Baselines already have (d6/1500 wd0.14 s0): muon 4.1806, muon_wwd 4.1683.
# Scout = d6 seed0, lr-swept for the shampoo arms (their optimal LR differs from muon). Waits for GPU.
set -u
cd /home/jonas/git/nanochat
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src PYTHONUNBUFFERED=1
PYBIN="uv run --project /home/jonas/git/tct-models python"
R=/home/jonas/git/gns/results
ts() { date '+%a %d. %b %H:%M:%S'; }
c="--depth 6 --aspect-ratio 64 --device-batch-size 16 --num-iterations 1500 --eval-every 100 --weight-decay 0.14 --seed 0 --matrix-lr-grid 0.01,0.02,0.04"

echo "### CURV+WWD scout — waiting for GPU $(ts) ###"
while pgrep -f "train_compare_precond.py" >/dev/null; do sleep 30; done
echo "### CURV+WWD scout (d6, seed0) $(ts) ###"
echo "--- ortho_shampoo + WWD (the amortized combo)  $(ts) ---"
$PYBIN scripts/train_compare_precond.py --arms ortho_shampoo --wwd --wwd-power 0.5 $c \
  --out "$R/curv_orthoshampoo_wwd_s0.json"
echo "--- ortho_shampoo, isotropic decay (control)  $(ts) ---"
$PYBIN scripts/train_compare_precond.py --arms ortho_shampoo $c \
  --out "$R/curv_orthoshampoo_iso_s0.json"
echo "### CURV+WWD scout DONE $(ts) ###"
