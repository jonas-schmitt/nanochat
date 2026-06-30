#!/usr/bin/env bash
# Idea 3 — spectral-denoising "soft Muon" — Phase A falsification driver.
#
# The clean shrinkage baseline is soft_muon@tau=0 (EXACT SVD polar), NOT the muon arm: muon uses the
# 5-step Newton-Schulz polar, so muon-vs-soft_muon conflates the ~0.002 svd-vs-fused map gap with the
# shrinkage we want to measure. soft_muon@tau=0 shares soft_muon's SVD code path, so tau=0 -> tau=c
# isolates pure spectral shrinkage. The muon arm is kept only as an absolute anchor in tau_sweep.
#
# Usage:
#   bash scripts/soft_muon_probe.sh tau_sweep            # A2: d6/b16/400, tau in {0,.05,.1,.2,.4}
#   bash scripts/soft_muon_probe.sh grid <best_tau>      # A3: aspect{64,96,128} x batch{8,16,64}
#   SEED=1 bash scripts/soft_muon_probe.sh grid 0.1      # repeat a pass at another seed
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src PYTHONUNBUFFERED=1
PYBIN="uv run --project /home/jonas/git/tct-models python"
R=/home/jonas/git/gns/results
MODE="${1:-tau_sweep}"
SEED="${SEED:-0}"
STEPS=400
ts() { date '+%a %d. %b %H:%M:%S'; }

# run <arm> <tau> <aspect> <batch> <out>
run() {
  echo "--- arm=$1 tau=$2 aspect=$3 batch=$4  $(ts) ---"
  $PYBIN scripts/train_compare_precond.py --arms "$1" --soft-tau "$2" --soft-q 2.0 \
    --depth 6 --aspect-ratio "$3" --device-batch-size "$4" \
    --num-iterations "$STEPS" --matrix-lr-grid 0.02 --eval-every 50 --seed "$SEED" \
    --out "$5"
}

if [ "$MODE" = "tau_sweep" ]; then
  echo "### SOFT-MUON tau-sweep  d6 a64 b16 /${STEPS}  seed${SEED}  $(ts) ###"
  run muon      0.0  64 16 "$R/soft_muon_tau_sweep_muon_s${SEED}.json"
  for TAU in 0.0 0.05 0.1 0.2 0.4; do
    run soft_muon "$TAU" 64 16 "$R/soft_muon_tau_sweep_t${TAU}_s${SEED}.json"
  done
  echo "### tau-sweep DONE  $(ts) ###"

elif [ "$MODE" = "grid" ]; then
  BEST_TAU="${2:?usage: grid <best_tau>}"
  echo "### SOFT-MUON batch x width grid  d6 /${STEPS}  tau0 vs tau${BEST_TAU}  seed${SEED}  $(ts) ###"
  for A in 64 96 128; do
    for B in 8 16 64; do
      run soft_muon 0.0        "$A" "$B" "$R/soft_muon_grid_a${A}_b${B}_t0_s${SEED}.json"
      run soft_muon "$BEST_TAU" "$A" "$B" "$R/soft_muon_grid_a${A}_b${B}_t${BEST_TAU}_s${SEED}.json"
    done
  done
  echo "### grid DONE  $(ts) ###"
else
  echo "unknown mode: $MODE (use tau_sweep | grid)"; exit 1
fi
