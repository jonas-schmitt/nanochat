#!/usr/bin/env bash
# Idea 3 / Bet B extensions — soft-Muon ESTIMATOR-gate falsification driver (E1 soft_muon_snr, E2 soft_muon_mp).
#
# Sibling of soft_muon_probe.sh: same regime + the SAME exact-SVD anchor (soft_muon@tau=0, NOT muon, to isolate
# pure shrinkage from the ~0.002 svd-vs-fused gap), but the gate is the PARAMETER-FREE estimator instead of a
# swept tau. soft_muon_snr@strength=0 == soft_muon@tau=0 (both return U V^T), so strength 0 -> c isolates the
# empirical-SNR shrinkage; soft_muon_mp has no knob (tau from the spectrum bulk). Distinct output filenames, so
# nothing here touches the in-flight soft_muon_{tau_sweep,grid} runs.
#
# Usage:
#   bash scripts/soft_muon_snr_probe.sh strength_sweep                  # d6/a64/b16, strength in {0,.5,1,2,4}
#   bash scripts/soft_muon_snr_probe.sh grid soft_muon_snr [strength]   # aspect{64,96,128} x batch{8,16,64}
#   bash scripts/soft_muon_snr_probe.sh grid soft_muon_mp               # parameter-free variant (no strength)
#   SEED=1 bash scripts/soft_muon_snr_probe.sh grid soft_muon_snr 1.0   # repeat a pass at another seed
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src PYTHONUNBUFFERED=1
PYBIN="uv run --project /home/jonas/git/tct-models python"
R=/home/jonas/git/gns/results
MODE="${1:-strength_sweep}"
SEED="${SEED:-0}"
STEPS=400
ts() { date '+%a %d. %b %H:%M:%S'; }

# run <arm> <strength> <aspect> <batch> <out>  (--snr-strength used by soft_muon_snr; harmless to others)
run() {
  echo "--- arm=$1 strength=$2 aspect=$3 batch=$4  $(ts) ---"
  $PYBIN scripts/train_compare_precond.py --arms "$1" --snr-strength "$2" --soft-q 2.0 \
    --depth 6 --aspect-ratio "$3" --device-batch-size "$4" \
    --num-iterations "$STEPS" --matrix-lr-grid 0.02 --eval-every 50 --seed "$SEED" \
    --out "$5"
}

# anchor = exact-SVD polar (soft_muon@tau=0); identical to soft_muon_snr@strength=0.
anchor() {
  echo "--- anchor soft_muon@tau=0 aspect=$1 batch=$2  $(ts) ---"
  $PYBIN scripts/train_compare_precond.py --arms soft_muon --soft-tau 0.0 --soft-q 2.0 \
    --depth 6 --aspect-ratio "$1" --device-batch-size "$2" \
    --num-iterations "$STEPS" --matrix-lr-grid 0.02 --eval-every 50 --seed "$SEED" \
    --out "$3"
}

if [ "$MODE" = "strength_sweep" ]; then
  echo "### SOFT-MUON-SNR strength-sweep  d6 a64 b16 /${STEPS}  seed${SEED}  $(ts) ###"
  for S in 0.0 0.5 1.0 2.0 4.0; do
    run soft_muon_snr "$S" 64 16 "$R/soft_muon_snr_sweep_s${S}_seed${SEED}.json"
  done
  echo "### strength-sweep DONE  $(ts) ###"

elif [ "$MODE" = "grid" ]; then
  ARM="${2:?usage: grid <soft_muon_snr|soft_muon_mp> [strength]}"
  STR="${3:-1.0}"
  echo "### ${ARM} batch x width grid  d6 /${STEPS}  vs soft_muon@tau=0  strength${STR}  seed${SEED}  $(ts) ###"
  for A in 64 96 128; do
    for B in 8 16 64; do
      anchor "$A" "$B"            "$R/soft_muon_snr_grid_a${A}_b${B}_anchor_s${SEED}.json"
      run "$ARM" "$STR" "$A" "$B" "$R/soft_muon_snr_grid_a${A}_b${B}_${ARM}_s${SEED}.json"
    done
  done
  echo "### grid DONE  $(ts) ###"
else
  echo "unknown mode: $MODE (use strength_sweep | grid)"; exit 1
fi
