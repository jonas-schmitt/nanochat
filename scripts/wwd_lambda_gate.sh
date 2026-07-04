#!/bin/bash
# WWD λ-gate — scientific closure FALLBACK (run only on fused-track KILL). TODO "FUSED EARLY-KILL GATE".
# Tests whether the "WWD edge grows with depth" headline survives PER-DEPTH λ tuning (1/D confound).
# Dense muon/muon_wwd pairs, downward λ-grid at d8+d12, then 3-seed confirm at each arm's λ*.
# RESUMABLE: every run is skipped iff its output already exists AND parses (validity-checked).
set -u
cd /home/jonas/git/nanochat
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src
export PYTHONUNBUFFERED=1
RUN="uv run --project /home/jonas/git/tct-models python"
R=/home/jonas/git/gns/results
DEPTHS="${DEPTHS:-8 12}"
LAMBDAS="${LAMBDAS:-0.05 0.07 0.105 0.14}"

LOG=$R/wwd_lambda_gate_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "logging to $LOG"

# valid = file exists AND parses as JSON with an 'arms' or '_done' key (else it is re-run)
valid () { python3 -c "import json,sys
try:
 d=json.load(open('$1')); sys.exit(0 if (d.get('arms') or d.get('_done')) else 1)
except Exception: sys.exit(1)" 2>/dev/null; }

pair () {  # pair <depth> <wd> <seed>  (arms muon+muon_wwd; WWD mechanism at defaults)
  local out=$R/wwdgate_d$1_wd$2_s$3.json
  if valid "$out"; then echo "  skip (done) $out"; return; fi
  $RUN scripts/train_compare_precond.py --depth $1 --num-iterations 1500 \
    --arms muon,muon_wwd --matrix-lr-grid 0.02 --weight-decay $2 \
    --wwd-power 0.5 --wwd-strength 1.0 \
    --shampoo-recompute-every 100 --ema-eval-beta 0.999 --eval-every 100 \
    --seed $3 --out "$out" || echo "RUN-FAIL: $out"
}

# Stage A: λ-search, seed 0
for D in $DEPTHS; do for WD in $LAMBDAS; do pair $D $WD 0; done; done
# Stage B: seeds 1,2 across the whole grid (the judge picks λ* and uses the matching files)
for D in $DEPTHS; do for WD in $LAMBDAS; do for S in 1 2; do pair $D $WD $S; done; done; done

echo "=== JUDGE"
$RUN scripts/judge_wwd_gate.py --results-dir $R \
  --depths "$(echo $DEPTHS | tr ' ' ',')" --lambdas "$(echo $LAMBDAS | tr ' ' ',')" --seeds 0,1,2
echo "WWD λ-GATE COMPLETE"
