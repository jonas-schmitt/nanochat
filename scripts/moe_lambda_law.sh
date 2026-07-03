#!/bin/bash
# λ-adaptive law follow-up (pre-registered addendum, gns notes/moe-wwd-preregistration.md):
#   E=8 @ wd0.28 (prediction 4: edge >= +0.02) and E=4 @ wd0.56 (prediction 5: λ* grows with E).
# Launch AFTER moe_law_chain.sh completes (do not run concurrently).
set -e
cd /home/jonas/git/nanochat
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src
export PYTHONUNBUFFERED=1
RUN="uv run --project /home/jonas/git/tct-models python"
R=/home/jonas/git/gns/results

LOG=$R/moe_lambda_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "logging to $LOG"

for CFG in "8 0.28" "4 0.56"; do
  set -- $CFG
  echo "=== E=$1 wd=$2 s0"
  $RUN scripts/train_compare_precond.py --depth 6 --num-iterations 1500 \
    --arms muon,muon_wwd --matrix-lr-grid 0.02 --weight-decay $2 \
    --shampoo-recompute-every 100 --ema-eval-beta 0.999 --eval-every 100 \
    --moe-experts $1 --seed 0 --out $R/moe_e$1_wd$2_s0.json
done
echo "LAMBDA-LAW FOLLOW-UP COMPLETE"
