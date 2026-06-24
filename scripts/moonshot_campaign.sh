#!/usr/bin/env bash
# Overnight moonshot campaign — rigorous, critique-proof experiments (multi-seed, matched-LR,
# final-val, honest significance verdict). Each scaling_ladder call is resumable (per (depth,seed,arm,lr)
# checkpointing), so the whole thing survives a crash/restart. Waits for the GPU to free first.
#
#   bash scripts/moonshot_campaign.sh        # runs all stages
# Individual stages can be run by copy-pasting a single $LADDER line with its --tag.
set -u
cd /home/jonas/git/nanochat
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src
LOG=/home/jonas/git/gns/results/campaign.log
echo "campaign armed $(date)" > "$LOG"
while nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; do sleep 60; done
echo "GPU free; campaign start $(date)" >> "$LOG"
LADDER="uv run --project /home/jonas/git/tct-models python -u scripts/scaling_ladder.py --mode fixed --batch-sweep= --compile --fixed-iters 1500"

echo "### C1 curvature scaling significance: muon vs ortho_shampoo + synth(0.5), matched LR 0.02, 3 seeds, d6/8/12 ###" >> "$LOG"
$LADDER --depths 6,8,12 --seeds 0,1,2 --matrix-lr-grid 0.02 \
  --arms muon,ortho_shampoo,synth --synth-alpha 0.5 --tag camp_curv >> "$LOG" 2>&1

echo "### C2 AdamW strong-baseline anchor: muon vs tuned adamw, d6/8 ###" >> "$LOG"
$LADDER --depths 6,8 --seeds 0,1,2 --matrix-lr-grid 0.005,0.02 \
  --arms muon,adamw --tag camp_adamw >> "$LOG" 2>&1

echo "### C3 grammar cheaper inverse-root preserves training: ortho 8-step grammar schedule, d6/8/12 ###" >> "$LOG"
$LADDER --depths 6,8,12 --seeds 0,1,2 --matrix-lr-grid 0.02 \
  --arms muon,ortho_shampoo --precond-coupled-orders 2,3,3,2,3,3,3,3 --tag camp_gram >> "$LOG" 2>&1

echo "campaign DONE $(date)" >> "$LOG"
