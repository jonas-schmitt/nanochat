#!/usr/bin/env bash
# FULL moonshot campaign — one resumable command for everything (significance gate + scaling + grammar).
#
# RESUMABLE + SINGLE-INSTANCE: every stage is a fixed-tag scaling_ladder call, and the ladder checkpoints
# per (depth, seed, arm, lr) — so this command CONTINUES all already-run parts (incl. the Gate-A stages
# launched earlier) and only computes what is missing. Completed stages finalise in seconds. An flock
# guard makes a second concurrent launch a no-op.
#
#   bash scripts/moonshot_campaign.sh        # run, or re-run any time to resume
set -u
cd /home/jonas/git/nanochat
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src
LOG=/home/jonas/git/gns/results/campaign.log
LOCK=/home/jonas/git/gns/results/campaign.lock

exec 9>"$LOCK"
if ! flock -n 9; then echo "[campaign] another instance is running — exiting"; exit 0; fi
echo "campaign (re)start $(date)" >> "$LOG"
while nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; do sleep 60; done
echo "GPU free; running stages $(date)" >> "$LOG"

LADDER="uv run --project /home/jonas/git/tct-models python -u scripts/scaling_ladder.py \
  --mode fixed --batch-sweep= --compile"
stage(){ local tag="$1"; shift
  echo "### stage $tag start $(date) ###" >> "$LOG"
  $LADDER --tag "$tag" "$@" >> "$LOG" 2>&1
  echo "### stage $tag done $(date) ###" >> "$LOG"; }

# --- Gate A (significance) — resumes the parts already trained -----------------------------------------
# 3-seed d8 significance: muon vs ortho_shampoo (alpha=1) and synth(alpha=0.5), matched LR, final-val, t-test
stage gateA_curv  --depths 8 --seeds 0,1,2 --fixed-iters 1500 --matrix-lr-grid 0.02 \
      --arms muon,ortho_shampoo,synth --synth-alpha 0.5
# tuned-AdamW strong-baseline anchor (each at its own LR)
stage gateA_adamw --depths 8 --seeds 0,1,2     --fixed-iters 1000 --matrix-lr-grid 0.003,0.01,0.03 \
      --arms muon,adamw
# --- Scaling + grammar (new) --------------------------------------------------------------------------
# curvature significance across the ladder (3 seeds, d6/8/12)
stage camp_curv   --depths 6,8,12 --seeds 0,1,2 --fixed-iters 1500 --matrix-lr-grid 0.02 \
      --arms muon,ortho_shampoo,synth --synth-alpha 0.5
# grammar cheaper inverse-root preserves training (8-step searched schedule vs muon)
stage camp_gram   --depths 6,8,12 --seeds 0,1,2 --fixed-iters 1500 --matrix-lr-grid 0.02 \
      --arms muon,ortho_shampoo --precond-coupled-orders 2,3,3,2,3,3,3,3

echo "campaign DONE $(date)" >> "$LOG"
