#!/usr/bin/env bash
# Overnight moonshot campaign — rigorous, critique-proof experiments (multi-seed, matched-LR,
# final-val, honest significance verdict).
#
# RESUMABLE + SINGLE-INSTANCE: every stage is a scaling_ladder call with a fixed --tag, and the ladder
# checkpoints per (depth, seed, arm, lr) — so re-running this script after ANY interruption continues
# each unfinished sub-experiment exactly where it stopped (completed stages finalize in seconds). An
# flock guard makes a second concurrent launch a no-op instead of a double-run.
#
#   bash scripts/moonshot_campaign.sh     # run, or re-run to resume
set -u
cd /home/jonas/git/nanochat
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src
LOG=/home/jonas/git/gns/results/campaign.log
LOCK=/home/jonas/git/gns/results/campaign.lock

# single-instance: if another campaign holds the lock, do nothing (don't double-run / contend)
exec 9>"$LOCK"
if ! flock -n 9; then echo "[campaign] another instance is running — exiting" ; exit 0; fi

echo "campaign (re)start $(date)" >> "$LOG"
# defer to any OTHER GPU job (e.g. Gate A) before starting; on a resume this exits immediately
while nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; do sleep 60; done
echo "GPU free; running stages $(date)" >> "$LOG"

LADDER="uv run --project /home/jonas/git/tct-models python -u scripts/scaling_ladder.py \
  --mode fixed --batch-sweep= --compile --fixed-iters 1500"

stage(){  # $1 = tag (resume key); rest = ladder args. Resumes via the ladder checkpoint.
  local tag="$1"; shift
  echo "### stage $tag start $(date) ###" >> "$LOG"
  $LADDER --tag "$tag" "$@" >> "$LOG" 2>&1
  echo "### stage $tag done $(date) ###" >> "$LOG"
}

# C1 — curvature scaling significance: muon vs ortho_shampoo + synth(0.5), matched LR, 3 seeds, d6/8/12
stage camp_curv  --depths 6,8,12 --seeds 0,1,2 --matrix-lr-grid 0.02 \
      --arms muon,ortho_shampoo,synth --synth-alpha 0.5
# C2 — tuned-AdamW strong-baseline anchor: muon vs adamw, d6/8
stage camp_adamw --depths 6,8 --seeds 0,1,2 --matrix-lr-grid 0.005,0.02 --arms muon,adamw
# C3 — grammar cheaper inverse-root preserves training: ortho 8-step grammar schedule, d6/8/12
stage camp_gram  --depths 6,8,12 --seeds 0,1,2 --matrix-lr-grid 0.02 \
      --arms muon,ortho_shampoo --precond-coupled-orders 2,3,3,2,3,3,3,3

echo "campaign DONE $(date)" >> "$LOG"
