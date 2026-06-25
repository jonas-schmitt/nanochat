#!/usr/bin/env bash
# FULL moonshot + adaptations campaign — one resumable command.
#
# RESUMABLE + SINGLE-INSTANCE: every stage is a fixed-tag scaling_ladder call; the ladder checkpoints per
# (depth, seed, arm, lr), so this CONTINUES all already-run parts and only computes what is missing.
# Completed stages finalise in seconds. flock guard = a second concurrent launch is a no-op.
# Stop/resume any time (Ctrl-C, then re-run). Comment out stages you don't want.
#
#   bash scripts/moonshot_campaign.sh
# After it finishes (or any time): python scripts/analyze_adaptations.py
set -u
cd /home/jonas/git/nanochat
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src
LOG=/home/jonas/git/gns/results/campaign.log
LOCK=/home/jonas/git/gns/results/campaign.lock
exec 9>"$LOCK"; if ! flock -n 9; then echo "[campaign] already running — exiting"; exit 0; fi
echo "campaign (re)start $(date)" >> "$LOG"
LADDER="uv run --project /home/jonas/git/tct-models python -u scripts/scaling_ladder.py \
  --mode fixed --batch-sweep= --compile --seeds 0,1,2"
stage(){ local tag="$1"; shift
  echo "### stage $tag start $(date) ###" >> "$LOG"
  $LADDER --tag "$tag" "$@" >> "$LOG" 2>&1
  echo "### stage $tag done $(date) ###" >> "$LOG"; }

# ===== prior moonshot stages (resume; finalise fast) ==================================================
stage gateA_curv  --depths 8     --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo,synth --synth-alpha 0.5
stage camp_curv   --depths 6,8,12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo,synth --synth-alpha 0.5
stage camp_gram   --depths 6,8,12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo --precond-coupled-orders 2,3,3,2,3,3,3,3

# ===== A1 — batch x scale (DECISIVE): does curvature win grow with batch at d12? (LR scaled ~sqrt(bs/16))
stage batch16   --depths 8,12 --device-batch-size 16  --fixed-iters 1500 --matrix-lr-grid 0.02  --arms muon,ortho_shampoo,synth --synth-alpha 0.5
stage batch64   --depths 8,12 --device-batch-size 64  --fixed-iters 1500 --matrix-lr-grid 0.04  --arms muon,ortho_shampoo,synth --synth-alpha 0.5
stage batch128  --depths 8,12 --device-batch-size 128 --fixed-iters 1500 --matrix-lr-grid 0.057 --arms muon,ortho_shampoo,synth --synth-alpha 0.5
stage batch256  --depths 8,12 --device-batch-size 256 --fixed-iters 1500 --matrix-lr-grid 0.08  --arms muon,ortho_shampoo,synth --synth-alpha 0.5

# ===== A3 — curvature strength alpha*(scale) ==========================================================
stage alpha0p25 --depths 6,8,12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,synth --synth-alpha 0.25
stage alpha0p50 --depths 6,8,12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,synth --synth-alpha 0.5
stage alpha0p75 --depths 6,8,12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,synth --synth-alpha 0.75
stage alpha1p00 --depths 6,8,12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,synth --synth-alpha 1.0

# ===== A4 — SOAP (Adam in the Kronecker eigenbasis), own Adam-LR ======================================
stage soap      --depths 6,8,12 --fixed-iters 1500 --matrix-lr-grid 0.003,0.01,0.03 --arms muon,soap

# ===== A0 — shrinkage spread (exp29 schedules + matched ridge): which shrinkage holds at scale? =======
stage shrink1e6 --depths 6,8,12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo --precond-coupled-orders 2,2,2,2,2,2,3,3,2,2,2,2 --shampoo-ridge 1e-6
stage shrink1e4 --depths 6,8,12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo --precond-coupled-orders 2,3,3,2,2,3,3,3 --shampoo-ridge 1e-4
stage shrink1e3 --depths 6,8,12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo --precond-coupled-orders 3,3,3,3,3,3 --shampoo-ridge 1e-3
stage shrink1e2 --depths 6,8,12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo --precond-coupled-orders 2,2,3,3,3 --shampoo-ridge 1e-2

# ===== A2 — under-training check (longer d12) =========================================================
stage d12_long  --depths 12 --fixed-iters 3000 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo,synth --synth-alpha 0.5

# tuned-AdamW anchor (resume)
stage gateA_adamw --depths 8 --fixed-iters 1000 --matrix-lr-grid 0.003,0.01,0.03 --arms muon,adamw
echo "campaign DONE $(date)" >> "$LOG"
