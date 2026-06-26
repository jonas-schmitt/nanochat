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
  --mode fixed --batch-sweep= --compile"
# stage3 = 3-seed significance pass (for stages with existing 3-seed checkpoints — resume in seconds).
stage3(){ local tag="$1"; shift
  echo "### stage $tag start (3-seed) $(date) ###" >> "$LOG"
  $LADDER --seeds 0,1,2 --tag "$tag" "$@" >> "$LOG" 2>&1
  echo "### stage $tag done $(date) ###" >> "$LOG"; }
# stage1 = 1-seed SCOUT (triage only — NOT a significance result; the old single-seed "advantage grows
# with scale" verdict was a self-confirming artifact, fixed in scaling_ladder._fit_trend). Upgrade a
# scout that shows signal: re-run its tag with `--restart --seeds 0,1,2` (discards the 1-seed ckpt,
# starts the 3-seed significance pass). Cheap stages can also just be run at 3 seeds directly.
stage1(){ local tag="$1"; shift
  echo "### stage $tag start (1-seed SCOUT) $(date) ###" >> "$LOG"
  $LADDER --seeds 0 --tag "$tag" "$@" >> "$LOG" 2>&1
  echo "### stage $tag done $(date) ###" >> "$LOG"; }

# STAGE ORDER (2026-06-26 post-audit reorder): most promising directions first.
# All checkpoints were moved to audit-pre-backup/ — campaign starts FRESH with the audit-fixed harness
# (C3: production beta2=0.9/wd=0.28 cosine; H2: tiny matrices filtered from Muon; H1: SOAP rotation fix)
# and the corrected-cost grammar search schedules (exp28/exp29 rerun + large search 256×200×3 seeds).
# Resumable, so moving a stage is free. All PENDING stages run as 1-seed SCOUTS first.
# Upgrade a scout that shows signal: `scaling_ladder.py --tag <tag> --restart --seeds 0,1,2 ...`.

# ===== 1. CORE 3-SEED BASELINE (must go first — foundation for interpreting all scouts) ================
# These re-establish the d6/d8/d12 curvature result with the audit-fixed harness. ~2-3 hours total.
stage3 gateA_curv  --depths 8     --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo,synth --synth-alpha 0.5
stage3 camp_curv   --depths 6,8,12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo,synth --synth-alpha 0.5
stage3 camp_gram   --depths 6,8,12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo --precond-coupled-orders 2,2,2,3,3,3,3,3,3
stage3 batch16   --depths 8,12 --device-batch-size 16  --fixed-iters 1500 --matrix-lr-grid 0.02  --arms muon,ortho_shampoo,synth --synth-alpha 0.5

# ===== 2. MOST PROMISING NEW DIRECTIONS (1-seed scouts; ranked by P(win) × SOTA-relevance) ===========

# ===== W-d6 — WIDTH scaling at d6 (direction 1: MOST SOTA-relevant untested axis) =====================
# The campaign's "scale" axis is DEPTH (d6->d12); frontier SOTA is WIDE, not deep. If the ortho_shampoo
# win GROWS with width at fixed depth, the d12 erosion is a depth-specific artifact irrelevant to the
# regime where SOTA lives. Wide nets have more high-kappa MLP factors without depth-induced gradient-flow
# complications. P(win survives width) plausibly > P(survives depth).
#   d6 a96 -> w640 (~30M, ~= d8);  d6 a128 -> w768 (~42M, between d8 and d12)
# WIN SIGNAL: the ortho_shampoo gap vs muon becomes MORE negative as aspect grows (within a fixed depth).
# notes/scaling-directions-not-sampled.md direction 1. Cheapest new-bet data (d6 only).
stage1 width_d6_a96  --depths 6 --aspect-ratio 96  --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo,synth --synth-alpha 0.5
stage1 width_d6_a128 --depths 6 --aspect-ratio 128 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo,synth --synth-alpha 0.5

# ===== F — orthogonalization FREQUENCY (direction 2: cheapest d12 mechanism test) =====================
# Every arm applies the preconditioner EVERY step; at d12 the benefit saturates while cumulative
# disruption grows. Apply the polar/curvature map only every K steps, raw Nesterov in between (both arms
# at the same K so the gap comparison is fair). Tests whether over-orthogonalization at depth causes the
# d12 erosion. WIN SIGNAL: the ortho_shampoo gap at d12 becomes <0 as K increases.
# Also a wall-clock lever (fewer preconditioner applications = cheaper steps). d12-only, 1500 it.
stage1 orthK2_d12 --depths 12 --orth-every 2 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo,synth --synth-alpha 0.5
stage1 orthK4_d12 --depths 12 --orth-every 4 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo,synth --synth-alpha 0.5
stage1 orthK8_d12 --depths 12 --orth-every 8 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo,synth --synth-alpha 0.5

# ===== A3 — curvature strength alpha*(scale) (erodes least; continuous curvature picture) ===============
stage1 alpha0p25 --depths 6,8,12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,synth --synth-alpha 0.25
stage1 alpha0p50 --depths 6,8,12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,synth --synth-alpha 0.5
stage1 alpha0p75 --depths 6,8,12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,synth --synth-alpha 0.75
stage1 alpha1p00 --depths 6,8,12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,synth --synth-alpha 1.0

# ===== 3. COMPLETING THE PICTURE (1-seed scouts; heavier) ============================================

# ===== W-d8 — WIDTH scaling at d8 (completes the W picture; bigger) ====================================
#   d8 a96 -> w768 (~57M);  d8 a128 -> w1024 (~101M, > d12)
stage1 width_d8_a96  --depths 8 --aspect-ratio 96  --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo,synth --synth-alpha 0.5
stage1 width_d8_a128 --depths 8 --aspect-ratio 128 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo,synth --synth-alpha 0.5

# ===== A1 — batch x scale (DECISIVE but heavy: batch64-256 are 4-16x compute/step) =====================
# LR scaled ~sqrt(bs/16). The canonical regime where curvature beats first-order is LARGE BATCH.
stage1 batch64   --depths 8,12 --device-batch-size 64  --fixed-iters 1500 --matrix-lr-grid 0.04  --arms muon,ortho_shampoo,synth --synth-alpha 0.5
stage1 batch128  --depths 8,12 --device-batch-size 128 --fixed-iters 1500 --matrix-lr-grid 0.057 --arms muon,ortho_shampoo,synth --synth-alpha 0.5
stage1 batch256  --depths 8,12 --device-batch-size 256 --fixed-iters 1500 --matrix-lr-grid 0.08  --arms muon,ortho_shampoo,synth --synth-alpha 0.5

# ===== 4. CONTROLS AND REMAINING (1-seed scouts) =====================================================

# ===== A2 — under-training check (longer d12; 3000 it; required control) ==============================
stage1 d12_long  --depths 12 --fixed-iters 3000 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo,synth --synth-alpha 0.5

# ===== A4 — SOAP (Adam in the Kronecker eigenbasis, own Adam-LR; H1 fix applied) ======================
stage1 soap      --depths 6,8,12 --fixed-iters 1500 --matrix-lr-grid 0.003,0.01,0.03 --arms muon,soap

# ===== A0 — shrinkage spread (exp29 regimes; large-search schedules 256×200×3 seeds) ==================
# Schedules from experiments/search_precond_schedules.py --regimes 1e-6,1e-4,1e-3,1e-2 --pop 256 --gens 200
stage1 shrink1e6 --depths 6,8,12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo --precond-coupled-orders 2,2,2,2,3,3,3,3,3,3,3 --shampoo-ridge 1e-6
stage1 shrink1e4 --depths 6,8,12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo --precond-coupled-orders 2,2,2,3,3,3,3,3 --shampoo-ridge 1e-4
stage1 shrink1e3 --depths 6,8,12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo --precond-coupled-orders 3,3,3,3,3,3 --shampoo-ridge 1e-3
stage1 shrink1e2 --depths 6,8,12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo --precond-coupled-orders 2,2,3,3,3 --shampoo-ridge 1e-2

# ===== 5. PROVENANCE (3-seed; quick — re-establish with fixed harness for verification) ===============

# A-alloc (FALSIFIED 2026-06-26): per-factor kappa allocation underperformed even uniform curvature
# at d12 (gap +0.0091, worse than Muon). Kept for provenance/verification on resume.
stage3 layer_adapt --depths 6,8,12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,layer_adaptive

# tuned-AdamW anchor (3-seed; the strong conventional baseline)
stage3 gateA_adamw --depths 8 --fixed-iters 1000 --matrix-lr-grid 0.003,0.01,0.03 --arms muon,adamw
echo "campaign DONE $(date)" >> "$LOG"
