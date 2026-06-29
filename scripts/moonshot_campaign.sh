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

# ========================================================================================================
# TIERED REORDER (2026-06-29): value-first. The core-baseline block is DONE (gateA_curv/camp_curv/
# camp_gram/batch16-d8 all checkpointed → finalise in seconds). Verdict so far: d6/d8 ortho_shampoo win
# SURVIVED the stronger audit-fixed baseline (5× replicated, significant); d12 erosion replicated; the
# searched grammar schedule (camp_gram) gave the biggest d8 gap at the LOWEST cost. Open questions, ranked:
#   (1) is the win WIDTH-driven not depth-driven?  -> Tier 1 width stages (MoE-relevant: frontier is wide)
#   (2) can we ship "cheaper Muon, same quality"?  -> Tier 1 cost arms (the fallback deliverable)
# Heavy/low-value stages are GATED or commented out below (reversible) per the 2026-06-29 prune decision.
# ========================================================================================================

# ===== TIER 0. CORE 3-SEED BASELINE (DONE — resumes in seconds) ======================================
stage3 gateA_curv  --depths 8     --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo,synth --synth-alpha 0.5
stage3 camp_curv   --depths 6,8,12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo,synth --synth-alpha 0.5
stage3 camp_gram   --depths 6,8,12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo --precond-coupled-orders 2,2,2,3,3,3,3,3,3
# batch16: PRUNED. Its d8 rung is DONE and saved (results/scaling_ladder_batch16.json; analyze reads it).
# d12 was redundant (device-batch 16 == default, just re-replicates camp_curv d12). NOT re-invoked here:
# changing its --depths would hard-abort load_or_init (depths is a checkpoint key). Leaving it commented
# keeps the d8 data and avoids both the abort and the redundant ~1.7h d12 recompute.
# stage3 batch16   --depths 8,12 --device-batch-size 16 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo,synth --synth-alpha 0.5

# ===== TIER 1. DECISIVE + CHEAP (~3h) — run these first ==============================================

# --- WIDTH at fixed depth: THE width-vs-depth / MoE question (~1.5h) ---------------------------------
# The campaign's only scale axis was DEPTH (d6->d12, width grew with it) so "erodes at d12" is confounded:
# depth or total-scale? These vary WIDTH at fixed depth to separate them. Clean contrast:
#   width_d8_a96 (d8, w768)  vs  camp_curv d12 (d12, w768)  == SAME WIDTH, different depth.
# WIN = ortho_shampoo gap grows (more negative) as aspect grows at fixed depth -> erosion is depth-specific
# -> method is relevant to the WIDE (MoE) regime where SOTA lives, and the grammar/fp8 cost wins compound.
#   d6 a96->w576, a128->w768 ; d8 a96->w768, a128->w1024
stage1 width_d6_a96  --depths 6 --aspect-ratio 96  --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo,synth --synth-alpha 0.5
stage1 width_d6_a128 --depths 6 --aspect-ratio 128 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo,synth --synth-alpha 0.5
stage1 width_d8_a96  --depths 8 --aspect-ratio 96  --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo,synth --synth-alpha 0.5
stage1 width_d8_a128 --depths 8 --aspect-ratio 128 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo,synth --synth-alpha 0.5

# --- CHEAPER MUON: the practical deliverable if accuracy dies (~1.6h) --------------------------------
# C1 4-step polar (dir 2, P~65%): joint-opt coeffs (results/jointopt_grammar_probe.json) vs 5-step muon.
stage1 cost_4step_d8  --depths 8  --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,muon_4step,muon_3step
stage1 cost_4step_d12 --depths 12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,muon_4step,muon_3step
# C2 fp8 polar end-to-end (dir 1, P~70%): exp27 showed 1.76-2.58x kernel speedup at width>=8192, never
# validated in training. Shape guard (C8) falls back to bf16 for non-16-divisible matrices.
stage1 fp8_d8  --depths 8  --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,muon_fp8
stage1 fp8_d12 --depths 12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,muon_fp8

# ===== TIER 2. EXPENSIVE — GATED on a Tier-1 signal =================================================
# Prune-directive (2026-06-29): only the cheap Tier-1 triage runs unattended. Expensive (d12 / large-batch
# / 3-seed) compute is spent ONLY on a bet Tier 1 actually confirms — never on low-potential mechanism
# tests run blind. Uncomment a block only when its gate below fires.

# --- GATE: BATCH axis (the other "curvature wins at scale" bet). batch64_d8 is the cheap probe (~1.9h).
#     If the ortho gap GROWS vs batch16-d8 (-0.0138), unlock the full sweep; else leave it pruned.
stage1 batch64_d8 --depths 8 --device-batch-size 64 --fixed-iters 1500 --matrix-lr-grid 0.04 --arms muon,ortho_shampoo,synth --synth-alpha 0.5
# UNLOCK on a positive batch64_d8 signal (high-potential, expensive — the regime where curvature should win):
# stage1 batch64   --depths 8,12 --device-batch-size 64  --fixed-iters 1500 --matrix-lr-grid 0.04  --arms muon,ortho_shampoo,synth --synth-alpha 0.5
# stage1 batch128  --depths 8,12 --device-batch-size 128 --fixed-iters 1500 --matrix-lr-grid 0.057 --arms muon,ortho_shampoo,synth --synth-alpha 0.5
# stage1 batch256  --depths 8,12 --device-batch-size 256 --fixed-iters 1500 --matrix-lr-grid 0.08  --arms muon,ortho_shampoo,synth --synth-alpha 0.5

# UNLOCK on a positive WIDTH signal (Tier 1): upgrade to significance + push wider — the high-potential
# expensive confirmation of the MoE-relevance bet. e.g.:
#   scaling_ladder.py --tag width_d8_a96 --restart --seeds 0,1,2 --depths 8 --aspect-ratio 96 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo,synth --synth-alpha 0.5

# ===== PRUNED — LOW POTENTIAL (do NOT run unattended; revive only with a specific justification) =====
# Mechanism tests / low-P new methods / controls. Per the prune directive these get NO expensive (d12)
# compute unless a Tier-1 result specifically motivates them. Kept here, commented, for one-line revival.
# --- F orth-frequency at d12 (dir 3): d12 mechanism + free wall-clock ratios. MOOT if the win is
#     width-driven (lead hypothesis). Revive for the wall-clock lever IF cheaper-Muon becomes the deliverable.
# stage1 orthK2_d12 --depths 12 --orth-every 2 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo,synth --synth-alpha 0.5
# stage1 orthK4_d12 --depths 12 --orth-every 4 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo,synth --synth-alpha 0.5
# stage1 orthK8_d12 --depths 12 --orth-every 8 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo,synth --synth-alpha 0.5
# --- C3 low-rank orthogonalization (dir 4, P~40%): not yet a wall-clock win (full SVD > 5 polar matmuls).
#     Revive (d8 first) only if the WIDTH bet lands — then it's the "cheaper Muon for wide models" follow-up.
# stage1 lowrank_d8    --depths 8  --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,lowrank_orth --lowrank-k 64
# stage1 lowrank_d12   --depths 12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,lowrank_orth --lowrank-k 64
# stage1 lowrank_k32_d8  --depths 8  --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,lowrank_orth --lowrank-k 32
# stage1 lowrank_k32_d12 --depths 12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,lowrank_orth --lowrank-k 32
# --- E eigenbasis composition (dir 6, P~15%): low-P new method. d8 scout is cheap if ever wanted.
# stage1 eigen_d8  --depths 8  --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo,eigenbasis_shampoo
# stage1 eigen_d12 --depths 12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo,eigenbasis_shampoo
# --- A3 curvature-strength alpha continuum (d6,d8): secondary refinement, not a high-potential bet.
# stage1 alpha0p25 --depths 6,8 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,synth --synth-alpha 0.25
# stage1 alpha0p50 --depths 6,8 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,synth --synth-alpha 0.5
# stage1 alpha0p75 --depths 6,8 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,synth --synth-alpha 0.75
# stage1 alpha1p00 --depths 6,8 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,synth --synth-alpha 1.0

# --- PRUNED CONTROLS (commented; revive only if a direction above turns positive) --------------------
# A2 under-training control (longer d12, 3000it): revive only if d12 erosion looks like under-training.
# stage1 d12_long  --depths 12 --fixed-iters 3000 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo,synth --synth-alpha 0.5
# A4 SOAP (3-lr grid, heavy): control, low value given the verdict.
# stage1 soap      --depths 6,8,12 --fixed-iters 1500 --matrix-lr-grid 0.003,0.01,0.03 --arms muon,soap
# A0 shrinkage spread (4 stages): grammar-ridge refinement; camp_gram already shows the cost win.
# stage1 shrink1e6 --depths 6,8,12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo --precond-coupled-orders 2,2,2,2,3,3,3,3,3,3,3 --shampoo-ridge 1e-6
# stage1 shrink1e4 --depths 6,8,12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo --precond-coupled-orders 2,2,2,3,3,3,3,3 --shampoo-ridge 1e-4
# stage1 shrink1e3 --depths 6,8,12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo --precond-coupled-orders 3,3,3,3,3,3 --shampoo-ridge 1e-3
# stage1 shrink1e2 --depths 6,8,12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo --precond-coupled-orders 2,2,3,3,3 --shampoo-ridge 1e-2
# AN annealed alpha (dir 7, P~10%): revive only if a static-alpha point looks promising.
# stage1 alpha_anneal05_d8  --depths 8  --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,synth --synth-alpha 0.5 --synth-alpha-warmup 750
# stage1 alpha_anneal05_d12 --depths 12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,synth --synth-alpha 0.5 --synth-alpha-warmup 750
# stage1 alpha_anneal10_d8  --depths 8  --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,synth --synth-alpha 1.0 --synth-alpha-warmup 750
# stage1 alpha_anneal10_d12 --depths 12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,synth --synth-alpha 1.0 --synth-alpha-warmup 750
# A-alloc layer_adapt: FALSIFIED 2026-06-26 (d12 +0.0091, worse than Muon). No reason to re-run.
# stage3 layer_adapt --depths 6,8,12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,layer_adaptive
# gateA_adamw anchor: Muon already beats tuned AdamW by +1.36 nats — known, low value.
# stage3 gateA_adamw --depths 8 --fixed-iters 1000 --matrix-lr-grid 0.003,0.01,0.03 --arms muon,adamw
echo "campaign DONE $(date)" >> "$LOG"
