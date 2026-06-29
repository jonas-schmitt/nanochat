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
# REGIME RE-RUN + TIERED REORDER (2026-06-29). Two things happened today:
# (a) PRODUCTION-REGIME FIX: nanochat bf03701/6130fd0 closed C3(c) (non-matrix params now use production
#     per-group AdamW; embedding LR ~0.2 vs old 3e-3) and pinned TF32 OFF for every arm (coupled
#     inverse-root now GENUINE fp32, not ~10-bit TF32). These change the numerics for EVERY arm, so the
#     Jun-28 results/precond_* are STALE. The harness resume signature does NOT include regime/TF32, so a
#     resume would silently SKIP them and mix regimes — those checkpoints were therefore ARCHIVED to
#     gns results/stale-pre-regime-fix/. The Tier-0 block below now RE-COMPUTES fresh under production.
# (b) PRUNE/REORDER: value-first. Only the cheap high-potential Tier-1 triage runs after the baseline;
#     heavy/low-value stages are GATED or commented (reversible).
# Old-regime verdict (to be re-established under production): d6/d8 ortho_shampoo win significant,
# d12 erosion borderline; camp_gram gave the biggest d8 gap at lowest cost. The coupled cost-win search
# (102/124/90/72/56) is UNAFFECTED — coupled cost was always exact, no search re-run needed.
# Open questions after the baseline confirms direction:
#   (1) is the win WIDTH-driven not depth-driven?  -> Tier 1 width stages (MoE-relevant: frontier is wide)
#   (2) can we ship "cheaper Muon, same quality"?  -> Tier 1 cost arms (the fallback deliverable)
# ========================================================================================================

# ===== TIER 0. CORE 3-SEED BASELINE (d6/d8) — RE-RUN under production regime (~5h; was archived) ======
# d12 baseline DEFERRED to Tier 0b (runs AFTER the Tier-1 width/cost triage). Rationale: d12 is ~half the
# baseline cost (~5.8h, likely more under genuine-fp32) and the decisive width-vs-depth signal lives at
# d6/d8. d12 is KEPT, just run later — see Tier 0b. (2026-06-29 defer decision.)
stage3 gateA_curv  --depths 8   --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo,synth --synth-alpha 0.5
stage3 camp_curv   --depths 6,8 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo,synth --synth-alpha 0.5
stage3 camp_gram   --depths 6,8 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo --precond-coupled-orders 2,2,2,3,3,3,3,3,3
# batch16: PRUNED (redundant). device-batch 16 == the default, so batch16-d8 just re-replicates camp_curv
# d8 and batch16-d12 re-replicates camp_curv d12 — no new information. camp_curv already re-establishes
# d6/d8/d12 under the production regime. Revive only if you want a second independent d8 replication.
# stage3 batch16   --depths 8,12 --device-batch-size 16 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo,synth --synth-alpha 0.5

# ===== TIER 1. LARGE-BATCH SOTA TEST — the one live shot (2026-06-29 pivot) ==========================
# WHY: the d8 accuracy win does NOT survive iso-FLOP at batch-16 (Muon wins +0.18..0.30; see the iso-FLOP
# verdict in gns TODO.md). BUT the overhead is the optimizer's per-step matmuls (batch-INDEPENDENT) while
# fwd/bwd scales with batch -> total-step FLOP overhead amortizes: b16 2.05x -> b64 1.28x -> b256 1.07x.
# At large batch iso-FLOP ~ iso-step, AND low gradient noise favors second-order. So large batch is the one
# practically-relevant regime where curvature could beat Muon at EQUAL compute.
# WIN = ortho's iso-step edge HOLDS/GROWS at large batch AND the analyze ISO-FLOP readout goes <0.
# LR ~sqrt(batch/16); batch64 carries an LR grid (LR is the main confound). ortho = default schedule
# (matches the camp_curv d8 baseline -0.0249 exactly). batch64 is the GATE; 128/256 unlock on a held edge.
stage1 batch64_d8  --depths 8 --device-batch-size 64  --fixed-iters 1500 --matrix-lr-grid 0.03,0.04,0.06 --arms muon,ortho_shampoo
# UNLOCK 128/256 only if batch64 shows the iso-step edge HOLDS (then read the ISO-FLOP section there):
# stage1 batch128_d8 --depths 8 --device-batch-size 128 --fixed-iters 1500 --matrix-lr-grid 0.057,0.08 --arms muon,ortho_shampoo
# stage1 batch256_d8 --depths 8 --device-batch-size 256 --fixed-iters 1500 --matrix-lr-grid 0.08,0.11  --arms muon,ortho_shampoo

# ===== TIER 1b. CHEAPER-MUON FLOOR (cheap; the confirmed deliverable, runs alongside) ===============
# C1 4-step polar (dir 2): joint-opt coeffs (results/jointopt_grammar_probe.json) vs 5-step muon.
stage1 cost_4step_d8  --depths 8  --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,muon_4step,muon_3step
stage1 cost_4step_d12 --depths 12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,muon_4step,muon_3step
# C2 fp8 polar end-to-end (dir 1): exp27 showed 1.76-2.58x kernel speedup at width>=8192.
stage1 fp8_d8  --depths 8  --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,muon_fp8
stage1 fp8_d12 --depths 12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,muon_fp8

# ===== TIER 1b. ISO-FLOP DE-RISK — the decisive accuracy-moonshot gate (2026-06-29) ==================
# Existing-curve iso-FLOP (analyze_adaptations.py "ISO-FLOP" section) is strongly NEGATIVE at d8: at
# equal compute Muon wins by +0.18..+0.30 (recompute-10) and +0.05..+0.07 (optimistic recompute-50).
# The −0.025 win is an iso-STEP artifact. ONLY escape: long horizon (flat curve near convergence shrinks
# the step-deficit penalty). These stages test that escape; if they fail, the accuracy angle is dead and
# the deliverable is the cheaper-Muon floor (cost_4step / fp8 above).
# (1) Long-horizon iso-FLOP: run muon + searched-schedule ortho long at d8; analyze reports iso-FLOP gap
#     vs horizon. WIN = the iso-FLOP gap CLOSES toward <0 as steps grow.
# DEPRIORITIZED 2026-06-29: large batch (Tier 1) is the better escape; revive this as the secondary one.
# stage1 isoflop_long_d8 --depths 8 --fixed-iters 6000 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo --precond-coupled-orders 2,2,2,3,3,3,3,3,3
# (2) Recompute×quality: does the win survive the CHEAP amortisation (recompute 50/100) that the iso-FLOP
#     math needs? NEEDS WIRING FIRST: scaling_ladder.py does NOT yet forward --shampoo-recompute-every to
#     the harness (train_compare_precond.py accepts it). Add the passthrough, then run both arms at the
#     same interval so the gap is fair. Lower priority than (1) — even the optimistic recompute-50 iso-FLOP
#     projection still LOSES (+0.05..0.07), so this only matters in combination with a long-horizon win.
# stage1 recompute50_d8 --depths 8 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo --precond-coupled-orders 2,2,2,3,3,3,3,3,3 --shampoo-recompute-every 50

# ===== TIER 0b. DEFERRED d12 BASELINE (3-seed, ~5.8h) — the expensive coupled inverse-root at depth ===
# KEPT, not dropped. Runs AFTER the cheap triage so the width-vs-depth signal lands first; Ctrl-C here if
# the Tier-1 width result already settles whether a fresh d12 baseline is worth it. Re-confirms the d12
# erosion under the production regime and supplies the SAME-WIDTH depth-isolation contrast for the width
# question (width_d8_a96 d8/w768  vs  camp_curv_d12 d12/w768). Uses SEPARATE *_d12 tags (not --depths 12
# appended to camp_curv) so a re-run never hard-aborts load_or_init on the depths checkpoint key.
# Note: analyze_adaptations reads these under the camp_curv_d12 / camp_gram_d12 tags (not stitched into
# the camp_curv depth-trend) — compare the d12 gaps to camp_curv d6/d8 by hand.
# DEPRIORITIZED 2026-06-29: the d12 erosion matters less now the accuracy angle is iso-FLOP-dead at small
# batch. Revive if you still want the production-regime d12 baseline / same-width depth-isolation contrast.
# stage3 camp_curv_d12 --depths 12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo,synth --synth-alpha 0.5
# stage3 camp_gram_d12 --depths 12 --fixed-iters 1500 --matrix-lr-grid 0.02 --arms muon,ortho_shampoo --precond-coupled-orders 2,2,2,3,3,3,3,3,3

# ===== TIER 2. EXPENSIVE — GATED on a Tier-1 signal =================================================
# Prune-directive (2026-06-29): only the cheap Tier-1 triage runs unattended. Expensive (d12 / large-batch
# / 3-seed) compute is spent ONLY on a bet Tier 1 actually confirms — never on low-potential mechanism
# tests run blind. Uncomment a block only when its gate below fires.

# --- BATCH axis: now PROMOTED to Tier 1 above (the large-batch SOTA test). The old single-LR gate here is
#     SUPERSEDED — leaving it active would duplicate the batch64_d8 tag with a different config and abort
#     load_or_init. Commented out. (Old d8,12 multi-batch variants kept below for reference only.)
# stage1 batch64_d8 --depths 8 --device-batch-size 64 --fixed-iters 1500 --matrix-lr-grid 0.04 --arms muon,ortho_shampoo,synth --synth-alpha 0.5
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
