#!/bin/bash
# INNER-LOOP GATES + fused-track closure queue (auto-chained after diloco_probe.sh).
# Pre-registered decision experiments, cheapest/most-decisive first:
#  Q1 whitened-guard rerun — the ONE follow-up the step-1020 blowup earned. Beats 3.9287 or the
#     H2 outer-geometry limb closes for good.
#  Q2 EMA-eval confound — muon + muon_lookahead with --ema-eval-beta: if tuned-Muon+EMA closes the
#     Lookahead gap, the temporal lever collapses to prior art (decides Rung 2 / G2 search design).
#  Q3 data-repetition sweep — muon vs muon_wwd at 4x/16x repetition: WWD's regularizer mechanism
#     predicts the edge GROWS with repetition (decides the Rung-4 regime + a possible reframing).
#  Q4 WWD Rung-1 close — muon_wwd @ rc100 seeds 1,2 (d6) — turns the provisional iso-FLOP PASS
#     into a multi-seed result.
set -e
cd /home/jonas/git/nanochat
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src
export PYTHONUNBUFFERED=1
RUN="uv run --project /home/jonas/git/tct-models python"
R=/home/jonas/git/gns/results

echo "=== Q1: whitened outer + stability guard (one pre-registered rerun)"
$RUN scripts/train_diloco.py --depth 6 --num-iterations 1500 --workers 4 --preset muloco \
  --outer-transform whitened --outer-whiten-guard --outer-factor-ridge 1e-2 \
  --matrix-lr 0.02 --eval-every 100 --out $R/probe_outer_whitened_guarded.json

echo "=== Q2: EMA-eval confound gate (temporal lever vs free weight averaging)"
$RUN scripts/train_compare_precond.py --depth 6 --num-iterations 1500 \
  --arms muon,muon_lookahead --matrix-lr-grid 0.02 --eval-every 100 \
  --ema-eval-beta 0.999 --out $R/ema_confound_s0.json

echo "=== Q3: data-repetition sweep (WWD regularizer identity)"
for REP in 4 16; do
  $RUN scripts/train_compare_precond.py --depth 6 --num-iterations 1500 \
    --arms muon,muon_wwd --matrix-lr-grid 0.02 --weight-decay 0.14 \
    --shampoo-recompute-every 100 --data-repeat $REP --eval-every 100 \
    --out $R/datarep${REP}_s0.json
done

echo "=== Q4: WWD Rung-1 close (rc100 multi-seed, d6 seeds 1,2)"
for S in 1 2; do
  $RUN scripts/train_compare_precond.py --depth 6 --num-iterations 1500 \
    --arms muon,muon_wwd --matrix-lr-grid 0.02 --weight-decay 0.14 \
    --shampoo-recompute-every 100 --seed $S --eval-every 100 \
    --out $R/wwd_rc100_s${S}.json
done

# --- deferred probe arms (validation value only after H2 closed; reordered behind Q1-Q4
#     on 2026-07-02 owner request: decision-relevant experiments first) ---
COMMON="--depth 6 --num-iterations 1500 --matrix-lr 0.02 --eval-every 100"
echo "=== P4: DP anchor (muon at M*batch)"
$RUN scripts/train_compare_precond.py --depth 6 --num-iterations 1500 --arms muon \
  --device-batch-size 64 --matrix-lr-grid 0.02 --eval-every 100 --out $R/probe_dp_anchor64.json
echo "=== P5: DiLoCo baseline (AdamW-inner)"
$RUN scripts/train_diloco.py $COMMON --workers 4 --preset diloco --matrix-lr 0.003 \
  --out $R/probe_diloco_adamw.json
echo "=== P6: MuLoCo 2-bit + EF"
$RUN scripts/train_diloco.py $COMMON --workers 4 --preset muloco2bit --out $R/probe_muloco_2bit.json
echo "=== P7: whiten-vs-rotate (2-bit hadamard)"
$RUN scripts/train_diloco.py $COMMON --workers 4 --preset muloco2bit --basis hadamard \
  --out $R/probe_hadamard_2bit.json
echo "=== P8: comm curve h=100"
$RUN scripts/train_diloco.py $COMMON --workers 4 --preset muloco --h 100 --out $R/probe_muloco_h100.json

$RUN scripts/analyze_diloco_probe.py || true
echo "INNER-LOOP QUEUE COMPLETE"
