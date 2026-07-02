#!/bin/bash
# FALSIFICATION PROBE for the fused track (d6/1500, M=4, single-seed first-look).
# Ordered so the headline question is answered first if interrupted.
#
# Hypotheses + kill criteria (d6/1500 noise floor ~0.003; treat |gap|<0.005 as a tie):
#  H2 (the NOVEL claim, runs 1-3): outer geometry (polar/whitened) beats MuLoCo's sgd outer.
#      KILL: both outer_polar and outer_whitened <= muloco_fp32 + noise -> geometry lever dead at d6.
#  H1 (simulator validity, runs 1,4,5): muloco_fp32 > diloco_adamw (MuLoCo's claim), and both
#      within striking distance of the DP anchor. KILL: order inverted -> simulator/scale artifact,
#      fix before any further spending.
#  H3 (wire, runs 6,7): muloco_2bit ~ muloco_fp32 (paper replication; KILL if gap > 0.02);
#      hadamard_2bit vs muloco_2bit = the whiten-vs-rotate first data point, judged against the
#      logged delta_stats via the rounding model's prediction (C0).
#  Bonus (run 8): h=100 point for the comm-quality tradeoff curve.
set -e
cd /home/jonas/git/nanochat
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src
export PYTHONUNBUFFERED=1
RUN="uv run --project /home/jonas/git/tct-models python"
R=/home/jonas/git/gns/results
COMMON="--depth 6 --num-iterations 1500 --matrix-lr 0.02 --eval-every 100"

# 1. THE INCUMBENT: MuLoCo (Muon-inner, Nesterov outer lr8-EMA, fp32 sync, M=4 h=30)
$RUN scripts/train_diloco.py $COMMON --workers 4 --preset muloco --out $R/probe_muloco_fp32.json
# 2. NOVEL: outer-Muon (polar of the mean pseudo-gradient)
$RUN scripts/train_diloco.py $COMMON --workers 4 --preset muloco --outer-transform polar \
  --out $R/probe_outer_polar.json
# 3. NOVEL: whitened outer step (L,R factors over rounds, p=0.5)
$RUN scripts/train_diloco.py $COMMON --workers 4 --preset muloco --outer-transform whitened \
  --out $R/probe_outer_whitened.json
# 4. DP anchor: plain muon at M*batch (iso-token upper reference), via the incumbent harness
$RUN scripts/train_compare_precond.py --depth 6 --num-iterations 1500 --arms muon \
  --device-batch-size 64 --matrix-lr-grid 0.02 --eval-every 100 --out $R/probe_dp_anchor64.json
# 5. DiLoCo baseline (AdamW-inner, its own tuned Adam-range inner lr per the paper regime)
$RUN scripts/train_diloco.py $COMMON --workers 4 --preset diloco --matrix-lr 0.003 \
  --out $R/probe_diloco_adamw.json
# 6. MuLoCo 2-bit + error feedback (the paper's headline point)
$RUN scripts/train_diloco.py $COMMON --workers 4 --preset muloco2bit --out $R/probe_muloco_2bit.json
# 7. Whiten-vs-rotate, first data point: 2-bit + EF in the Hadamard basis
$RUN scripts/train_diloco.py $COMMON --workers 4 --preset muloco2bit --basis hadamard \
  --out $R/probe_hadamard_2bit.json
# 8. Comm-quality curve: the incumbent at h=100
$RUN scripts/train_diloco.py $COMMON --workers 4 --preset muloco --h 100 --out $R/probe_muloco_h100.json

echo "PROBE SUITE COMPLETE"
