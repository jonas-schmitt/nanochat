#!/bin/bash
# DiLoCo-simulator sanity gates G-A1/G-A2/G-A3 (fused track, Phase A3).
# Short d6 runs (300 steps) — enough eval points to catch any wiring divergence.
set -e
cd /home/jonas/git/nanochat
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src
export PYTHONUNBUFFERED=1
# bit-reproducible mode: the gates compare val traces for EQUALITY, and the harness's normal
# run-to-run nondeterminism (~6e-3, measured gate_ga3a vs gate_ga3a_repeat) would swamp them
export GNS_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
RUN="uv run --project /home/jonas/git/tct-models python"
R=/home/jonas/git/gns/results
N=300

# references: the incumbent harness's own arms (identical bytes the gates must reproduce)
$RUN scripts/train_compare_precond.py --depth 6 --num-iterations $N --arms muon \
  --matrix-lr-grid 0.02 --eval-every 50 --out $R/gate_ref_muon.json
$RUN scripts/train_compare_precond.py --depth 6 --num-iterations $N --arms muon_lookahead \
  --lookahead-k 5 --lookahead-alpha 0.5 --matrix-lr-grid 0.02 --eval-every 50 \
  --out $R/gate_ref_lookahead.json

# G-A1: 1-worker, sync every step, outer sgd lr=1 β=0 == plain muon
$RUN scripts/train_diloco.py --depth 6 --num-iterations $N --workers 1 --preset dp \
  --matrix-lr 0.02 --eval-every 50 --out $R/gate_ga1.json

# G-A2: 1-worker, h=5, outer sgd lr=0.5 β=0 == muon_lookahead(k=5, α=0.5)
$RUN scripts/train_diloco.py --depth 6 --num-iterations $N --workers 1 --preset dp \
  --h 5 --outer-lr 0.5 --matrix-lr 0.02 --eval-every 50 --out $R/gate_ga2.json

# G-A3: bits=32 wire with inert genes toggled must be IDENTICAL
$RUN scripts/train_diloco.py --depth 6 --num-iterations $N --workers 2 --preset muloco --h 15 \
  --matrix-lr 0.02 --eval-every 50 --out $R/gate_ga3a.json
$RUN scripts/train_diloco.py --depth 6 --num-iterations $N --workers 2 --preset muloco --h 15 \
  --basis hadamard --error-feedback 1 --stochastic-rounding 1 \
  --matrix-lr 0.02 --eval-every 50 --out $R/gate_ga3b.json

$RUN scripts/check_diloco_gates.py $R/gate_ref_muon.json $R/gate_ref_lookahead.json \
  $R/gate_ga1.json $R/gate_ga2.json $R/gate_ga3a.json $R/gate_ga3b.json
