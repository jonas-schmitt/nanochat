#!/usr/bin/env bash
# TODO NLA N3 — windowed Anderson (iterate-space) GPU falsification at d6/1500.
# CPU survivor: AA(5)>AA(1)>Muon. GPU baselines (freshbet_*): muon s0 4.1921 / s1 4.1995;
# muon_anderson (per-param momentum secant, window1) s0 4.1979 / s1 4.2058 (LOSES to muon).
# This tests the GLOBAL ITERATE formulation (the one CPU-validated). Stage 1 = seed-0 scout
# (window 5 @ lr {0.01,0.02} + window 1 formulation control); Stage 2 (seeds 0,1) gated on it.
set -u
cd /home/jonas/git/nanochat
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src PYTHONUNBUFFERED=1
PYBIN="uv run --project /home/jonas/git/tct-models python"
R=/home/jonas/git/gns/results
ts() { date '+%a %d. %b %H:%M:%S'; }
c="--depth 6 --aspect-ratio 64 --device-batch-size 16 --num-iterations 1500 --eval-every 100"

echo "### ANDWIN scout (seed 0)  $(ts) ###"
$PYBIN scripts/train_compare_precond.py --arms muon_anderson_win --anderson-window 5 $c \
  --matrix-lr-grid 0.02 --seed 0 --out "$R/andwin_w5_lr02_s0.json"
$PYBIN scripts/train_compare_precond.py --arms muon_anderson_win --anderson-window 5 $c \
  --matrix-lr-grid 0.01 --seed 0 --out "$R/andwin_w5_lr01_s0.json"
$PYBIN scripts/train_compare_precond.py --arms muon_anderson_win --anderson-window 1 $c \
  --matrix-lr-grid 0.02 --seed 0 --out "$R/andwin_w1_lr02_s0.json"
echo "### ANDWIN scout DONE  $(ts) ###"
