#!/usr/bin/env bash
# Minimal μP / match-RMS width-trend probe (Rung-4-adjacent, decides if role/rms genes enter the grammar).
# The harness `muon` applies NO shape scale, so `muon_roles 1,1,2,0.5` = the μP match-RMS rule
# (√(fan_out/fan_in): attn 1, mlp_in ×2 [d→4d], mlp_out ×0.5 [4d→d]). The multipliers are WIDTH-INDEPENDENT,
# so the ONLY μP signal is the WIDTH TREND: does the RMS advantage GROW from a64→a128? (At one width it's
# just per-role reallocation, which io_split showed is noise.) 2 widths × 2 arms × lr{0.02,0.04}, seed0, iso decay.
# SIGNATURE: gap(muon−rms) larger at a128 than a64 ⇒ μP alive ⇒ escalate to a192/a256 + multi-seed; flat/neg ⇒ drop.
set -u
cd /home/jonas/git/nanochat
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src PYTHONUNBUFFERED=1
PYBIN="uv run --project /home/jonas/git/tct-models python"
R=/home/jonas/git/gns/results
ts() { date '+%a %d. %b %H:%M:%S'; }
base="--depth 6 --device-batch-size 16 --num-iterations 1500 --eval-every 100 --weight-decay 0.14 --seed 0 --matrix-lr-grid 0.02,0.04"

echo "### μP/RMS width probe — waiting for GPU $(ts) ###"
while pgrep -f "train_compare_precond.py" >/dev/null; do sleep 30; done
for AR in 64 128; do
  echo "--- muon a$AR  $(ts) ---"
  $PYBIN scripts/train_compare_precond.py --arms muon --aspect-ratio "$AR" $base \
    --out "$R/mup_muon_a${AR}_s0.json"
  echo "--- muon_rms (1,1,2,0.5) a$AR  $(ts) ---"
  $PYBIN scripts/train_compare_precond.py --arms muon_roles --role-lr-mults 1,1,2,0.5 --aspect-ratio "$AR" $base \
    --out "$R/mup_rms_a${AR}_s0.json"
done
echo "### μP/RMS width probe DONE $(ts) ###"
