#!/usr/bin/env bash
# Idea 4 width/scale gate — the decisive μP test for io_split (the d6-screen winner).
# Question: does io_split's +0.064 margin over muon GROW with width (a64→a96→a128)? A μP/coordination lever
# should widen; a mis-tuned-LR artifact should not. Persistent + resume-skippable.
set -u
cd /home/jonas/git/nanochat
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src PYTHONUNBUFFERED=1
PYBIN="uv run --project /home/jonas/git/tct-models python"
R=/home/jonas/git/gns/results
SEED="${SEED:-0}"; STEPS=400; IO="1.3,0.7,1.3,0.7"
ts() { date '+%a %d. %b %H:%M:%S'; }
common="--depth 6 --device-batch-size 16 --num-iterations $STEPS --matrix-lr-grid 0.02 --eval-every 50 --seed $SEED"

echo "### IDEA4 WIDTH GATE  io_split=$IO  seed${SEED}  $(ts) ###"
for A in 64 96 128; do
  echo "--- muon        aspect=$A  $(ts) ---"
  $PYBIN scripts/train_compare_precond.py --arms muon $common --aspect-ratio "$A" \
    --out "$R/rolesw_muon_a${A}_s${SEED}.json"
  echo "--- muon_roles  aspect=$A  $(ts) ---"
  $PYBIN scripts/train_compare_precond.py --arms muon_roles --role-lr-mults "$IO" $common --aspect-ratio "$A" \
    --out "$R/rolesw_io_a${A}_s${SEED}.json"
done
echo "### WIDTH GATE DONE  $(ts) ###"
