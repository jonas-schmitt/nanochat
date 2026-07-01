#!/usr/bin/env bash
# Idea 4 LR-confound control — the #1 reviewer objection. Sweep the base matrix-LR for BOTH muon and
# io_split; io_split must win at each's OWN optimum, else the +0.064 is just a better effective LR.
# d6/a64/b16/400. Persistent + resume-skippable.
set -u
cd /home/jonas/git/nanochat
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src PYTHONUNBUFFERED=1
PYBIN="uv run --project /home/jonas/git/tct-models python"
R=/home/jonas/git/gns/results
SEED="${SEED:-0}"; STEPS=400; IO="1.3,0.7,1.3,0.7"
ts() { date '+%a %d. %b %H:%M:%S'; }
common="--depth 6 --aspect-ratio 64 --device-batch-size 16 --num-iterations $STEPS --eval-every 50 --seed $SEED"

echo "### IDEA4 LR-CONFOUND SWEEP  io=$IO  seed${SEED}  $(ts) ###"
for LR in 0.010 0.014 0.020 0.028 0.040; do
  echo "--- muon        lr=$LR  $(ts) ---"
  $PYBIN scripts/train_compare_precond.py --arms muon --matrix-lr-grid "$LR" $common \
    --out "$R/roleslr_muon_lr${LR}_s${SEED}.json"
  echo "--- muon_roles  lr=$LR  $(ts) ---"
  $PYBIN scripts/train_compare_precond.py --arms muon_roles --role-lr-mults "$IO" --matrix-lr-grid "$LR" $common \
    --out "$R/roleslr_io_lr${LR}_s${SEED}.json"
done
echo "### LR-CONFOUND SWEEP DONE  $(ts) ###"
