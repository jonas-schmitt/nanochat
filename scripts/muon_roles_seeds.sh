#!/usr/bin/env bash
# Idea 4 multi-seed confirmation — the harness is nondeterministic (~0.01-0.025 same-seed scatter,
# comparable to io_split's effect), so the single-seed +0.064 is unreliable. Paired muon vs io_split
# across seeds to get mean ± std of the gap. d6/a64/b16/400.
set -u
cd /home/jonas/git/nanochat
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src PYTHONUNBUFFERED=1
PYBIN="uv run --project /home/jonas/git/tct-models python"
R=/home/jonas/git/gns/results
IO="1.3,0.7,1.3,0.7"
ts() { date '+%a %d. %b %H:%M:%S'; }
common="--depth 6 --aspect-ratio 64 --device-batch-size 16 --num-iterations 400 --matrix-lr-grid 0.02 --eval-every 50"

echo "### IDEA4 MULTI-SEED  io=$IO  $(ts) ###"
for S in 0 1 2 3 4; do
  echo "--- muon        seed=$S  $(ts) ---"
  $PYBIN scripts/train_compare_precond.py --arms muon $common --seed "$S" \
    --out "$R/rolesseed_muon_s${S}.json"
  echo "--- muon_roles  seed=$S  $(ts) ---"
  $PYBIN scripts/train_compare_precond.py --arms muon_roles --role-lr-mults "$IO" $common --seed "$S" \
    --out "$R/rolesseed_io_s${S}.json"
done
echo "### MULTI-SEED DONE  $(ts) ###"
