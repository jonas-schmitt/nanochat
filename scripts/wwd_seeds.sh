#!/usr/bin/env bash
# Whitened WD multi-seed confirmation. Seed-0 scout: muon_wwd wd0.14 (4.1683) beat the RETUNED-isotropic
# control muon wd0.14 (4.1806) by +0.012 at d6/1500 (~4x the 1500-step noise). But single-seed screens
# have fooled us (io_split). Paired muon vs muon_wwd at wd0.14 over seeds 1,2 (seed0 already done) => a
# 3-seed paired mean. Waits for the running scout to free the GPU first. Resumable.
set -u
cd /home/jonas/git/nanochat
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src PYTHONUNBUFFERED=1
PYBIN="uv run --project /home/jonas/git/tct-models python"
R=/home/jonas/git/gns/results
ts() { date '+%a %d. %b %H:%M:%S'; }

echo "### WWD-SEEDS waiting for the GPU to free (scout finishing) $(ts) ###"
while pgrep -f "train_compare_precond.py" >/dev/null; do sleep 30; done
echo "### GPU free — WWD multi-seed confirmation (seeds 1,2) $(ts) ###"
c="--depth 6 --aspect-ratio 64 --device-batch-size 16 --num-iterations 1500 --matrix-lr-grid 0.02 --eval-every 100 --weight-decay 0.14"
for S in 1 2; do
  echo "--- muon wd0.14 seed=$S  $(ts) ---"
  $PYBIN scripts/train_compare_precond.py --arms muon $c --seed "$S" --out "$R/wwd_muon_wd0.14_s${S}.json"
  echo "--- muon_wwd s1.0 wd0.14 seed=$S  $(ts) ---"
  $PYBIN scripts/train_compare_precond.py --arms muon_wwd --wwd-strength 1.0 $c --seed "$S" \
    --out "$R/wwd_wwd_wd0.14_s${S}.json"
done
echo "### WWD-SEEDS DONE $(ts) ###"
