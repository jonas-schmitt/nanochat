#!/usr/bin/env bash
# Whitened-WD decay-POWER sweep (informs the grammar gene menu + finds the best power). We have
# seed0 @ wd0.14: power 0.0(iso)=4.1806, power 0.5=4.1683. This adds power 0.25 (gentle) and 0.75
# (strong) at seed0 to map the axis. Waits for the d8 gate to free the GPU first. Resumable.
set -u
cd /home/jonas/git/nanochat
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src PYTHONUNBUFFERED=1
PYBIN="uv run --project /home/jonas/git/tct-models python"
R=/home/jonas/git/gns/results
ts() { date '+%a %d. %b %H:%M:%S'; }
c="--depth 6 --aspect-ratio 64 --device-batch-size 16 --num-iterations 1500 --matrix-lr-grid 0.02 --eval-every 100 --weight-decay 0.14 --seed 0"

echo "### WWD power sweep — waiting for GPU (d8 gate running) $(ts) ###"
while pgrep -f "train_compare_precond.py" >/dev/null; do sleep 30; done
echo "### WWD power sweep (seed 0) $(ts) ###"
for P in 0.25 0.75; do
  echo "--- muon_wwd power=$P wd0.14 seed0  $(ts) ---"
  $PYBIN scripts/train_compare_precond.py --arms muon_wwd --wwd-strength 1.0 --wwd-power "$P" $c \
    --out "$R/wwd_pow${P}_s0.json"
done
echo "### WWD power sweep DONE $(ts) ###"
