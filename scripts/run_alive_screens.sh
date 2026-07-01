#!/usr/bin/env bash
# One-shot orchestration of the alive-idea d6 screens. Persistent (survives session teardown),
# resumable (each train_compare_precond run resume-skips its own completed out file).
set -u
cd /home/jonas/git/nanochat
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src PYTHONUNBUFFERED=1
PYBIN="uv run --project /home/jonas/git/tct-models python"
R=/home/jonas/git/gns/results

echo "########## ALIVE-IDEA SCREENS  $(date) ##########"
echo "===== Idea 4: muon_roles (muon+lookahead anchors resume-skip) ====="
bash scripts/muon_roles_probe.sh
echo "===== Idea 3 E1: soft_muon_snr strength-sweep ====="
bash scripts/soft_muon_snr_probe.sh strength_sweep
echo "===== Idea 3 E2: soft_muon_mp base ====="
$PYBIN scripts/train_compare_precond.py --arms soft_muon_mp --soft-q 2.0 \
  --depth 6 --aspect-ratio 64 --device-batch-size 16 --num-iterations 400 \
  --matrix-lr-grid 0.02 --eval-every 50 --seed 0 --out "$R/soft_muon_mp_base_s0.json"
echo "########## SCREENS DONE  $(date) ##########"
