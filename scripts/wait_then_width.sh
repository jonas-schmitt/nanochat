#!/usr/bin/env bash
# Wait for the alive-screens orchestration to finish (avoid single-GPU contention), then run the
# Idea 4 width gate. Persistent + resumable.
set -u
cd /home/jonas/git/nanochat
L=/home/jonas/git/gns/results/alive_screens.log
until grep -q "SCREENS DONE" "$L" 2>/dev/null; do sleep 30; done
echo "[wait_then_width] screens done -> launching width gate  $(date)"
bash scripts/muon_roles_width.sh
