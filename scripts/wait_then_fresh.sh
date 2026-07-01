#!/usr/bin/env bash
# Wait for the knee0 recheck to finish (its out json appears), then run the fresh-bet falsification.
set -u
cd /home/jonas/git/nanochat
OUT=/home/jonas/git/gns/results/knee0_recheck_d6_seeds012.json
until [ -f "$OUT" ]; do sleep 30; done
echo "[wait_then_fresh] knee0 recheck done -> launching fresh-bet screen  $(date)"
bash scripts/fresh_bets_probe.sh
