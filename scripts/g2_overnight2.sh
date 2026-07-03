#!/bin/bash
# Overnight chain #2 (2026-07-03):
#   1. knee0-ORIGINAL EMA-fair d6->d8 gate — does the temporal claim (+0.046 raw over Lookahead)
#      survive when every arm is judged at its best eval protocol? (the G2 searched variant did NOT)
#   2. Fused-track Phase B replication triple at d6/1500, M=4 workers, seed 0:
#      muloco2bit (the MuLoCo headline recipe), dp reference (h=1 fp32), diloco (AdamW-inner).
set -e
cd /home/jonas/git/nanochat
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src
export PYTHONUNBUFFERED=1
RUN="uv run --project /home/jonas/git/tct-models python"
R=/home/jonas/git/gns/results

LOG=$R/overnight2_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "logging to $LOG"

echo "=== PHASE 1: knee0-original EMA-fair trend gate (d6,d8 x seeds 0,1,2 x 1500)"
$RUN -u scripts/eval_program_trend.py --depths 6,8 --seeds 0,1,2 --steps 1500 \
  --lr 0.02 --weight-decay 0.28 \
  --search-json $R/search_program.json --n-knees 1 \
  --out $R/knee0_emafair_gate.json
echo "PHASE 1 DONE -> $R/knee0_emafair_gate.json"

echo "=== PHASE 2: MuLoCo Phase-B replication triple (d6/1500, M4, seed 0)"
$RUN scripts/train_diloco.py --depth 6 --num-iterations 1500 --workers 4 \
  --preset muloco2bit --h 30 --matrix-lr 0.02 --eval-every 100 \
  --out $R/probe_muloco_2bit.json
$RUN scripts/train_diloco.py --depth 6 --num-iterations 1500 --workers 4 \
  --preset dp --matrix-lr 0.02 --eval-every 100 \
  --out $R/probe_dp_m4.json
$RUN scripts/train_diloco.py --depth 6 --num-iterations 1500 --workers 4 \
  --preset diloco --h 30 --matrix-lr 0.02 --eval-every 100 \
  --out $R/probe_diloco_adamw.json
echo "PHASE 2 DONE"
echo "OVERNIGHT CHAIN 2 COMPLETE"
