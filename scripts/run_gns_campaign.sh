#!/usr/bin/env bash
# GNS downstream campaign (4h budget, single GB10). Maximizes information about whether
# orthogonalization-schedule choice changes real LM training (steps-to-loss) and converts
# to wall-clock (Round 2, real fp8). Runs base_train per (arm, matrix-lr) on a small
# RESOLVING model. Ordered so the most informative runs land first.
#
# Launch:  bash scripts/run_gns_campaign.sh
set -u
cd "$(dirname "$0")/.."
PY="uv run --project /path/to/tct-models python"
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src
export PYTHONUNBUFFERED=1
OUT=/home/jonas/git/gns/results/campaign
mkdir -p "$OUT"
TSV="$OUT/summary.tsv"
[ -f "$TSV" ] || echo -e "arm\tlr\tfp8\tfinal_val_bpb\twall_s\tstatus" > "$TSV"

# --- small resolving model: ~small enough for many runs, large enough to rank arms ---
MODEL="--depth=8 --max-seq-len=1024 --window-pattern=L --device-batch-size=16 \
 --total-batch-size=131072 --num-iterations=400 --eval-every=100 --eval-tokens=262144 \
 --core-metric-every=-1 --sample-every=-1 --warmup-steps=30 --run=dummy"

run () {  # arm lr fp8
  local arm=$1 lr=$2 fp8=$3
  local tag="${arm}__lr${lr}__${fp8}"
  local log="$OUT/${tag}.log"
  echo "[$(date +%H:%M:%S)] >>> $tag"
  local t0=$(date +%s)
  if [ "$arm" = "fused" ]; then
    $PY -m scripts.base_train $MODEL --muon-schedule=fused --matrix-lr=$lr > "$log" 2>&1
  else
    $PY -m scripts.base_train $MODEL --muon-schedule=$arm --muon-fp8=$fp8 --matrix-lr=$lr > "$log" 2>&1
  fi
  local rc=$? t1=$(date +%s)
  local bpb=$(grep -oiE "bpb:?[[:space:]]+[0-9.]+" "$log" | tail -1 | grep -oE "[0-9.]+$")
  echo -e "${arm}\t${lr}\t${fp8}\t${bpb:-NA}\t$((t1-t0))\t$([ $rc -eq 0 ] && echo ok || echo fail)" >> "$TSV"
  echo "[$(date +%H:%M:%S)] <<< $tag  bpb=${bpb:-NA}  ${rc}  $((t1-t0))s"
}

# Phase 1 — key comparison at default lr (baseline, oracle, control, both extreme frontier pts)
for arm in fused svd none jordan5_bf16_control all_fp8_control frontier_cost_6p625 frontier_cost_8p875; do
  run "$arm" 0.02 real
done
# Phase 2 — coarse lr co-tune for the decisive arms
for lr in 0.01 0.04; do
  for arm in fused svd jordan5_bf16_control frontier_cost_6p625 frontier_cost_8p875; do
    run "$arm" "$lr" real
  done
done
# Phase 3 — remaining frontier points at default lr (fills time if budget remains)
for arm in frontier_cost_7p625 frontier_cost_7p75 frontier_cost_8p625 frontier_cost_8p75; do
  run "$arm" 0.02 real
done
# Phase 4 — Round-1 quality cross-check (simulated fp8) for the fp8-bearing arms
for arm in all_fp8_control frontier_cost_6p625 frontier_cost_8p875; do
  run "$arm" 0.02 sim
done

echo "DONE. Summary:"; column -t "$TSV"
