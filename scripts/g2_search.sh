#!/bin/bash
# G2 Stage-1 launcher: the grammar search over DEFAULT_GENES (temporal × wwd × recompute × ema ×
# wd_scale × lr_scale), seeded with STANDARD solvers only (knee0/wwd/ema are must-beat references,
# not seeds), THEN the WWD×EMA confound run (~25 min). The confound is only needed for the
# knee-judging reference table, not for the search — the search measures the wwd×ema interaction
# over the whole plane itself, so nothing about the gene set depends on the confound outcome.
#
# Usage:            bash scripts/g2_search.sh
# Search only:      SKIP_CONFOUND=1 bash scripts/g2_search.sh
# Bigger search:    POP=20 GENS=8 bash scripts/g2_search.sh     (~15.5 h instead of ~11 h)
set -e
cd /home/jonas/git/nanochat
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src
export PYTHONUNBUFFERED=1
RUN="uv run --project /home/jonas/git/tct-models python"
R=/home/jonas/git/gns/results

echo "=== G2 Stage-1 search (pop ${POP:-16}, gens ${GENS:-6}, steps ${STEPS:-700})"
$RUN scripts/search_program.py --depth 6 --steps "${STEPS:-700}" --pop "${POP:-16}" \
  --gens "${GENS:-6}" --lr 0.02 --weight-decay 0.28 --seed "${SEED:-0}" \
  --out $R/g2_stage1.json
echo "G2 STAGE-1 DONE -> $R/g2_stage1.json (knees feed eval_program_trend.py / 1500-step re-eval)"

if [ "${SKIP_CONFOUND:-0}" != "1" ]; then
  echo "=== reference-bar run: WWD x EMA confound (needed before knee judging, not before the search)"
  $RUN scripts/train_compare_precond.py --depth 6 --num-iterations 1500 \
    --arms muon,muon_wwd --matrix-lr-grid 0.02 --weight-decay 0.14 \
    --shampoo-recompute-every 100 --ema-eval-beta 0.999 --eval-every 100 \
    --out $R/wwd_ema_confound_s0.json
fi
