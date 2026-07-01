#!/usr/bin/env bash
# Idea 4 / Bet A — muon_roles per-role LR-multiplier d6 screen (falsification).
# candidate_vectors() from gns.module_lr, ROLES order = attn_qkv,attn_o,mlp_in,mlp_out; uniform(1,1,1,1)==muon.
# Screen: best candidate must beat BOTH muon (anchor) and muon_lookahead (floor) beyond seed scatter.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src PYTHONUNBUFFERED=1
PYBIN="uv run --project /home/jonas/git/tct-models python"
R=/home/jonas/git/gns/results
SEED="${SEED:-0}"; STEPS=400
ts() { date '+%a %d. %b %H:%M:%S'; }
common="--depth 6 --aspect-ratio 64 --device-batch-size 16 --num-iterations $STEPS --matrix-lr-grid 0.02 --eval-every 50 --seed $SEED"

echo "### MUON_ROLES d6 screen  seed${SEED}  $(ts) ###"
$PYBIN scripts/train_compare_precond.py --arms muon           $common --out "$R/roles_muon_s${SEED}.json"
$PYBIN scripts/train_compare_precond.py --arms muon_lookahead $common --out "$R/roles_lookahead_s${SEED}.json"

# label -> CSV (attn_qkv,attn_o,mlp_in,mlp_out), mirroring gns.module_lr.candidate_vectors()
run_role() {
  echo "--- muon_roles $1 = $2  $(ts) ---"
  $PYBIN scripts/train_compare_precond.py --arms muon_roles --role-lr-mults "$2" $common \
    --out "$R/roles_$1_s${SEED}.json"
}
run_role io_split 1.3,0.7,1.3,0.7
run_role out_damp 1.0,0.7,1.0,0.7
run_role mlp_up   1.0,1.0,1.5,1.0
run_role attn_up  1.3,1.0,1.0,1.0
run_role mlp_dom  0.8,0.8,1.5,1.0
echo "### MUON_ROLES screen DONE  $(ts) ###"
