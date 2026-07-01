#!/usr/bin/env bash
# (1) finalize lr for ortho_shampoo+WWD (add 0.06 — lr0.04 looked best/edge-of-grid), then
# (2) PRIORITY: recompute-frequency sweep for BOTH muon_wwd and ortho_shampoo+WWD — does the win
#     survive STALE factors (recompute 50/100 vs 10)? Decay should tolerate it (=> muon_wwd cheap);
#     the open question is whether the shampoo DESCENT does too. Measures val AND wall (the cost decider).
# Baselines (d6/1500 s0 wd0.14, recompute=10): muon 4.1806/680s, muon_wwd 4.1683/842s,
#   ortho_shampoo+WWD 4.1648/834s @lr0.02. Waits for the running scout to free the GPU.
set -u
cd /home/jonas/git/nanochat
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src PYTHONUNBUFFERED=1
PYBIN="uv run --project /home/jonas/git/tct-models python"
R=/home/jonas/git/gns/results
ts() { date '+%a %d. %b %H:%M:%S'; }
c="--depth 6 --aspect-ratio 64 --device-batch-size 16 --num-iterations 1500 --eval-every 100 --weight-decay 0.14 --seed 0"

echo "### CURV+WWD next — waiting for GPU $(ts) ###"
while pgrep -f "train_compare_precond.py" >/dev/null; do sleep 30; done

echo "### (1) lr 0.06 for ortho_shampoo+WWD $(ts) ###"
$PYBIN scripts/train_compare_precond.py --arms ortho_shampoo --wwd --wwd-power 0.5 $c \
  --matrix-lr-grid 0.06 --out "$R/curv_wwd_lr06_s0.json"

# pick best ortho+wwd lr across {0.02,0.04 (scout file), 0.06}
BEST=$($PYBIN - <<'PY'
import json
cand={}
for f in ["/home/jonas/git/gns/results/curv_orthoshampoo_wwd_s0.json",
          "/home/jonas/git/gns/results/curv_wwd_lr06_s0.json"]:
    try:
        d=json.load(open(f))
        for k,v in d.get("_done",{}).items():
            cand[float(k.split("|")[1])]=v["best_val"]
    except Exception: pass
best=min(cand,key=cand.get)
print(f"{best:g}")
PY
)
echo "### best ortho+WWD lr = $BEST ; (2) recompute-freq sweep $(ts) ###"
for RC in 50 100; do
  echo "--- muon_wwd lr0.02 recompute=$RC  $(ts) ---"
  $PYBIN scripts/train_compare_precond.py --arms muon_wwd --wwd-power 0.5 $c \
    --matrix-lr-grid 0.02 --shampoo-recompute-every "$RC" --out "$R/wwd_rc${RC}_s0.json"
  echo "--- ortho_shampoo+WWD lr$BEST recompute=$RC  $(ts) ---"
  $PYBIN scripts/train_compare_precond.py --arms ortho_shampoo --wwd --wwd-power 0.5 $c \
    --matrix-lr-grid "$BEST" --shampoo-recompute-every "$RC" --out "$R/curvwwd_rc${RC}_s0.json"
done
echo "### CURV+WWD next DONE $(ts) ###"
