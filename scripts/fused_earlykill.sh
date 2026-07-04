#!/bin/bash
# FUSED EARLY-KILL GATE orchestrator (TODO "FUSED EARLY-KILL GATE"; plan snug-doodling-mist.md).
# The full-paper bet, cheaply: record one anchor -> validate replay fidelity -> replay-vs-real RANK
# gate (reusing existing real runs) -> tier-1 replay search -> tier-2 validate the front -> verdict.
# RESUMABLE: every step is skip-if-valid; stop any time (Ctrl-C / kill) and re-run this script.
set -u
cd /home/jonas/git/nanochat
export PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src
export PYTHONUNBUFFERED=1
RUN="uv run --project /home/jonas/git/tct-models python"
R=/home/jonas/git/gns/results
A=$R/fused_anchor_d6M4H30.pt          # the recorded anchor

LOG=$R/fused_earlykill_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "logging to $LOG"

valid_json () { python3 -c "import json,sys; d=json.load(open('$1')); sys.exit(0 if d.get('best_val') is not None else 1)" 2>/dev/null; }
valid_pt ()  { python3 -c "import torch,sys; d=torch.load('$1',map_location='cpu',weights_only=False); sys.exit(0 if d.get('rounds') else 1)" 2>/dev/null; }

diloco () {  # diloco <out.json> <extra args...>   (skip if the result already exists & parses)
  local out=$1; shift
  if valid_json "$out"; then echo "  skip (done) $out"; return 0; fi
  $RUN scripts/train_diloco.py --depth 6 --num-iterations 1500 --workers 4 --h 30 \
    --matrix-lr 0.02 --eval-every 100 --out "$out" "$@" || { echo "RUN-FAIL: $out"; return 1; }
}

echo "=== STEP 1: record fp32 anchor (central outer sgd@olr4)"
if valid_pt "$A"; then echo "  skip (anchor exists) $A"; else
  $RUN scripts/train_diloco.py --depth 6 --num-iterations 1500 --workers 4 --preset muloco --h 30 \
    --outer-lr 4 --matrix-lr 0.02 --eval-every 100 \
    --record-deltas "$A" --record-dtype float32 \
    --out $R/fused_anchor_d6M4H30.json || { echo "ANCHOR FAILED — abort"; exit 1; }
fi

echo "=== STEP 2: full-scale self-consistency gate (HARD prerequisite)"
if $RUN scripts/replay_policy.py --recording "$A" --gate --gate-tol 1e-3 \
     --out $R/fused_selfconsistency.json; then
  echo "  self-consistency PASS"
else
  echo "  self-consistency FAIL — replay is not faithful at full scale; STOP and fix."; exit 1
fi

echo "=== STEP 3a: produce 3 NEW precision-family references (real runs)"
diloco $R/ref_8bit_olr4.json        --preset muloco --outer-lr 4 --delta-bits 8
diloco $R/ref_4bit_hadamard_olr4.json --preset muloco --outer-lr 4 --delta-bits 4 --basis hadamard --error-feedback 1
diloco $R/ref_2bit_noEF_olr4.json   --preset muloco --outer-lr 4 --delta-bits 2 --error-feedback 0

echo "=== STEP 3b: replay-vs-real RANK gate (reuse existing reals + the 3 new + the anchor)"
REFS="$R/fused_anchor_d6M4H30.json,$R/phaseb_muloco_olr2.json,$R/phaseb_muloco_olr6.json"
REFS="$REFS,$R/probe_outer_polar_olr4.json,$R/probe_outer_whitened_olr4.json"
REFS="$REFS,$R/probe_muloco_2bit_olr4.json,$R/ref_8bit_olr4.json"
REFS="$REFS,$R/ref_4bit_hadamard_olr4.json,$R/ref_2bit_noEF_olr4.json"
$RUN scripts/replay_rank_gate.py --recording "$A" --results "$REFS" \
  --out $R/fused_rank_gate.json || { echo "RANK GATE FAILED — abort"; exit 1; }

# geometry search only if the rank gate trusts it
GEOM=$(python3 -c "import json; print('--include-geometry' if json.load(open('$R/fused_rank_gate.json'))['trust'].get('geometry') else '')" 2>/dev/null)
echo "  geometry flag for tier-1 search: '${GEOM:-<precision-only>}'"

echo "=== STEP 4: tier-1 replay search (resumable)"
$RUN scripts/search_policy.py --recording "$A" --pop 20 --gens 6 --resume $GEOM \
  --out $R/fused_search_tier1.json || echo "  (search interrupted — re-run to resume)"

echo "=== STEP 5: tier-2 validation of the top-3 front knees (real runs)"
python3 - "$R/fused_search_tier1.json" > /tmp/fused_front_genomes.txt <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for i, f in enumerate(d.get("front", [])[:3]):
    print(f"{i}\t{json.dumps(f['genome_dict'])}")
PY
while IFS=$'\t' read -r i gj; do
  [ -z "$i" ] && continue
  out=$R/fused_tier2_knee$i.json
  if valid_json "$out"; then echo "  skip (done) $out"; continue; fi
  $RUN scripts/train_diloco.py --depth 6 --num-iterations 1500 --workers 4 --h 30 \
    --matrix-lr 0.02 --eval-every 100 --genome-json "$gj" --out "$out" || echo "RUN-FAIL: $out"
done < /tmp/fused_front_genomes.txt

echo "=== STEP 6: final verdict"
$RUN scripts/judge_fused_earlykill.py \
  --rank-gate $R/fused_rank_gate.json --search $R/fused_search_tier1.json \
  --tier2-glob "$R/fused_tier2_knee*.json" \
  --incumbent-val 3.8210 --incumbent-bits 32
echo "FUSED EARLY-KILL GATE COMPLETE"
