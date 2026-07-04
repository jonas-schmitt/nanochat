"""Final verdict for the fused early-kill gate (TODO "FUSED EARLY-KILL GATE").

Combines the replay-rank-gate trust verdict with the tier-2-validated Pareto front to emit
GO-strong / GO-floor / KILL, per notes/fused-earlykill-preregistration.md.

  GO-strong: replay trusted on >= the precision axis AND a tier-2 (REAL) front point dominates or
             matches the tuned MuLoCo incumbent FRONTIER on the val x comm-bits plane.
  GO-floor : replay trusted + a non-trivial frontier (>=3 pts) even if none beats the incumbents.
  KILL     : replay untrusted even for precision, or a degenerate frontier.

Objective is comm_bits_per_param_step (analytic; fp32/H30 = 1.0667, 2-bit/H30 = 0.0667) — NOT the raw
delta_bits. Incumbents are the tuned hand-picked points; both are passed as result JSONs and their
(best_val, bits) are computed from their own genomes, so the bar can never be mis-specified by hand.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/jonas/git/gns/src")
from gns.policy_grammar import from_dict  # noqa: E402


def _load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def _val_bits(path):
    """(best_val, comm_bits_per_param_step) from a train_diloco result JSON, or None."""
    d = _load(path)
    if not d or "best_val" not in d:
        return None
    try:
        bits = from_dict(d["config"]["genome"]).comm_bits_per_param_step()
    except Exception:
        return None
    return float(d["best_val"]), float(bits)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rank-gate", required=True)
    p.add_argument("--search", required=True)
    p.add_argument("--tier2-glob", required=True)
    p.add_argument("--incumbents", required=True,
                   help="comma-separated tuned-incumbent result JSONs (val,bits read from each)")
    p.add_argument("--noise", type=float, default=0.003, help="d6/1500 val noise floor (val-tie band)")
    p.add_argument("--out", default="/home/jonas/git/gns/results/fused_earlykill_verdict.json")
    a = p.parse_args()

    rg = _load(a.rank_gate) or {}
    trust = rg.get("trust", {})
    replay_trusted = bool(trust.get("precision"))  # >= precision axis

    search = _load(a.search) or {}
    front = search.get("front", [])

    incumbents = []
    for path in a.incumbents.split(","):
        vb = _val_bits(path.strip())
        if vb:
            incumbents.append({"file": os.path.basename(path.strip()), "val": vb[0], "bits": vb[1]})
    if not incumbents:
        print("ERROR: no readable incumbents — cannot judge dominance", flush=True)
        sys.exit(2)

    # tier-2: REAL runs of front knees; bits computed from each run's OWN genome (robust, no matching)
    tier2 = []
    for path in sorted(glob.glob(a.tier2_glob)):
        vb = _val_bits(path)
        if vb:
            tier2.append({"file": os.path.basename(path), "val": vb[0], "bits": vb[1]})

    # P (searched) "dominates or matches" incumbent I iff P.val <= I.val + noise AND P.bits <= I.bits,
    # with a strict improvement on >=1 axis (val by more than noise, or fewer bits). Checked against
    # EACH incumbent; a single win over either the fp32 or the 2-bit point advances the hand frontier.
    def wins(P, I):
        no_worse = P["val"] <= I["val"] + a.noise and P["bits"] <= I["bits"] + 1e-9
        strict = P["val"] < I["val"] - a.noise or P["bits"] < I["bits"] - 1e-9
        return no_worse and strict

    # A pure tie (same val AND same bits) is REDISCOVERY of an incumbent, not a win -> GO-floor, not
    # GO-strong. Only genuine Pareto improvement (`wins`) counts as strong.
    def ties(P, I):
        return (abs(P["val"] - I["val"]) <= a.noise and abs(P["bits"] - I["bits"]) <= 1e-9)

    winners = []
    for P in tier2:
        hit = [I for I in incumbents if wins(P, I)]
        tie = [I for I in incumbents if ties(P, I) and I["file"] not in [x["file"] for x in hit]]
        P["dominates"] = [I["file"] for I in hit]
        P["rediscovers"] = [I["file"] for I in tie]
        if hit:
            winners.append(P)

    nontrivial_front = len(front) >= 3
    strong = any(P["dominates"] for P in tier2)   # only genuine Pareto improvement

    if not replay_trusted:
        verdict, why = "KILL", "replay not trusted even for precision genes (cheap search is invalid)"
    elif strong:
        verdict, why = "GO-strong", "a tier-2 point dominates/matches the tuned incumbent frontier"
    elif nontrivial_front:
        verdict, why = "GO-floor", "replay trusted + non-trivial frontier (methods-paper floor)"
    else:
        verdict, why = "KILL", "degenerate frontier"

    print("=== FUSED EARLY-KILL VERDICT ===")
    print(f"  replay trust: precision={trust.get('precision')} geometry={trust.get('geometry')}")
    print("  incumbent frontier:")
    for I in incumbents:
        print(f"    val {I['val']:.4f}  bits/p/s {I['bits']:.4f}  {I['file']}")
    print(f"  tier-1 front size: {len(front)}   tier-2 validated: {len(tier2)}")
    for P in tier2:
        tag = ""
        if P["dominates"]:
            tag = "  <-- DOMINATES " + ",".join(P["dominates"])
        elif P["rediscovers"]:
            tag = "  (rediscovers " + ",".join(P["rediscovers"]) + " — floor, not a win)"
        print(f"    val {P['val']:.4f}  bits/p/s {P['bits']:.4f}  {P['file']}{tag}")
    print(f"\n  => {verdict}  ({why})")
    print("  GO-* : re-plan the full fused paper.   KILL : run WWD λ-gate closure, then tct.")

    tmp = a.out + ".tmp"
    json.dump({"verdict": verdict, "why": why, "replay_trusted": replay_trusted,
               "incumbents": incumbents, "tier2": tier2, "winners": winners,
               "front_size": len(front)}, open(tmp, "w"), indent=1)
    os.replace(tmp, a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
