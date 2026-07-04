"""Final verdict for the fused early-kill gate (TODO "FUSED EARLY-KILL GATE").

Combines the replay-rank-gate trust verdict with the tier-2-validated Pareto front to emit
GO-strong / GO-floor / KILL, per the pre-registered criteria in
notes/fused-earlykill-preregistration.md.

  GO-strong: replay trusted on >= the precision axis AND a tier-2 (REAL) front point dominates or
             matches the tuned MuLoCo incumbent on the val x bits frontier (beyond the noise floor).
  GO-floor : replay trusted + a non-trivial frontier that rediscovers sensible points, even if none
             strictly beats the incumbent.
  KILL     : replay untrusted even for precision, or a degenerate frontier.

Inputs: the rank-gate JSON, the tier-1 search JSON (for the front + its replay vals), and the tier-2
REAL result JSONs (train_diloco format) for the front knees that were re-run.
"""
import argparse
import glob
import json
import os


def _load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rank-gate", required=True)
    p.add_argument("--search", required=True)
    p.add_argument("--tier2-glob", required=True, help="glob of tier-2 REAL result JSONs")
    p.add_argument("--incumbent-val", type=float, default=3.8210, help="tuned MuLoCo best val (olr2)")
    p.add_argument("--incumbent-bits", type=float, default=32.0, help="incumbent bits/param/step")
    p.add_argument("--noise", type=float, default=0.003, help="d6/1500 val noise floor")
    p.add_argument("--out", default="/home/jonas/git/gns/results/fused_earlykill_verdict.json")
    a = p.parse_args()

    rg = _load(a.rank_gate) or {}
    trust = rg.get("trust", {})
    replay_trusted = bool(trust.get("precision"))  # >= precision axis

    search = _load(a.search) or {}
    front = search.get("front", [])

    # tier-2: REAL runs of front knees. Match by canonical genome; take (real best_val, bits).
    tier2 = []
    for path in sorted(glob.glob(a.tier2_glob)):
        d = _load(path)
        if not d:
            continue
        g = d.get("config", {}).get("genome_canonical") or d.get("config", {}).get("genome")
        bits = None
        # bits/param/step: prefer the analytic field if the run recorded it, else derive from front
        for f in front:
            if f.get("genome") == g:
                bits = f.get("bits_per_param_step")
        tier2.append({"canonical": g, "real_best": float(d["best_val"]), "bits": bits,
                      "file": os.path.basename(path)})

    # dominance: real val <= incumbent+noise AND bits <= incumbent bits (a Pareto win or match)
    dominators = [t for t in tier2 if t["bits"] is not None
                  and t["real_best"] <= a.incumbent_val + a.noise
                  and t["bits"] <= a.incumbent_bits]
    nontrivial_front = len(front) >= 3

    if not replay_trusted:
        verdict = "KILL"; why = "replay not trusted even for precision genes (cheap search is invalid)"
    elif dominators:
        verdict = "GO-strong"; why = f"{len(dominators)} tier-2 point(s) dominate/match tuned MuLoCo"
    elif nontrivial_front:
        verdict = "GO-floor"; why = "replay trusted + non-trivial frontier (methods-paper floor)"
    else:
        verdict = "KILL"; why = "degenerate frontier"

    print("=== FUSED EARLY-KILL VERDICT ===")
    print(f"  replay trust: precision={trust.get('precision')} geometry={trust.get('geometry')}")
    print(f"  tier-1 front size: {len(front)}   tier-2 validated: {len(tier2)}")
    for t in tier2:
        mark = "  <-- dominates/matches" if t in dominators else ""
        b = f"{t['bits']:.4f}" if t["bits"] is not None else "?"
        print(f"    real {t['real_best']:.4f}  bits/p/s {b}  {t['file']}{mark}")
    print(f"  incumbent: val {a.incumbent_val:.4f}  bits {a.incumbent_bits:g}")
    print(f"\n  => {verdict}  ({why})")
    print("  GO-* : re-plan the full fused paper.   KILL : run WWD λ-gate closure, then tct.")

    tmp = a.out + ".tmp"
    json.dump({"verdict": verdict, "why": why, "replay_trusted": replay_trusted,
               "dominators": dominators, "tier2": tier2, "front_size": len(front)},
              open(tmp, "w"), indent=1)
    os.replace(tmp, a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
