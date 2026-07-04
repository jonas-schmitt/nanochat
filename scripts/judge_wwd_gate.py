"""Judge the WWD λ-gate (fallback closure, TODO "FUSED EARLY-KILL GATE" > fallback).

Tests whether the "WWD edge grows with depth" headline survives PER-DEPTH λ tuning (the 1/D-scaling
confound: optimal independent λ ~ 1/D, we only ever tested λ≥0.14). For each (arm, depth) it picks
λ* = argmin on-EMA best_val over the grid, then compares muon@λ*_muon vs wwd@λ*_wwd on 3 seeds.

Consumes the dense-pair result JSONs written by wwd_lambda_gate.sh (train_compare_precond format:
`_done`/`arms` with per-arm best_val + curve.val_ema). Reuses the same `_done`/val_ema parsing used
throughout the campaign. Emits REAL / ARTIFACT and a verdict JSON. No GPU.

  python scripts/judge_wwd_gate.py --results-dir /home/jonas/git/gns/results \
     --depths 8,12 --lambdas 0.05,0.07,0.105,0.14 --seeds 0,1,2
"""
import argparse
import glob
import json
import os

import numpy as np


def _arm_emabest(path, arm):
    """on-EMA best (min over val_ema curve) for `arm` in a train_compare result, or None."""
    if not os.path.exists(path):
        return None
    try:
        d = json.load(open(path))
    except Exception:
        return None
    arms = d.get("arms") or {k.split("|")[0]: v for k, v in d.get("_done", {}).items()}
    v = arms.get(arm)
    if not v:
        return None
    ema = v.get("curve", {}).get("val_ema")
    return float(min(ema)) if ema else float(v["best_val"])


def _fname(d, depth, wd, seed):
    # matches wwd_lambda_gate.sh naming: wwdgate_d{depth}_wd{wd}_s{seed}.json
    return os.path.join(d, f"wwdgate_d{depth}_wd{wd:g}_s{seed}.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="/home/jonas/git/gns/results")
    p.add_argument("--depths", default="8,12")
    p.add_argument("--lambdas", default="0.05,0.07,0.105,0.14")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--out", default="/home/jonas/git/gns/results/wwd_lambda_gate_verdict.json")
    a = p.parse_args()
    depths = [int(x) for x in a.depths.split(",")]
    lambdas = [float(x) for x in a.lambdas.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]

    report, edges = {}, {}
    for depth in depths:
        # λ-search on seed 0 (min on-EMA) per arm; flag grid-edge minima
        lam_star = {}
        for arm in ("muon", "muon_wwd"):
            vals = {wd: _arm_emabest(_fname(a.results_dir, depth, wd, 0), arm) for wd in lambdas}
            avail = {k: v for k, v in vals.items() if v is not None}
            if not avail:
                lam_star[arm] = None; continue
            best = min(avail, key=avail.get)
            lam_star[arm] = best
            edge_flag = best in (min(lambdas), max(lambdas))
            report.setdefault(f"d{depth}", {})[f"{arm}_lambda_star"] = best
            report[f"d{depth}"][f"{arm}_lambda_curve"] = {f"{k:g}": v for k, v in avail.items()}
            if edge_flag:
                report[f"d{depth}"].setdefault("warnings", []).append(
                    f"{arm} λ* at grid edge ({best:g}) — extend the grid")
        lm, lw = lam_star.get("muon"), lam_star.get("muon_wwd")
        if lm is None or lw is None:
            report.setdefault(f"d{depth}", {})["status"] = "incomplete"; continue
        # best-vs-best, 3 seeds
        per_seed = []
        for s in seeds:
            mv = _arm_emabest(_fname(a.results_dir, depth, lm, s), "muon")
            wv = _arm_emabest(_fname(a.results_dir, depth, lw, s), "muon_wwd")
            if mv is not None and wv is not None:
                per_seed.append(mv - wv)
        if len(per_seed) < 2:
            report[f"d{depth}"]["status"] = "incomplete (need >=2 seeds)"; continue
        arr = np.array(per_seed)
        mean, sem = float(arr.mean()), float(arr.std() / max(1, len(arr) - 1) ** 0.5)
        edges[depth] = mean
        report[f"d{depth}"].update({
            "edge_mean": mean, "edge_sem": sem, "edge_per_seed": per_seed,
            "sign_consistent": bool(all(x > 0 for x in per_seed)),
            "gt_2sem": bool(mean > 2 * sem)})

    # REAL iff at every depth: mean>2·SEM, sign 3/3, and the deepest edge does not shrink >0.005
    depth_ok = all(report.get(f"d{d}", {}).get("gt_2sem") and report[f"d{d}"].get("sign_consistent")
                   for d in depths if f"d{d}" in report)
    trend_ok = True
    if len(edges) >= 2:
        ds = sorted(edges)
        trend_ok = edges[ds[-1]] >= edges[ds[0]] - 0.005
    verdict = "REAL" if (depth_ok and trend_ok and len(edges) == len(depths)) else "ARTIFACT"

    print("=== WWD λ-GATE VERDICT (best-vs-best, per-depth λ-tuned, on-EMA) ===")
    for d in depths:
        r = report.get(f"d{d}", {})
        if "edge_mean" in r:
            print(f"  d{d}: λ*_muon={r['muon_lambda_star']:g} λ*_wwd={r['muon_wwd_lambda_star']:g}  "
                  f"edge {r['edge_mean']:+.4f} ± {r['edge_sem']:.4f}  "
                  f"sign3/3={r['sign_consistent']} >2SEM={r['gt_2sem']}")
        else:
            print(f"  d{d}: {r.get('status', 'missing')}")
        for w in r.get("warnings", []):
            print(f"      ! {w}")
    print(f"\n  depth-gates {'PASS' if depth_ok else 'FAIL'}, trend {'PASS' if trend_ok else 'FAIL'} "
          f"-> {verdict}")

    tmp = a.out + ".tmp"
    json.dump({"verdict": verdict, "depth_ok": depth_ok, "trend_ok": trend_ok,
               "edges": edges, "report": report}, open(tmp, "w"), indent=1)
    os.replace(tmp, a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
