"""Stage-2 trend gate for the program-grammar search.

Takes the Stage-1 Pareto knees (from search_program.json) plus the reference points
(Muon, cheaper-Muon=polar4, vanilla Lookahead) and asks the question that actually decides
whether the composed grammar is real: does each candidate's margin over LOOKAHEAD (the free,
known floor) HOLD OR GROW from d6 to d8 at production length and multiple seeds?

This is the gate that the curvature arm failed: a point-d6 win that shrinks at d8 is a
surrogate artifact, not a method. All candidates are trained through the SAME gns program
path as the search (search_program.train_genome), so the comparison is apples-to-apples.

  PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src \
  uv run --project /home/jonas/git/tct-models python -u scripts/eval_program_trend.py \
    --depths 6,8 --seeds 0,1,2 --steps 1500 --lr 0.02 \
    --search-json /home/jonas/git/gns/results/search_program.json --n-knees 3
"""
from __future__ import annotations

import argparse, json, time
import numpy as np
import torch

from search_program import build_context, train_genome, _git_sha, _polar_by_name
from gns import program_grammar as pg


def _cfg():
    p = argparse.ArgumentParser()
    p.add_argument("--depths", type=str, default="6,8")
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--aspect-ratio", type=int, default=64); p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--max-seq-len", type=int, default=1024); p.add_argument("--device-batch-size", type=int, default=16)
    p.add_argument("--momentum", type=float, default=0.95); p.add_argument("--beta2", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.28); p.add_argument("--ns-steps", type=int, default=5)
    p.add_argument("--warmup-steps", type=int, default=20); p.add_argument("--n-val-batches", type=int, default=16)
    p.add_argument("--search-json", type=str, default="/home/jonas/git/gns/results/search_program.json")
    p.add_argument("--n-knees", type=int, default=3, help="how many top-val Stage-1 knees to gate")
    p.add_argument("--out", type=str, default="/home/jonas/git/gns/results/program_trend_gate.json")
    p.add_argument("--no-compile", action="store_true")
    return p.parse_args()


def _candidates(args):
    """Reference points + Stage-1 knees, as {label: genome}. Labels are stable for reporting."""
    cands = {
        "muon": pg.muon_program(),
        "cheaper_muon": pg.ProgramGenome(temporal=pg.muon_program().temporal,
                                         polar_fine=_polar_by_name("polar4"),
                                         polar_coarse=_polar_by_name("polar4")),
        "lookahead": pg.lookahead_program(6, 0.5),
    }
    try:
        s = json.load(open(args.search_json))
        knees = s.get("stage2_knees", [])[: args.n_knees]
        for i, k in enumerate(knees):
            g = pg.from_dict(k["genome"]) if isinstance(k, dict) and "genome" in k else None
            if g is not None and pg.canonical(g) not in {pg.canonical(v) for v in cands.values()}:
                cands[f"knee{i}"] = g
    except FileNotFoundError:
        print(f"  (no search json at {args.search_json}; gating reference points only)", flush=True)
    return cands


def main():
    args = _cfg()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    depths = [int(d) for d in args.depths.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    cands = _candidates(args)
    print(f"=== TREND GATE: {list(cands)} over depths {depths} x seeds {seeds} "
          f"({args.steps} steps, lr {args.lr}) ===", flush=True)
    for lab, g in cands.items():
        print(f"  {lab:14s} cost {g.extra_matmuls():+.3f}  {pg.canonical(g)}", flush=True)

    # EMA-fairness (2026-07-02): every arm is judged at its BEST eval protocol (raw vs EMA-0.99 vs
    # EMA-0.999) — train_genome prices all of them from one trained run. Judging candidates at their
    # ema gene against RAW references inflates the margin by the free Polyak-averaging win (~0.15 at
    # short horizon) and would make a PASS meaningless.
    # vals[label][depth] = list over seeds (best-protocol); vals_own = at the genome's own ema gene
    vals: dict[str, dict[int, list[float]]] = {lab: {d: [] for d in depths} for lab in cands}
    vals_own: dict[str, dict[int, list[float]]] = {lab: {d: [] for d in depths} for lab in cands}
    proto_used: dict[str, dict[int, list[float]]] = {lab: {d: [] for d in depths} for lab in cands}
    for d in depths:
        for s in seeds:
            t0 = time.time()
            ctx = build_context(args, d, s, args.steps, compile_=not args.no_compile)
            print(f"[d{d} s{s}] context built ({time.time()-t0:.0f}s)", flush=True)
            for lab, g in cands.items():
                tg = time.time()
                res = train_genome(ctx, g, args)   # dict since the ema-densification upgrade
                if isinstance(res, dict):
                    protos = {b: v for b, v in res.items() if isinstance(b, float)}
                    b_best, v_best = min(protos.items(), key=lambda kv: kv[1])
                    v_own = res[g.eval_ema_beta]
                else:
                    b_best, v_best, v_own = g.eval_ema_beta, res, res
                vals[lab][d].append(v_best); vals_own[lab][d].append(v_own)
                proto_used[lab][d].append(b_best)
                print(f"  [d{d} s{s}] {lab:14s} val {v_best:.4f} (best-proto ema{b_best:g}; "
                      f"own {v_own:.4f})  ({time.time()-tg:.0f}s)", flush=True)
            del ctx
            torch.cuda.empty_cache()

    mean = {lab: {d: float(np.mean(vs)) for d, vs in dd.items()} for lab, dd in vals.items()}
    sem = {lab: {d: float(np.std(vs) / max(1, len(vs) - 1) ** 0.5) for d, vs in dd.items()}
           for lab, dd in vals.items()}
    dmin, dmax = min(depths), max(depths)
    refs = [r for r in ("muon", "cheaper_muon", "lookahead") if r in cands]
    # the must-beat bar at each depth: the BEST reference, each at ITS best eval protocol
    bestref = {d: min(mean[r][d] for r in refs) for d in depths}
    la = "lookahead"
    report = {}
    for lab in cands:
        margin_la = {d: mean[la][d] - mean[lab][d] for d in depths}
        margin_ref = {d: bestref[d] - mean[lab][d] for d in depths}
        slope = margin_ref[dmax] - margin_ref[dmin]
        passes = (lab not in refs and margin_ref[dmin] > 0.0 and slope >= -0.005)
        report[lab] = {
            "canonical": pg.canonical(cands[lab]),
            "cost": cands[lab].extra_matmuls(),
            "val_mean": mean[lab],
            "val_sem": sem[lab],
            "eval_protocols_used": proto_used[lab],
            "margin_over_lookahead": margin_la,
            "margin_over_bestref": margin_ref,
            "trend_slope_d{}_to_d{}".format(dmin, dmax): slope,
            "verdict": "PASS" if passes else ("ref" if lab in refs else "FAIL"),
        }

    out = {
        "config": {**vars(args), "depths": depths, "seeds": seeds,
                   "gns_sha": _git_sha("/home/jonas/git/gns"),
                   "nanochat_sha": _git_sha("/home/jonas/git/nanochat")},
        "val_seedmean": mean,
        "val_seeds_bestproto": {lab: dd for lab, dd in vals.items()},
        "val_seeds_owngene": {lab: dd for lab, dd in vals_own.items()},
        "candidates": report,
    }
    json.dump(out, open(args.out, "w"), indent=1)

    print(f"\n=== TREND GATE RESULT (all arms at their BEST eval protocol) ===")
    print(f"  {'label':14s} {'d'+str(dmin):>8s} {'d'+str(dmax):>8s}  "
          f"{'vs-ref@d'+str(dmin):>11s} {'vs-ref@d'+str(dmax):>11s} {'slope':>8s}  verdict")
    for lab, r in report.items():
        m = r["margin_over_bestref"]
        print(f"  {lab:14s} {mean[lab][dmin]:8.4f} {mean[lab][dmax]:8.4f}  "
              f"{m[dmin]:+11.4f} {m[dmax]:+11.4f} {r['trend_slope_d%d_to_d%d'%(dmin,dmax)]:+8.4f}  {r['verdict']}")
    print(f"\n  PASS = beats the BEST EMA-matched reference at d{dmin} AND margin holds/grows to d{dmax}.")


if __name__ == "__main__":
    main()
