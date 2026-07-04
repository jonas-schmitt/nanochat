"""Replay-vs-real RANK gate (fused early-kill, TODO "FUSED EARLY-KILL GATE").

The core de-risking step: replay is off-policy evaluation, trustworthy only near the recording
anchor. Before trusting the tier-1 search's replay numbers, we must measure — PER GENE FAMILY —
how well replay-val rank-correlates with REAL train_diloco val.

Cheap by construction: the "real" side REUSES existing `train_diloco` result JSONs already on disk
(Gate-B + Phase-C runs share the anchor's inner config: d6/1500 M4 H30 inner=muon s0). Each such
result carries `config.genome` + `best_val` (real ground truth); we reconstruct the genome, replay it
against the recorded anchor (seconds), and pair (replay_best, real_best). New references (hadamard /
8-bit / EF-off) are produced as ordinary train_diloco runs by the orchestrator and consumed here the
same way — this script is pure analysis over {anchor recording} + {a set of result JSONs}.

Families are tagged relative to the anchor: a reference varies PRECISION (bits/basis/EF/rounding),
GEOMETRY (outer transform/lr/momentum), or both. We report Spearman over all refs and over the
precision-only and geometry-only subsets. Expectation: precision/basis/EF trustworthy (perturbative);
outer-geometry doubtful (trajectory divergence).

  python scripts/replay_rank_gate.py --recording ANCHOR.pt \
    --results a.json,b.json,... --out rank_gate.json
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/jonas/git/gns/src")

import replay_policy as rp  # noqa: E402
from gns.policy_grammar import canonical, from_dict as genome_from_dict  # noqa: E402


def _atomic_json(path, obj):
    tmp = f"{path}.tmp.{os.getpid()}"
    json.dump(obj, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def _rankdata(a):
    order = np.argsort(np.asarray(a, float))
    ranks = np.empty(len(a))
    ranks[order] = np.arange(len(a))
    return ranks


def _spearman(x, y):
    if len(x) < 3:
        return float("nan")
    rx, ry = _rankdata(x), _rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _family(g, anchor):
    prec = (g.delta_bits != anchor.delta_bits or g.basis != anchor.basis
            or g.error_feedback != anchor.error_feedback
            or g.stochastic_rounding != anchor.stochastic_rounding)
    geom = (g.outer_transform != anchor.outer_transform
            or abs(g.outer_lr - anchor.outer_lr) > 1e-9
            or tuple(g.outer.momentum_betas) != tuple(anchor.outer.momentum_betas)
            or g.outer.nesterov != anchor.outer.nesterov)
    return prec, geom


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--recording", type=str, required=True)
    p.add_argument("--results", type=str, required=True,
                   help="comma-separated result JSON paths and/or globs (real ground truth to reuse)")
    p.add_argument("--eval-every", type=int, default=100, help="replay eval cadence (match real runs)")
    p.add_argument("--spearman-pass", type=float, default=0.8)
    p.add_argument("--out", type=str, default="/home/jonas/git/gns/results/replay_rank_gate.json")
    cli = p.parse_args()

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = "cuda"
    rec = rp.load_recording(cli.recording)
    anchor = rec["anchor_genome"]
    args = rp.rec_args(rec)
    model, val_batches = rp.build_eval_context(rec, device)

    paths = []
    for tok in cli.results.split(","):
        tok = tok.strip()
        paths.extend(sorted(glob.glob(tok)) if any(c in tok for c in "*?[") else [tok])

    rows = []
    for path in paths:
        if not os.path.exists(path):
            print(f"  (skip missing {path})", flush=True); continue
        try:
            d = json.load(open(path))
            g = genome_from_dict(d["config"]["genome"]); real = float(d["best_val"])
        except Exception as e:
            print(f"  (skip unreadable {path}: {e})", flush=True); continue
        if g.h != anchor.h or g.inner != anchor.inner:
            print(f"  (skip {os.path.basename(path)}: inner/h differ from anchor — not replayable)",
                  flush=True); continue
        t0 = time.time()
        log = rp.replay(rec, g, args, model, val_batches, device, eval_every=cli.eval_every)
        rep = float(min(log["val"]))
        prec, geom = _family(g, anchor)
        fam = "both" if prec and geom else "precision" if prec else "geometry" if geom else "anchor"
        rows.append({"label": os.path.basename(path), "canonical": canonical(g), "family": fam,
                     "real_best": real, "replay_best": rep,
                     "bits_per_param_step": g.comm_bits_per_param_step()})
        print(f"  {fam:9s} real {real:.4f}  replay {rep:.4f}  d {rep-real:+.4f}  "
              f"({time.time()-t0:.0f}s)  {os.path.basename(path)}", flush=True)

    def corr(subset):
        r = [x for x in rows if x["family"] in subset or x["family"] == "anchor"]
        return _spearman([x["replay_best"] for x in r], [x["real_best"] for x in r]), len(r)

    sp_all, n_all = _spearman([x["replay_best"] for x in rows], [x["real_best"] for x in rows]), len(rows)
    sp_prec, n_prec = corr({"precision"})
    sp_geom, n_geom = corr({"geometry"})
    prec_ok = not np.isnan(sp_prec) and sp_prec >= cli.spearman_pass
    geom_ok = not np.isnan(sp_geom) and sp_geom >= cli.spearman_pass

    print("\n=== REPLAY-vs-REAL RANK GATE ===")
    print(f"  all       n={n_all:2d}  Spearman {sp_all:+.3f}")
    print(f"  precision n={n_prec:2d}  Spearman {sp_prec:+.3f}  -> {'TRUST' if prec_ok else 'DOUBT'}")
    print(f"  geometry  n={n_geom:2d}  Spearman {sp_geom:+.3f}  -> {'TRUST' if geom_ok else 'DOUBT'}")
    print(f"\n  search-trust: precision={'YES' if prec_ok else 'NO'} "
          f"geometry={'YES' if geom_ok else 'NO'} (pass threshold {cli.spearman_pass})")
    print("  => tier-1 search should include-geometry ONLY if geometry=YES; else precision axes + "
          "M=1 proxy for outer-lr.")

    _atomic_json(cli.out, {
        "anchor": canonical(anchor), "spearman_pass": cli.spearman_pass,
        "spearman": {"all": sp_all, "precision": sp_prec, "geometry": sp_geom},
        "n": {"all": n_all, "precision": n_prec, "geometry": n_geom},
        "trust": {"precision": prec_ok, "geometry": geom_ok},
        "rows": rows,
    })
    print(f"wrote {cli.out}")


if __name__ == "__main__":
    main()
