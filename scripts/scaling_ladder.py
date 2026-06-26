"""scaling_ladder.py — does an optimizer arm's advantage over Muon hold/grow with model size?

The evidence engine for the frontier-suitability argument. Single-scale (d6/d12) optimizer tests do
not predict frontier behaviour; this sweeps a depth ladder and measures, per size, how a candidate
arm compares to the Muon baseline on three axes that extrapolate:

  * loss gap        : best_val(candidate) - best_val(muon)   -> constant/shrinking-toward-better = scalable
  * wall overhead   : (wall_cand - wall_muon)/wall_muon      -> should decay ~ d/(batch*seq) toward frontier
  * LR transfer     : argmin matrix-lr per arm per depth      -> stable across width = config transfers

It reuses scripts/train_compare_precond.py as the evaluator via subprocess (same pattern as
search_optimizer.py:evaluate) and reads back its JSON — no duplicated training loop.

Token budget (--mode):
  fixed    : same --num-iterations at every depth (matched-step advantage-vs-scale trend).
  optimal  : ISO-RATIO compute — tokens proportional to NON-embedding params (the scaling-law shape),
             normalised so the LARGEST rung lands at --opt-max-iters (full Chinchilla 20 tok/param is
             infeasible here; iso-ratio keeps constant tokens/param while staying affordable).
  both     : fixed pass then optimal pass.

Run (tct-models env + repos on PYTHONPATH):
  cd /home/jonas/git/nanochat
  PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src \
    uv run --project /home/jonas/git/tct-models python -u scripts/scaling_ladder.py \
      --depths 6,8,10,12,16 --arms muon,ortho_shampoo --matrix-lr-grid 0.005,0.01,0.02 --compile
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HARNESS = str(Path(__file__).resolve().parent / "train_compare_precond.py")
HARNESS_OUT = Path("/home/jonas/git/gns/results/precond_train_compare.json")
RESULTS_DIR = Path("/home/jonas/git/gns/results")
ASPECT, HEAD_DIM = 64, 128  # must match train_compare_precond.build_model
# one-sided 97.5% Student-t critical values by dof (n-1); ~2 sigma with small-sample correction
_TCRIT = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def width(depth: int, aspect: int = ASPECT) -> int:
    base = depth * aspect
    return ((base + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM


def nonembed_params(depth: int, aspect: int = ASPECT) -> int:
    d = width(depth, aspect)
    return 12 * depth * d * d  # ~ attn(4d^2) + mlp(8d^2) per layer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--depths", type=str, default="6,8,10,12,16")
    p.add_argument("--arms", type=str, default="muon,ortho_shampoo",
                   help="first arm is the baseline (muon); the rest are candidates")
    p.add_argument("--matrix-lr-grid", type=str, default="0.005,0.01,0.02")
    p.add_argument("--seeds", type=str, default="0",
                   help="comma list of harness --seed values per rung; the gap is averaged "
                        "(paired) across them. Default '0' keeps the single-seed behaviour; "
                        "pass e.g. '0,1,2' for the load-bearing pass. Seeds vary BOTH weight "
                        "init AND the train data window (different parquet shards -> genuinely "
                        "different documents); VAL is held fixed across seeds.")
    p.add_argument("--mode", choices=["fixed", "optimal", "both"], default="both")
    p.add_argument("--fixed-iters", type=int, default=2000)
    p.add_argument("--opt-max-iters", type=int, default=2500, help="iters at the largest depth in optimal mode")
    p.add_argument("--opt-min-iters", type=int, default=300)
    p.add_argument("--device-batch-size", type=int, default=16)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--batch-sweep", type=str, default="16,32,64,128",
                   help="device-batch sizes for the overhead-AND-gap-vs-batch probe; empty to "
                        "skip. Swept UPWARD toward frontier batch: overhead amortizes ~1/batch, "
                        "and the val-loss gap is recorded to test whether the advantage SURVIVES "
                        "at large batch (the quality gate). Single-seed (seed 0).")
    p.add_argument("--batch-sweep-depth", type=int, default=6)
    p.add_argument("--batch-sweep-iters", type=int, default=600)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--synth-alpha", type=float, default=1.0, help="passed to the harness synth arm")
    p.add_argument("--synth-ortho", type=int, default=1, help="passed to the harness synth arm")
    p.add_argument("--precond-coupled-orders", type=str, default="",
                   help="grammar CoupledStep order sequence passed to the harness inverse-root")
    p.add_argument("--shampoo-ridge", type=str, default="",
                   help="harness shampoo-ridge (effective shrinkage); empty = harness default 1e-4")
    p.add_argument("--aspect-ratio", type=int, default=64,
                   help="model width = depth * aspect-ratio (rounded to head_dim). Default 64 = the "
                        "depth-scaling campaign. Higher aspect at fixed depth = the WIDTH-scaling probe "
                        "(direction 1, notes/scaling-directions-not-sampled.md): does the curvature win "
                        "grow with width at fixed depth? Frontier SOTA is wide, not deep.")
    p.add_argument("--orth-every", type=int, default=1,
                   help="apply the polar/preconditioner every K steps (passed to the harness). Default 1 "
                        "= every step. Tests the over-orthogonalization-at-depth hypothesis (direction 2, "
                        "notes/scaling-directions-not-sampled.md).")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--restart", action="store_true",
                   help="ignore any existing checkpoint for this --tag and start fresh")
    return p.parse_args()


def run_harness(depth, arms, iters, lr_grid, dbs, seq, compile_, out_path, seed=0,
                synth_alpha=1.0, synth_ortho=1, precond_orders="", shampoo_ridge="",
                aspect_ratio=64, orth_every=1):
    # unique per-call --out (race-free) that the harness ALSO checkpoints per (arm,lr) to, so a driver
    # restart resumes a half-done depth instead of recomputing it. The harness resume signature includes
    # seed, so a seed-specific out_path resumes each seed independently.
    cmd = [sys.executable, "-u", HARNESS, "--depth", str(depth), "--num-iterations", str(iters),
           "--arms", arms, "--matrix-lr-grid", lr_grid, "--device-batch-size", str(dbs),
           "--max-seq-len", str(seq), "--seed", str(seed), "--synth-alpha", str(synth_alpha),
           "--synth-ortho", str(synth_ortho), "--precond-coupled-orders", precond_orders,
           "--aspect-ratio", str(aspect_ratio), "--orth-every", str(orth_every),
           "--out", str(out_path)]
    if shampoo_ridge:
        cmd += ["--shampoo-ridge", str(shampoo_ridge)]
    if compile_:
        cmd.append("--compile")
    env = {**os.environ, "PYTHONPATH": "/home/jonas/git/nanochat:/home/jonas/git/gns/src"}
    subprocess.run(cmd, check=True, env=env)  # inherits stdout -> streams live
    return json.loads(Path(out_path).read_text())


def collect(res, arms_list):
    s = res["arms"]
    return {a: {"best_val": s[a]["best_val"], "final_val": s[a]["final_val"],
                "lr": s[a]["lr"], "wall_s": s[a]["total_wall_s"]} for a in arms_list}


def optimal_iters(depth, max_depth, args):
    r = nonembed_params(depth, args.aspect_ratio) / nonembed_params(max_depth, args.aspect_ratio)
    return max(args.opt_min_iters, round(args.opt_max_iters * r))


def _save(payload, path):
    """Atomic + durable checkpoint write (tmp + fsync + os.replace). A transient FS error warns but
    does not crash the run — the previous good checkpoint stays valid."""
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=1); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    except OSError as e:
        print(f"  [warn] checkpoint save failed ({e}); continuing — previous checkpoint intact")


def _fit_trend(rungs, candidates, name):
    """Honest verdict: a negative 2-point slope means NOTHING if the candidate is not a SIGNIFICANT
    win (>2 sigma over seeds) at the rungs. Gate on significance + level sign, not slope alone — the
    old slope-only verdict announced 'advantage grows with scale' for runs where the candidate lost
    at every rung."""
    trend = {}
    for c in candidates:
        levels = [r["candidates"][c]["gap"] for r in rungs]
        sig_rungs = [bool(r["candidates"][c].get("significant")) for r in rungs]
        slope = None
        if len(rungs) >= 2:
            slope = float(np.polyfit(np.log([r["width"] for r in rungs]), levels, 1)[0])
        if not any(sig_rungs):
            verdict = "no significant win at any rung (effect within noise)"
        elif all(sig_rungs) and slope is not None and slope < 0:
            verdict = "significant win at every rung; negative scaling slope (trend, slope-CI not established)"
        elif all(sig_rungs):
            verdict = "significant win at every rung; slope flat/positive"
        else:
            verdict = "significant only at some rungs (mixed)"
        trend[c] = {"gap_vs_logwidth_slope": slope, "levels": levels,
                    "significant_rungs": sig_rungs, "verdict": verdict}
        lv = " ".join(f"{g:+.4f}{'*' if s else ''}" for g, s in zip(levels, sig_rungs))
        print(f"  [{name}] TREND {c}: levels {lv}  slope "
              f"{('%.4f' % slope) if slope is not None else 'n/a'}  -> {verdict}")
    return trend


def run_pass(payload, name, depths, iters_of, args, arms_list, baseline, candidates, out_json, seeds):
    """Resumable: each completed (pass, depth) rung is checkpointed to out_json; on a restart with the
    same --tag, already-done rungs are skipped. Granularity is per-depth — an interrupted depth re-runs
    its seeds, but each seed's harness call resumes from its own seed-specific file.

    Multi-seed: the candidate gap is the PAIRED difference best_val(cand,s) - best_val(muon,s) averaged
    over seeds (pairing cancels the per-seed common-mode noise); `gap` is the mean (keeps the key that
    _fit_trend/make_figure read), with `gap_std`/`gap_seeds`/`n_seeds` for significance."""
    print(f"\n########## ladder pass: {name}  (seeds {seeds}) ##########")
    pz = payload.setdefault(f"pass_{name}", {"iters_mode": name, "rungs": [], "trend": {}})
    done = {r["depth"] for r in pz["rungs"]}
    for d in depths:
        if d in done:
            print(f"  [resume] {name} d{d} already checkpointed — skipping")
            continue
        it = iters_of(d)
        print(f"\n===== depth {d} (width {width(d, args.aspect_ratio)}, "
              f"~{nonembed_params(d, args.aspect_ratio)/1e6:.1f}M non-embed) "
              f"x {it} iters x {len(seeds)} seed(s) =====")
        per_seed = []  # collect()-style dict per seed
        for s in seeds:
            sub_out = RESULTS_DIR / f"precond_{args.tag or 'default'}_{name}_d{d}_s{s}.json"
            res = run_harness(d, ",".join(arms_list), it, args.matrix_lr_grid,
                              args.device_batch_size, args.max_seq_len, args.compile, sub_out, seed=s,
                              synth_alpha=args.synth_alpha, synth_ortho=args.synth_ortho,
                              precond_orders=args.precond_coupled_orders, shampoo_ridge=args.shampoo_ridge,
                              aspect_ratio=args.aspect_ratio, orth_every=args.orth_every)
            per_seed.append(collect(res, arms_list))
        rec = {"depth": d, "width": width(d, args.aspect_ratio),
               "nonembed_params": nonembed_params(d, args.aspect_ratio),
               "iters": it, "seeds": {s: per_seed[i] for i, s in enumerate(seeds)}, "candidates": {}}
        # arms summary = seed-mean of best_val/wall_s (lr reported as the per-seed list)
        rec["arms"] = {a: {"best_val": float(np.mean([ps[a]["best_val"] for ps in per_seed])),
                           "wall_s": float(np.mean([ps[a]["wall_s"] for ps in per_seed])),
                           "lr": [ps[a]["lr"] for ps in per_seed]} for a in arms_list}
        for c in candidates:
            # gap on FINAL val (no best-checkpoint selection); paired per seed (cancels common-mode noise)
            gaps = np.array([ps[c]["final_val"] - ps[baseline]["final_val"] for ps in per_seed])
            over = np.array([(ps[c]["wall_s"] - ps[baseline]["wall_s"]) / ps[baseline]["wall_s"]
                             for ps in per_seed])
            n = len(gaps); gmean = float(gaps.mean())
            gstd = float(gaps.std(ddof=1)) if n > 1 else 0.0
            sem = gstd / np.sqrt(n) if n > 1 and gstd > 0 else float("inf")
            tstat = gmean / sem if np.isfinite(sem) and sem > 0 else 0.0
            # one-sided t-test that the paired mean gap is < 0 (candidate beats baseline). This is the
            # rigorous "2 sigma" significance (uncertainty of the MEAN, small-sample t-critical), not an
            # effect-size |mean|>2*SD which would reject real p~0.01 effects.
            sig = bool(n > 1 and gmean < 0 and tstat < -_TCRIT.get(n - 1, 2.0))
            rec["candidates"][c] = {
                "gap": gmean,                                       # <0 = candidate beats baseline (seed-mean, FINAL val)
                "gap_std": gstd, "gap_sem": (gstd / np.sqrt(n) if n > 1 else None),
                "t_stat": tstat, "t_crit": _TCRIT.get(n - 1, 2.0),
                "gap_seeds": [float(g) for g in gaps], "n_seeds": n,
                "significant": sig,   # one-sided t-test mean<0 at ~95% — the ONLY thing we interpret
                "overhead": float(over.mean()),                     # per-sweep wall overhead (seed-mean)
                "lr": [ps[c]["lr"] for ps in per_seed], "muon_lr": [ps[baseline]["lr"] for ps in per_seed]}
        pz["rungs"].append(rec)
        pz["rungs"].sort(key=lambda r: r["depth"])
        _save(payload, out_json)  # checkpoint after each completed rung
        for c in candidates:
            cc = rec["candidates"][c]
            tg = "SIGNIF beats baseline (>2sigma)" if cc["significant"] else "NOT significant (within noise)"
            print(f"  [{name}] d{d}: {c} gap {cc['gap']:+.4f} +/-{cc['gap_std']:.4f} (n={cc['n_seeds']}) "
                  f"-> {tg}  overhead {cc['overhead']:+.1%}  [checkpointed]")
    pz["trend"] = _fit_trend(pz["rungs"], candidates, name)
    _save(payload, out_json)


_CKPT_KEYS = ("depths", "arms", "matrix_lr_grid", "seeds", "fixed_iters", "opt_max_iters",
              "opt_min_iters", "device_batch_size", "max_seq_len", "synth_alpha", "synth_ortho",
              "precond_coupled_orders", "shampoo_ridge", "aspect_ratio", "orth_every")


def load_or_init(path, args, depths, baseline, candidates):
    """Resume from an existing checkpoint for this --tag. Refuses to SILENTLY overwrite prior work: a
    corrupt file or a config mismatch is a hard error (use --restart to discard, or a new --tag)."""
    fresh = {"config": vars(args), "depths": depths, "baseline": baseline, "candidates": candidates}
    if args.restart:
        if path.exists():
            print(f"[restart] discarding existing checkpoint {path.name}")
        return fresh
    if not path.exists():
        return fresh
    try:
        p = json.loads(path.read_text())
    except Exception as e:
        raise SystemExit(f"[abort] checkpoint {path} is unreadable/corrupt ({e}). "
                         f"Pass --restart to discard it, or use a different --tag.")
    old = p.get("config", {})
    # backward-compatible: only abort on keys that EXIST in the old checkpoint and actually differ;
    # knobs added to the code after the checkpoint was written are tolerated (their default applies).
    diffs = [k for k in _CKPT_KEYS if k in old and old[k] != getattr(args, k)]
    if diffs:
        raise SystemExit(f"[abort] checkpoint {path.name} config differs on {diffs} (would mix "
                         f"incompatible runs). Use a new --tag, or --restart to discard.")
    tmp = path.with_suffix(path.suffix + ".tmp")  # clear stale tmp from a prior crash mid-write
    if tmp.exists():
        tmp.unlink()
    done = {m: [r["depth"] for r in p.get(f"pass_{m}", {}).get("rungs", [])]
            for m in ("fixed", "optimal") if f"pass_{m}" in p}
    print(f"[resume] loaded {path.name}: completed rungs {done}")
    return p


def batch_sweep(args, candidates, baseline, arms_list):
    if not args.batch_sweep.strip():
        return None
    print(f"\n########## gap-AND-overhead-vs-batch probe (depth {args.batch_sweep_depth}) ##########")
    out = []
    for bs in [int(x) for x in args.batch_sweep.split(",")]:
        sub_out = RESULTS_DIR / f"precond_{args.tag or 'default'}_batch{bs}.json"
        res = run_harness(args.batch_sweep_depth, ",".join(arms_list), args.batch_sweep_iters,
                          args.matrix_lr_grid, bs, args.max_seq_len, args.compile, sub_out,
                          synth_alpha=args.synth_alpha, synth_ortho=args.synth_ortho,
                          precond_orders=args.precond_coupled_orders, shampoo_ridge=args.shampoo_ridge,
                          aspect_ratio=args.aspect_ratio, orth_every=args.orth_every)
        arms = collect(res, arms_list)
        wmuon = arms[baseline]["wall_s"]; bmuon = arms[baseline]["best_val"]
        # record the val-loss GAP, not just overhead: the quality question is whether the
        # candidate's advantage SURVIVES as batch grows and overhead amortizes toward 0.
        row = {"batch": bs, "tokens_per_step": bs * args.max_seq_len,
               "overhead": {c: (arms[c]["wall_s"] - wmuon) / wmuon for c in candidates},
               "gap": {c: arms[c]["best_val"] - bmuon for c in candidates}}
        out.append(row)
        for c in candidates:
            print(f"  batch {bs}: {c} gap {row['gap'][c]:+.4f}  overhead {row['overhead'][c]:+.1%}")
    # headline: does the advantage survive at the largest (frontier) batch vs the smallest?
    if len(out) >= 2:
        lo, hi = out[0], out[-1]
        for c in candidates:
            print(f"  [batch] {c}: gap {lo['gap'][c]:+.4f} (batch {lo['batch']}) -> "
                  f"{hi['gap'][c]:+.4f} (batch {hi['batch']}); overhead "
                  f"{lo['overhead'][c]:+.1%} -> {hi['overhead'][c]:+.1%}  "
                  f"-> {'advantage survives at frontier batch' if hi['gap'][c] <= 0 else 'advantage erodes at large batch'}")
    return out


def make_figure(payload, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(figure skipped: {e})")
        return
    passes = [v for k, v in payload.items() if k.startswith("pass_")]
    cands = list(passes[0]["trend"].keys()) if passes else []
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for pz in passes:
        ws = [r["width"] for r in pz["rungs"]]
        for c in cands:
            g = [r["candidates"][c]["gap"] for r in pz["rungs"]]
            gstd = [r["candidates"][c].get("gap_std", 0.0) for r in pz["rungs"]]
            o = [r["candidates"][c]["overhead"] for r in pz["rungs"]]
            # lr is a per-seed list per rung; plot the candidate's per-seed LRs (transfer = flat)
            lr = [np.mean(r["candidates"][c]["lr"]) if isinstance(r["candidates"][c]["lr"], list)
                  else r["candidates"][c]["lr"] for r in pz["rungs"]]
            ax[0].errorbar(ws, g, yerr=gstd, fmt="o-", capsize=3, label=f"{c}/{pz['iters_mode']}")
            ax[1].plot(ws, o, "o-", label=f"{c}/{pz['iters_mode']}")
            ax[2].plot(ws, lr, "o-", label=f"{c}/{pz['iters_mode']}")
    ax[0].axhline(0, color="k", lw=0.7); ax[0].set_title("loss gap vs Muon (<0 better)")
    ax[1].axhline(0, color="k", lw=0.7); ax[1].set_title("wall overhead vs Muon")
    ax[2].set_title("argmin matrix-LR (transfer)")
    for a in ax:
        a.set_xlabel("model width"); a.set_xscale("log"); a.legend(fontsize=7); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=110)
    print(f"[figure] {path}")


def main():
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    depths = [int(x) for x in args.depths.split(",")]
    arms_list = args.arms.split(",")
    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    baseline, candidates = arms_list[0], arms_list[1:]
    assert baseline == "muon", "first arm must be the muon baseline"
    max_depth = max(depths)

    suffix = f"_{args.tag}" if args.tag else ""
    out_json = RESULTS_DIR / f"scaling_ladder{suffix}.json"
    payload = load_or_init(out_json, args, depths, baseline, candidates)

    modes = ["fixed", "optimal"] if args.mode == "both" else [args.mode]
    for m in modes:
        iters_of = ((lambda d: args.fixed_iters) if m == "fixed"
                    else (lambda d: optimal_iters(d, max_depth, args)))
        run_pass(payload, m, depths, iters_of, args, arms_list, baseline, candidates, out_json, seeds)
    if "batch_sweep" not in payload:
        payload["batch_sweep"] = batch_sweep(args, candidates, baseline, arms_list)
        _save(payload, out_json)
    else:
        print("[resume] batch_sweep already present — skipping")

    make_figure(payload, RESULTS_DIR / f"scaling_ladder{suffix}.png")
    print(f"\n[saved] {out_json}")
    for m in modes:
        for c, t in payload[f"pass_{m}"]["trend"].items():
            slope = t["gap_vs_logwidth_slope"]
            slope_str = f"{slope:+.4f}" if slope is not None else "n/a"
            print(f"  {m:8s} {c}: slope {slope_str}  ({t['verdict']})")


if __name__ == "__main__":
    main()
