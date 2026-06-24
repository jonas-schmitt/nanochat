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


def width(depth: int) -> int:
    base = depth * ASPECT
    return ((base + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM


def nonembed_params(depth: int) -> int:
    d = width(depth)
    return 12 * depth * d * d  # ~ attn(4d^2) + mlp(8d^2) per layer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--depths", type=str, default="6,8,10,12,16")
    p.add_argument("--arms", type=str, default="muon,ortho_shampoo",
                   help="first arm is the baseline (muon); the rest are candidates")
    p.add_argument("--matrix-lr-grid", type=str, default="0.005,0.01,0.02")
    p.add_argument("--mode", choices=["fixed", "optimal", "both"], default="both")
    p.add_argument("--fixed-iters", type=int, default=2000)
    p.add_argument("--opt-max-iters", type=int, default=2500, help="iters at the largest depth in optimal mode")
    p.add_argument("--opt-min-iters", type=int, default=300)
    p.add_argument("--device-batch-size", type=int, default=16)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--batch-sweep", type=str, default="8,16,32",
                   help="device-batch sizes for the overhead-vs-batch probe; empty to skip")
    p.add_argument("--batch-sweep-depth", type=int, default=6)
    p.add_argument("--batch-sweep-iters", type=int, default=600)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--restart", action="store_true",
                   help="ignore any existing checkpoint for this --tag and start fresh")
    return p.parse_args()


def run_harness(depth, arms, iters, lr_grid, dbs, seq, compile_, out_path):
    # unique per-call --out (race-free) that the harness ALSO checkpoints per (arm,lr) to, so a driver
    # restart resumes a half-done depth instead of recomputing it.
    cmd = [sys.executable, "-u", HARNESS, "--depth", str(depth), "--num-iterations", str(iters),
           "--arms", arms, "--matrix-lr-grid", lr_grid, "--device-batch-size", str(dbs),
           "--max-seq-len", str(seq), "--out", str(out_path)]
    if compile_:
        cmd.append("--compile")
    env = {**os.environ, "PYTHONPATH": "/home/jonas/git/nanochat:/home/jonas/git/gns/src"}
    subprocess.run(cmd, check=True, env=env)  # inherits stdout -> streams live
    return json.loads(Path(out_path).read_text())


def collect(res, arms_list):
    s = res["arms"]
    return {a: {"best_val": s[a]["best_val"], "lr": s[a]["lr"], "wall_s": s[a]["total_wall_s"]}
            for a in arms_list}


def optimal_iters(depth, max_depth, args):
    r = nonembed_params(depth) / nonembed_params(max_depth)
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
    trend = {}
    if len(rungs) >= 2:
        lw = np.log([r["width"] for r in rungs])
        for c in candidates:
            gaps = np.array([r["candidates"][c]["gap"] for r in rungs])
            slope = float(np.polyfit(lw, gaps, 1)[0])
            trend[c] = {"gap_vs_logwidth_slope": slope,
                        "verdict": ("advantage grows with scale" if slope < 0
                                    else "advantage erodes with scale")}
            print(f"  [{name}] TREND {c}: d(gap)/d(log width) = {slope:+.4f}  -> {trend[c]['verdict']}")
    return trend


def run_pass(payload, name, depths, iters_of, args, arms_list, baseline, candidates, out_json):
    """Resumable: each completed (pass, depth) rung is checkpointed to out_json; on a restart with the
    same --tag, already-done rungs are skipped. Granularity is per-depth — an interrupted depth re-runs."""
    print(f"\n########## ladder pass: {name} ##########")
    pz = payload.setdefault(f"pass_{name}", {"iters_mode": name, "rungs": [], "trend": {}})
    done = {r["depth"] for r in pz["rungs"]}
    for d in depths:
        if d in done:
            print(f"  [resume] {name} d{d} already checkpointed — skipping")
            continue
        it = iters_of(d)
        print(f"\n===== depth {d} (width {width(d)}, ~{nonembed_params(d)/1e6:.1f}M non-embed) "
              f"x {it} iters =====")
        sub_out = RESULTS_DIR / f"precond_{args.tag or 'default'}_{name}_d{d}.json"
        res = run_harness(d, ",".join(arms_list), it, args.matrix_lr_grid,
                          args.device_batch_size, args.max_seq_len, args.compile, sub_out)
        arms = collect(res, arms_list)
        rec = {"depth": d, "width": width(d), "nonembed_params": nonembed_params(d),
               "iters": it, "arms": arms, "candidates": {}}
        bmuon = arms[baseline]["best_val"]; wmuon = arms[baseline]["wall_s"]
        for c in candidates:
            rec["candidates"][c] = {
                "gap": arms[c]["best_val"] - bmuon,                 # <0 = candidate beats muon
                "overhead": (arms[c]["wall_s"] - wmuon) / wmuon,    # per-sweep wall overhead
                "lr": arms[c]["lr"], "muon_lr": arms[baseline]["lr"]}
        pz["rungs"].append(rec)
        pz["rungs"].sort(key=lambda r: r["depth"])
        _save(payload, out_json)  # checkpoint after each completed rung
        for c in candidates:
            cc = rec["candidates"][c]
            print(f"  [{name}] d{d}: {c} gap {cc['gap']:+.4f}  overhead {cc['overhead']:+.1%}  "
                  f"lr {cc['lr']} (muon {cc['muon_lr']})  [checkpointed]")
    pz["trend"] = _fit_trend(pz["rungs"], candidates, name)
    _save(payload, out_json)


_CKPT_KEYS = ("depths", "arms", "matrix_lr_grid", "fixed_iters", "opt_max_iters",
              "opt_min_iters", "device_batch_size", "max_seq_len")


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
    if [old.get(k) for k in _CKPT_KEYS] != [getattr(args, k) for k in _CKPT_KEYS]:
        raise SystemExit(f"[abort] checkpoint {path.name} config differs from current args "
                         f"(would mix incompatible runs). Use a new --tag, or --restart to discard.")
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
    print(f"\n########## overhead-vs-batch probe (depth {args.batch_sweep_depth}) ##########")
    out = []
    for bs in [int(x) for x in args.batch_sweep.split(",")]:
        sub_out = RESULTS_DIR / f"precond_{args.tag or 'default'}_batch{bs}.json"
        res = run_harness(args.batch_sweep_depth, ",".join(arms_list), args.batch_sweep_iters,
                          args.matrix_lr_grid, bs, args.max_seq_len, args.compile, sub_out)
        arms = collect(res, arms_list)
        wmuon = arms[baseline]["wall_s"]
        row = {"batch": bs, "tokens_per_step": bs * args.max_seq_len,
               "overhead": {c: (arms[c]["wall_s"] - wmuon) / wmuon for c in candidates}}
        out.append(row)
        for c in candidates:
            print(f"  batch {bs}: {c} overhead {row['overhead'][c]:+.1%}")
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
            o = [r["candidates"][c]["overhead"] for r in pz["rungs"]]
            lr = [r["candidates"][c]["lr"] for r in pz["rungs"]]
            ax[0].plot(ws, g, "o-", label=f"{c}/{pz['iters_mode']}")
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
        run_pass(payload, m, depths, iters_of, args, arms_list, baseline, candidates, out_json)
    if "batch_sweep" not in payload:
        payload["batch_sweep"] = batch_sweep(args, candidates, baseline, arms_list)
        _save(payload, out_json)
    else:
        print("[resume] batch_sweep already present — skipping")

    make_figure(payload, RESULTS_DIR / f"scaling_ladder{suffix}.png")
    print(f"\n[saved] {out_json}")
    for m in modes:
        for c, t in payload[f"pass_{m}"]["trend"].items():
            print(f"  {m:8s} {c}: slope {t['gap_vs_logwidth_slope']:+.4f}  ({t['verdict']})")


if __name__ == "__main__":
    main()
