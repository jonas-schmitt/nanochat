"""DEPRECATED (2026-06-25) — superseded by the real typed-grammar search.

This was a flat coefficient GA over polar (a,b,c) triples. The moonshot now uses the existing typed
grammar + NSGA-II in gns (`gns.search.nsga2` over `gns.coupled_grammar`, driven by
`gns/experiments/exp28_coupled_regime_zoo.py`) scored by the exact scalar surrogate, with grammar
schedules injected into training via the harness `--precond-coupled-orders` flag and evaluated by
`scaling_ladder.py`. Kept for provenance; do not extend.

③ (scalability moonshot) — search a polar Newton–Schulz schedule that SCALES better than Muon.

Single-scale loss is the wrong objective: a schedule that wins at d6 tells you ~nothing about d=16384
and often anti-correlates with scale. So fitness here is a SCALABILITY proxy — each candidate is judged
on how its advantage over Muon behaves between a small and a large model:

    gap_d = best_val(candidate, depth d) - best_val(muon, depth d)        (negative = beats Muon)
    F     = gap_large + lambda * relu(gap_large - gap_small)              (MINIMISE)

i.e. reward beating Muon at the larger size AND penalise any candidate whose advantage erodes from
small->large. Candidates are first passed through a cheap SCALAR-STABILITY filter (the polar map must
drive singular values toward 1 without blowing up) — this enforces the width-covariant/contractive
prior for free and avoids spending GPU on divergent schedules. The elite winner is then put through
horizon-transfer (4x iters) and scale-transfer (advantage persists across a depth ladder) gates.

Evaluator: scripts/train_compare_precond.py's `searched_polar` arm (runs the candidate through
gns.fused) — no duplicated training loop. GPU-gated.

Run: PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src \
     uv run --project /home/jonas/git/tct-models python -u scripts/search_optimizer.py \
       --gens 6 --pop 8 --depth-small 6 --depth-large 12 --compile
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HARNESS = str(Path(__file__).resolve().parent / "train_compare_precond.py")
HARNESS_OUT = Path("/home/jonas/git/gns/results/precond_train_compare.json")
OUT = Path("/home/jonas/git/gns/results/exp27_optimizer_search.json")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--depth-small", type=int, default=6)
    p.add_argument("--depth-large", type=int, default=12)
    p.add_argument("--lambda-slope", type=float, default=1.0, help="penalty weight on advantage erosion")
    p.add_argument("--eval-iters", type=int, default=150)       # short eval per candidate
    p.add_argument("--transfer-iters", type=int, default=600)   # 4x horizon gate
    p.add_argument("--ladder-depths", type=str, default="6,8,12,16")  # scale-transfer gate
    p.add_argument("--matrix-lr", type=str, default="0.02")
    p.add_argument("--pop", type=int, default=8)
    p.add_argument("--gens", type=int, default=6)
    p.add_argument("--n-steps", type=int, default=5)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--restart", action="store_true",
                   help="ignore any existing checkpoint and start the search fresh")
    return p.parse_args()


OUT_CKPT = OUT.with_suffix(".ckpt.json")
_SEARCH_KEYS = ("depth_small", "depth_large", "eval_iters", "n_steps", "pop", "lambda_slope",
                "matrix_lr", "seed")


def _save_ckpt(state):
    """Atomic + durable checkpoint (tmp + fsync + os.replace). A transient FS error warns but does not
    crash the search — the previous good checkpoint stays valid."""
    try:
        tmp = OUT_CKPT.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(state, f, indent=1); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, OUT_CKPT)
    except OSError as e:
        print(f"  [warn] checkpoint save failed ({e}); continuing — previous checkpoint intact")


def _ser_cache(cache):
    return {f"{d},{i}": v for (d, i), v in cache.items()}


def _deser_cache(d):
    return {tuple(int(x) for x in k.split(",")): v for k, v in d.items()}


def load_ckpt(args):
    """Resume the search unless --restart. Refuses to SILENTLY overwrite: corrupt/mismatch is a hard
    error (use --restart to discard)."""
    if args.restart:
        if OUT_CKPT.exists():
            print(f"[restart] discarding existing search checkpoint {OUT_CKPT.name}")
        return None
    if not OUT_CKPT.exists():
        return None
    try:
        s = json.loads(OUT_CKPT.read_text())
    except Exception as e:
        raise SystemExit(f"[abort] search checkpoint {OUT_CKPT} unreadable/corrupt ({e}). "
                         f"Pass --restart to discard it.")
    old = s.get("config", {})
    if [old.get(k) for k in _SEARCH_KEYS] != [getattr(args, k) for k in _SEARCH_KEYS]:
        raise SystemExit("[abort] search checkpoint config differs from current args. "
                         "Pass --restart to discard it.")
    tmp = OUT_CKPT.with_suffix(".tmp")
    if tmp.exists():
        tmp.unlink()
    print(f"[resume] loaded {OUT_CKPT.name}")
    return s


# ---------------------------- grammar (width-covariant prior) ----------------------------
def random_schedule(rng, n_steps):
    # perturb around the cubic Newton–Schulz (1.5,-0.5,0); precision per step in {fp8e4m3,bf16}
    triples, precs = [], []
    for _ in range(n_steps):
        a = float(rng.uniform(1.2, 4.5)); b = float(rng.uniform(-3.0, -0.3)); c = float(rng.uniform(-0.1, 0.7))
        triples.append((a, b, c))
        precs.append("fp8e4m3" if rng.random() < 0.5 else "bf16")
    return triples, precs


CUBIC_NS = (1.5, -0.5, 0.0)        # canonical Newton–Schulz
MUON_QUINTIC = (3.4445, -4.775, 2.0315)  # Muon's tuned 5th-order step


def seed_population(n_steps, pop, rng):
    """Start the search from KNOWN-GOOD orthogonalisers (cubic NS + Muon quintic) and explore around
    them, rather than broad-random (which is mostly divergent -> filtered -> wasted evals). The goal is
    to find a schedule that *beats* Muon, so Muon's own iteration is the right starting point."""
    seeds = [([CUBIC_NS] * n_steps, ["bf16"] * n_steps),
             ([MUON_QUINTIC] * n_steps, ["bf16"] * n_steps)]
    out = [([tuple(x) for x in t], list(p)) for t, p in seeds]
    while len(out) < pop:
        out.append(mutate(seeds[int(rng.integers(len(seeds)))], rng))
    return out[:pop]


def mutate(sched, rng):
    triples, precs = [list(t) for t in sched[0]], list(sched[1])
    i = rng.integers(len(triples))
    j = rng.integers(3)
    triples[i][j] += float(rng.normal(0, 0.3))
    if rng.random() < 0.3:
        precs[i] = "fp8e4m3" if precs[i] == "bf16" else "bf16"
    return [tuple(t) for t in triples], precs


def scalar_stable(triples, hi=1.0):
    """Cheap divergence guard (no GPU). run_polar_2d pre-normalises its input (fused.py: Y/=Y.norm()*1.01),
    so the schedule acts on singular values s in (0,1] via s -> s*(a + b s^2 + c s^4). We reject only
    schedules that BLOW UP on that domain — orthogonalisation *quality* is judged by the two-scale
    fitness (a poor orthogonaliser simply yields poor loss), so the filter must not exclude valid
    Muon-family schedules (e.g. the quintic, which deliberately does not fix s=1). The width-covariant
    prior is carried by sampling around the cubic in random_schedule + this guard + fitness selection."""
    s = np.linspace(0.0, hi, 40)
    for (a, b, c) in triples:
        s = s * (a + b * s ** 2 + c * s ** 4)
        if not np.all(np.isfinite(s)) or np.max(np.abs(s)) > 5.0:
            return False
    return True


# ---------------------------- evaluator (reuses the harness) ----------------------------
def _run(depth, iters, args, polar=None):
    fd, out = tempfile.mkstemp(suffix=".json"); os.close(fd)  # unique per call -> race-free
    cmd = [sys.executable, "-u", HARNESS, "--depth", str(depth), "--num-iterations", str(iters),
           "--matrix-lr-grid", args.matrix_lr, "--eval-every", str(iters), "--out", out]
    if polar is None:
        cmd += ["--arms", "muon"]
    else:
        triples, precs = polar
        coeffs = ";".join(f"{a},{b},{c}" for a, b, c in triples)
        cmd += ["--arms", "searched_polar", "--polar-coeffs", coeffs, "--polar-precs", ",".join(precs)]
    if args.compile:
        cmd.append("--compile")
    env = {**os.environ, "PYTHONPATH": "/home/jonas/git/nanochat:/home/jonas/git/gns/src"}
    arm = "muon" if polar is None else "searched_polar"
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=3600, env=env)
        return float(json.loads(Path(out).read_text())["arms"][arm]["best_val"])
    except Exception as e:
        print(f"   run failed ({arm} d{depth}): {type(e).__name__}")
        return float("inf")
    finally:
        try:
            os.remove(out)
        except OSError:
            pass


def muon_baseline(args, depth, iters, cache):
    key = (depth, iters)
    if key not in cache:
        cache[key] = _run(depth, iters, args, polar=None)
        print(f"  muon baseline d{depth} x{iters}: {cache[key]:.4f}")
    return cache[key]


def fitness(sched, args, muon_cache):
    """Two-scale scalability fitness (minimise). Returns (F, info)."""
    if not scalar_stable(sched[0]):
        return float("inf"), {"reason": "unstable"}
    cand_s = _run(args.depth_small, args.eval_iters, args, polar=sched)
    cand_l = _run(args.depth_large, args.eval_iters, args, polar=sched)
    gap_s = cand_s - muon_baseline(args, args.depth_small, args.eval_iters, muon_cache)
    gap_l = cand_l - muon_baseline(args, args.depth_large, args.eval_iters, muon_cache)
    F = gap_l + args.lambda_slope * max(0.0, gap_l - gap_s)
    return F, {"gap_small": gap_s, "gap_large": gap_l, "cand_small": cand_s, "cand_large": cand_l}


# ---------------------------- gates on the winner ----------------------------
def scale_transfer_gate(sched, args, muon_cache):
    """Advantage must persist/grow across the depth ladder (slope of gap vs log width <= ~0)."""
    depths = [int(x) for x in args.ladder_depths.split(",")]
    rows = []
    for d in depths:
        cand = _run(d, args.transfer_iters, args, polar=sched)
        gap = cand - muon_baseline(args, d, args.transfer_iters, muon_cache)
        rows.append({"depth": d, "gap": gap})
        print(f"  scale-transfer d{d}: gap {gap:+.4f}")
    lw = np.log([64 * d if (64 * d) % 128 == 0 else (((64 * d) // 128) + 1) * 128 for d in depths])
    slope = float(np.polyfit(lw, [r["gap"] for r in rows], 1)[0]) if len(rows) > 1 else float("nan")
    return {"rows": rows, "gap_vs_logwidth_slope": slope, "pass": slope <= 0.0}


def main():
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    rng = np.random.default_rng(args.seed)
    ck = load_ckpt(args)
    gates = {}
    if ck:  # resume
        muon_cache = _deser_cache(ck["muon_cache"])
        pop = [([tuple(t) for t in tr], list(pr)) for tr, pr in ck["population"]]
        history = ck["history"]
        best = (ck["best"][0], ([tuple(t) for t in ck["best"][1][0]], ck["best"][1][1])) if ck.get("best") else None
        gates = ck.get("gates", {})
        rng.bit_generator.state = ck["rng_state"]
        if ck.get("in_gen") is not None:  # killed mid-generation
            start_gen = ck["in_gen"]
            resume_partial = [(F, ([tuple(t) for t in tr], pr)) for F, (tr, pr) in ck.get("partial", [])]
        else:
            start_gen = ck["gen_done"] + 1
            resume_partial = []
        print(f"[resume] start_gen={start_gen}, {len(resume_partial)} candidate(s) already scored in it")
    else:  # fresh — prime the two fitness baselines, seed from cubic NS + Muon quintic
        muon_cache = {}
        muon_baseline(args, args.depth_small, args.eval_iters, muon_cache)
        muon_baseline(args, args.depth_large, args.eval_iters, muon_cache)
        pop = seed_population(args.n_steps, args.pop, rng)
        history, best, start_gen, resume_partial = [], None, 0, []

    def snapshot(gen_done, in_gen, partial):
        return {"config": vars(args), "gen_done": gen_done, "in_gen": in_gen, "partial": partial,
                "population": pop, "history": history, "best": best, "gates": gates,
                "muon_cache": _ser_cache(muon_cache), "rng_state": rng.bit_generator.state}

    for gen in range(start_gen, args.gens):
        scored = resume_partial if gen == start_gen else []
        resume_partial = []
        for idx in range(len(scored), len(pop)):
            sched = pop[idx]
            F, info = fitness(sched, args, muon_cache)
            scored.append((F, sched))
            tag = info.get("reason", f"gap_s {info.get('gap_small', float('nan')):+.3f} "
                                     f"gap_l {info.get('gap_large', float('nan')):+.3f}")
            print(f"gen {gen} F {F:.4f}  {tag}  precs {sched[1]}")
            _save_ckpt(snapshot(gen - 1, gen, scored))  # per-candidate checkpoint
        scored.sort(key=lambda t: t[0])
        history.append({"gen": gen, "best_F": scored[0][0]})
        if best is None or scored[0][0] < best[0]:
            best = scored[0]
        elites = [s for _, s in scored[: max(2, args.pop // 4)]]
        pop = elites + [mutate(elites[int(rng.integers(len(elites)))], rng)
                        for _ in range(args.pop - len(elites))]
        _save_ckpt(snapshot(gen, None, []))  # generation-complete checkpoint
        print(f"[checkpoint] gen {gen} complete -> {OUT_CKPT.name}")

    # gates on the winner (each checkpointed so a stop resumes without re-running completed gates)
    print("\n=== gates on winner ===")
    if best is None or not np.isfinite(best[0]):
        print("no viable (stable, finite-fitness) candidate found — skipping gates")
        OUT.write_text(json.dumps({"config": vars(args), "history": history,
                                   "winner": None, "note": "no viable candidate"}, indent=1))
        return
    if "horizon" not in gates:
        hv = _run(args.depth_large, args.transfer_iters, args, polar=best[1])
        hm = muon_baseline(args, args.depth_large, args.transfer_iters, muon_cache)
        gates["horizon"] = {"val": hv, "muon": hm, "pass": bool(hv < hm)}
        _save_ckpt(snapshot(args.gens - 1, None, []))
    if "scale" not in gates:
        gates["scale"] = scale_transfer_gate(best[1], args, muon_cache)
        _save_ckpt(snapshot(args.gens - 1, None, []))
    result = {"config": vars(args), "history": history,
              "winner": {"triples": best[1][0], "precs": best[1][1], "best_F": best[0]},
              "gates": {"horizon_transfer": gates["horizon"], "scale_transfer": gates["scale"]}}
    OUT.write_text(json.dumps(result, indent=1))
    print(f"\nwinner F {best[0]:.4f}  horizon {'PASS' if gates['horizon']['pass'] else 'FAIL'}"
          f"  scale-slope {gates['scale']['gap_vs_logwidth_slope']:+.4f} "
          f"{'PASS' if gates['scale']['pass'] else 'FAIL'}  -> {OUT}")


if __name__ == "__main__":
    main()
