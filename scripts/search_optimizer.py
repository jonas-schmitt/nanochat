"""③ (moonshot) — search the optimizer (a polar Newton–Schulz schedule) against real training loss.

Evolutionary search over PolarSchedules (per-step Gram-poly triples a,b,c + precision), each candidate
scored by a SHORT real nanochat run via scripts/train_compare_precond.py's `searched_polar` arm (which
runs the candidate through gns.fused). Objective: minimise val loss at a fixed step budget (a wall-clock
term can be added once the fused fp8 timing of exp26 is in). The winner is then re-checked at a 4× horizon
to guard short-horizon overfit (Gate ③ = transfers and beats Muon there).

This is the high-variance research bet (P<10%); it reuses the harness as the evaluator so there is no
duplicated training loop. GPU-gated.

Run: PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src \
     uv run --project /home/jonas/git/tct-models python scripts/search_optimizer.py --gens 6 --pop 8
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

HARNESS = str(Path(__file__).resolve().parent / "train_compare_precond.py")
OUT = Path("/home/jonas/git/gns/results/exp27_optimizer_search.json")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--eval-iters", type=int, default=150)   # short eval per candidate
    p.add_argument("--transfer-iters", type=int, default=600)  # 4x for the transfer gate
    p.add_argument("--pop", type=int, default=8)
    p.add_argument("--gens", type=int, default=6)
    p.add_argument("--n-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def random_schedule(rng, n_steps):
    # perturb around the cubic Newton–Schulz (1.5,-0.5,0); precision per step in {fp8e4m3,bf16}
    triples, precs = [], []
    for _ in range(n_steps):
        a = float(rng.uniform(1.2, 4.5)); b = float(rng.uniform(-3.0, -0.3)); c = float(rng.uniform(-0.1, 0.7))
        triples.append((a, b, c))
        precs.append("fp8e4m3" if rng.random() < 0.5 else "bf16")
    return triples, precs


def mutate(sched, rng):
    triples, precs = [list(t) for t in sched[0]], list(sched[1])
    i = rng.integers(len(triples))
    j = rng.integers(3)
    triples[i][j] += float(rng.normal(0, 0.3))
    if rng.random() < 0.3:
        precs[i] = "fp8e4m3" if precs[i] == "bf16" else "bf16"
    return [tuple(t) for t in triples], precs


def evaluate(triples, precs, args, iters):
    coeffs = ";".join(f"{a},{b},{c}" for a, b, c in triples)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        outpath = f.name
    cmd = ["python", HARNESS, "--depth", str(args.depth), "--num-iterations", str(iters),
           "--arms", "searched_polar", "--matrix-lr-grid", "0.02", "--eval-every", str(iters),
           "--polar-coeffs", coeffs, "--polar-precs", ",".join(precs)]
    env = {"PYTHONPATH": "/home/jonas/git/nanochat:/home/jonas/git/gns/src"}
    import os
    env = {**os.environ, **env, "GNS_TRAIN_OUT": outpath}
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=1800, env=env)
        res = json.loads(Path("/home/jonas/git/gns/results/precond_train_compare.json").read_text())
        return float(res["arms"]["searched_polar"]["best_val"])
    except Exception as e:  # diverged / errored candidate
        print(f"   candidate failed: {type(e).__name__}")
        return float("inf")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    pop = [random_schedule(rng, args.n_steps) for _ in range(args.pop)]
    history = []
    best = None
    for gen in range(args.gens):
        scored = []
        for sched in pop:
            v = evaluate(sched[0], sched[1], args, args.eval_iters)
            scored.append((v, sched))
            print(f"gen {gen} val {v:.4f}  precs {sched[1]}")
        scored.sort(key=lambda t: t[0])
        history.append({"gen": gen, "best_val": scored[0][0]})
        if best is None or scored[0][0] < best[0]:
            best = scored[0]
        # next gen: elites + mutations
        elites = [s for _, s in scored[: max(2, args.pop // 4)]]
        pop = elites + [mutate(elites[int(rng.integers(len(elites)))], rng)
                        for _ in range(args.pop - len(elites))]

    # transfer gate: re-run the winner at 4x horizon
    transfer_val = evaluate(best[1][0], best[1][1], args, args.transfer_iters)
    OUT.write_text(json.dumps({
        "config": vars(args), "history": history,
        "winner": {"triples": best[1][0], "precs": best[1][1], "eval_val": best[0],
                   "transfer_val": transfer_val},
        "note": "Gate ③: compare transfer_val against a Muon run at the same horizon (run the harness "
                "with --arms muon --num-iterations <transfer-iters>).",
    }, indent=1))
    print(f"\nwinner eval {best[0]:.4f}  transfer {transfer_val:.4f}  -> {OUT}")


if __name__ == "__main__":
    main()
