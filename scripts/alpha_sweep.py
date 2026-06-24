"""alpha_sweep.py — Stage-2 probe: is there a useful Muon<->Shampoo MIDDLE?

The `synth` arm interpolates curvature strength: D = L^(-alpha/4) g R^(-alpha/4) (+ optional polar).
alpha=0 is Muon (no curvature), alpha=1 is full Shampoo inverse-root. This sweeps alpha in [0,1],
paired with muon in the same harness call (shared init + batches), and records the matched-step
gap(alpha, depth) = best_val(synth) - best_val(muon).

Decisive moonshot go/no-go: does any INTERIOR alpha (0<alpha<1) beat BOTH endpoints by more than the
~4e-4 GPU-nondeterminism floor? If yes, partial curvature is a real, unexplored optimum and the full
spectral-shaping grammar search is justified. If the gap is monotone in alpha (no interior win), the
"useful middle" premise is weak — a cheap kill.

Resumable: each (alpha, depth) is a separate harness call with a unique --out; the harness itself
checkpoints per (arm,lr), so a restart re-finalizes completed cells in seconds.

Run:
  cd /home/jonas/git/nanochat
  PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src \
    uv run --project /home/jonas/git/tct-models python -u scripts/alpha_sweep.py \
      --alphas 0,0.25,0.5,0.75,1.0 --depths 6,12 --num-iterations 1500 --compile --tag a1
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HARNESS = str(Path(__file__).resolve().parent / "train_compare_precond.py")
RES = Path("/home/jonas/git/gns/results")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0")
    p.add_argument("--depths", default="6,12")
    p.add_argument("--num-iterations", type=int, default=1500)
    p.add_argument("--matrix-lr-grid", default="0.01,0.02")
    p.add_argument("--synth-ortho", type=int, default=1)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--tag", default="a1")
    return p.parse_args()


def run_cell(alpha, depth, args):
    out = RES / f"precond_alpha_{args.tag}_a{alpha}_d{depth}.json"
    cmd = [sys.executable, "-u", HARNESS, "--depth", str(depth), "--num-iterations",
           str(args.num_iterations), "--arms", "muon,synth", "--matrix-lr-grid", args.matrix_lr_grid,
           "--synth-alpha", str(alpha), "--synth-ortho", str(args.synth_ortho), "--out", str(out)]
    if args.compile:
        cmd.append("--compile")
    env = {**os.environ, "PYTHONPATH": "/home/jonas/git/nanochat:/home/jonas/git/gns/src"}
    subprocess.run(cmd, check=True, env=env)
    a = json.loads(out.read_text())["arms"]
    return a["synth"]["best_val"] - a["muon"]["best_val"], a["synth"]["best_val"], a["muon"]["best_val"]


def main():
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    alphas = [float(x) for x in args.alphas.split(",")]
    depths = [int(x) for x in args.depths.split(",")]
    gaps = {}
    for d in depths:
        print(f"\n===== depth {d} =====")
        for al in alphas:
            gap, sv, mv = run_cell(al, d, args)
            gaps[(al, d)] = gap
            print(f"  alpha {al:.2f} d{d}: gap {gap:+.5f}  (synth {sv:.4f}, muon {mv:.4f})")
            # persist after each cell (re-derivable, but cheap and live)
            (RES / f"alpha_sweep_{args.tag}.json").write_text(json.dumps(
                {"config": vars(args), "gaps": {f"a{a}_d{dp}": gaps[(a, dp)] for (a, dp) in gaps}}, indent=1))
        interior = [a for a in alphas if 0 < a < 1]
        best = min(alphas, key=lambda a: gaps[(a, d)])
        best_int = min(interior, key=lambda a: gaps[(a, d)]) if interior else None
        msg = f"  d{d} VERDICT: best alpha={best} gap {gaps[(best, d)]:+.5f}"
        if best_int is not None:
            beats_both = gaps[(best_int, d)] < min(gaps[(0.0, d)], gaps[(1.0, d)]) - 5e-4 \
                if (0.0, d) in gaps and (1.0, d) in gaps else None
            msg += f"; best interior alpha={best_int} gap {gaps[(best_int, d)]:+.5f}  beats_both_endpoints={beats_both}"
        print(msg)
    print(f"\n[saved] {RES / f'alpha_sweep_{args.tag}.json'}")


if __name__ == "__main__":
    main()
