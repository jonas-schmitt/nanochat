"""Tier-1 delta-replay evaluator (TODO.md "RE-ENTRY SEARCH ARCHITECTURE").

Evaluates wire/outer PolicyGenome variants against a recorded anchor run (train_diloco.py
--record-deltas) WITHOUT retraining: replays the recorded RAW per-round per-worker deltas
through the candidate's wire (error-feedback -> basis -> quantize) and outer step, using the
SAME functions the simulator runs (policy_grammar.compress_delta / outer_direction,
train_diloco._outer_transform_fn) — single source of truth.

VALIDITY (off-policy evaluation — trust is gene-family dependent, see TODO):
  quantizer/bits/basis/EF genes: perturbative around the anchor trajectory -> screening-grade.
  outer-transform/outer-lr genes: large trajectory divergence -> DOUBTFUL, rank-gate first.
  h / inner / workers: change the stream itself -> MUST match the anchor (asserted).

Self-consistency gate (run before trusting anything):
  python scripts/replay_policy.py --recording R.pt --gate
replays the anchor's own genome and compares the val trace to the recorded one (fp32-recorded
deltas + identical ops => must match to ~kernel-nondeterminism tolerance).
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/jonas/git/gns/src")

import train_compare_precond as tcp  # noqa: E402
import train_diloco as td  # noqa: E402
from gns.policy_grammar import (  # noqa: E402
    canonical,
    compress_delta,
    from_dict as genome_from_dict,
    init_outer_state,
    outer_direction,
)
from gns.temporal_grammar import lookahead_sync as tg_lookahead_sync  # noqa: E402


def load_recording(path):
    rec = torch.load(path, map_location="cpu", weights_only=False)
    rec["anchor_genome"] = genome_from_dict(rec["genome"])
    return rec


def rec_args(rec, **over):
    """Namespace with the recorded run's config (model/regime/outer-factor knobs) + overrides."""
    a = SimpleNamespace(**rec["config"])
    for k, v in over.items():
        setattr(a, k, v)
    return a


def replay(rec, genome, args, model, val_batches, device, eval_every=None):
    """Replay the recorded delta stream under `genome`. Returns a log dict like run_policy's.

    eval_every=None evaluates ONLY after the final round (search mode); an int mirrors the
    simulator's round-end eval cadence (gate mode).
    """
    anchor = rec["anchor_genome"]
    workers = len(rec["rounds"][0])
    assert genome.h == anchor.h, f"h must match the anchor ({anchor.h}), got {genome.h}"
    assert genome.inner == anchor.inner, "inner optimizer genes are baked into the stream"
    all_params = list(model.parameters())
    mp = tcp.matrix_params(model)
    mp_ids = {id(p): j for j, p in enumerate(mp)}

    theta = [t.to(device).float().clone() for t in rec["theta0"]]
    ef_acc: list[dict[int, torch.Tensor]] = [{} for _ in range(workers)]
    gens = [torch.Generator(device=device).manual_seed(10_000 + m) for m in range(workers)]
    outer_state = [init_outer_state(genome, t) for t in theta]
    outer_factors = {j: {"L": torch.zeros(p.shape[0], p.shape[0], device=device),
                         "R": torch.zeros(p.shape[1], p.shape[1], device=device),
                         "Linv": None, "Rinv": None}
                     for j, p in enumerate(mp)} if genome.outer_transform == "whitened" else {}
    transform = td._outer_transform_fn(genome, args, outer_factors)

    n_params_synced = sum(p.numel() for p in all_params)
    log = {"step": [], "val": [], "comm_bits": []}

    def evaluate(step_idx):
        for p, t in zip(all_params, theta):
            p.data.copy_(t.to(p.dtype))
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([float(model(vx, vy).item()) for vx, vy in val_batches]))
        model.train()
        bits = genome.comm_bits_per_param_step() * n_params_synced * step_idx * workers
        log["step"].append(step_idx); log["val"].append(vl); log["comm_bits"].append(bits)

    step_now = 0
    next_eval = eval_every if eval_every else None
    for r, worker_deltas in enumerate(rec["rounds"]):
        steps_this = rec["steps_per_round"][r]
        delta_sum = [torch.zeros_like(t) for t in theta]
        with torch.no_grad():
            for m, deltas in enumerate(worker_deltas):
                for i, raw in enumerate(deltas):
                    raw = raw.to(device=device, dtype=torch.float32)
                    sent, new_acc = compress_delta(raw, genome, ef_acc[m].get(i), gen=gens[m])
                    if new_acc is not None:
                        ef_acc[m][i] = new_acc
                    delta_sum[i] += sent
            for i, t in enumerate(theta):
                mean_delta = delta_sum[i] / workers
                j = mp_ids.get(id(all_params[i]))
                if j is not None and genome.outer_transform == "whitened":
                    st = outer_factors[j]
                    st["L"].mul_(args.outer_factor_beta).add_(mean_delta @ mean_delta.t(),
                                                              alpha=1 - args.outer_factor_beta)
                    st["R"].mul_(args.outer_factor_beta).add_(mean_delta.t() @ mean_delta,
                                                              alpha=1 - args.outer_factor_beta)
                    st["Linv"] = tcp.inv_fourth_root(st["L"], args.outer_factor_ridge,
                                                     args.outer_factor_coupled_steps)
                    st["Rinv"] = tcp.inv_fourth_root(st["R"], args.outer_factor_ridge,
                                                     args.outer_factor_coupled_steps)
                d = outer_direction(genome, mean_delta, outer_state[i],
                                    lambda name, x, _j=j: transform(name, x, key=_j,
                                                                    is_matrix=_j is not None))
                t += genome.outer_lr * d
                if genome.outer.lookahead_levels:
                    tg_lookahead_sync(genome.outer, t, outer_state[i], r + 1)
        step_now += steps_this
        if next_eval is not None and next_eval <= step_now:
            evaluate(step_now)
            while next_eval <= step_now:
                next_eval += eval_every
    if not log["step"] or log["step"][-1] != step_now:
        evaluate(step_now)
    return log


def build_eval_context(rec, device):
    """Model + val batches exactly as the recording run built them (same seed/protocol)."""
    a = rec_args(rec)
    tok = tcp.get_tokenizer(); vocab = tok.get_vocab_size()
    loader = tcp.tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tok, a.device_batch_size, a.max_seq_len, split="val", device=device,
        resume_state_dict=None)
    val_batches = []
    for _ in range(a.n_val_batches):
        x, y, _ = next(loader)
        val_batches.append((x.clone(), y.clone()))
    model, _ = tcp.build_model(a, vocab, device, a.seed)
    return model, val_batches


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--recording", type=str, required=True)
    p.add_argument("--gate", action="store_true",
                   help="self-consistency: replay the anchor genome, compare to the recorded trace")
    p.add_argument("--gate-tol", type=float, default=5e-4)
    p.add_argument("--genome-json", type=str, default="", help="candidate genome (default: anchor)")
    p.add_argument("--eval-every", type=int, default=0, help="0 = final-round val only")
    p.add_argument("--out", type=str, default="")
    cli = p.parse_args()

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = "cuda"
    rec = load_recording(cli.recording)
    genome = (genome_from_dict(json.loads(cli.genome_json)) if cli.genome_json
              else rec["anchor_genome"])
    args = rec_args(rec)
    model, val_batches = build_eval_context(rec, device)

    eval_every = cli.eval_every or (args.eval_every if cli.gate else 0) or None
    t0 = time.time()
    log = replay(rec, genome, args, model, val_batches, device, eval_every=eval_every)
    dt = time.time() - t0
    print(f"replayed {canonical(genome)}")
    print(f"  final val {log['val'][-1]:.4f}  comm {log['comm_bits'][-1]/8e9:.2f}GB  ({dt:.1f}s)")

    if cli.gate:
        ref = dict(zip(rec["log"]["step"], rec["log"]["val"]))
        got = dict(zip(log["step"], log["val"]))
        common = sorted(set(ref) & set(got))
        assert common, f"no common eval steps: {sorted(ref)} vs {sorted(got)}"
        worst = max(abs(ref[s] - got[s]) for s in common)
        print(f"  GATE: {len(common)} common eval points, max |dval| = {worst:.2e} "
              f"(tol {cli.gate_tol:.0e}) -> {'PASS' if worst <= cli.gate_tol else 'FAIL'}")
        if worst > cli.gate_tol:
            sys.exit(1)
    if cli.out:
        json.dump({"genome": canonical(genome), "log": log, "replay_seconds": dt},
                  open(cli.out, "w"), indent=1)


if __name__ == "__main__":
    main()
