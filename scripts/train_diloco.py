"""DiLoCo-family simulator: sequential M-worker low-communication training on ONE GPU.

The fused-track harness (gns TODO "THE FUSED TRACK"): executes a gns.policy_grammar.PolicyGenome
— {sync period/levels × delta precision/basis/error-feedback × outer geometry/recurrence × inner
optimizer} — by simulating M DiLoCo workers SEQUENTIALLY. The algorithm is exactly the one a real
cluster would run (parallelism is only a speedup, not part of the math), so quality-vs-comm-bits
claims transfer; wall-clock/bandwidth claims do NOT (out of scope by design).

Structure per round: for each worker m — load θ_global, restore m's inner state, run H inner
steps on m's OWN data shard, form the delta Δ_m = θ_m − θ_global, push it through the simulated
wire (error-feedback accumulate → basis → quantize → inverse basis; gns.policy_grammar.
compress_delta) — then one OUTER step on the mean delta (gns.policy_grammar.outer_direction:
temporal recurrence over rounds × {sgd | polar "outer-Muon" | whitened} transform).

Everything inner-step reuses train_compare_precond (imported as a module — the same bytes the
incumbent arms run): build_model, matrix_params, _nesterov, polar_express_orth,
apply_norm_caution_update, _build_other_adam, inv_fourth_root, data materialisation, VAL protocol,
results-JSON shape.

SANITY GATES (bit-exactness by construction — deltas and outer updates are computed in fp32,
where bf16 differences and their re-addition are exact):
  G-A1: M=1, H=1, outer=sgd lr=1 β=0, fp32 wire  ==  the `muon` arm's val trace.
  G-A2: M=1, H=k, outer=sgd lr=α β=0, fp32 wire  ==  the `muon_lookahead` arm (Lookahead IS
        1-worker DiLoCo: slow += α(fast − slow); fast ← slow  ≡  θ += α·Δ; workers restart at θ).
  G-A3: bits=32 wire (any basis/EF setting) == the no-quantization path (exact pass-through).
Run them with scripts/diloco_gates.sh; comparator scripts/check_diloco_gates.py.

Iso-token convention (papers'): --num-iterations = inner steps PER WORKER; total tokens =
M × iterations × batch. DP anchors are the plain arms at M× batch. Comm bits are reported
exactly via genome.comm_bits_total.

Run (tct-models env + gns on path), e.g. tuned MuLoCo incumbent at d6:
  cd /home/jonas/git/nanochat
  PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src PYTHONUNBUFFERED=1 \
    uv run --project /home/jonas/git/tct-models python scripts/train_diloco.py \
      --depth 6 --num-iterations 1500 --workers 4 --preset muloco --matrix-lr 0.02
"""
import argparse
import json
import math
import sys
import time
from dataclasses import replace as dc_replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/jonas/git/gns/src")

import train_compare_precond as tcp  # noqa: E402  — the incumbent harness, reused verbatim
from gns.policy_grammar import (  # noqa: E402
    PolicyGenome,
    canonical,
    compress_delta,
    diloco_genome,
    dp_genome,
    from_dict as genome_from_dict,
    init_outer_state,
    muloco_2bit_genome,
    muloco_genome,
    outer_direction,
    to_dict as genome_to_dict,
)
from gns.quant_model import DeltaStats  # noqa: E402
from gns.temporal_grammar import TemporalGenome, lookahead_sync as tg_lookahead_sync  # noqa: E402

GNS_OUT = Path("/home/jonas/git/gns/results/diloco_run.json")

PRESETS = {
    "dp": dp_genome,
    "diloco": diloco_genome,
    "muloco": muloco_genome,
    "muloco2bit": muloco_2bit_genome,
}


def parse_args():
    p = argparse.ArgumentParser()
    # model/regime (mirrors train_compare_precond defaults)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--aspect-ratio", type=int, default=64)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--device-batch-size", type=int, default=16)
    p.add_argument("--num-iterations", type=int, default=1500, help="inner steps PER WORKER")
    p.add_argument("--matrix-lr", type=float, default=0.02)
    p.add_argument("--adam-lr", type=float, default=3e-3)
    p.add_argument("--momentum", type=float, default=0.95)
    p.add_argument("--beta2", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.28)
    p.add_argument("--ns-steps", type=int, default=5)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--n-val-batches", type=int, default=16)
    p.add_argument("--eval-every", type=int, default=50, help="in INNER steps (aligned to round ends)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="")
    # policy genome: preset + overrides (overrides only applied when the flag is given)
    p.add_argument("--preset", type=str, default="muloco", choices=tuple(PRESETS))
    p.add_argument("--genome-json", type=str, default="", help="full PolicyGenome dict (overrides preset+flags)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--h", type=int, default=-1, help="sync period in inner steps (-1: preset's)")
    p.add_argument("--delta-bits", type=int, default=-1)
    p.add_argument("--basis", type=str, default="")
    p.add_argument("--error-feedback", type=int, default=-1, help="0/1 (-1: preset's)")
    p.add_argument("--ef-beta", type=float, default=-1.0)
    p.add_argument("--stochastic-rounding", type=int, default=-1, help="0/1 (-1: preset's)")
    p.add_argument("--outer-transform", type=str, default="")
    p.add_argument("--outer-lr", type=float, default=-1.0)
    p.add_argument("--outer-beta", type=float, default=-1.0, help="single-timescale outer momentum β (EMA form)")
    p.add_argument("--outer-nesterov", type=int, default=-1, help="0/1 (-1: preset's)")
    p.add_argument("--outer-whiten-power", type=float, default=-1.0)
    p.add_argument("--inner", type=str, default="", choices=("", "muon", "adamw"))
    # whitened-outer factor maintenance
    p.add_argument("--outer-factor-beta", type=float, default=0.9, help="EMA over rounds for outer L,R")
    p.add_argument("--outer-factor-ridge", type=float, default=1e-4)
    p.add_argument("--outer-factor-coupled-steps", type=int, default=24)
    return p.parse_args()


def build_genome(args) -> PolicyGenome:
    if args.genome_json:
        return genome_from_dict(json.loads(args.genome_json))
    g = PRESETS[args.preset]()
    if args.h > 0:
        g = dc_replace(g, h=args.h)
    if args.delta_bits > 0:
        g = dc_replace(g, delta_bits=args.delta_bits)
    if args.basis:
        g = dc_replace(g, basis=args.basis)
    if args.error_feedback >= 0:
        g = dc_replace(g, error_feedback=bool(args.error_feedback))
    if args.ef_beta >= 0:
        g = dc_replace(g, ef_beta=args.ef_beta)
    if args.stochastic_rounding >= 0:
        g = dc_replace(g, stochastic_rounding=bool(args.stochastic_rounding))
    if args.outer_transform:
        g = dc_replace(g, outer_transform=args.outer_transform)
    if args.outer_lr > 0:
        g = dc_replace(g, outer_lr=args.outer_lr)
    if args.outer_whiten_power > 0:
        g = dc_replace(g, outer_whiten_power=args.outer_whiten_power)
    if args.outer_beta >= 0 or args.outer_nesterov >= 0:
        beta = args.outer_beta if args.outer_beta >= 0 else g.outer.momentum_betas[0]
        nest = bool(args.outer_nesterov) if args.outer_nesterov >= 0 else g.outer.nesterov
        g = dc_replace(g, outer=TemporalGenome(momentum_betas=(beta,), nesterov=nest))
    if args.inner:
        g = dc_replace(g, inner=args.inner)
    return g


class Worker:
    """One simulated DiLoCo worker: its inner-optimizer state + data shard + wire state.

    The MODEL object is shared (θ is loaded per turn); param identity therefore stays stable,
    so per-worker torch.optim.AdamW instances keep their own state across rounds correctly.
    """

    def __init__(self, idx, model, mp, mp_set, args, genome, device):
        self.idx = idx
        self.mom = [torch.zeros_like(p) for p in mp]                      # inner Muon momentum
        self.norm_state = [{"mom": self.mom[j], "v2": None} for j in range(len(mp))]
        self.adam = tcp._build_other_adam(model, mp_set)                  # non-matrix params
        self.madam = (torch.optim.AdamW(mp, lr=args.matrix_lr, betas=(0.9, 0.95), weight_decay=0.01)
                      if genome.inner == "adamw" else None)
        self.ef_acc: dict[int, torch.Tensor] = {}                        # error-feedback per param
        # stochastic-rounding RNG must live on the DELTAS' device (torch.rand device must match)
        self.gen = torch.Generator(device=device).manual_seed(10_000 + idx)
        self.steps_done = 0


def _outer_transform_fn(genome, args, outer_factors):
    """Builds transform_fn(name, delta, key, is_matrix) closures for outer_direction."""

    def fn(name, delta, key=None, is_matrix=False):
        if name == "sgd" or not is_matrix:
            return delta
        if name == "polar":
            # outer-Muon: orthogonalize the mean pseudo-gradient; norm-matched to ‖Δ‖_F so
            # outer_lr keeps the same meaning across transforms.
            d = tcp.polar_express_orth(delta.bfloat16(), args.ns_steps).float()
            if not torch.isfinite(d).all():
                return delta
            return d * (delta.norm() / d.norm().clamp_min(1e-12))
        if name == "whitened":
            st = outer_factors.get(key)
            if st is None or st.get("Linv") is None:
                return delta
            k = int(round(4 * genome.outer_whiten_power))
            d = delta
            for _ in range(k):
                d = st["Linv"] @ d
            for _ in range(k):
                d = d @ st["Rinv"]
            return d * (delta.norm() / d.norm().clamp_min(1e-12))
        raise ValueError(name)

    return fn


def run_policy(genome, args, model, train_shards, val_batches, device):
    mp = tcp.matrix_params(model)
    mp_set = {id(p) for p in mp}
    all_params = list(model.parameters())
    mp_ids = {id(p): j for j, p in enumerate(mp)}

    theta = [p.detach().float().clone() for p in all_params]              # θ_global, fp32 master
    workers = [Worker(m, model, mp, mp_set, args, genome, device) for m in range(args.workers)]
    outer_state = [init_outer_state(genome, t) for t in theta]            # recurrence over rounds
    outer_factors = {j: {"L": torch.zeros(p.shape[0], p.shape[0], device=device),
                         "R": torch.zeros(p.shape[1], p.shape[1], device=device),
                         "Linv": None, "Rinv": None}
                     for j, p in enumerate(mp)} if genome.outer_transform == "whitened" else {}
    transform = _outer_transform_fn(genome, args, outer_factors)

    h = genome.h
    n_rounds = math.ceil(args.num_iterations / h)
    delta_stats_log: list[tuple] = []
    log = {"step": [], "val": [], "wall_ms": [], "comm_bits": []}
    n_params_synced = sum(p.numel() for p in all_params)
    ev_start = torch.cuda.Event(enable_timing=True); ev_start.record()
    # Evals fire at round ends when an eval_every boundary was crossed, labeled with the ACTUAL
    # inner step (θ_global only changes at round ends). When h divides eval_every the labels align
    # exactly with train_compare_precond's cadence — required by the G-A1/G-A2 comparators.
    next_eval = args.eval_every

    def evaluate(step_idx):
        for p, t in zip(all_params, theta):
            p.data.copy_(t.to(p.dtype))
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([float(model(vx, vy).item()) for vx, vy in val_batches]))
        model.train()
        torch.cuda.synchronize()
        ev_now = torch.cuda.Event(enable_timing=True); ev_now.record(); torch.cuda.synchronize()
        bits = genome.comm_bits_per_param_step() * n_params_synced * step_idx * args.workers
        log["step"].append(step_idx); log["val"].append(vl)
        log["wall_ms"].append(ev_start.elapsed_time(ev_now)); log["comm_bits"].append(bits)
        print(f"  [diloco M{args.workers} h{h}] step {step_idx:4d}/{args.num_iterations}  "
              f"val {vl:.4f}  comm {bits/8e9:.2f}GB  ({log['wall_ms'][-1]/1000:.1f}s)")

    for r in range(n_rounds):
        steps_this = min(h, args.num_iterations - r * h)
        delta_sum = [torch.zeros_like(t) for t in theta]
        for w in workers:
            for p, t in zip(all_params, theta):                           # θ ← θ_global
                p.data.copy_(t.to(p.dtype))
            for j in range(steps_this):
                step = r * h + j + 1
                x, y = train_shards[w.idx][step - 1]
                loss = model(x, y)
                model.zero_grad(set_to_none=True)
                w.adam.zero_grad(set_to_none=True)
                loss.backward()
                lrm = min(1.0, step / max(1, args.warmup_steps))
                cos_wd = args.weight_decay * 0.5 * (1 + math.cos(math.pi * step / args.num_iterations))
                with torch.no_grad():
                    if genome.inner == "muon":
                        for jj, p in enumerate(mp):
                            if p.grad is None:
                                continue
                            gm = tcp._nesterov(p.grad, w.norm_state[jj], args.momentum)
                            D = tcp.polar_express_orth(gm, args.ns_steps)
                            if not torch.isfinite(D).all():
                                D = p.grad.lerp(w.norm_state[jj]["mom"], args.momentum)
                            tcp.apply_norm_caution_update(D, p, w.norm_state[jj],
                                                          args.matrix_lr * lrm, cos_wd, args.beta2)
                for gpar in w.adam.param_groups:
                    gpar["lr"] = gpar["base_lr"] * lrm
                w.adam.step()
                if w.madam is not None:
                    for gpar in w.madam.param_groups:
                        gpar["lr"] = args.matrix_lr * lrm
                    w.madam.step()
                w.steps_done += 1
            with torch.no_grad():                                          # Δ_m through the wire
                for i, (p, t) in enumerate(zip(all_params, theta)):
                    raw = p.detach().float() - t
                    if w.idx == 0 and id(p) in mp_ids and len(delta_stats_log) < 400:
                        # C0 evidence: the REAL pseudo-gradient outlier statistics the rounding
                        # model's rotation prediction is conditioned on (worker 0, matrix params)
                        s = DeltaStats.measure(raw)
                        delta_stats_log.append((s.absmax_over_std, s.rho_hit, s.hit_fraction,
                                                s.outlier_energy_fraction))
                    sent, new_acc = compress_delta(raw, genome, w.ef_acc.get(i), gen=w.gen)
                    if new_acc is not None:
                        w.ef_acc[i] = new_acc
                    delta_sum[i] += sent
        with torch.no_grad():                                              # outer step on mean Δ
            step_idx = r * h + steps_this
            for i, t in enumerate(theta):
                mean_delta = delta_sum[i] / args.workers
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
                # outer multilevel lookahead (V-cycle over ROUNDS), if the genome carries levels
                if genome.outer.lookahead_levels:
                    tg_lookahead_sync(genome.outer, t, outer_state[i], r + 1)
        step_now = r * h + steps_this
        if next_eval <= step_now:
            evaluate(step_now)
            while next_eval <= step_now:
                next_eval += args.eval_every
    if delta_stats_log:
        arr = np.array(delta_stats_log)
        log["delta_stats"] = {"rho_clean": float(arr[:, 0].mean()), "rho_hit": float(arr[:, 1].mean()),
                              "hit_fraction": float(arr[:, 2].mean()),
                              "outlier_energy_fraction": float(arr[:, 3].mean()),
                              "n_measured": len(delta_stats_log)}
        print(f"  delta stats (worker0 matrix deltas, n={len(delta_stats_log)}): "
              f"rho_clean {log['delta_stats']['rho_clean']:.2f}  rho_hit {log['delta_stats']['rho_hit']:.2f}  "
              f"hit {log['delta_stats']['hit_fraction']:.2f}  w {log['delta_stats']['outlier_energy_fraction']:.2f}")
    return log


def main():
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    _, _, _, world, device = tcp.compute_init("cuda")
    assert world == 1
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    import os
    if os.environ.get("GNS_DETERMINISTIC") == "1":
        # same hook as train_compare_precond: bit-reproducible mode for the equivalence gates
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("[deterministic] GNS_DETERMINISTIC=1: deterministic algorithms + math SDPA")
    genome = build_genome(args)
    print(f"policy: {canonical(genome)}")
    print(f"comm: {genome.comm_bits_per_param_step():.4f} bits/param/step "
          f"({genome.comm_reduction_vs_dp():.0f}x less than DP)")

    tok = tcp.get_tokenizer(); vocab = tok.get_vocab_size()

    def materialise(split, n, resume_state_dict=None):
        loader = tcp.tokenizing_distributed_data_loader_with_state_bos_bestfit(
            tok, args.device_batch_size, args.max_seq_len, split=split, device=device,
            resume_state_dict=resume_state_dict)
        out = []
        for _ in range(n):
            x, y, _ = next(loader)
            out.append((x.clone(), y.clone()))
        return out

    # Disjoint per-worker data shards: worker m starts at parquet shard seed+m (worker 0 with
    # seed 0 uses the default stream == train_compare_precond's, which the G-A1/G-A2 gates need).
    train_shards = []
    for m in range(args.workers):
        off = args.seed + m
        resume = {"pq_idx": off, "rg_idx": 0, "epoch": 1} if off else None
        train_shards.append(materialise("train", args.num_iterations, resume_state_dict=resume))
    val_batches = materialise("val", args.n_val_batches)
    print(f"depth {args.depth}, vocab {vocab}, {args.workers}x{args.num_iterations} train + "
          f"{len(val_batches)} val batches")

    model, _ = tcp.build_model(args, vocab, device, args.seed)
    t0 = time.time()
    log = run_policy(genome, args, model, train_shards, val_batches, device)

    out_path = Path(args.out) if args.out else GNS_OUT
    n_par = sum(p.numel() for p in model.parameters())
    cfg = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    cfg["genome"] = genome_to_dict(genome)
    cfg["genome_canonical"] = canonical(genome)
    cfg["n_params"] = n_par
    result = {"config": cfg,
              "best_val": min(log["val"]), "final_val": log["val"][-1],
              "total_wall_s": log["wall_ms"][-1] / 1000,
              "total_comm_bits": log["comm_bits"][-1],
              "comm_reduction_vs_dp": genome.comm_reduction_vs_dp(),
              "curve": log,
              "runtime_min": (time.time() - t0) / 60.0}
    tcp._atomic_write(out_path, result)
    print(f"\nwrote {out_path}  best val {result['best_val']:.4f}  "
          f"comm {result['total_comm_bits']/8e9:.2f}GB  ({result['runtime_min']:.1f} min)")
    tcp.compute_cleanup()


if __name__ == "__main__":
    main()
