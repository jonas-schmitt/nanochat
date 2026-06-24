"""SOTA-hunt harness: fair matrix-preconditioner comparison inside a real nanochat GPT run.

Every arm shares Muon's EXACT post-direction machinery — Nesterov momentum, NorMuon variance
reduction, and cautious weight decay/update (lifted from nanochat.optim.muon_step_unfused) — and
varies ONLY the matrix DIRECTION map D(g, state). So the `muon` arm is the real SOTA Muon (not bare
polar + an RMS-norm hack), and every other arm is judged against it on equal footing.

Direction maps:
  sgd            : D = nesterov(g)                                        (no-precond floor)
  muon           : D = polar_express_orth(nesterov(g))                    (SOTA baseline)
  shampoo        : D = L^(-1/4) @ nesterov(g) @ R^(-1/4)                  (Shampoo via gns.coupled)
  ortho_shampoo  : D = polar_express_orth( L^(-1/4) g R^(-1/4) )          (① curvature dir + Muon robustness)
  layer_adaptive : per-factor: ortho_shampoo if kappa_proxy > thresh else muon   (② allocation)

L,R = EMA Kronecker factors (G Gᵀ, Gᵀ G); L^(-1/4) by the coupled Newton–Schulz production (gns.coupled,
fp32, power-iteration λmax, relative ridge), recomputed every K steps. Non-matrix params: shared AdamW.
Identical model init + pre-materialised climbmix batches across arms; per-arm matrix-LR sweep (fair
tuning). Logs val loss vs BOTH step and wall-clock.

Run (tct-models env + gns on path):
  cd /home/jonas/git/nanochat
  PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src \
    uv run --project /home/jonas/git/tct-models python scripts/train_compare_precond.py \
      --depth 6 --num-iterations 2000 --arms muon,shampoo,ortho_shampoo,layer_adaptive,sgd \
      --matrix-lr-grid 0.01,0.02,0.04
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from nanochat.common import compute_init, compute_cleanup
from nanochat.gpt import GPT, GPTConfig
from nanochat.optim import polar_express_orth
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit

import sys
sys.path.insert(0, "/home/jonas/git/gns/src")
from gns import coupled  # noqa: E402
from gns.coupled import CoupledInit, CoupledStep  # noqa: E402
from gns.fused import PolarStep, run_polar_2d  # noqa: E402
from gns.precision import TORCH_DTYPE, Prec  # noqa: E402

GNS_OUT = Path("/home/jonas/git/gns/results/precond_train_compare.json")


def _atomic_write(path, payload):
    """Crash-safe JSON write: tmp + fsync + os.replace (never leaves a partial/corrupt file)."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--aspect-ratio", type=int, default=64)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--device-batch-size", type=int, default=16)
    p.add_argument("--num-iterations", type=int, default=2000)
    p.add_argument("--matrix-lr-grid", type=str, default="0.02")
    p.add_argument("--adam-lr", type=float, default=3e-3)
    p.add_argument("--momentum", type=float, default=0.95)
    p.add_argument("--beta2", type=float, default=0.95)          # NorMuon second-moment EMA
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--ns-steps", type=int, default=5)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--shampoo-beta", type=float, default=0.95)
    p.add_argument("--shampoo-ridge", type=float, default=1e-4)
    p.add_argument("--shampoo-coupled-steps", type=int, default=24)
    p.add_argument("--shampoo-recompute-every", type=int, default=10)
    p.add_argument("--kappa-threshold", type=float, default=1e4)  # layer_adaptive routing
    p.add_argument("--synth-alpha", type=float, default=1.0,
                   help="synth arm: curvature strength in [0,1]. D = L^(-alpha/4) g R^(-alpha/4); "
                        "alpha=0 -> Muon (no curvature), alpha=1 -> full Shampoo inverse-root. The "
                        "alpha in (0,1) interior is the unexplored Muon<->Shampoo spectral middle.")
    p.add_argument("--synth-ortho", type=int, default=1,
                   help="synth arm: 1 = polar-orthogonalize the (curvature-shaped) direction (Muon "
                        "robustness on top), 0 = raw Shampoo-style direction.")
    p.add_argument("--n-val-batches", type=int, default=16)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--arms", type=str, default="muon,shampoo,ortho_shampoo,layer_adaptive,sgd")
    p.add_argument("--polar-coeffs", type=str, default="",
                   help="③ searched_polar arm: ';'-separated 'a,b,c' Gram-poly triples")
    p.add_argument("--polar-precs", type=str, default="",
                   help="③ per-step precision (comma list, e.g. fp8e4m3,bf16,...); default all bf16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the model fwd/bwd (math-preserving; speeds every arm equally, "
                        "so fairness/wall-clock gate stay valid). Worth it for the heavier d12 runs.")
    p.add_argument("--out", type=str, default="",
                   help="output JSON path (default: the shared GNS_OUT). Per-(arm,lr) results are "
                        "checkpointed here so an interrupted run resumes, skipping completed sub-runs.")
    return p.parse_args()


def build_polar_schedule(args):
    if not args.polar_coeffs:
        return None
    triples = [tuple(float(x) for x in t.split(",")) for t in args.polar_coeffs.split(";")]
    precs = (args.polar_precs.split(",") if args.polar_precs
             else ["bf16"] * len(triples))
    return tuple(PolarStep(coeffs=t, prec=Prec(p)) for t, p in zip(triples, precs))


def build_model(args, vocab_size, device, seed):
    base_dim = args.depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    cfg = GPTConfig(sequence_len=args.max_seq_len, vocab_size=vocab_size,
                    n_layer=args.depth, n_head=num_heads, n_kv_head=num_heads,
                    n_embd=model_dim, window_pattern="L")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device=device)
    model.init_weights()
    return model, model_dim


def matrix_params(model):
    # robust to a torch.compile OptimizedModule wrapper (params live on _orig_mod)
    m = getattr(model, "_orig_mod", model)
    return [p for p in m.transformer.h.parameters() if p.dim() == 2]


# ----------------------------- coupled inverse 1/4-root -----------------------------
def _power_iter_max(L, iters=18):
    n = L.shape[0]
    v = torch.randn(n, device=L.device, dtype=L.dtype)
    v /= v.norm() + 1e-30
    for _ in range(iters):
        v = L @ v
        v /= v.norm() + 1e-30
    return float(v @ (L @ v))


def _kappa_proxy(L, lam_max=None):
    """Cheap condition-number proxy: lam_max / lam_min via two power-iteration sequences
    (the second on (lam_max I - L) gives lam_max - lam_min). Eigendecomposition-free."""
    Ls = 0.5 * (L + L.t())
    lam_max = _power_iter_max(Ls) if lam_max is None else lam_max
    if not (lam_max > 0 and np.isfinite(lam_max)):
        return 1.0
    n = Ls.shape[0]
    shifted = lam_max * torch.eye(n, device=Ls.device, dtype=Ls.dtype) - Ls
    gap = _power_iter_max(shifted)           # ~ lam_max - lam_min
    lam_min = max(lam_max - gap, 0.0)
    return lam_max / max(lam_min, lam_max * 1e-12)


def inv_fourth_root(L, ridge, k, prec=Prec.fp32):
    L = 0.5 * (L + L.t())
    lam = _power_iter_max(L)
    if not (lam > 0 and np.isfinite(lam)):
        return torch.eye(L.shape[0], device=L.device, dtype=torch.float32)
    n = L.shape[0]
    Ln = L / lam + ridge * torch.eye(n, device=L.device, dtype=L.dtype)
    chain = (CoupledInit(root=4, lambda_min=ridge, lambda_max=1.0, prec=prec),) + tuple(
        CoupledStep(root=4, prec=prec) for _ in range(k))
    Y = coupled.matrix_apply(Ln.to(TORCH_DTYPE[prec]), chain).float()
    if not torch.isfinite(Y).all():
        return torch.eye(L.shape[0], device=L.device, dtype=torch.float32)
    return Y * (lam ** -0.25)


# ----------------------------- shared post-direction machinery (Muon's) -----------------------------
def apply_norm_caution_update(D, p, st, lr, wd, beta2):
    """NorMuon variance reduction + cautious WD/update, lifted from muon_step_unfused (per-param).
    `D` is the arm's already-Nesterov-smoothed direction; this is identical across all arms."""
    red_dim = -1 if p.shape[-2] >= p.shape[-1] else -2
    g = D
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    rds = g.size(red_dim)
    v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * rds).sqrt()
    if st.get("v2") is None or st["v2"].shape != v_mean.shape:
        st["v2"] = torch.zeros_like(v_mean)
    st["v2"].lerp_(v_mean, 1 - beta2)
    step_size = st["v2"].clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * rds) * step_size.square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    mask = (g * p) >= 0
    p.sub_((lr * g + lr * wd * p * mask).to(p.dtype))


def _nesterov(grad, st, momentum):
    st["mom"].lerp_(grad, 1 - momentum)
    return grad.lerp(st["mom"], momentum)


def _shampoo_dir(gm, st):
    return (st["Linv"] @ gm.float() @ st["Rinv"]).to(gm.dtype)


def _spd_power(M, a):
    """SPD matrix raised to scalar power a (eigendecomp). a=1 -> M, a=0 -> I. Used by the synth arm to
    interpolate curvature strength: (L^(-1/4))^a = L^(-a/4)."""
    if a == 1.0:
        return M
    n = M.shape[0]
    if a == 0.0:
        return torch.eye(n, device=M.device, dtype=M.dtype)
    evals, evecs = torch.linalg.eigh(0.5 * (M + M.t()))
    return (evecs * evals.clamp_min(0).pow(a)) @ evecs.t()


def _synth_dir(gm, st, args):
    """Curvature-strength-interpolated direction L^(-a/4) g R^(-a/4), optional polar finisher."""
    D = (st["Linv_a"] @ gm.float() @ st["Rinv_a"]).to(gm.dtype)
    return polar_express_orth(D, args.ns_steps) if args.synth_ortho else D


def direction(arm, p, st, grad, args, step):
    gm = _nesterov(grad, st, args.momentum)
    use_precond = step >= args.warmup_steps and st.get("Linv") is not None
    if arm == "sgd":
        return gm
    if arm == "muon":
        return polar_express_orth(gm, args.ns_steps)
    if arm == "searched_polar":  # ③: a searched gns.fused polar schedule as the U->O map
        return run_polar_2d(gm, args._polar_schedule).to(gm.dtype)
    if arm == "shampoo":
        return _shampoo_dir(gm, st) if use_precond else gm
    if arm == "ortho_shampoo":
        if not use_precond:
            return polar_express_orth(gm, args.ns_steps)
        return polar_express_orth(_shampoo_dir(gm, st), args.ns_steps)
    if arm == "layer_adaptive":
        if not use_precond:
            return polar_express_orth(gm, args.ns_steps)
        if st.get("kappa", 1.0) > args.kappa_threshold:
            return polar_express_orth(_shampoo_dir(gm, st), args.ns_steps)
        return polar_express_orth(gm, args.ns_steps)
    if arm == "synth":  # Muon<->Shampoo spectral interpolation (curvature strength alpha)
        if not use_precond:
            return polar_express_orth(gm, args.ns_steps) if args.synth_ortho else gm
        return _synth_dir(gm, st, args)
    raise ValueError(arm)


def run_arm(arm, lr, args, model, train_batches, val_batches, device):
    mp = matrix_params(model)
    mp_set = {id(p) for p in mp}
    other = [p for p in model.parameters() if id(p) not in mp_set]
    adam = torch.optim.AdamW(other, lr=args.adam_lr, betas=(0.9, 0.95), weight_decay=0.0)
    # `adamw` baseline: matrix params on standard AdamW too (the conventional strong baseline, not just
    # the SGD floor). Its LR is the swept matrix-lr, so pass an Adam-range --matrix-lr-grid for it.
    madam = (torch.optim.AdamW(mp, lr=lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
             if arm == "adamw" else None)
    state = [{"mom": torch.zeros_like(p), "v2": None,
              "L": torch.zeros(p.shape[0], p.shape[0], device=device),
              "R": torch.zeros(p.shape[1], p.shape[1], device=device),
              "Linv": None, "Rinv": None, "kappa": 1.0} for p in mp]
    needs_factors = arm in ("shampoo", "ortho_shampoo", "layer_adaptive", "synth")

    log = {"step": [], "val": [], "wall_ms": []}
    ev_start = torch.cuda.Event(enable_timing=True); ev_start.record()
    for step in range(1, args.num_iterations + 1):
        x, y = train_batches[step - 1]
        loss = model(x, y)
        model.zero_grad(set_to_none=True); adam.zero_grad(set_to_none=True)
        loss.backward()
        lrm = min(1.0, step / max(1, args.warmup_steps))
        with torch.no_grad():
            for j, p in enumerate(mp):
                if p.grad is None:
                    continue
                if arm == "adamw":   # matrix params handled by the AdamW optimizer (madam) below
                    continue
                st = state[j]
                if needs_factors:
                    gf = p.grad.float()
                    st["L"].mul_(args.shampoo_beta).add_(gf @ gf.t(), alpha=1 - args.shampoo_beta)
                    st["R"].mul_(args.shampoo_beta).add_(gf.t() @ gf, alpha=1 - args.shampoo_beta)
                    if step >= args.warmup_steps and (st["Linv"] is None
                                                      or step % args.shampoo_recompute_every == 0):
                        st["Linv"] = inv_fourth_root(st["L"], args.shampoo_ridge, args.shampoo_coupled_steps)
                        st["Rinv"] = inv_fourth_root(st["R"], args.shampoo_ridge, args.shampoo_coupled_steps)
                        if arm == "layer_adaptive":
                            st["kappa"] = max(_kappa_proxy(st["L"]), _kappa_proxy(st["R"]))
                        if arm == "synth":
                            st["Linv_a"] = _spd_power(st["Linv"], args.synth_alpha)
                            st["Rinv_a"] = _spd_power(st["Rinv"], args.synth_alpha)
                D = direction(arm, p, st, p.grad, args, step)
                if not torch.isfinite(D).all():
                    D = _nesterov(p.grad, st, args.momentum)  # robustness fallback
                apply_norm_caution_update(D, p, st, lr * lrm, args.weight_decay, args.beta2)
        for gpar in adam.param_groups:
            gpar["lr"] = args.adam_lr * lrm
        adam.step()
        if madam is not None:
            for g in madam.param_groups:
                g["lr"] = lr * lrm
            madam.step()
        if step % args.eval_every == 0 or step == 1:
            torch.cuda.synchronize()
            ev_now = torch.cuda.Event(enable_timing=True); ev_now.record(); torch.cuda.synchronize()
            with torch.no_grad():
                vl = float(np.mean([float(model(vx, vy).item()) for vx, vy in val_batches]))
            log["step"].append(step); log["val"].append(vl)
            log["wall_ms"].append(ev_start.elapsed_time(ev_now))
            print(f"  [{arm:14s} lr{lr:.3f}] step {step:4d}/{args.num_iterations}  "
                  f"val {vl:.4f}  ({log['wall_ms'][-1]/1000:.1f}s)")
    return log


def main():
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # stream logs live to file (no block-buffering blind spot)
    _, _, _, world, device = compute_init("cuda")
    assert world == 1
    torch.set_float32_matmul_precision("high")
    tok = get_tokenizer(); vocab = tok.get_vocab_size()

    def materialise(split, n, resume_state_dict=None):
        loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
            tok, args.device_batch_size, args.max_seq_len, split=split, device=device,
            resume_state_dict=resume_state_dict)
        out = []
        for _ in range(n):
            x, y, _ = next(loader)
            out.append((x.clone(), y.clone()))
        return out
    # Seed varies WEIGHT INIT (build_model) AND the TRAIN data window: starting at a
    # different parquet shard per seed gives genuinely different documents, so the gap's
    # spread across seeds is an honest replication-variance estimate (not init-only, which
    # the paired difference would make artificially tight). VAL is held FIXED across seeds
    # (common measuring stick). seed 0 -> offset 0 == the original single-seed behaviour.
    train_resume = ({"pq_idx": args.seed, "rg_idx": 0, "epoch": 1} if args.seed else None)
    train_batches = materialise("train", args.num_iterations, resume_state_dict=train_resume)
    val_batches = materialise("val", args.n_val_batches)
    print(f"depth {args.depth}, vocab {vocab}, {len(train_batches)} train + {len(val_batches)} val batches")

    args._polar_schedule = build_polar_schedule(args)
    arms = args.arms.split(",")
    lr_grid = [float(x) for x in args.matrix_lr_grid.split(",")]
    out_path = Path(args.out) if args.out else GNS_OUT
    # exclude private attrs (e.g. _polar_schedule = tuple of PolarStep) — not JSON-serializable
    cfg = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    base_dim = args.depth * args.aspect_ratio
    cfg["model_dim"] = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim

    # resume: reuse completed (arm,lr) sub-runs from a prior partial run of THIS out file
    done = {}
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text()); pc = prev.get("config", {})
            sig = ("depth", "seed", "arms", "matrix_lr_grid", "num_iterations", "device_batch_size")
            if [pc.get(k) for k in sig] == [cfg[k] for k in sig]:
                done = prev.get("_done", {})
                if done:
                    print(f"[resume] {out_path.name}: {len(done)} (arm,lr) sub-runs already complete")
        except Exception:
            done = {}

    results = {"config": cfg, "_done": done, "arms": {}}
    t0 = time.time()
    for arm in arms:
        for lr in lr_grid:
            key = f"{arm}|{lr:.6g}"
            if key in done:
                print(f"=== arm: {arm}  lr={lr}  [resume: skip] ===")
                continue
            print(f"\n=== arm: {arm}  lr={lr} ===")
            model, _ = build_model(args, vocab, device, args.seed)
            if args.compile:
                model = torch.compile(model, dynamic=False)  # rebuilt per (arm,lr): warmup amortised over num-iterations
            log = run_arm(arm, lr, args, model, train_batches, val_batches, device)
            del model; torch.cuda.empty_cache()
            done[key] = {"arm": arm, "lr": lr, "best_val": min(log["val"]), "final_val": log["val"][-1],
                         "total_wall_s": log["wall_ms"][-1] / 1000, "curve": log}
            _atomic_write(out_path, {"config": cfg, "_done": done})  # checkpoint after each sub-run

    # finalize: best lr per arm from the completed sub-runs
    for arm in arms:
        cand = [v for v in done.values() if v["arm"] == arm]
        b = min(cand, key=lambda v: v["best_val"])
        results["arms"][arm] = {"lr": b["lr"], "best_val": b["best_val"], "final_val": b["final_val"],
                                "total_wall_s": b["total_wall_s"], "curve": b["curve"]}

    # summary + gates
    print("\n" + "=" * 72)
    print(f"{'arm':16s} {'best lr':>8s} {'best val':>9s} {'final val':>9s} {'wall(s)':>9s}")
    for arm, b in results["arms"].items():
        print(f"{arm:16s} {b['lr']:8.3f} {b['best_val']:9.4f} {b['final_val']:9.4f} {b['total_wall_s']:9.1f}")
    s = results["arms"]
    if "muon" in s:
        mu = s["muon"]["best_val"]
        for arm in ("shampoo", "ortho_shampoo", "layer_adaptive"):
            if arm in s:
                results.setdefault("gates", {})[f"{arm}_beats_muon_val"] = bool(s[arm]["best_val"] < mu)
        if "ortho_shampoo" in s and "shampoo" in s:
            results["gates"]["gate1_ortho_beats_both"] = bool(
                s["ortho_shampoo"]["best_val"] < min(mu, s["shampoo"]["best_val"]))
    results["runtime_min"] = (time.time() - t0) / 60.0
    _atomic_write(out_path, results)
    print(f"\nwrote {out_path}  ({results['runtime_min']:.1f} min)  gates: {results.get('gates', {})}")
    compute_cleanup()


if __name__ == "__main__":
    main()
