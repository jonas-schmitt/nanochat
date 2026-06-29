"""SOTA-hunt harness: fair matrix-preconditioner comparison inside a real nanochat GPT run.

Every arm shares Muon's EXACT post-direction machinery — Nesterov momentum, NorMuon variance
reduction, and cautious weight decay/update (lifted from nanochat.optim.muon_step_unfused) — and
varies ONLY the matrix DIRECTION map D(g, state). The `muon` arm uses production-matched Muon
hyperparameters (beta2=0.9, weight_decay=0.28 cosine-annealed) so that every other arm is judged
against the real SOTA Muon baseline on equal footing.

Direction maps:
  sgd            : D = nesterov(g) then NorMuon + cautious WD (no polar
                   orthogonalization — NOT vanilla SGD, this is Muon-without-orth)
  muon           : D = polar_express_orth(nesterov(g))                    (SOTA baseline)
  shampoo        : D = L^(-1/4) @ nesterov(g) @ R^(-1/4)                  (Shampoo via gns.coupled)
  ortho_shampoo  : D = polar_express_orth( L^(-1/4) g R^(-1/4) )          (① curvature dir + Muon robustness)
  layer_adaptive : per-factor: ortho_shampoo if kappa_proxy > thresh else muon   (② allocation)

L,R = EMA Kronecker factors (G Gᵀ, Gᵀ G); L^(-1/4) by the coupled Newton–Schulz production (gns.coupled,
fp32, power-iteration λmax, relative ridge), recomputed every K steps. Non-matrix params: AdamW with
the production gpt.py per-group config (per-group LR/betas/eps/wd), identical across arms, so the
comparison runs in the real-nanochat regime rather than one mistuned shared-LR group (audit C3(c)).
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
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from nanochat.common import compute_init, compute_cleanup
from nanochat.gpt import GPT, GPTConfig
from nanochat.optim import polar_express_orth, polar_express_coeffs
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit

import sys
sys.path.insert(0, "/home/jonas/git/gns/src")
from gns import coupled  # noqa: E402
from gns.coupled import CoupledInit, CoupledStep  # noqa: E402
from gns.fused import PolarStep, run_polar_2d  # noqa: E402
from gns.precision import TORCH_DTYPE, Prec  # noqa: E402

GNS_OUT = Path("/home/jonas/git/gns/results/precond_train_compare.json")

# Joint-optimized 4-step polar coefficients (from results/jointopt_grammar_probe.json).
# Match polar_express 5-step quality (L∞ 4.3e-3 vs polar_express tightness) at 20% lower cost.
# Direction 2: "4-step polar" — cheaper Muon, same quality.
JOINTOPT_4STEP_COEFFS = [
    (3.78665, -5.89811, 2.26281),
    (3.39514, -5.67786, 2.57043),
    (2.36862, -3.56152, 1.79741),
    (2.84245, -4.23397, 2.73999),
]

# Joint-optimized 3-step polar coefficients (lower quality, 40% cheaper — stress test).
JOINTOPT_3STEP_COEFFS = [
    (4.14125, -6.02906, 2.45243),
    (3.44765, -4.70214, 1.72506),
    (3.26689, -4.35382, 1.91733),
]


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
    p.add_argument("--beta2", type=float, default=0.9)           # NorMuon second-moment EMA (production Muon)
    p.add_argument("--weight-decay", type=float, default=0.28)   # production Muon cautious WD (cosine-annealed)
    p.add_argument("--ns-steps", type=int, default=5)
    p.add_argument("--orth-every", type=int, default=1,
                   help="apply the polar/preconditioner direction map only every K steps; on off-steps "
                        "the raw Nesterov momentum is used (no polar/curvature map). Default 1 = every "
                        "step (current behavior). Tests the over-orthogonalization-at-depth hypothesis "
                        "(direction 2, notes/scaling-directions-not-sampled.md).")
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
    p.add_argument("--precond-coupled-orders", type=str, default="",
                   help="grammar-searched CoupledStep order sequence (e.g. '3,2,2,1') used for the "
                        "Shampoo inverse-root instead of the uniform 24-step chain; empty = default.")
    p.add_argument("--soap-refresh-every", type=int, default=50,
                   help="SOAP eigenbasis refresh interval (steps). Must be >> Shampoo's recompute (10) so "
                        "the in-basis Adam second moment can adapt in a stable basis.")
    p.add_argument("--soap-beta2", type=float, default=0.99, help="SOAP second-moment EMA (Adam-typical).")
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
    # --- practical-relevance extensions (directions 1,2,4,6,7 in TODO.md) ---
    p.add_argument("--lowrank-k", type=int, default=64,
                   help="lowrank_orth arm: rank-k SVD approximation for cheaper orthogonalization. "
                        "Direction 4: cheaper Muon for wide models. k=0 = full-rank (disabled).")
    p.add_argument("--synth-alpha-warmup", type=int, default=0,
                   help="synth arm: linearly ramp alpha from 0 to --synth-alpha over this many steps. "
                        "Direction 7: annealed curvature strength (early=Muon robust, late=Shampoo curvature). "
                        "0 = static alpha (current behavior).")
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


# Minimum dimension for a matrix to go through the Muon orthogonalization path.
# Tiny gate matrices (e.g. ve_gate at 12×n_kv_head) have too few samples for stable
# NorMuon variance reduction and no theoretical justification for orthogonalization.
# Production gpt.py groups by shape and the MuonAdamW docstring warns against applying
# Muon to embeddings/final-FC/small matrices. Matrices below this threshold go to AdamW.
MIN_MUON_DIM = 32


def matrix_params(model):
    # robust to a torch.compile OptimizedModule wrapper (params live on _orig_mod)
    m = getattr(model, "_orig_mod", model)
    return [p for p in m.transformer.h.parameters() if p.dim() == 2 and min(p.shape) >= MIN_MUON_DIM]


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
    (the second on (lam_max I - L) gives lam_max - lam_min). Eigendecomposition-free.

    Limitations: the proxy is NUMERICALLY ROBUST (power iteration is stable) but NOT
    TIGHT — (1) only 18 power iters means lam_max/lam_min are approximate (especially
    lam_min via the shifted gap, which amplifies error when the spectrum is clustered);
    (2) the lam_min floor `lam_max * 1e-12` caps the reported kappa at 1e12, hiding
    true ill-conditioning beyond that; (3) it is per-factor (L or R alone), not the
    true Kronecker condition number. Sufficient for layer_adaptive routing (a coarse
    threshold gate), not for quantitative conditioning claims."""
    Ls = 0.5 * (L + L.t())
    lam_max = _power_iter_max(Ls) if lam_max is None else lam_max
    if not (lam_max > 0 and np.isfinite(lam_max)):
        return 1.0
    n = Ls.shape[0]
    shifted = lam_max * torch.eye(n, device=Ls.device, dtype=Ls.dtype) - Ls
    gap = _power_iter_max(shifted)           # ~ lam_max - lam_min
    lam_min = max(lam_max - gap, 0.0)
    return lam_max / max(lam_min, lam_max * 1e-12)


def inv_fourth_root(L, ridge, k, prec=Prec.fp32, orders=None):
    L = 0.5 * (L + L.t())
    lam = _power_iter_max(L)
    if not (lam > 0 and np.isfinite(lam)):
        return torch.eye(L.shape[0], device=L.device, dtype=torch.float32)
    n = L.shape[0]
    Ln = L / lam + ridge * torch.eye(n, device=L.device, dtype=L.dtype)
    # `orders` (a grammar-searched CoupledStep order sequence, e.g. exp28's cheaper schedule) replaces
    # the k uniform steps — same fp32-floor accuracy at fewer matmuls.
    steps = (tuple(CoupledStep(root=4, order=o, prec=prec) for o in orders) if orders
             else tuple(CoupledStep(root=4, prec=prec) for _ in range(k)))
    chain = (CoupledInit(root=4, lambda_min=ridge, lambda_max=1.0, prec=prec),) + steps
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


def _polar_with_coeffs(gm, coeffs):
    """Polar orthogonalization using arbitrary (a,b,c) coefficient triples via the gns.fused path.
    Used by muon_4step / muon_3step / muon_fp8 arms — cheaper or faster-precision Muon."""
    schedule = tuple(PolarStep(coeffs=t, prec=Prec.bf16) for t in coeffs)
    return run_polar_2d(gm, schedule).to(gm.dtype)


def _polar_fp8(gm, coeffs):
    """fp8 polar orthogonalization via gns.fused run_polar_2d with fp8e4m3 precision.
    Direction 1: 1.8× faster Muon (fp8 tensor cores), same quality (polar is well-conditioned)."""
    schedule = tuple(PolarStep(coeffs=t, prec=Prec.fp8e4m3) for t in coeffs)
    return run_polar_2d(gm, schedule).to(gm.dtype)


def _eigenbasis_shampoo_dir(gm, st, args):
    """Direction 6: polar within the Kronecker eigenbasis (composition order change).
    Instead of polar_express_orth(L^{-1/4} G R^{-1/4}) in the STANDARD basis, diagonalize
    L=Q_L D_L Q_L^T, R=Q_R D_R Q_R^T, precondition in the eigenbasis (diagonal scaling),
    apply polar there, then rotate back. Keeps the iterate in the commuting symmetric subspace
    (where exp22/G2 showed matrix stability holds)."""
    if st.get("Q_L") is None or st.get("Q_R") is None:
        # Not yet refreshed — fall back to standard ortho_shampoo
        return polar_express_orth(_shampoo_dir(gm, st), args.ns_steps)
    QL, QR = st["Q_L"], st["Q_R"]
    # Rotate gradient into eigenbasis (Q are fp32 from eigh)
    ghat = QL.t() @ gm.float() @ QR
    # Diagonal preconditioning in eigenbasis (eigenvalues of L^{-1/4}, R^{-1/4})
    dL = st.get("dL")
    dR = st.get("dR")
    if dL is not None and dR is not None:
        ghat = (dL.unsqueeze(1) * ghat) * dR.unsqueeze(0)
    # Polar in eigenbasis (polar_express_orth returns bf16; cast for matmul)
    ghat_orth = polar_express_orth(ghat, args.ns_steps).float()
    # Rotate back
    return (QL @ ghat_orth @ QR.t()).to(gm.dtype)


def _lowrank_orth(gm, k):
    """Direction 4: low-rank SVD orthogonalization for cheaper Muon on wide models.
    Instead of 5 polar_express matmuls on the full (m×n) gradient, compute a rank-k SVD
    approximation and return U_k @ V_k^h. For k << min(m,n) this is dramatically cheaper
    (one SVD vs 5 matmuls). Quality: close to full-rank if G has fast spectral decay
    (gradients often do early in training). Uses truncated SVD for correctness first;
    a randomized SVD can replace it for wall-clock speedup."""
    gf = gm.float()
    U, S, Vh = torch.linalg.svd(gf, full_matrices=False)
    k = min(k, S.shape[0])
    # U: (..., m, k), Vh: (..., k, n) → U_k @ V_k^h = U[..., :k] @ Vh[..., :k, :]
    return (U[..., :k] @ Vh[..., :k, :]).to(gm.dtype)


# ----------------------------- SOAP: Adam in the Kronecker eigenbasis -----------------------------
def _soap_refresh_basis(st):
    """Recompute the factor eigenbases Q_L,Q_R and rotate the in-basis momentum to stay aligned
    (the first moment is gradient-covariant; the per-element second moment re-adapts in the new basis)."""
    QLn = torch.linalg.eigh(0.5 * (st["L"] + st["L"].t())).eigenvectors
    QRn = torch.linalg.eigh(0.5 * (st["R"] + st["R"].t())).eigenvectors
    if st["msoap"] is not None:
        if st["Q_L"] is not None:
            # subsequent refresh: rotate from old eigenbasis to new
            RL = QLn.t() @ st["Q_L"]; RR = st["Q_R"].t() @ QRn
            st["msoap"] = RL @ st["msoap"] @ RR
        else:
            # first refresh: msoap accumulated in identity basis (Q_old = I), rotate to new
            st["msoap"] = QLn.t() @ st["msoap"] @ QRn
    st["Q_L"], st["Q_R"] = QLn, QRn


def _soap_step(p, st, lr, args, eps=1e-8):
    """SOAP update: rotate g into the eigenbasis, run Adam there, rotate back. Bypasses NorMuon."""
    g = p.grad.float()
    QL, QR = st["Q_L"], st["Q_R"]
    ghat = (QL.t() @ g @ QR) if QL is not None else g     # identity basis before the first refresh
    if st["msoap"] is None or st["msoap"].shape != ghat.shape:
        st["msoap"] = torch.zeros_like(ghat); st["vsoap"] = torch.zeros_like(ghat)
    st["msoap"].mul_(args.momentum).add_(ghat, alpha=1 - args.momentum)
    st["vsoap"].mul_(args.soap_beta2).add_(ghat * ghat, alpha=1 - args.soap_beta2)
    phat = st["msoap"] / (st["vsoap"].sqrt() + eps)
    D = (QL @ phat @ QR.t()) if QL is not None else phat   # rotate back
    p.sub_((lr * D).to(p.dtype))


def direction(arm, p, st, grad, args, step):
    gm = _nesterov(grad, st, args.momentum)
    use_precond = step >= args.warmup_steps and st.get("Linv") is not None
    # orth-every: on off-steps skip the polar/curvature direction map entirely and use raw Nesterov.
    # This applies the preconditioner every K steps instead of every step, testing whether
    # over-orthogonalization at depth causes the d12 erosion (direction 2,
    # notes/scaling-directions-not-sampled.md). Default orth_every=1 is a no-op (every step).
    if args.orth_every > 1 and (step % args.orth_every != 0):
        return gm
    if arm == "sgd":
        return gm
    if arm == "muon":
        return polar_express_orth(gm, args.ns_steps)
    if arm == "muon_4step":  # Direction 2: 4-step joint-opt polar (20% cheaper Muon)
        return _polar_with_coeffs(gm, JOINTOPT_4STEP_COEFFS)
    if arm == "muon_3step":  # 3-step joint-opt polar (40% cheaper, stress test)
        return _polar_with_coeffs(gm, JOINTOPT_3STEP_COEFFS)
    if arm == "muon_fp8":   # Direction 1: fp8 polar (1.8× faster Muon via fp8 tensor cores)
        return _polar_fp8(gm, polar_express_coeffs)
    if arm == "lowrank_orth":  # Direction 4: low-rank SVD orthogonalization (cheaper Muon)
        return _lowrank_orth(gm, args.lowrank_k) if args.lowrank_k > 0 else polar_express_orth(gm, args.ns_steps)
    if arm == "eigenbasis_shampoo":  # Direction 6: polar within Kronecker eigenbasis
        if not use_precond:
            return polar_express_orth(gm, args.ns_steps)
        return _eigenbasis_shampoo_dir(gm, st, args)
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
        # Direction 7: annealed alpha — ramp from 0 (pure Muon) to args.synth_alpha over warmup steps.
        if args.synth_alpha_warmup > 0 and step <= args.synth_alpha_warmup:
            alpha = args.synth_alpha * step / args.synth_alpha_warmup
            # Recompute Linv_a/Rinv_a at the current annealed alpha (lightweight: just power of existing Linv)
            st["Linv_a"] = _spd_power(st["Linv"], alpha)
            st["Rinv_a"] = _spd_power(st["Rinv"], alpha)
        if not use_precond:
            return polar_express_orth(gm, args.ns_steps) if args.synth_ortho else gm
        return _synth_dir(gm, st, args)
    raise ValueError(arm)


def _build_other_adam(model, mp_ids):
    """Production-matched AdamW for the non-matrix params (audit C3(c)).

    Previously the harness drove ALL non-matrix params (embeddings, lm_head, value-embeds,
    scalars, ve_gate) through a single plain AdamW at --adam-lr with weight_decay=0. That is
    fair across arms (identical for every arm) but runs the model in a very different REGIME
    from real nanochat training (e.g. embeddings at 3e-3 vs production ~0.2), which weakens
    the EXTERNAL validity of "arm X beats Muon" (does it transfer to production?). Mirror
    gpt.py setup_optimizer's per-group config exactly so the comparison happens in the
    production regime. Matrix params (handled by the arm's direction map) are excluded; each
    group carries base_lr for the warmup ramp. NOTE: --adam-lr no longer affects these groups."""
    m = getattr(model, "_orig_mod", model)
    s = (m.config.n_embd / 768) ** -0.5  # dmodel_lr_scale, matching gpt.py
    groups = [
        dict(params=list(m.lm_head.parameters()),          base_lr=0.004 * s,     betas=(0.8, 0.96),  eps=1e-10, weight_decay=0.01),
        dict(params=list(m.transformer.wte.parameters()),  base_lr=0.2 * s,       betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
        dict(params=list(m.value_embeds.parameters()),     base_lr=0.2 * s * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
        dict(params=[m.resid_lambdas],                     base_lr=0.5 * 0.01,    betas=(0.8, 0.95),  eps=1e-10, weight_decay=0.05),
        dict(params=[m.x0_lambdas],                        base_lr=0.5,           betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        dict(params=[m.smear_gate.weight, m.smear_lambda, m.backout_lambda], base_lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
    ]
    # ve_gate (sub-tile block matrices, filtered out of the matrix path by MIN_MUON_DIM):
    # grouped with the value-embedding path it gates, matching the gpt.py H2 fix.
    named = {id(p) for g in groups for p in g["params"]}
    small = [p for p in m.transformer.h.parameters() if id(p) not in mp_ids and id(p) not in named]
    if small:
        groups.append(dict(params=small, base_lr=0.2 * s * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01))
    # every non-matrix param must be covered exactly once (mirrors gpt.py setup_optimizer's assert)
    covered = {id(p) for g in groups for p in g["params"]}
    other_ids = {id(p) for p in m.parameters() if id(p) not in mp_ids}
    assert covered == other_ids, "non-matrix param partition mismatch (model structure changed?)"
    return torch.optim.AdamW([dict(g, lr=g["base_lr"]) for g in groups])


def run_arm(arm, lr, args, model, train_batches, val_batches, device):
    mp = matrix_params(model)
    mp_set = {id(p) for p in mp}
    adam = _build_other_adam(model, mp_set)
    # `adamw` baseline: matrix params on standard AdamW too (the conventional strong baseline, not just
    # the SGD floor). Its LR is the swept matrix-lr, so pass an Adam-range --matrix-lr-grid for it.
    # AdamW uses a lower weight decay than Muon (0.01 vs 0.28) — using Muon's WD mistunes the baseline.
    adam_wd = 0.01 if arm == "adamw" else args.weight_decay
    madam = (torch.optim.AdamW(mp, lr=lr, betas=(0.9, 0.95), weight_decay=adam_wd)
             if arm == "adamw" else None)
    state = [{"mom": torch.zeros_like(p), "v2": None,
              "L": torch.zeros(p.shape[0], p.shape[0], device=device),
              "R": torch.zeros(p.shape[1], p.shape[1], device=device),
              "Linv": None, "Rinv": None, "kappa": 1.0,
              "Q_L": None, "Q_R": None, "msoap": None, "vsoap": None} for p in mp]
    needs_factors = arm in ("shampoo", "ortho_shampoo", "layer_adaptive", "synth", "soap",
                            "eigenbasis_shampoo")

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
                    fresh = (st["Q_L"] is None) if arm == "soap" else (st["Linv"] is None)
                    recompute_n = args.soap_refresh_every if arm == "soap" else args.shampoo_recompute_every
                    if step >= args.warmup_steps and (fresh or step % recompute_n == 0):
                        if arm == "soap":
                            _soap_refresh_basis(st)
                        else:
                            st["Linv"] = inv_fourth_root(st["L"], args.shampoo_ridge, args.shampoo_coupled_steps, orders=args._precond_orders)
                            st["Rinv"] = inv_fourth_root(st["R"], args.shampoo_ridge, args.shampoo_coupled_steps, orders=args._precond_orders)
                            if arm == "layer_adaptive":
                                st["kappa"] = max(_kappa_proxy(st["L"]), _kappa_proxy(st["R"]))
                            if arm == "synth":
                                if args.synth_alpha_warmup > 0 and step <= args.synth_alpha_warmup:
                                    alpha = args.synth_alpha * step / args.synth_alpha_warmup
                                else:
                                    alpha = args.synth_alpha
                                st["Linv_a"] = _spd_power(st["Linv"], alpha)
                                st["Rinv_a"] = _spd_power(st["Rinv"], alpha)
                            if arm == "eigenbasis_shampoo":
                                # Eigendecompose L,R for in-eigenbasis polar (direction 6)
                                evals_L, evecs_L = torch.linalg.eigh(0.5 * (st["L"] + st["L"].t()))
                                evals_R, evecs_R = torch.linalg.eigh(0.5 * (st["R"] + st["R"].t()))
                                st["Q_L"], st["Q_R"] = evecs_L, evecs_R
                                st["dL"] = evals_L.clamp_min(1e-12).pow(-0.25)
                                st["dR"] = evals_R.clamp_min(1e-12).pow(-0.25)
                if arm == "soap":   # Adam in the eigenbasis; bypasses NorMuon, own Adam-range LR
                    _soap_step(p, st, lr * lrm, args)
                    continue
                D = direction(arm, p, st, p.grad, args, step)
                if not torch.isfinite(D).all():
                    # robustness fallback: plain Nesterov direction. direction() already
                    # advanced st["mom"] once (its first line calls _nesterov); re-running
                    # _nesterov here would lerp the gradient into st["mom"] a SECOND time
                    # (double EMA update). Reuse the already-advanced buffer instead.
                    D = p.grad.lerp(st["mom"], args.momentum)
                # Cosine-annealed weight decay (matching production base_train.py:get_weight_decay)
                cos_wd = args.weight_decay * 0.5 * (1 + math.cos(math.pi * step / args.num_iterations))
                apply_norm_caution_update(D, p, st, lr * lrm, cos_wd, args.beta2)
        for gpar in adam.param_groups:
            gpar["lr"] = gpar["base_lr"] * lrm
        adam.step()
        if madam is not None:
            for g in madam.param_groups:
                g["lr"] = lr * lrm
            madam.step()
        if step % args.eval_every == 0 or step == 1:
            torch.cuda.synchronize()
            ev_now = torch.cuda.Event(enable_timing=True); ev_now.record(); torch.cuda.synchronize()
            model.eval()
            with torch.no_grad():
                vl = float(np.mean([float(model(vx, vy).item()) for vx, vy in val_batches]))
            model.train()
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
    # TF32 OFF for genuine fp32 accumulation (audit C2, matching base_train and the gns
    # executor / DESIGN convention 6). The fp32 matmuls here — the gns.coupled inverse-root
    # (shampoo/ortho_shampoo/synth/eigenbasis arms) and _shampoo_dir — must accumulate in
    # true fp32 for the gns rounding model (u_fp32 = 2^-24) to be predictive; under "high"
    # they used ~10-bit TF32. Uniform across arms, so the cross-arm gate stays fair.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
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
    args._precond_orders = ([int(x) for x in args.precond_coupled_orders.split(",")]
                            if args.precond_coupled_orders else None)
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
            # resume: reuse completed (arm,lr) sub-runs from a prior partial run of THIS out file.
            # Backward-compatible: a key ABSENT from an old checkpoint defaults to the current value
            # (so adding --orth-every does not re-run pre-existing 3-seed work that never had it).
            sig = ("depth", "seed", "arms", "matrix_lr_grid", "num_iterations", "device_batch_size",
                   "orth_every", "aspect_ratio", "lowrank_k", "synth_alpha_warmup",
                   "synth_alpha", "synth_ortho", "precond_coupled_orders",
                   "shampoo_ridge", "shampoo_recompute_every", "shampoo_coupled_steps",
                   "ns_steps", "weight_decay", "momentum", "beta2",
                   "warmup_steps", "n_val_batches", "eval_every")
            if [pc.get(k, cfg[k]) for k in sig] == [cfg[k] for k in sig]:
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
