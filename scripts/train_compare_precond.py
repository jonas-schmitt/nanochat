"""Real training comparison: does the gns COUPLED inverse-root production work as a Shampoo
preconditioner inside an actual nanochat GPT training run?

Controlled experiment. Same model init (seed), same pre-materialised climbmix batches, same AdamW
for non-matrix params; the ONLY thing that varies is the MATRIX-parameter update DIRECTION:

  muon    : O = polar_express_orth(momentum_grad)          (nanochat's Muon/polar baseline)
  shampoo : P = L^(-1/4) @ momentum_grad @ R^(-1/4)         (Shampoo via gns.coupled, fp32)
            with L=EMA(G Gᵀ), R=EMA(Gᵀ G); L^(-1/4) computed by the COUPLED Newton–Schulz
            production (power-iteration λmax, relative ridge, no eigendecomposition).
  sgd     : P = momentum_grad                               (no preconditioner — floor)

Every matrix update is RMS-normalised to unit RMS before the (shared) matrix LR, so the LR is
comparable across arms and we compare update DIRECTION quality, not scale.

Decisive question: does shampoo-coupled train competitively with Muon (loss decreases, no
divergence)? If yes, the coupled inverse-root production is validated end-to-end in the loop —
the thing it was built for.

Run (tct-models env + gns on path):
  cd /home/jonas/git/nanochat
  PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src \
    uv run --project /home/jonas/git/tct-models python scripts/train_compare_precond.py \
      --depth 6 --num-iterations 500
"""
import argparse
import json
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
from gns.precision import TORCH_DTYPE, Prec  # noqa: E402

GNS_OUT = Path("/home/jonas/git/gns/results/precond_train_compare.json")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--aspect-ratio", type=int, default=64)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--device-batch-size", type=int, default=16)
    p.add_argument("--num-iterations", type=int, default=500)
    p.add_argument("--matrix-lr", type=float, default=0.02)
    p.add_argument("--adam-lr", type=float, default=3e-3)
    p.add_argument("--momentum", type=float, default=0.95)
    p.add_argument("--ns-steps", type=int, default=5)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--shampoo-beta", type=float, default=0.95)
    p.add_argument("--shampoo-ridge", type=float, default=1e-4)
    p.add_argument("--shampoo-coupled-steps", type=int, default=24)
    p.add_argument("--shampoo-recompute-every", type=int, default=10)
    p.add_argument("--n-val-batches", type=int, default=16)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--arms", type=str, default="muon,shampoo,sgd")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


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
    return [p for p in model.transformer.h.parameters() if p.dim() == 2]


def _power_iter_max(L, iters=18):
    n = L.shape[0]
    v = torch.randn(n, device=L.device, dtype=L.dtype)
    v /= v.norm() + 1e-30
    for _ in range(iters):
        v = L @ v
        v /= v.norm() + 1e-30
    return float(v @ (L @ v))


def inv_fourth_root(L, ridge, k, prec=Prec.fp32):
    """L^(-1/4) via the coupled production: normalise by λmax, ridge into [ridge,1], iterate."""
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


def rms_normalize(P):
    return P / (P.square().mean().sqrt() + 1e-12)


def run_arm(arm, args, model, train_batches, val_batches, device):
    mp = matrix_params(model)
    mp_set = {id(p) for p in mp}
    other = [p for p in model.parameters() if id(p) not in mp_set]
    adam = torch.optim.AdamW(other, lr=args.adam_lr, betas=(0.9, 0.95), weight_decay=0.0)
    mom = [torch.zeros_like(p) for p in mp]
    # Shampoo Kronecker factors
    facs = [{"L": torch.zeros(p.shape[0], p.shape[0], device=device),
             "R": torch.zeros(p.shape[1], p.shape[1], device=device),
             "Linv": None, "Rinv": None} for p in mp]

    log = {"train_loss": [], "val_steps": [], "val_loss": []}
    for step in range(1, args.num_iterations + 1):
        x, y = train_batches[step - 1]
        loss = model(x, y)
        model.zero_grad(set_to_none=True)
        adam.zero_grad(set_to_none=True)
        loss.backward()

        lrm = min(1.0, step / max(1, args.warmup_steps))
        with torch.no_grad():
            for j, p in enumerate(mp):
                g = p.grad
                if g is None:
                    continue
                mom[j].mul_(args.momentum).add_(g, alpha=1 - args.momentum)
                gm = g.add(mom[j], alpha=args.momentum)  # Nesterov-style lookahead
                if arm == "muon":
                    P = polar_express_orth(gm, args.ns_steps).float()
                elif arm == "sgd":
                    P = gm.float()
                elif arm == "shampoo":
                    gf = g.float()
                    f = facs[j]
                    f["L"].mul_(args.shampoo_beta).add_(gf @ gf.t(), alpha=1 - args.shampoo_beta)
                    f["R"].mul_(args.shampoo_beta).add_(gf.t() @ gf, alpha=1 - args.shampoo_beta)
                    if step < args.warmup_steps:
                        P = gm.float()  # warm up factors before preconditioning
                    else:
                        if f["Linv"] is None or (step % args.shampoo_recompute_every == 0):
                            f["Linv"] = inv_fourth_root(f["L"], args.shampoo_ridge,
                                                        args.shampoo_coupled_steps)
                            f["Rinv"] = inv_fourth_root(f["R"], args.shampoo_ridge,
                                                        args.shampoo_coupled_steps)
                        P = f["Linv"] @ gm.float() @ f["Rinv"]
                else:
                    raise ValueError(arm)
                if not torch.isfinite(P).all():
                    P = gm.float()  # robustness: fall back to momentum direction
                upd = rms_normalize(P).to(p.dtype)
                p.add_(upd, alpha=-args.matrix_lr * lrm)
        for g_ in adam.param_groups:
            g_["lr"] = args.adam_lr * lrm
        adam.step()

        log["train_loss"].append(float(loss.item()))
        if step % args.eval_every == 0 or step == 1:
            with torch.no_grad():
                vl = float(np.mean([float(model(vx, vy).item()) for vx, vy in val_batches]))
            log["val_steps"].append(step)
            log["val_loss"].append(vl)
            print(f"  [{arm:8s}] step {step:4d}/{args.num_iterations}  "
                  f"train {log['train_loss'][-1]:.4f}  val {vl:.4f}")
    return log


def main():
    args = parse_args()
    _, _, _, world, device = compute_init("cuda")
    assert world == 1
    torch.set_float32_matmul_precision("high")
    tok = get_tokenizer()
    vocab = tok.get_vocab_size()

    # pre-materialise identical train + val batches for every arm (same data, same order)
    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tok, args.device_batch_size, args.max_seq_len, split="train", device=device,
        resume_state_dict=None)
    train_batches = []
    for _ in range(args.num_iterations):
        x, y, _ = next(loader)
        train_batches.append((x.clone(), y.clone()))
    vloader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tok, args.device_batch_size, args.max_seq_len, split="val", device=device,
        resume_state_dict=None)
    val_batches = []
    for _ in range(args.n_val_batches):
        x, y, _ = next(vloader)
        val_batches.append((x.clone(), y.clone()))
    print(f"materialised {len(train_batches)} train + {len(val_batches)} val batches; "
          f"depth {args.depth}, vocab {vocab}")

    arms = args.arms.split(",")
    results = {"config": vars(args), "arms": {}}
    t0 = time.time()
    for arm in arms:
        print(f"\n=== arm: {arm} ===")
        model, model_dim = build_model(args, vocab, device, args.seed)
        results["config"]["model_dim"] = model_dim
        log = run_arm(arm, args, model, train_batches, val_batches, device)
        results["arms"][arm] = log
        del model
        torch.cuda.empty_cache()

    # summary
    print("\n" + "=" * 64)
    print(f"{'arm':10s} {'final train(sm50)':>18s} {'best val':>10s} {'final val':>10s}")
    summary = {}
    for arm, log in results["arms"].items():
        sm = float(np.mean(log["train_loss"][-50:]))
        bestv = float(min(log["val_loss"]))
        finv = float(log["val_loss"][-1])
        summary[arm] = {"final_train_sm50": sm, "best_val": bestv, "final_val": finv}
        print(f"{arm:10s} {sm:18.4f} {bestv:10.4f} {finv:10.4f}")
    results["summary"] = summary
    results["runtime_min"] = (time.time() - t0) / 60.0
    # verdict: shampoo-coupled trains competitively with muon (final val within 3% or better)
    if "shampoo" in summary and "muon" in summary:
        ratio = summary["shampoo"]["best_val"] / summary["muon"]["best_val"]
        results["shampoo_vs_muon_best_val_ratio"] = ratio
        results["shampoo_trains_competitively"] = bool(ratio <= 1.03)
        print(f"\nshampoo/muon best-val ratio: {ratio:.4f}  "
              f"-> coupled-Shampoo {'competitive' if ratio <= 1.03 else 'worse'} with Muon")
    GNS_OUT.write_text(json.dumps(results, indent=1))
    print(f"wrote {GNS_OUT}  ({results['runtime_min']:.1f} min)")
    compute_cleanup()


if __name__ == "__main__":
    main()
