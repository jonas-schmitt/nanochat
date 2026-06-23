"""Harvest REAL Shampoo/SOAP preconditioner factors from a short nanochat run.

For the GNS Pillar-2 (inverse-root) potential assessment. nanochat keeps training with
its stock Muon optimizer; we only OBSERVE the matrix-parameter gradients to accumulate
the Kronecker factors a Shampoo/SOAP optimizer would precondition with:

    L_t = beta * L_{t-1} + (1-beta) * G_t G_t^T   (m x m)
    R_t = beta * R_{t-1} + (1-beta) * G_t^T G_t   (n x n)

for each selected weight G (m x n) in model.transformer.h. Both are SPD. We snapshot
L, R at a few steps for a representative subset of layers, save them (fp32) plus an
eigvalsh manifest into the GNS results tree, so the GNS scalar search / matrix fidelity
experiments can be re-anchored to the conditioning real GPT preconditioners actually have.

Optimizer-agnostic (reads .grad before optimizer.step), real climbmix data, real tokenizer.

Run (tct-models-compatible env, single GPU):
  cd /home/jonas/git/nanochat
  uv run --project /home/jonas/git/tct-models python scripts/harvest_precond.py \
      --depth 4  --num-iterations 300
  uv run --project /home/jonas/git/tct-models python scripts/harvest_precond.py \
      --depth 12 --num-iterations 300
"""
import argparse
import json
import os
from pathlib import Path

import torch

from nanochat.common import compute_init, compute_cleanup, get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit

# GNS results tree (sibling repo); overridable
GNS_PRECOND_DIR = Path(os.environ.get(
    "GNS_PRECOND_DIR", "/home/jonas/git/gns/results/precond"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--aspect-ratio", type=int, default=64)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--device-batch-size", type=int, default=16)
    p.add_argument("--num-iterations", type=int, default=300)
    p.add_argument("--matrix-lr", type=float, default=0.02)
    p.add_argument("--warmup-steps", type=int, default=20,
                   help="linear LR warmup (mirrors nanochat base_train stability)")
    p.add_argument("--ema-beta", type=float, default=0.95,
                   help="EMA decay for the Kronecker factor accumulators")
    p.add_argument("--window-pattern", type=str, default="L")
    p.add_argument("--snapshot-fracs", type=str, default="0.1,0.5,1.0",
                   help="fractions of num-iterations at which to snapshot factors")
    return p.parse_args()


def build_model(args, vocab_size, device):
    base_dim = args.depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=args.depth, n_head=num_heads, n_kv_head=num_heads,
        n_embd=model_dim, window_pattern=args.window_pattern,
    )
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    return model, config, model_dim


def select_harvest_params(model):
    """A representative subset: one attn proj + both MLP matrices, in {first, mid, last}
    layer. Returns list of (name, param) with 2-D weights."""
    h = model.transformer.h
    n = len(h)
    layers = sorted({0, n // 2, n - 1})
    out = []
    for li in layers:
        blk = h[li]
        for mname, mod in (("attn.c_q", blk.attn.c_q),
                           ("mlp.c_fc", blk.mlp.c_fc),
                           ("mlp.c_proj", blk.mlp.c_proj)):
            w = mod.weight
            assert w.dim() == 2, (mname, tuple(w.shape))
            out.append((f"L{li}.{mname}", w))
    return out


def main():
    args = parse_args()
    _, ddp_rank, _, world, device = compute_init("cuda")
    assert world == 1, "harvest is single-GPU"
    torch.set_float32_matmul_precision("high")

    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    model, config, model_dim = build_model(args, vocab_size, device)
    print(f"d{args.depth}: model_dim={model_dim} n_layer={config.n_layer} "
          f"n_head={config.n_head} vocab={vocab_size}")

    optimizer = model.setup_optimizer(matrix_lr=args.matrix_lr, weight_decay=0.0,
                                      muon_orth="fused")
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])
    train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer, args.device_batch_size, args.max_seq_len, split="train",
        device=device, resume_state_dict=None)
    x, y, _ = next(train_loader)

    harvest = select_harvest_params(model)
    beta = args.ema_beta
    facs = {}  # name -> {"L":Tensor m x m, "R":Tensor n x n, "shape":(m,n)}
    for name, w in harvest:
        m, n = w.shape
        facs[name] = {
            "L": torch.zeros(m, m, dtype=torch.float32, device=device),
            "R": torch.zeros(n, n, dtype=torch.float32, device=device),
            "shape": (int(m), int(n)),
        }
    print(f"harvesting {len(harvest)} matrices: "
          + ", ".join(f"{nm}{tuple(w.shape)}" for nm, w in harvest))

    snap_steps = sorted({max(1, int(round(f * args.num_iterations)))
                         for f in (float(s) for s in args.snapshot_fracs.split(","))})
    print(f"snapshot steps: {snap_steps}")

    out_dir = GNS_PRECOND_DIR / f"d{args.depth}"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"depth": args.depth, "model_dim": model_dim,
                "n_layer": config.n_layer, "ema_beta": beta,
                "num_iterations": args.num_iterations,
                "matrix_lr": args.matrix_lr, "seq_len": args.max_seq_len,
                "device_batch_size": args.device_batch_size,
                "snapshots": {}}

    losses = []
    for step in range(1, args.num_iterations + 1):
        lrm = min(1.0, step / max(1, args.warmup_steps))
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
        loss = model(x, y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        x, y, _ = next(train_loader)
        # --- harvest: accumulate Kronecker factors from gradients (pre-step) ---
        with torch.no_grad():
            for name, w in harvest:
                g = w.grad
                if g is None:
                    continue
                g = g.float()
                f = facs[name]
                f["L"].mul_(beta).add_(g @ g.t(), alpha=1.0 - beta)
                f["R"].mul_(beta).add_(g.t() @ g, alpha=1.0 - beta)
        optimizer.step()
        losses.append(loss.item())
        if step % max(1, args.num_iterations // 10) == 0 or step == 1:
            print(f"  step {step:4d}/{args.num_iterations}  loss {losses[-1]:.4f}")

        if step in snap_steps:
            snap = {}
            for name, w in harvest:
                f = facs[name]
                rec = {"shape": f["shape"], "factors": {}}
                for side in ("L", "R"):
                    A = f[side]
                    # symmetrize for numerical cleanliness, then SPD eigenvalues
                    As = 0.5 * (A + A.t())
                    evals = torch.linalg.eigvalsh(As.double()).cpu()
                    evals = torch.clamp(evals, min=0.0)
                    fname = f"{name}.{side}_{step}.pt"
                    torch.save(As.cpu(), out_dir / fname)
                    lam = evals.numpy()
                    lam_max = float(lam[-1])
                    lam_min_pos = float(lam[lam > 0][0]) if (lam > 0).any() else 0.0
                    rec["factors"][side] = {
                        "file": fname, "dim": A.shape[0],
                        "lambda_max": lam_max, "lambda_min": float(lam[0]),
                        "lambda_min_pos": lam_min_pos,
                        "cond_pos": (lam_max / lam_min_pos) if lam_min_pos > 0 else float("inf"),
                        "eigvals": [float(v) for v in lam],
                    }
                snap[name] = rec
            manifest["snapshots"][str(step)] = snap
            print(f"  [snapshot @ step {step}] saved {len(harvest)*2} factor files")

    manifest["final_loss"] = losses[-1]
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nwrote {out_dir}/manifest.json  (final loss {losses[-1]:.4f})")
    compute_cleanup()


if __name__ == "__main__":
    main()
