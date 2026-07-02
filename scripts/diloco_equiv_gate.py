"""G-A1 bisection: run ONE training step through train_compare_precond.run_arm and through
train_diloco.run_policy (dp genome, M=1, h=1) from identical inits/data, then compare every
parameter. Localizes which subsystem (matrix/Muon path, non-matrix/Adam path, or gradients)
introduces the first divergence. Run with GNS_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8."""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/jonas/git/gns/src")
import train_compare_precond as tcp  # noqa: E402
import train_diloco as td  # noqa: E402
from dataclasses import replace as dc_replace  # noqa: E402
from gns.policy_grammar import dp_genome  # noqa: E402

N_STEPS = 12
MODE = "ga2"


def make_args(**over):
    a = SimpleNamespace(
        depth=6, aspect_ratio=64, head_dim=128, max_seq_len=1024, device_batch_size=16,
        num_iterations=N_STEPS, matrix_lr=0.02, adam_lr=3e-3, momentum=0.95, beta2=0.9,
        weight_decay=0.28, ns_steps=5, warmup_steps=20, n_val_batches=1, eval_every=10**9,
        seed=0, workers=1, out="", h=1,
        # tcp extras its run_arm/direction path reads for the muon arm
        orth_every=1, shampoo_beta=0.95, shampoo_ridge=1e-4, shampoo_coupled_steps=24,
        shampoo_recompute_every=10, soap_refresh_every=50, soap_beta2=0.99,
        kappa_threshold=1e4, synth_alpha=1.0, synth_ortho=1, synth_alpha_warmup=0,
        lookahead_k=6, lookahead_alpha=0.5, anderson_window=5, anderson_reg=1e-8,
        anderson_restart=0, wwd_strength=1.0, wwd_power=0.5, wwd=False,
        subspace_k=32, subspace_refresh=16, subspace_buffer=64, subspace_lr=0.5,
        subspace_ridge=1e-4, soft_tau=0.1, soft_q=2.0, soft_mode="frac", snr_strength=1.0,
        soft_no_renorm=False, role_lr_mults="1,1,1,1", lowrank_k=64,
        precond_coupled_orders="", polar_coeffs="", polar_precs="",
        outer_factor_beta=0.9, outer_factor_ridge=1e-4, outer_factor_coupled_steps=24,
        _polar_schedule=None, _precond_orders=None,
    )
    for k, v in over.items():
        setattr(a, k, v)
    return a


def main():
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    device = "cuda"
    args = make_args()
    tok = tcp.get_tokenizer(); vocab = tok.get_vocab_size()

    loader = tcp.tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tok, args.device_batch_size, args.max_seq_len, split="train", device=device,
        resume_state_dict=None)
    train = []
    for _ in range(N_STEPS):
        x, y, _ = next(loader)
        train.append((x.clone(), y.clone()))
    vloader = tcp.tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tok, args.device_batch_size, args.max_seq_len, split="val", device=device,
        resume_state_dict=None)
    vx, vy, _ = next(vloader)
    val = [(vx.clone(), vy.clone())]

    def record_losses(model, sink, hsink):
        orig = model.forward
        wte = model.transformer.wte.weight
        lmh = model.lm_head.weight
        def wrapped(x, y=None, *a, **k):
            if model.training:
                hsink.append((float(wte.float().abs().sum().item()),
                              float(lmh.float().abs().sum().item())))
            out = orig(x, y, *a, **k) if y is not None else orig(x, *a, **k)
            if torch.is_tensor(out) and out.dim() == 0 and model.training:
                sink.append(float(out.item()))
            return out
        model.forward = wrapped

    if MODE == "ga1":
        arm, genome = "muon", dp_genome()
    else:
        arm = "muon_lookahead"
        args.lookahead_k, args.lookahead_alpha = 5, 0.5
        genome = dc_replace(dp_genome(), h=5, outer_lr=0.5)

    # --- side A: the reference harness's run_arm ---
    model_a, _ = tcp.build_model(args, vocab, device, args.seed)
    names = {id(p): n for n, p in model_a.named_parameters()}
    loss_a = []; hash_a = []
    record_losses(model_a, loss_a, hash_a)
    tcp.run_arm(arm, args.matrix_lr, args, model_a, train, val, device)
    snap_a = {names[id(p)]: p.detach().float().clone() for p in model_a.parameters()}
    buf_a = {n: b.detach().float().clone() for n, b in model_a.named_buffers()}

    # --- side B: the diloco simulator, dp genome (M=1, h=1, sgd lr1 beta0) ---
    model_b, _ = tcp.build_model(args, vocab, device, args.seed)
    names_b = {id(p): n for n, p in model_b.named_parameters()}
    loss_b = []; hash_b = []
    record_losses(model_b, loss_b, hash_b)
    td.run_policy(genome, args, model_b, [train], val, device)
    snap_b = {names_b[id(p)]: p.detach().float().clone() for p in model_b.parameters()}
    buf_b = {n: b.detach().float().clone() for n, b in model_b.named_buffers()}

    print("\nper-step param hashes at forward entry (wte, lm_head):")
    for i,(ha,hb) in enumerate(zip(hash_a,hash_b),1):
        m = "" if ha==hb else "   <-- DIFFERS"
        print(f"  step {i}: A{ha} B{hb}{m}")
    print("\nper-step TRAIN losses (A=run_arm, B=run_policy):")
    for i, (la, lb) in enumerate(zip(loss_a, loss_b), 1):
        mark = "" if la == lb else "   <-- DIFFERS"
        print(f"  step {i}: {la:.9f}  {lb:.9f}{mark}")
    if len(loss_a) != len(loss_b):
        print(f"  (count mismatch: A={len(loss_a)} B={len(loss_b)})")
    for n in buf_a:
        d = (buf_a[n] - buf_b[n]).abs().max().item() if n in buf_b else float("nan")
        if d > 0 or d != d:
            print(f"  buffer differs: {n} {d:.3e}")

    mp_names = {names_b[id(p)] for p in tcp.matrix_params(model_b)}
    print(f"\n{'param':44s} {'group':7s} {'max|diff|':>12s}")
    worst = []
    for n in sorted(snap_a):
        d = (snap_a[n] - snap_b[n]).abs().max().item()
        grp = "matrix" if n in mp_names else "other"
        worst.append((d, n, grp))
        if d > 0:
            print(f"{n:44s} {grp:7s} {d:12.3e}")
    worst.sort(reverse=True)
    nz = [w for w in worst if w[0] > 0]
    print(f"\n{len(nz)}/{len(worst)} params differ after {N_STEPS} step(s)")
    if nz:
        print("worst:", nz[0])
    else:
        print("BIT-IDENTICAL after", N_STEPS, "step(s)")

    # THE GATE: loss-prefix bit-identity. Same-process, both harnesses, identical inputs;
    # divergence beyond the prefix is cross-program cuBLAS/allocator ulp drift (documented
    # in check_diloco_gates.py), so the wiring claim rests on the exact prefix match.
    k = sum(1 for la, lb in zip(loss_a, loss_b) if la == lb)
    print(f"\nloss-prefix identity: {k}/{min(len(loss_a), len(loss_b))} steps ({MODE})")
    # Thresholds: ga1 k>=3 proves one full delta->outer->copy-back cycle exact; ga2 k>=6 proves
    # the step-5 sync cycle exact. Beyond the prefix, cross-program cuBLAS/allocator context
    # causes ulp-level drift (same-program runs are bit-identical; see check_diloco_gates.py),
    # so the first divergence must be ulp-scale, not wiring-scale.
    need = 3 if MODE == "ga1" else 6
    assert k >= need, f"{MODE}: wiring divergence within {k} steps (need >= {need}) — REAL bug"
    if k < len(loss_a):
        rel = abs(loss_a[k] - loss_b[k]) / abs(loss_a[k])
        print(f"first divergence at step {k+1}: rel {rel:.2e}")
        assert rel < 1e-3, f"first divergence too large ({rel:.2e}) — wiring-scale, not ulp drift"
    print(f"{MODE} EQUIV GATE PASS")


if __name__ == "__main__":
    main()
