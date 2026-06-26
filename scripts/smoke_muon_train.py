"""Minimal end-to-end smoke for the GNS Muon integration: drive a tiny real GPT through
a few optimizer steps for several arms on random token batches (no data pipeline needed).

Validates that setup_optimizer(muon_orth=...) and the eager pluggable Muon path work
inside a real GPT training loop on the actual matrix-parameter shapes, on this device.

Run: PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src \
     uv run --project /path/to/tct-models python scripts/smoke_muon_train.py [cuda|cpu]
"""
import sys
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.muon_schedules import get_orth_method, set_fp8_real

set_fp8_real(False)  # Round-1 faithful simulated fp8 (no _scaled_mm shape constraints)

DEV = "cuda" if (torch.cuda.is_available() and (len(sys.argv) < 2 or sys.argv[1] != "cpu")) else "cpu"
print(f"device: {DEV}")

cfg = GPTConfig(sequence_len=64, vocab_size=128, n_layer=2, n_head=2, n_kv_head=2,
                n_embd=128, window_pattern="SL")
B, T, STEPS, LR = 8, 64, 15, 0.02

ARMS = ["fused", "polar_express", "svd", "none",
        "jordan5_bf16_control", "all_fp8_control", "frontier_cost_6p625"]


def run_arm(arm):
    torch.manual_seed(0)
    model = GPT(cfg).to(DEV)
    orth = "fused" if arm == "fused" else get_orth_method(arm)
    opt = model.setup_optimizer(matrix_lr=LR, weight_decay=0.0, muon_orth=orth)
    g = torch.Generator(device=DEV).manual_seed(123)
    losses = []
    for _ in range(STEPS):
        idx = torch.randint(0, cfg.vocab_size, (B, T), generator=g, device=DEV)
        tgt = torch.randint(0, cfg.vocab_size, (B, T), generator=g, device=DEV)
        loss = model(idx, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


ok = True
print(f"{'arm':24s} {'loss[0]':>9s} {'loss[-1]':>9s}  finite  Δ")
for arm in ARMS:
    try:
        L = run_arm(arm)
        finite = all(x == x and abs(x) < 1e9 for x in L)
        ok = ok and finite
        print(f"{arm:24s} {L[0]:9.4f} {L[-1]:9.4f}  {str(finite):5s}  {L[-1]-L[0]:+.4f}")
    except Exception as e:
        ok = False
        print(f"{arm:24s} ERROR: {type(e).__name__}: {e}")

print(f"\nSMOKE: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
