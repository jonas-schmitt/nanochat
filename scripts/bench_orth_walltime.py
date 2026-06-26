"""Round 2 / 2b — Blackwell (GB10) orthogonalization wall-clock microbench.

For each arm, time the orthogonalization map U -> O on large GPU matrices (matmul cost
dominates Python overhead), with REAL fp8 (_scaled_mm) enabled. Answers the cost->wall-clock
question: does a precision-aware GNS schedule reach lower wall-clock than the bf16 baselines
on tier-2 Blackwell? Reproduces exp10's mechanism (1.30x on RTX 4070) on the GB10.

Run: PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src \
     uv run --project /path/to/tct-models python scripts/bench_orth_walltime.py
"""
import os, sys, time, json
import torch
from nanochat.optim import orthogonalize_eager
from nanochat.muon_schedules import build_registry, set_fp8_real

# CAVEAT: orthogonalize_eager runs a per-matrix Python loop (the experiment harness, not the
# fused/compiled production kernel). These ms/call numbers therefore carry per-call Python
# overhead and are an UPPER BOUND on a true batched-fused-fp8 schedule. They rank the arms'
# relative orthogonalization cost on real GB10 fp8 matmuls; they are NOT end-to-end training
# wall-clock (that needs the batched fused kernel, not yet built — see TODO Open questions).
assert torch.cuda.is_available(), "need the GB10"
_cap = torch.cuda.get_device_capability(0)
_fp8_capable = _cap[0] >= 89  # fp8 _scaled_mm requires sm_89+ (Ada/Hopper/Blackwell)
real = set_fp8_real(True) if _fp8_capable else False
print(f"GPU: {torch.cuda.get_device_name(0)} (sm_{_cap[0]}{_cap[1]}); "
      f"fp8 real (_scaled_mm) available/enabled: {real}", flush=True)
DEV = "cuda"
NS = int(os.environ.get("BENCH_NS", "5"))
SHAPES = [(4096, 4096), (2048, 8192), (8192, 2048)]
ITERS = int(os.environ.get("BENCH_ITERS", "20"))
REG = build_registry()
ARMS = ["polar_express", "jordan5_bf16_control", "all_fp8_control",
        "frontier_cost_6p625", "frontier_cost_7p75", "frontier_cost_8p75",
        "frontier_cost_8p875", "svd"]


def timed(method, X):
    for _ in range(3):  # warmup
        orthogonalize_eager(X, method, NS)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        orthogonalize_eager(X, method, NS)
    torch.cuda.synchronize()
    return (time.time() - t0) / ITERS * 1e3  # ms/call


results = {}
for (m, n) in SHAPES:
    X = torch.randn(1, m, n, device=DEV)  # stacked K=1
    print(f"\nshape {m}x{n} (ms/call, {ITERS} iters):", flush=True)
    base = None
    rows = {}
    for arm in ARMS:
        method = "polar_express" if arm == "polar_express" else REG[arm]
        try:
            ms = timed(method, X)
            if arm == "polar_express":
                base = ms
            rows[arm] = ms
            spd = (base / ms) if base else float("nan")
            print(f"  {arm:24s} {ms:8.3f} ms   speedup vs polar {spd:5.2f}x", flush=True)
        except Exception as e:
            print(f"  {arm:24s} ERROR {type(e).__name__}: {e}", flush=True)
            rows[arm] = None
    results[f"{m}x{n}"] = rows

out = "/home/jonas/git/gns/results/gb10_orth_walltime.json"
json.dump({"device": torch.cuda.get_device_name(0), "fp8_real": real,
           "iters": ITERS, "ns_steps": NS, "ms_per_call": results}, open(out, "w"), indent=1)
print(f"\nsaved -> {out}")
