"""Analyze the adaptation campaign — recompute the one-sided t-test from stored per-seed gaps (so the
verdict is independent of which code wrote the flags) and report the decisive number per adaptation:

  A1 batch x scale : gap(d12, batch) — does the curvature win grow with batch?
  A3 alpha         : gap(depth, alpha) — is there an alpha*(depth) that holds at d12?
  A4 SOAP          : gap(depth) vs muon
  A0 shrink        : gap(depth) per shrinkage level

  cd /home/jonas/git/nanochat && python scripts/analyze_adaptations.py
"""
import json
import math
import re
from pathlib import Path

import numpy as np

RES = Path("/home/jonas/git/gns/results")
TCRIT = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

# ---- implementation-independent matmul-FLOP overhead (audit: harness wall-clock OVER-penalises the ----
# curvature arms because the optimizer's matmuls are GPU-inefficient as implemented, not because they
# dominate FLOPs). We report TOTAL-STEP FLOPs = fwd/bwd (6·N·tokens) + optimizer matmuls, as a ratio vs
# muon. This is the number the iso-FLOP accuracy question actually needs. All counts in MACs (the 2×/MAC
# cancels in the ratio). Documented assumptions; ortho overhead is sensitive to the recompute interval.
NS_STEPS_DEFAULT = 5        # --ns-steps (muon/ortho polar iterations)
RECOMPUTE_DEFAULT = 10      # --shampoo-recompute-every (inverse-root amortisation interval)
COUPLED_STEPS_DEFAULT = 24  # --shampoo-coupled-steps (uniform fallback chain length)
_POLAR_STEPS = {"muon_4step": 4, "muon_3step": 3}  # cost arms: fixed polar step count (else ns_steps)


def cfg(tag):
    p = RES / f"scaling_ladder_{tag}.json"
    return json.loads(p.read_text()).get("config", {}) if p.exists() else {}


def _coupled_raw_matmuls(orders, k):
    """Raw matmul count of one inverse-4th-root, matching gns.coupled.schedule_cost (sans prec mult):
    base = ceil(log2(root=4)) + 2 = 4 ; each CoupledStep = base + (order-1)."""
    base = math.ceil(math.log2(4)) + 2  # = 4
    return sum(base + (o - 1) for o in orders) if orders else k * base  # default order=1 -> base


def _block_matrices(d):
    """Dominant 2D matrices optimised per transformer layer (nanochat gpt.py), as (out,in):
    attn c_q,c_proj = (d,d); MLP c_fc = (4d,d), c_proj = (d,4d). (small c_k/c_v omitted — they
    barely move the ratio.)"""
    return [(d, d), (d, d), (4 * d, d), (d, 4 * d)]


def _opt_macs_per_step(arm, d, depth, cf):
    """Optimizer matmul-MACs/step summed over the layer's dominant matrices × n_layer (=depth).
    Precision-independent (fp8 has the SAME MACs as bf16 — its win is tensor-core throughput)."""
    ns = cf.get("ns_steps", NS_STEPS_DEFAULT)
    recompute = cf.get("shampoo_recompute_every", RECOMPUTE_DEFAULT)
    oo = cf.get("precond_coupled_orders") or ""
    orders = [int(x) for x in oo.split(",")] if oo else None
    raw = _coupled_raw_matmuls(orders, cf.get("shampoo_coupled_steps", COUPLED_STEPS_DEFAULT))
    steps = _POLAR_STEPS.get(arm, ns)
    tot = 0.0
    for (M, N) in _block_matrices(d):
        b = min(M, N)
        polar = 2 * M * N * b + b ** 3                      # one NS polar step
        if arm in ("muon", "muon_fp8", "muon_4step", "muon_3step"):
            tot += steps * polar
        elif arm in ("ortho_shampoo", "synth"):
            tot += ns * polar + (M * M * N + M * N * N) + raw * (M ** 3 + N ** 3) / recompute
        elif arm == "shampoo":
            tot += (M * M * N + M * N * N) + raw * (M ** 3 + N ** 3) / recompute
        else:
            return None  # lowrank_orth (SVD), adamw, sgd — not modelled
    return tot * depth


def flop_overhead(arm, r, cf):
    """TOTAL-step-FLOP overhead of `arm` vs muon at rung r (fwd/bwd 6·N·tok + optimizer). Returns
    (total_ratio, opt_only_multiple) or (None,None)."""
    d, depth = r.get("width"), r.get("depth")
    nonembed = r.get("nonembed_params")
    if d is None or nonembed is None:
        return None, None
    toks = cf.get("device_batch_size", 16) * cf.get("max_seq_len", 1024)
    fb = 3.0 * nonembed * toks  # fwd+bwd in MACs (6·N·tok FLOPs = 3·N·tok MACs)
    a, mu = _opt_macs_per_step(arm, d, depth, cf), _opt_macs_per_step("muon", d, depth, cf)
    if a is None or not mu:
        return None, None
    return (fb + a) / (fb + mu), a / mu


def rungs(tag):
    p = RES / f"scaling_ladder_{tag}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("pass_fixed", {}).get("rungs", [])


def stat(cc):
    g = np.array(cc["gap_seeds"]); n = len(g); m = g.mean()
    sd = g.std(ddof=1) if n > 1 else 0.0
    t = m / (sd / np.sqrt(n)) if sd > 0 else 0.0
    sig = m < 0 and t < -TCRIT.get(n - 1, 2.0)
    return m, sd, t, n, sig


def line(label, cc):
    m, sd, t, n, sig = stat(cc)
    print(f"    {label:22s} gap {m:+.4f} ±{sd:.4f} (n={n}) t={t:+.2f}  {'** SIGNIF' if sig else 'n.s.'}")


def main():
    tags = sorted(p.stem.replace("scaling_ladder_", "") for p in RES.glob("scaling_ladder_*.json"))
    print("available result tags:", tags)

    print("\n========== A1 — batch x scale (does the curvature win grow with batch?) ==========")
    for bs in ("16", "64", "128", "256"):
        rs = rungs(f"batch{bs}")
        if not rs:
            continue
        for r in sorted(rs, key=lambda r: r["depth"]):
            for c, cc in r["candidates"].items():
                line(f"batch{bs} d{r['depth']} {c}", cc)

    print("\n========== A3 — curvature strength alpha*(depth) ==========")
    for a in ("0p25", "0p50", "0p75", "1p00"):
        rs = rungs(f"alpha{a}")
        if not rs:
            continue
        for r in sorted(rs, key=lambda r: r["depth"]):
            for c, cc in r["candidates"].items():
                line(f"alpha{a} d{r['depth']} {c}", cc)

    print("\n========== A4 — SOAP vs muon ==========")
    for r in sorted(rungs("soap") or [], key=lambda r: r["depth"]):
        for c, cc in r["candidates"].items():
            line(f"soap d{r['depth']} {c}", cc)

    print("\n========== A0 — shrinkage spread (ortho_shampoo at each level) ==========")
    for lv in ("1e6", "1e4", "1e3", "1e2"):
        rs = rungs(f"shrink{lv}")
        if not rs:
            continue
        for r in sorted(rs, key=lambda r: r["depth"]):
            for c, cc in r["candidates"].items():
                line(f"shrink{lv} d{r['depth']} {c}", cc)

    print("\n========== A2 — under-training check (longer d12) ==========")
    for r in rungs("d12_long") or []:
        for c, cc in r["candidates"].items():
            line(f"d12_long d{r['depth']} {c}", cc)

    print("\n========== W — WIDTH scaling (does the curvature win grow with WIDTH at fixed depth?) ==========")
    print("    (compare to camp_curv at aspect 64 = the depth-scaling reference. WIN = gap more negative")
    print("     as aspect grows within a fixed depth — the d12 erosion would be a depth-specific artifact.)")
    for tag in ("width_d6_a96", "width_d6_a128", "width_d8_a96", "width_d8_a128"):
        rs = rungs(tag)
        if not rs:
            continue
        cf = cfg(tag)
        for r in sorted(rs, key=lambda r: r["depth"]):
            for c, cc in r["candidates"].items():
                m, sd, t, n, sig = stat(cc)
                fr, om = flop_overhead(c, r, cf)
                fstr = f"  flop={fr:.2f}x" if fr is not None else ""
                print(f"    {tag} d{r['depth']} {c:14s} gap {m:+.4f} ±{sd:.4f} (n={n}) t={t:+.2f}"
                      f"  {'** SIGNIF' if sig else 'n.s.'}{fstr}")

    print("\n========== F — orthogonalization FREQUENCY (does less frequent orth hold at d12?) ==========")
    print("    (both arms at the same K; WIN = ortho_shampoo gap at d12 becomes significant as K grows.)")
    for k in ("2", "4", "8"):
        rs = rungs(f"orthK{k}_d12")
        if not rs:
            continue
        for r in sorted(rs, key=lambda r: r["depth"]):
            for c, cc in r["candidates"].items():
                line(f"orthK{k}_d12 d{r['depth']} {c}", cc)

    print("\n========== A-alloc — per-factor kappa allocation (layer_adaptive vs muon) ==========")
    print("    (compare to camp_curv ortho_shampoo at the same depth = uniform-curvature reference)")
    for r in sorted(rungs("layer_adapt") or [], key=lambda r: r["depth"]):
        for c, cc in r["candidates"].items():
            line(f"layer_adapt d{r['depth']} {c}", cc)
    for r in sorted(rungs("camp_curv") or [], key=lambda r: r["depth"]):
        for c, cc in r["candidates"].items():
            if c == "ortho_shampoo":
                line(f"[ref] camp_curv d{r['depth']} {c}", cc)

    print("\n========== C — cheaper/faster Muon (directions 1,2: cost-reduction arms) ==========")
    print("    (WIN = same quality as muon at lower cost. wall_overhead = harness wall-clock (impl-")
    print("     dependent, over-penalises curvature); flop = TOTAL-step-FLOP ratio vs muon (impl-INDEP,")
    print("     opt=optimizer-only matmul multiple). fp8: same FLOPs, ~1.8-2.6x tensor-core throughput.)")
    for tag in ("cost_4step_d8", "cost_4step_d12", "cost_3step_d8",
                "fp8_d8", "fp8_d12",
                "lowrank_d8", "lowrank_d12",
                "lowrank_k32_d8", "lowrank_k32_d12"):
        rs = rungs(tag)
        if not rs:
            continue
        cf = cfg(tag)
        for r in sorted(rs, key=lambda r: r["depth"]):
            for c, cc in r["candidates"].items():
                m, sd, t, n, sig = stat(cc)
                oh = cc.get("overhead", float("nan"))
                fr, om = flop_overhead(c, r, cf)
                fstr = (f"  flop={fr:.2f}x (opt {om:.1f}x)" if fr is not None else "  flop=n/a")
                fp8 = "  [fp8≈same-FLOPs,~2x throughput]" if "fp8" in c else ""
                print(f"    {tag} d{r['depth']} {c:16s} gap {m:+.4f} ±{sd:.4f} (n={n}) t={t:+.2f}"
                      f"  {'** SIGNIF' if sig else 'n.s.'}  wall={oh:+.1%}{fstr}{fp8}")

    print("\n========== E — eigenbasis composition (direction 6: polar within Kronecker eigenbasis) ==========")
    print("    (compare to camp_curv ortho_shampoo at the same depth = standard-basis reference)")
    for tag in ("eigen_d8", "eigen_d12"):
        rs = rungs(tag)
        if not rs:
            continue
        for r in sorted(rs, key=lambda r: r["depth"]):
            for c, cc in r["candidates"].items():
                line(f"{tag} d{r['depth']} {c}", cc)

    print("\n========== AN — annealed alpha (direction 7: adaptive curvature strength over training) ==========")
    print("    (compare to alpha* stages at the same final alpha = static reference)")
    for tag in ("alpha_anneal05_d8", "alpha_anneal05_d12",
                "alpha_anneal10_d8", "alpha_anneal10_d12"):
        rs = rungs(tag)
        if not rs:
            continue
        for r in sorted(rs, key=lambda r: r["depth"]):
            for c, cc in r["candidates"].items():
                line(f"{tag} d{r['depth']} {c}", cc)


if __name__ == "__main__":
    main()
