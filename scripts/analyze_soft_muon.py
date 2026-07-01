#!/usr/bin/env python3
"""Idea 3 soft-Muon (spectral-denoising) readout — Phase A.

Reads the per-run jsons written by scripts/soft_muon_probe.sh and prints:
  tau_sweep : final-val per tau + the pure-shrinkage Δ = val(tau=0) − val(tau)  (>0 ⇒ shrinkage helps).
  grid      : the Δ(batch,width) signature table + the three gate checks.

The shrinkage baseline is soft_muon@tau=0 (exact SVD polar), NOT the muon arm — muon's 5-step polar adds
the ~0.002 svd-vs-fused map gap that would otherwise contaminate Δ. muon is shown only as an anchor.

Usage:
  python scripts/analyze_soft_muon.py tau_sweep [--seed 0] [--results DIR]
  python scripts/analyze_soft_muon.py grid --tau 0.1 [--seed 0] [--results DIR]
"""
from __future__ import annotations
import argparse, glob, json, os

DEF_RESULTS = "/home/jonas/git/gns/results"


def _final_val(path):
    """final val for a single-arm run (falls back to best_val); None if missing/NaN."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    arms = d.get("arms") or {}
    if not arms:
        return None
    rec = next(iter(arms.values()))            # single arm per soft-muon run
    v = rec.get("final_val", rec.get("best_val"))
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if v != v else v               # NaN guard


def tau_sweep(results, seed):
    taus = ["0.0", "0.05", "0.1", "0.2", "0.4"]
    muon = _final_val(f"{results}/soft_muon_tau_sweep_muon_s{seed}.json")
    base = _final_val(f"{results}/soft_muon_tau_sweep_t0.0_s{seed}.json")
    print(f"\nsoft-Muon τ-sweep  (d6 a64 b16, seed {seed})")
    print(f"  anchor muon (5-step polar): {muon if muon is None else f'{muon:.4f}'}")
    print(f"  baseline soft_muon@τ=0 (exact SVD polar): {base if base is None else f'{base:.4f}'}")
    print(f"\n  {'τ':>6} {'final_val':>10} {'Δ=val(0)−val(τ)':>16}  (Δ>0 ⇒ shrinkage helps)")
    best = (None, -1e9)
    for t in taus:
        v = _final_val(f"{results}/soft_muon_tau_sweep_t{t}_s{seed}.json")
        if v is None:
            print(f"  {t:>6} {'—':>10} {'—':>16}")
            continue
        d = (base - v) if (base is not None) else float("nan")
        flag = "  <= best" if (d == d and d > best[1] and t != "0.0") else ""
        if d == d and t != "0.0" and d > best[1]:
            best = (t, d)
        print(f"  {t:>6} {v:>10.4f} {d:>16.4f}{flag}")
    print()
    if best[0] is None:
        print("  VERDICT: no τ>0 evaluated. ")
    elif best[1] <= 0:
        print("  VERDICT: best τ>0 does NOT beat τ=0 → no spectral lever here (mirrors Idea-2 lr=0). STOP.")
    else:
        print(f"  VERDICT: best τ={best[0]} beats τ=0 by {best[1]:.4f} → carry τ={best[0]} into the grid (A3).")
    print()


def grid(results, seed, tau):
    aspects = [64, 96, 128]          # width: model_dim 384 / 576 / 768
    batches = [8, 16, 64]
    tstr = str(tau)
    print(f"\nsoft-Muon batch×width grid  (d6/400, seed {seed})  Δ = val(τ=0) − val(τ={tstr})")
    print(f"  rows = aspect/width, cols = device batch.  Δ>0 ⇒ shrinkage helps.\n")
    D = {}
    hdr = "  width\\batch " + "".join(f"{b:>10}" for b in batches)
    print(hdr)
    for a in aspects:
        cells = []
        for b in batches:
            v0 = _final_val(f"{results}/soft_muon_grid_a{a}_b{b}_t0_s{seed}.json")
            vt = _final_val(f"{results}/soft_muon_grid_a{a}_b{b}_t{tstr}_s{seed}.json")
            d = (v0 - vt) if (v0 is not None and vt is not None) else None
            D[(a, b)] = d
            cells.append("—".rjust(10) if d is None else f"{d:>+10.4f}")
        print(f"  a{a:<4}({a*6:>4}d)" + "".join(cells))

    # Gate checks
    def trend(vals):
        vals = [x for x in vals if x is not None]
        return (vals[-1] - vals[0]) if len(vals) >= 2 else None

    print("\n  GATE checks (need all three):")
    base_ok = D.get((64, 16)) is not None and D[(64, 16)] > 0
    print(f"   (1) Δ>0 at base cell (a64,b16): {D.get((64,16))}  -> {'PASS' if base_ok else 'FAIL'}")
    # (2) shrinks with batch at fixed width
    batch_trends = {a: trend([D.get((a, b)) for b in batches]) for a in aspects}
    batch_ok = all(t is not None and t < 0 for t in batch_trends.values())
    print(f"   (2) Δ shrinks as batch↑ (per width, Δ[b64]−Δ[b8]<0): {batch_trends} -> {'PASS' if batch_ok else 'CHECK'}")
    # (3) grows with width at fixed batch
    width_trends = {b: trend([D.get((a, b)) for a in aspects]) for b in batches}
    width_ok = all(t is not None and t > 0 for t in width_trends.values())
    print(f"   (3) Δ grows as width↑ (per batch, Δ[a128]−Δ[a64]>0): {width_trends} -> {'PASS' if width_ok else 'CHECK'}")
    print()
    if base_ok and batch_ok and width_ok:
        print("  SIGNATURE: live-denoiser (Δ>0, shrinks w/ batch, GROWS w/ width) → seed-confirm + Phase B.")
    elif width_ok is False and all(t is not None and t < 0 for t in width_trends.values()):
        print("  SIGNATURE: Δ shrinks with WIDTH too → same vanishing axis as curvature. STOP.")
    else:
        print("  SIGNATURE: mixed/incomplete — inspect cells above before deciding.")
    print()


def controls(results, seed):
    """E3 (soft_muon --soft-no-renorm tau-sweep) + E1 (soft_muon_snr strength-sweep) deciders."""
    def show(title, base_path, points, path_fn, label):
        base = _final_val(base_path)
        print(f"\n{title}")
        print(f"  {label:>9} {'final_val':>10} {'Δ=val(0)−val(·)':>16}  (Δ>0 ⇒ gate helps)")
        best = (None, 0.0)
        for pt in points:
            v = _final_val(path_fn(pt))
            if v is None:
                print(f"  {pt:>9} {'—':>10} {'—':>16}"); continue
            d = (base - v) if base is not None else float("nan")
            if d == d and pt != points[0] and d > best[1]:
                best = (pt, d)
            print(f"  {pt:>9} {v:>10.4f} {d:>16.4f}")
        return best

    e3 = show("E3 control — soft_muon, --soft-no-renorm (d6/a64/b16/400)",
              f"{results}/soft_muon_norenorm_t0.0_s{seed}.json", ["0.0", "0.1", "0.2"],
              lambda t: f"{results}/soft_muon_norenorm_t{t}_s{seed}.json", "τ")
    e1 = show("E1 — soft_muon_snr, --soft-no-renorm (estimated per-direction SNR)",
              f"{results}/soft_muon_snr_norenorm_str0.0_s{seed}.json", ["0.0", "0.5", "1.0", "2.0"],
              lambda s: f"{results}/soft_muon_snr_norenorm_str{s}_s{seed}.json", "strength")
    print("\n  VERDICT:")
    print(f"   E3: {'renorm was NOT masking — σ-magnitude shrinkage robustly hurts' if (e3[0] is None or e3[1] <= 0) else f'no-renorm REVIVES it (best τ={e3[0]}, +{e3[1]:.4f}) → A2 was renorm-masking'}")
    if e1[0] is None or e1[1] <= 0:
        print("   E1: estimated-SNR gating does NOT beat strength=0 → spectral-denoising bet DEAD (spectrum is signal). STOP.")
    else:
        print(f"   E1: strength={e1[0]} beats 0 by {e1[1]:.4f} → ALIVE; carry to the batch×width grid (where batch-adaptivity is tested).")
    print()


def roles(results, seed):
    """Idea 4 muon_roles d6 screen: each candidate vs muon anchor AND muon_lookahead floor."""
    muon = _final_val(f"{results}/roles_muon_s{seed}.json")
    look = _final_val(f"{results}/roles_lookahead_s{seed}.json")
    cands = ["io_split", "out_damp", "mlp_up", "attn_up", "mlp_dom"]
    print(f"\nIdea 4 — muon_roles d6 screen (d6/a64/b16/400, seed {seed})")
    print(f"  anchor muon={_fmt(muon)}   floor muon_lookahead={_fmt(look)}")
    print(f"\n  {'config':>10} {'final_val':>10} {'Δ vs muon':>10} {'Δ vs look':>10}  (Δ>0 ⇒ better)")
    best = (None, -1e9)
    for c in cands:
        v = _final_val(f"{results}/roles_{c}_s{seed}.json")
        if v is None:
            print(f"  {c:>10} {'—':>10} {'—':>10} {'—':>10}"); continue
        dm = (muon - v) if muon is not None else float("nan")
        dl = (look - v) if look is not None else float("nan")
        if dm == dm and dm > best[1]:
            best = (c, dm)
        print(f"  {c:>10} {v:>10.4f} {dm:>+10.4f} {dl:>+10.4f}")
    print()
    if best[0] is None:
        print("  VERDICT: no candidate completed.")
    elif best[1] <= 0:
        print("  VERDICT: no role vector beats uniform Muon → Idea 4 FALSIFIED at d6. STOP.")
    else:
        vl = _final_val(f"{results}/roles_{best[0]}_s{seed}.json")
        beats_look = (look is not None and vl < look)
        print(f"  VERDICT: best={best[0]} beats muon by {best[1]:+.4f}; "
              f"{'ALSO beats lookahead → carry to width/scale gate.' if beats_look else 'but does NOT beat lookahead — weak.'}")
    print()


def e1_sweep(results, seed):
    """E1 soft_muon_snr strength-sweep (renorm-on, the user's soft_muon_snr_probe.sh naming)."""
    pts = ["0.0", "0.5", "1.0", "2.0", "4.0"]
    base = _final_val(f"{results}/soft_muon_snr_sweep_s0.0_seed{seed}.json")
    print(f"\nE1 — soft_muon_snr strength-sweep (d6/a64/b16/400, seed {seed}); baseline strength=0 = exact-SVD polar")
    print(f"  {'strength':>9} {'final_val':>10} {'Δ=val(0)−val(·)':>16}  (Δ>0 ⇒ SNR gate helps)")
    best = (None, 0.0)
    for s in pts:
        v = _final_val(f"{results}/soft_muon_snr_sweep_s{s}_seed{seed}.json")
        if v is None:
            print(f"  {s:>9} {'—':>10} {'—':>16}"); continue
        d = (base - v) if base is not None else float("nan")
        if d == d and s != "0.0" and d > best[1]:
            best = (s, d)
        print(f"  {s:>9} {v:>10.4f} {d:>16.4f}")
    print()
    if best[0] is None or best[1] <= 0:
        print("  VERDICT: no strength>0 beats strength=0 → E1 (empirical-SNR gate) FALSIFIED at d6. STOP.")
    else:
        print(f"  VERDICT: strength={best[0]} beats 0 by {best[1]:.4f} → ALIVE; carry to batch×width.")
    print()


def _fmt(v):
    return "—" if v is None else f"{v:.4f}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("tau_sweep", "grid", "controls", "roles", "e1_sweep"))
    ap.add_argument("--seed", default="0")
    ap.add_argument("--tau", default="0.1", help="grid mode: the swept τ compared against τ=0")
    ap.add_argument("--results", default=DEF_RESULTS)
    a = ap.parse_args()
    {"tau_sweep": lambda: tau_sweep(a.results, a.seed),
     "grid": lambda: grid(a.results, a.seed, a.tau),
     "controls": lambda: controls(a.results, a.seed),
     "roles": lambda: roles(a.results, a.seed),
     "e1_sweep": lambda: e1_sweep(a.results, a.seed)}[a.mode]()
