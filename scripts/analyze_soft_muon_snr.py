#!/usr/bin/env python3
"""Idea 3 / Bet B estimator-gate readout (E1 soft_muon_snr, E2 soft_muon_mp) — sibling of analyze_soft_muon.py.

Reads scripts/soft_muon_snr_probe.sh outputs:
  strength_sweep : final-val per strength + Δ = val(strength=0) − val(strength)  (>0 ⇒ SNR shrinkage helps).
  grid           : Δ(batch,width) = val(anchor soft_muon@τ=0) − val(arm) + the three gate checks.

Anchor is soft_muon@τ=0 (exact-SVD polar), identical to soft_muon_snr@strength=0 — isolates pure shrinkage
from the ~0.002 svd-vs-fused gap. Reuses analyze_soft_muon._final_val so the JSON parsing has one source.

Usage:
  python scripts/analyze_soft_muon_snr.py strength_sweep [--seed 0] [--results DIR]
  python scripts/analyze_soft_muon_snr.py grid --arm soft_muon_snr [--seed 0] [--results DIR]
"""
from __future__ import annotations
import argparse

from analyze_soft_muon import _final_val      # scripts/ is on sys.path[0] when run as a script

DEF_RESULTS = "/home/jonas/git/gns/results"


def strength_sweep(results, seed):
    strengths = ["0.0", "0.5", "1.0", "2.0", "4.0"]
    base = _final_val(f"{results}/soft_muon_snr_sweep_s0.0_seed{seed}.json")
    print(f"\nsoft-Muon-SNR strength-sweep  (d6 a64 b16, seed {seed})")
    print(f"  baseline strength=0 (exact-SVD polar ≡ Muon): {base if base is None else f'{base:.4f}'}")
    print(f"\n  {'strength':>9} {'final_val':>10} {'Δ=val(0)−val(s)':>16}  (Δ>0 ⇒ SNR shrinkage helps)")
    best = (None, -1e9)
    for s in strengths:
        v = _final_val(f"{results}/soft_muon_snr_sweep_s{s}_seed{seed}.json")
        if v is None:
            print(f"  {s:>9} {'—':>10} {'—':>16}")
            continue
        d = (base - v) if base is not None else float("nan")
        if d == d and s != "0.0" and d > best[1]:
            best = (s, d)
        print(f"  {s:>9} {v:>10.4f} {d:>16.4f}")
    print()
    if best[0] is None or best[1] <= 0:
        print("  VERDICT: no strength>0 beats strength=0 → no empirical-SNR lever here (mirrors the τ=0 / lr=0 deaths). STOP.")
    else:
        print(f"  VERDICT: best strength={best[0]} beats 0 by {best[1]:.4f} → carry into the grid.")
    print()


def grid(results, seed, arm):
    aspects = [64, 96, 128]
    batches = [8, 16, 64]
    print(f"\n{arm} batch×width grid  (d6/400, seed {seed})  Δ = val(soft_muon@τ=0) − val({arm})")
    print("  rows = width, cols = batch.  Δ>0 ⇒ shrinkage helps.\n")
    D = {}
    print("  width\\batch " + "".join(f"{b:>10}" for b in batches))
    for a in aspects:
        cells = []
        for b in batches:
            v0 = _final_val(f"{results}/soft_muon_snr_grid_a{a}_b{b}_anchor_s{seed}.json")
            vt = _final_val(f"{results}/soft_muon_snr_grid_a{a}_b{b}_{arm}_s{seed}.json")
            d = (v0 - vt) if (v0 is not None and vt is not None) else None
            D[(a, b)] = d
            cells.append("—".rjust(10) if d is None else f"{d:>+10.4f}")
        print(f"  a{a:<4}({a*6:>4}d)" + "".join(cells))

    def trend(vals):
        vals = [x for x in vals if x is not None]
        return (vals[-1] - vals[0]) if len(vals) >= 2 else None

    print("\n  GATE checks (need all three):")
    base_ok = D.get((64, 16)) is not None and D[(64, 16)] > 0
    print(f"   (1) Δ>0 at base (a64,b16): {D.get((64,16))} -> {'PASS' if base_ok else 'FAIL'}")
    bt = {a: trend([D.get((a, b)) for b in batches]) for a in aspects}
    batch_ok = all(t is not None and t < 0 for t in bt.values())
    print(f"   (2) Δ shrinks as batch↑: {bt} -> {'PASS' if batch_ok else 'CHECK'}")
    wt = {b: trend([D.get((a, b)) for a in aspects]) for b in batches}
    width_ok = all(t is not None and t > 0 for t in wt.values())
    print(f"   (3) Δ grows as width↑: {wt} -> {'PASS' if width_ok else 'CHECK'}")
    print()
    if base_ok and batch_ok and width_ok:
        print("  SIGNATURE: live-denoiser (Δ>0, shrinks w/ batch, GROWS w/ width) → seed-confirm + Phase B.")
    elif all(t is not None and t < 0 for t in wt.values()):
        print("  SIGNATURE: Δ shrinks with WIDTH too → same vanishing axis as curvature. STOP.")
    else:
        print("  SIGNATURE: mixed/incomplete — inspect the cells above.")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("strength_sweep", "grid"))
    ap.add_argument("--seed", default="0")
    ap.add_argument("--arm", default="soft_muon_snr", choices=("soft_muon_snr", "soft_muon_mp"))
    ap.add_argument("--results", default=DEF_RESULTS)
    a = ap.parse_args()
    if a.mode == "strength_sweep":
        strength_sweep(a.results, a.seed)
    else:
        grid(a.results, a.seed, a.arm)
