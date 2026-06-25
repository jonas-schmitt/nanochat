"""Analyze the adaptation campaign — recompute the one-sided t-test from stored per-seed gaps (so the
verdict is independent of which code wrote the flags) and report the decisive number per adaptation:

  A1 batch x scale : gap(d12, batch) — does the curvature win grow with batch?
  A3 alpha         : gap(depth, alpha) — is there an alpha*(depth) that holds at d12?
  A4 SOAP          : gap(depth) vs muon
  A0 shrink        : gap(depth) per shrinkage level

  cd /home/jonas/git/nanochat && python scripts/analyze_adaptations.py
"""
import json
import re
from pathlib import Path

import numpy as np

RES = Path("/home/jonas/git/gns/results")
TCRIT = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


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


if __name__ == "__main__":
    main()
