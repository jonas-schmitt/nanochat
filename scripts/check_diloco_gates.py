"""Comparator for the DiLoCo-simulator sanity gates (G-A1/G-A2/G-A3).

G-A1: train_diloco (M=1, h=1, outer sgd lr=1 β=0, fp32 wire) val trace == `muon` arm.
G-A2: train_diloco (M=1, h=k, outer sgd lr=α β=0, fp32 wire) val trace == `muon_lookahead` arm
      (compare only steps where both evaluated: multiples of lcm(h, eval_every)).
G-A3: two train_diloco runs at bits=32 differing only in inert wire genes (basis/EF/rounding)
      must produce identical traces.

Usage: check_diloco_gates.py <muon.json> <lookahead.json> <ga1.json> <ga2.json> <ga3a.json> <ga3b.json>
Exits non-zero on any gate failure.
"""
import json
import sys


def _trace(path, arm=None):
    d = json.load(open(path))
    if arm is not None:  # train_compare_precond format
        runs = [v for v in d["_done"].values() if v["arm"] == arm]
        assert len(runs) == 1, f"{path}: expected exactly one {arm} run, got {len(runs)}"
        c = runs[0]["curve"]
    else:                # train_diloco format
        c = d["curve"]
    return dict(zip(c["step"], c["val"]))


def _compare(name, a, b, tol):
    common = sorted(set(a) & set(b))
    assert common, f"{name}: no common eval steps ({sorted(a)} vs {sorted(b)})"
    worst = max(abs(a[s] - b[s]) for s in common)
    ok = worst <= tol
    print(f"{name}: {len(common)} common eval points, max |dval| = {worst:.2e} "
          f"(tol {tol:.0e}) -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    muon, lookahead, ga1, ga2, ga3a, ga3b = sys.argv[1:7]
    ok = True
    # Tolerance design (measured, 2026-07-02): SAME-program runs under GNS_DETERMINISTIC=1 are
    # bit-identical (G-A3, det_smoke repeats), but CROSS-program comparisons drift at fp32-ulp
    # scale (different allocator/cuBLAS context -> different, individually-deterministic kernel
    # selections; first divergence measured at ~1e-7 relative), which chaos amplifies to ~3.5e-3
    # in val over 300 steps. Bit-exactness across programs is therefore not achievable; the SHARP
    # wiring test is scripts/diloco_equiv_gate.py (in-process loss-prefix bit-identity: ga1 >= 3
    # steps = one full delta->outer->copy-back cycle, ga2 >= 6 steps = one sync cycle). These
    # trace gates catch wiring-scale errors above the measured drift ceiling.
    ok &= _compare("G-A1 (diloco~muon)", _trace(ga1), _trace(muon, "muon"), tol=5e-3)
    ok &= _compare("G-A2 (diloco~lookahead)", _trace(ga2), _trace(lookahead, "muon_lookahead"), tol=5e-3)
    ok &= _compare("G-A3 (bits32 inert, same-program: bit-exact)", _trace(ga3a), _trace(ga3b), tol=0.0)
    print("ALL GATES PASS" if ok else "GATE FAILURE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
