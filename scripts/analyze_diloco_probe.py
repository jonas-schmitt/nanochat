"""Falsification verdict for the fused-track probe suite (scripts/diloco_probe.sh).

Reads gns/results/probe_*.json and prints the H1/H2/H3 verdicts with the kill criteria
from the driver. Single-seed d6/1500: noise floor ~0.003, ties declared under 0.005.
Also evaluates the rounding model's rotation prediction against the LOGGED delta stats (C0).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/jonas/git/gns/src")
from gns.quant_model import DeltaStats, predicts_rotation_helps  # noqa: E402

R = Path("/home/jonas/git/gns/results")
NOISE = 0.005


def best_val(name, arm=None):
    p = R / name
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if arm is not None:
        runs = [v for v in d.get("_done", {}).values() if v["arm"] == arm]
        return min(v["best_val"] for v in runs) if runs else None
    return d.get("best_val")


def stats_of(name):
    p = R / name
    if not p.exists():
        return None
    ds = json.loads(p.read_text()).get("curve", {}).get("delta_stats")
    return ds and DeltaStats(std=1.0, absmax_over_std=ds["rho_clean"], rho_hit=ds["rho_hit"],
                             hit_fraction=ds["hit_fraction"],
                             outlier_energy_fraction=ds["outlier_energy_fraction"])


def main():
    vals = {
        "muloco_fp32": best_val("probe_muloco_fp32.json"),
        "outer_polar": best_val("probe_outer_polar.json"),
        "outer_whitened": best_val("probe_outer_whitened.json"),
        "dp_anchor64": best_val("probe_dp_anchor64.json", arm="muon"),
        "diloco_adamw": best_val("probe_diloco_adamw.json"),
        "muloco_2bit": best_val("probe_muloco_2bit.json"),
        "hadamard_2bit": best_val("probe_hadamard_2bit.json"),
        "muloco_h100": best_val("probe_muloco_h100.json"),
    }
    print(f"{'arm':16s} {'best val':>9s}   (d6/1500, M=4 h=30 unless noted; single-seed)")
    for k, v in vals.items():
        print(f"{k:16s} {v:9.4f}" if v is not None else f"{k:16s} {'—':>9s}")
    mu = vals["muloco_fp32"]
    if mu is None:
        print("\nincumbent missing — no verdicts yet")
        return

    print("\n--- VERDICTS ---")
    # H1: simulator validity
    if vals["diloco_adamw"] is not None:
        d = vals["diloco_adamw"] - mu
        print(f"H1a MuLoCo>DiLoCo: gap {d:+.4f} -> "
              f"{'PASS' if d > NOISE else 'TIE' if abs(d) <= NOISE else 'FAIL (simulator suspect)'}")
    if vals["dp_anchor64"] is not None:
        d = mu - vals["dp_anchor64"]
        print(f"H1b MuLoCo vs DP anchor: {d:+.4f} behind DP (papers: close; large gap = check regime)")
    # H2: the novel outer-geometry claim
    for arm in ("outer_polar", "outer_whitened"):
        if vals[arm] is not None:
            d = mu - vals[arm]  # positive = novel arm better (lower val)
            v = "BEATS incumbent" if d > NOISE else "tie" if d >= -NOISE else "LOSES"
            print(f"H2  {arm}: {d:+.4f} vs muloco -> {v}")
    if all(vals[a] is not None for a in ("outer_polar", "outer_whitened")):
        if max(mu - vals["outer_polar"], mu - vals["outer_whitened"]) <= NOISE:
            print("H2  KILL CRITERION MET: outer-geometry lever dead at d6 (both <= incumbent + noise)")
    # H3: wire
    if vals["muloco_2bit"] is not None:
        d = mu - vals["muloco_2bit"]
        print(f"H3a 2bit+EF vs fp32: {d:+.4f} -> "
              f"{'replicates (|gap|<=0.02)' if abs(d) <= 0.02 else 'FAILS replication — check wire'}"
              f"  [16x fewer sync bits]")
    if vals["hadamard_2bit"] is not None and vals["muloco_2bit"] is not None:
        d = vals["muloco_2bit"] - vals["hadamard_2bit"]  # positive = rotation better
        st = stats_of("probe_muloco_2bit.json")
        pred = predicts_rotation_helps(st, bits=2, stochastic=False) if st else None
        print(f"H3b whiten-vs-rotate: hadamard {d:+.4f} vs identity; model predicted "
              f"{'HELPS' if pred else 'HURTS' if pred is not None else '(no stats)'} -> "
              f"{'model CONFIRMED' if pred is not None and (d > 0) == pred else 'model WRONG (C0 evidence!)' if pred is not None else ''}")
        if st:
            print(f"    measured delta stats: rho_clean {st.absmax_over_std:.2f} rho_hit {st.rho_hit:.2f} "
                  f"hit {st.hit_fraction:.2f} w {st.outlier_energy_fraction:.2f}")
    if vals["muloco_h100"] is not None:
        print(f"cur h=100 point: {vals['muloco_h100']:.4f} ({vals['muloco_h100']-mu:+.4f} vs h=30, "
              f"3.3x fewer syncs) — the comm-quality curve's next knot")


if __name__ == "__main__":
    main()
