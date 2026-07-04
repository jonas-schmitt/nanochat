"""Tier-1 policy search: NSGA-II on the CHEAP objectives (replay-val x analytic comm-bits).

Runs entirely against a recorded anchor (train_diloco --record-deltas + replay_policy.replay):
no training per genome — one replay (~seconds) prices a candidate's wire/outer genes; comm-bits
is analytic (genome.comm_bits_per_param_step). Search space = the replay-eligible genes at the
anchor's fixed (h, inner, workers):
  wire: delta_bits x basis x error_feedback x ef_beta x stochastic_rounding   [screening-grade]
  outer: outer_lr x outer (beta, nesterov)                                    [DOUBTFUL — off-policy;
         rank-gate before trusting; --include-geometry adds outer_transform, same caveat]

QUALIFICATION CHECK (the point of the exercise): the tier-1 front must
  (a) contain >= 3 distinct non-dominated points, and
  (b) contain a point that matches-or-dominates the hand-picked MuLoCo-2bit wire
      (bits <= its bits AND replay-val <= its replay-val + margin).
Exit code reflects PASS/FAIL. Front candidates then go to tier-2 (real sim) validation.
"""
import argparse
import json
import sys
import time
from dataclasses import replace as dc_replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/jonas/git/gns/src")

import replay_policy as rp  # noqa: E402
from gns.policy_grammar import (  # noqa: E402
    BASIS_MENU,
    BITS_MENU,
    EF_BETA_MENU,
    OUTER_LR_MENU,
    OUTER_TRANSFORM_MENU,
    canonical,
    from_dict as genome_from_dict,
    to_dict as genome_to_dict,
)
from gns.search import _rank_and_crowd, fast_nondominated_sort  # noqa: E402
from gns.temporal_grammar import TemporalGenome  # noqa: E402

OUTER_BETA_MENU = (0.0, 0.9, 0.95)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--recording", type=str, required=True)
    p.add_argument("--pop", type=int, default=16)
    p.add_argument("--gens", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--qual-margin", type=float, default=0.002,
                   help="val slack for 'matches the muloco2bit wire' in the qualification check")
    p.add_argument("--include-geometry", action="store_true",
                   help="also search outer_transform (off-policy DOUBTFUL — rank-gate first)")
    p.add_argument("--out", type=str, default="/home/jonas/git/gns/results/search_policy_tier1.json")
    p.add_argument("--resume", action="store_true",
                   help="reload the <out>.ckpt.json search state (seen cache + RNG + generation) if present")
    return p.parse_args()


# Gene families, split so the search only varies what the rank gate TRUSTS. PRECISION genes are
# replay-faithful (perturbative around the anchor stream); GEOMETRY genes (outer lr / momentum /
# transform) are off-policy and only searched when --include-geometry (rank gate trusted them).
# When geometry is untrusted, ALL outer genes stay pinned to the anchor's (tuned) values — otherwise
# the search would optimize against replay numbers we just declared untrustworthy.
def _sample(rng, anchor, include_geometry):
    g = dc_replace(
        anchor,
        delta_bits=int(rng.choice(BITS_MENU)),
        basis=str(rng.choice(BASIS_MENU)),
        error_feedback=bool(rng.random() < 0.5),
        ef_beta=float(rng.choice(EF_BETA_MENU)),
        stochastic_rounding=bool(rng.random() < 0.5),
    )
    if include_geometry:
        g = dc_replace(
            g,
            outer_lr=float(rng.choice(OUTER_LR_MENU)),
            outer=TemporalGenome(momentum_betas=(float(rng.choice(OUTER_BETA_MENU)),),
                                 nesterov=bool(rng.random() < 0.5)),
            outer_transform=str(rng.choice(OUTER_TRANSFORM_MENU)),
        )
    return g


def _mutate(rng, g, anchor, include_geometry):
    which = int(rng.integers(0, 5 + (3 if include_geometry else 0)))
    if which == 0:
        return dc_replace(g, delta_bits=int(rng.choice(BITS_MENU)))
    if which == 1:
        return dc_replace(g, basis=str(rng.choice(BASIS_MENU)))
    if which == 2:
        return dc_replace(g, error_feedback=not g.error_feedback)
    if which == 3:
        return dc_replace(g, ef_beta=float(rng.choice(EF_BETA_MENU)))
    if which == 4:
        return dc_replace(g, stochastic_rounding=not g.stochastic_rounding)
    # geometry mutations (only reachable when include_geometry)
    if which == 5:
        return dc_replace(g, outer_lr=float(rng.choice(OUTER_LR_MENU)))
    if which == 6:
        return dc_replace(g, outer=TemporalGenome(momentum_betas=(float(rng.choice(OUTER_BETA_MENU)),),
                                                  nesterov=bool(rng.random() < 0.5)))
    return dc_replace(g, outer_transform=str(rng.choice(OUTER_TRANSFORM_MENU)))


def _atomic_json(path, obj):
    """Write JSON atomically (temp + rename) so a kill mid-write never leaves a half-file."""
    import os
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _save_ckpt(path, seen, genomes, rng, gen_done, population):
    """Resumable search state: the seen genome->objectives cache (each entry = one replay eval),
    the genome dicts, the RNG bit-generator state, the last completed generation, and the current
    population (by canonical key). Atomic — safe to kill at any point."""
    _atomic_json(path, {
        "seen": {k: list(v) for k, v in seen.items()},
        "genomes": {k: genome_to_dict(g) for k, g in genomes.items()},
        "rng_state": rng.bit_generator.state,
        "gen_done": gen_done,
        "population": [canonical(g) for g in population],
    })


def _crossover(rng, a, b):
    pick = lambda x, y: x if rng.random() < 0.5 else y  # noqa: E731
    return dc_replace(
        a,
        delta_bits=pick(a.delta_bits, b.delta_bits),
        basis=pick(a.basis, b.basis),
        error_feedback=pick(a.error_feedback, b.error_feedback),
        ef_beta=pick(a.ef_beta, b.ef_beta),
        stochastic_rounding=pick(a.stochastic_rounding, b.stochastic_rounding),
        outer_lr=pick(a.outer_lr, b.outer_lr),
        outer=pick(a.outer, b.outer),
        outer_transform=pick(a.outer_transform, b.outer_transform),
    )


def main():
    cli = parse_args()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = "cuda"
    rng = np.random.default_rng(cli.seed)

    rec = rp.load_recording(cli.recording)
    anchor = rec["anchor_genome"]
    args = rp.rec_args(rec)
    model, val_batches = rp.build_eval_context(rec, device)
    print(f"anchor: {canonical(anchor)}  ({len(rec['rounds'])} rounds x "
          f"{len(rec['rounds'][0])} workers recorded)")

    seen: dict[str, tuple[float, float]] = {}
    genomes: dict[str, object] = {}

    def objs_of(g):
        key = canonical(g)
        if key not in seen:
            t0 = time.time()
            try:
                log = rp.replay(rec, g, args, model, val_batches, device, eval_every=None)
                v = log["val"][-1]
                if not np.isfinite(v):
                    v = float("inf")
            except Exception as e:  # a broken genome must not kill the search
                print(f"  replay error ({e}) — pruning  {key}", flush=True)
                v = float("inf")
            bits = g.comm_bits_per_param_step()
            seen[key] = (v, bits); genomes[key] = g
            print(f"  val {v:8.4f}  bits/p/s {bits:7.4f}  ({time.time()-t0:4.1f}s)  {key}",
                  flush=True)
        return seen[key]

    # seeds: the fp32 anchor + the hand-picked MuLoCo-2bit wire (the must-match reference) + randoms
    ref2bit = dc_replace(anchor, delta_bits=2, error_feedback=True, ef_beta=0.9,
                         stochastic_rounding=False)
    ckpt_path = cli.out + ".ckpt.json"
    start_gen = 1
    import os
    if cli.resume and os.path.exists(ckpt_path):
        ck = json.load(open(ckpt_path))
        seen.update({k: tuple(v) for k, v in ck["seen"].items()})
        genomes.update({k: genome_from_dict(v) for k, v in ck["genomes"].items()})
        rng.bit_generator.state = ck["rng_state"]
        start_gen = ck["gen_done"] + 1
        population = [genomes[k] for k in ck["population"]]
        print(f"=== RESUMED tier-1 search from {ckpt_path}: {len(seen)} evals cached, "
              f"resuming at gen {start_gen}/{cli.gens} ===", flush=True)
    else:
        pop = [anchor, ref2bit] + [_sample(rng, anchor, cli.include_geometry)
                                   for _ in range(max(0, cli.pop - 2))]
        print(f"=== TIER-1 policy search (pop {cli.pop}, gens {cli.gens}) on (replay-val, comm-bits) ===")
        for g in pop:
            objs_of(g)
        population = list(pop)
        _save_ckpt(ckpt_path, seen, genomes, rng, 0, population)
    for gen in range(start_gen, cli.gens + 1):
        objs = [objs_of(g) for g in population]
        rank, crowd = _rank_and_crowd(objs)
        order = sorted(range(len(population)), key=lambda i: (rank[i], -crowd[i]))
        elite = [population[i] for i in order[: max(2, cli.pop // 2)]]
        children = []
        while len(children) < cli.pop - len(elite):
            i, j = (int(x) for x in rng.integers(0, len(elite), 2))
            c = _crossover(rng, elite[i], elite[j])
            if rng.random() < 0.9:
                c = _mutate(rng, c, anchor, cli.include_geometry)
            children.append(c)
        print(f"[gen {gen}] evaluating {len(children)} offspring:", flush=True)
        for c in children:
            objs_of(c)
        pool = {canonical(g): g for g in elite + children}
        pool[canonical(anchor)] = anchor; pool[canonical(ref2bit)] = ref2bit
        members = list(pool.values())
        m_objs = [objs_of(g) for g in members]
        m_rank, m_crowd = _rank_and_crowd(m_objs)
        m_order = sorted(range(len(members)), key=lambda i: (m_rank[i], -m_crowd[i]))
        population = [members[i] for i in m_order[: cli.pop]]
        _save_ckpt(ckpt_path, seen, genomes, rng, gen, population)

    finite = {k: v for k, v in seen.items() if np.isfinite(v[0])}
    keys = list(finite)
    fronts = fast_nondominated_sort([finite[k] for k in keys])
    front = sorted((keys[i] for i in fronts[0]), key=lambda k: finite[k][1])
    ref_v, ref_b = seen[canonical(ref2bit)]

    print("\n=== TIER-1 PARETO FRONT (cheap objectives — tier-2 validation still required) ===")
    for k in front:
        v, b = finite[k]
        print(f"  val {v:8.4f}  bits/p/s {b:7.4f}  {k}")
    print(f"  reference (hand-picked muloco2bit wire): val {ref_v:.4f}  bits/p/s {ref_b:.4f}")

    # QUALIFICATION: non-trivial front + a point matching-or-dominating the hand recipe
    qual_pts = [k for k in front
                if finite[k][1] <= ref_b and finite[k][0] <= ref_v + cli.qual_margin]
    ok = len(front) >= 3 and len(qual_pts) > 0
    print(f"\nQUALIFICATION: front size {len(front)} (need >=3); "
          f"{len(qual_pts)} point(s) match-or-dominate the muloco2bit wire "
          f"(margin {cli.qual_margin}) -> {'PASS' if ok else 'FAIL'}")

    json.dump({"anchor": canonical(anchor), "n_evaluated": len(seen),
               "front": [{"genome": k, "val": finite[k][0], "bits_per_param_step": finite[k][1],
                          "genome_dict": genome_to_dict(genomes[k])} for k in front],
               "reference_2bit": {"genome": canonical(ref2bit), "val": ref_v,
                                  "bits_per_param_step": ref_b},
               "qualified": ok,
               "all": {k: {"val": v[0], "bits": v[1]} for k, v in seen.items()}},
              open(cli.out, "w"), indent=1)
    print(f"wrote {cli.out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
