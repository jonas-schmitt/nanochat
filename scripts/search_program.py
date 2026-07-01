"""Program grammar search — the full matrix-primitive program search (Stage-1).

Searches the composed space (temporal grammar × phase-coupled polar schedule) for a
genome that DOMINATES vanilla Lookahead — the known, free wrapper that is the real floor
(Muon is the weaker reference) — at iso-FLOP. The coupling is the candidate-novel part:
on lookahead-sync steps the polar can be cheaper, on intermediate steps stronger.

This is **Stage-1** of a two-stage pipeline:
  * Stage-1 (this script): cheap d6 search, but ranked by NSGA-II on (val, polar-FLOP cost)
    — NOT single-point val — so cheaper-equal-quality genomes survive and the Pareto knees
    are surfaced. Headline = margin over Lookahead, with the cheaper-Muon (polar4) floor
    reported separately so the genuinely-temporal residual is isolated.
  * Stage-2 (scripts/eval_program_trend.py): take the Stage-1 Pareto knees + Muon + Lookahead
    and check the d6->d8 TREND at multiple seeds. A point-d6 win that shrinks at d8 is a
    surrogate artifact (the trap that falsified the curvature arm) and is rejected there.

  PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src \
  uv run --project /home/jonas/git/tct-models python -u scripts/search_program.py \
    --depth 6 --steps 1200 --pop 8 --gens 3 --lr 0.02 --seed 0
"""
from __future__ import annotations

import argparse, copy, json, math, subprocess, time, types
import numpy as np
import torch

from train_compare_precond import (build_model, matrix_params, apply_norm_caution_update,
                                    polar_express_orth, _polar_with_coeffs, _build_other_adam)
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit
from gns import program_grammar as pg
from gns.program_grammar import POLAR_MENU, PolarVariant
from gns.search import fast_nondominated_sort, _rank_and_crowd


def _cfg():
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--aspect-ratio", type=int, default=64); p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--max-seq-len", type=int, default=1024); p.add_argument("--device-batch-size", type=int, default=16)
    p.add_argument("--steps", type=int, default=600); p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--momentum", type=float, default=0.95); p.add_argument("--beta2", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.28); p.add_argument("--ns-steps", type=int, default=5)
    p.add_argument("--warmup-steps", type=int, default=20); p.add_argument("--n-val-batches", type=int, default=16)
    p.add_argument("--pop", type=int, default=8); p.add_argument("--gens", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="/home/jonas/git/gns/results/search_program.json")
    p.add_argument("--no-compile", action="store_true", help="disable torch.compile (for debugging)")
    return p.parse_args()


def _git_sha(repo: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _make_polar_fn():
    """Dispatch a PolarVariant to its actual orthogonalization on a frob-normalized g.

    polar4/polar5 are polar_express truncations (the established, validated cheaper-Muon /
    Muon path). Every other variant carries its OWN (a,b,c) schedule in ``variant.coeffs``
    (polar2no/polar3no are sound few-step Remez schedules — NOT truncations; polar6 is
    polar_express + a Jordan polish; jordan5/jointopt4 are their own coeffs), applied
    via the gns.fused path. ``ns_steps`` selects the prefix length.
    """
    def polar_fn(variant: PolarVariant, g: torch.Tensor) -> torch.Tensor:
        g = g / (g.float().norm() + 1e-12)
        if variant.name in ("polar4", "polar5"):
            return polar_express_orth(g, variant.ns_steps)
        return _polar_with_coeffs(g, variant.coeffs[: variant.ns_steps])
    return polar_fn


def build_context(args, depth: int, seed: int, steps: int, compile_: bool = True):
    """Build a (model, init snapshot, train/val batches, polar_fn) context for one (depth, seed).

    Shared by the Stage-1 search (one context) and the Stage-2 trend gate (one per rung).
    Genomes are evaluated by resetting the model to ``init_params`` and replaying ``train``.
    """
    a = copy.copy(args); a.depth = depth
    dev = "cuda"
    tok = get_tokenizer(); vocab = tok.get_vocab_size()
    model, _ = build_model(a, vocab, dev, seed)
    if compile_:
        model = torch.compile(model, dynamic=False)
    init_params = [p.detach().clone() for p in model.parameters()]

    def materialise(split, n):
        ld = tokenizing_distributed_data_loader_with_state_bos_bestfit(
            tok, a.device_batch_size, a.max_seq_len, split=split, device=dev)
        return [tuple(t.clone() for t in next(ld)[:2]) for _ in range(n)]

    train = materialise("train", steps)
    val = materialise("val", a.n_val_batches)
    return types.SimpleNamespace(model=model, init_params=init_params, train=train,
                                 val=val, polar_fn=_make_polar_fn(), depth=depth, seed=seed)


def train_genome(ctx, genome: pg.ProgramGenome, args) -> float:
    """Train ``genome`` from ctx's init snapshot over ctx.train; return mean val loss.

    Identical optimizer math to the previous evaluate(): per-matrix-param temporal state +
    phase-coupled polar via program_step, NorMuon/cautious update, production AdamW on the
    rest, and multilevel lookahead sync over ALL params. ``inf`` on non-finite directions.
    """
    model, init_params, train, val, polar_fn = (ctx.model, ctx.init_params, ctx.train,
                                                 ctx.val, ctx.polar_fn)
    steps = len(train)
    with torch.no_grad():
        for p, init in zip(model.parameters(), init_params):
            p.copy_(init)
    mp = matrix_params(model); mp_set = {id(p) for p in mp}
    adam = _build_other_adam(model, mp_set)
    st_mp = [pg.init_state(genome, p) for p in mp]
    # COORDINATION genes: per-matrix LR multiplier from role_scales × optional μP RMS (scalars, iso-FLOP).
    from gns.module_lr import classify_role
    _m = getattr(model, "_orig_mod", model)
    _name_by_id = {id(p): n for n, p in _m.transformer.h.named_parameters()}
    role_mult = [pg.matrix_lr_mult(genome, classify_role(_name_by_id[id(p)]), p.shape[0], p.shape[1])
                 for p in mp]
    # NOTE: the decay-geometry gene (genome.wwd_power) is DEFERRED here — it needs Shampoo-factor
    # maintenance (L,R + inverse roots) in this eval loop, added only after the d8 scale gate confirms
    # whitened WD holds at scale. Until then the search should be launched with wwd disabled (its cost
    # is already priced in extra_matmuls, so an unwired wwd_power>0 genome is just penalized, not used).
    if genome.wwd_power > 0:
        raise NotImplementedError("wwd_power search wiring pending d8 gate; launch search with wwd off")
    from gns.temporal_grammar import init_state as tg_init, lookahead_sync as tg_la_sync
    st_all = [tg_init(genome.temporal, p) for p in model.parameters()]

    for step in range(1, steps + 1):
        x, y = train[step - 1]
        loss = model(x, y)
        model.zero_grad(set_to_none=True); adam.zero_grad(set_to_none=True); loss.backward()
        lrm = min(1.0, step / max(1, args.warmup_steps))
        cos_wd = args.weight_decay * 0.5 * (1 + math.cos(math.pi * step / steps))
        with torch.no_grad():
            for j, p in enumerate(mp):
                if p.grad is None: continue
                D = pg.program_step(genome, p.grad, st_mp[j], step, polar_fn)
                if not torch.isfinite(D).all(): return float("inf")
                apply_norm_caution_update(D, p, st_mp[j], args.lr * lrm * role_mult[j], cos_wd, args.beta2)
        for gp in adam.param_groups: gp["lr"] = gp["base_lr"] * lrm
        adam.step()
        for pi, p in enumerate(model.parameters()):
            with torch.no_grad():
                tg_la_sync(genome.temporal, p, st_all[pi], step)
    model.eval()
    with torch.no_grad():
        vl = float(np.mean([float(model(vx, vy).item()) for vx, vy in val]))
    model.train()
    return vl


def main():
    args = _cfg()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False

    ctx = build_context(args, args.depth, args.seed, args.steps, compile_=not args.no_compile)

    # Reference points, all evaluated through the SAME path as the candidates (apples-to-apples):
    #   muon       — the weak reference (Muon)
    #   cheaper    — Muon with polar4 (the validated "cheaper-Muon" floor; isolates the cost win)
    #   lookahead  — the REAL floor to beat (a known, free wrapper)
    muon_g = pg.muon_program()
    look_g = pg.lookahead_program(6, 0.5)
    cheaper_g = pg.ProgramGenome(temporal=muon_g.temporal,
                                 polar_fine=_polar_by_name("polar4"),
                                 polar_coarse=_polar_by_name("polar4"))
    anchors = {pg.canonical(g): g for g in (muon_g, look_g, cheaper_g)}

    seen: dict[str, tuple[float, float]] = {}       # canonical -> (val, cost)
    genomes: dict[str, pg.ProgramGenome] = {}       # canonical -> genome object

    def objs_of(g) -> tuple[float, float]:
        key = pg.canonical(g)
        if key not in seen:
            t0 = time.time()
            v = train_genome(ctx, g, args)
            c = g.extra_matmuls()                    # polar-matmul cost over Muon (cheaper < 0)
            seen[key] = (v, c); genomes[key] = g
            print(f"  val {v:.4f}  cost {c:+.3f}  ({time.time()-t0:.0f}s)  {key}", flush=True)
        return seen[key]

    def val_of(g):  return objs_of(g)[0]

    rng = np.random.default_rng(args.seed)
    pop = [muon_g, look_g, cheaper_g] + [pg.random_program(rng) for _ in range(max(0, args.pop - 3))]

    print(f"=== PROGRAM grammar search (d{args.depth}, {args.steps} steps, lr {args.lr}, "
          f"pop {args.pop}, gens {args.gens}) — NSGA-II on (val, polar-cost) ===", flush=True)
    print(f"Polar menu: {[v.name for v in POLAR_MENU]}", flush=True)
    print("[gen 0] evaluating seed population (muon + cheaper + lookahead + random):", flush=True)
    for g in pop:
        objs_of(g)

    population = list(pop)
    for gen in range(1, args.gens + 1):
        # NSGA-II selection on (val, cost): rank by Pareto front then crowding distance.
        objs = [objs_of(g) for g in population]
        rank, crowd = _rank_and_crowd(objs)
        order = sorted(range(len(population)), key=lambda i: (rank[i], -crowd[i]))
        elite = [population[i] for i in order[: max(2, args.pop // 2)]]
        children = []
        while len(children) < args.pop - len(elite):
            i, j = (int(x) for x in rng.integers(0, len(elite), 2))
            c = pg.crossover(elite[i], elite[j], rng)
            if rng.random() < 0.9:
                c = pg.mutate(c, rng)
            children.append(c)
        print(f"[gen {gen}] evaluating {len(children)} offspring:", flush=True)
        for c in children:
            objs_of(c)
        pool = {pg.canonical(g): g for g in elite + children}
        pool.update(anchors)                          # never lose the reference points
        members = list(pool.values())
        m_objs = [objs_of(g) for g in members]
        m_rank, m_crowd = _rank_and_crowd(m_objs)
        m_order = sorted(range(len(members)), key=lambda i: (m_rank[i], -m_crowd[i]))
        population = [members[i] for i in m_order[: args.pop]]

    # ---- report ----
    all_keys = list(seen)
    all_objs = [seen[k] for k in all_keys]
    fronts = fast_nondominated_sort(all_objs)
    pareto = sorted((all_keys[i] for i in fronts[0]), key=lambda k: seen[k][0])
    muon_v = seen[pg.canonical(muon_g)][0]
    look_v = seen[pg.canonical(look_g)][0]
    cheap_v = seen[pg.canonical(cheaper_g)][0]
    best_key = min(seen, key=lambda k: seen[k][0]); best_v = seen[best_key][0]

    ranking = sorted(({"genome": k, "val": seen[k][0], "cost": seen[k][1],
                       "is_muon": k == pg.canonical(muon_g),
                       "is_cheaper_muon": k == pg.canonical(cheaper_g),
                       "is_lookahead": k == pg.canonical(look_g)} for k in seen),
                     key=lambda r: r["val"])
    summary = {
        "config": {**vars(args), "polar_menu": [v.name for v in POLAR_MENU],
                   "gns_sha": _git_sha("/home/jonas/git/gns"),
                   "nanochat_sha": _git_sha("/home/jonas/git/nanochat"),
                   "n_evaluated": len(seen)},
        "muon_val": muon_v, "cheaper_muon_val": cheap_v, "lookahead_val": look_v,
        "best_val": best_v, "best_genome": best_key,
        "best_beats_muon_by": muon_v - best_v,
        "best_beats_lookahead_by": look_v - best_v,        # <-- the honest headline
        "cheaper_muon_beats_muon_by": muon_v - cheap_v,    # the cost-only floor
        "temporal_residual_over_cheaper": cheap_v - best_v,  # what the temporal structure adds
        "pareto_front": [{"genome": k, "val": seen[k][0], "cost": seen[k][1]} for k in pareto],
        # Exact knee genomes (round-trip via pg.from_dict) for the Stage-2 trend gate:
        "stage2_knees": [{"canonical": k, "val": seen[k][0], "cost": seen[k][1],
                          "genome": pg.to_dict(genomes[k])} for k in pareto],
        "ranking": ranking[:12],
    }
    json.dump(summary, open(args.out, "w"), indent=1)
    print(f"\n=== RESULT ===")
    print(f"  muon {muon_v:.4f} | cheaper-muon(polar4) {cheap_v:.4f} | lookahead {look_v:.4f} | best {best_v:.4f}")
    print(f"  best beats LOOKAHEAD by {look_v-best_v:+.4f}  (vs muon {muon_v-best_v:+.4f})")
    print(f"  of which cheaper-muon floor {muon_v-cheap_v:+.4f}, temporal residual {cheap_v-best_v:+.4f}")
    print(f"  Pareto knees (-> Stage-2): {pareto}")
    print(f"  best genome: {best_key}")


def _polar_by_name(name):
    from gns.program_grammar import _POLAR_BY_NAME
    return _POLAR_BY_NAME[name]


if __name__ == "__main__":
    main()
