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
                                    polar_express_orth, _polar_with_coeffs, _build_other_adam,
                                    inv_fourth_root)

# Whitened weight decay / curvature-descent reuse the Shampoo factors; the inverse-root refresh interval is
# an optimizable gene (genome.recompute_every) since the decay metric tolerates stale factors (rc100 held the
# +0.0137 at ~1.05x wall in the recompute sweep). See gns.program_grammar RECOMPUTE_MENU.
def _wwd_target(p, Linv, Rinv, power):
    """Norm-matched whitened decay target L^-p·W·R^-p (Linv=L^-1/4 ⇒ k=round(4p) applications each side)."""
    k = int(round(power * 4))
    pw = p.float()
    for _ in range(k):
        pw = Linv @ pw
    for _ in range(k):
        pw = pw @ Rinv
    return (pw * (p.float().norm() / pw.norm().clamp_min(1e-12))).to(p.dtype)
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit
from gns import program_grammar as pg
from gns.program_grammar import POLAR_MENU, PolarVariant
from gns.search import fast_nondominated_sort, _rank_and_crowd
from gns.temporal_stability import is_temporally_stable


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


def train_genome(ctx, genome: pg.ProgramGenome, args, screen=None) -> dict:
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
    # Factor family (genome.wwd_power>0 and/or curvature_descent): maintain Shampoo factors L,R + inverse roots
    # refreshed every genome.recompute_every (the decay metric tolerates stale factors). Factors feed the DESCENT
    # (curvature_descent: precondition g → Linv·g·Rinv before program_step) and/or the DECAY (whitened WD).
    use_fac = genome.wwd_power > 0 or genome.curvature_descent
    rc = max(1, genome.recompute_every)
    fac = ([{"L": torch.zeros(p.shape[0], p.shape[0], device=p.device),
             "R": torch.zeros(p.shape[1], p.shape[1], device=p.device),
             "Linv": None, "Rinv": None} for p in mp] if use_fac else None)
    from gns.temporal_grammar import init_state as tg_init, lookahead_sync as tg_la_sync
    st_all = [tg_init(genome.temporal, p) for p in model.parameters()]
    # L3 (ema-axis densification): the eval-EMA gene does not affect TRAINING, so one trained run
    # yields the fitness of all three protocol siblings (raw + both EMA betas) — two extra fp32
    # buffers, ~zero extra time; the search caches all siblings from one GPU run.
    ema_betas = [b for b in pg.EVAL_EMA_MENU if b > 0]
    emas = {b: [p.detach().float().clone().zero_() for p in model.parameters()] for b in ema_betas}

    for step in range(1, steps + 1):
        x, y = train[step - 1]
        loss = model(x, y)
        model.zero_grad(set_to_none=True); adam.zero_grad(set_to_none=True); loss.backward()
        lrm = min(1.0, step / max(1, args.warmup_steps))
        # wd_scale gene: the genome owns its decay STRENGTH (shape = wwd_power); base lambda from args
        cos_wd = genome.wd_scale * args.weight_decay * 0.5 * (1 + math.cos(math.pi * step / steps))
        with torch.no_grad():
            for j, p in enumerate(mp):
                if p.grad is None: continue
                fj = fac[j] if use_fac else None
                if use_fac:                                   # update factors BEFORE the descent uses them
                    gf = p.grad.float()
                    fj["L"].mul_(0.95).add_(gf @ gf.t(), alpha=0.05)
                    fj["R"].mul_(0.95).add_(gf.t() @ gf, alpha=0.05)
                    if step >= args.warmup_steps and (fj["Linv"] is None or step % rc == 0):
                        fj["Linv"] = inv_fourth_root(fj["L"], 1e-4, 24)
                        fj["Rinv"] = inv_fourth_root(fj["R"], 1e-4, 24)
                grad_in = p.grad
                if genome.curvature_descent and fj is not None and fj["Linv"] is not None:
                    grad_in = ((fj["Linv"] @ p.grad.float()) @ fj["Rinv"]).to(p.grad.dtype)  # ortho_shampoo dir
                D = pg.program_step(genome, grad_in, st_mp[j], step, polar_fn)
                if not torch.isfinite(D).all(): return float("inf")
                wd_target = None
                if genome.wwd_power > 0 and fj is not None and fj["Linv"] is not None:
                    wd_target = _wwd_target(p, fj["Linv"], fj["Rinv"], genome.wwd_power)
                apply_norm_caution_update(D, p, st_mp[j], genome.lr_scale * args.lr * lrm * role_mult[j], cos_wd, args.beta2,
                                          wd_target=wd_target)
        for gp in adam.param_groups: gp["lr"] = genome.adam_lr_scale * gp["base_lr"] * lrm
        adam.step()
        for pi, p in enumerate(model.parameters()):
            with torch.no_grad():
                tg_la_sync(genome.temporal, p, st_all[pi], step)
        with torch.no_grad():
            for b, bufs in emas.items():
                for e, p in zip(bufs, model.parameters()):
                    e.mul_(b).add_(p.detach().float(), alpha=1 - b)
        # L2 disaster screen (margin calibrated from measured rank-inversion data: inversions reach
        # ~0.2-0.4, so this only kills GARBAGE, never sorts contenders). screen = (step, best, margin).
        if screen is not None and step == screen[0]:
            model.eval()
            with torch.no_grad():
                v200 = float(np.mean([float(model(vx, vy).item()) for vx, vy in val[:4]]))
            model.train()
            if screen[1] is not None and v200 > screen[1] + screen[2]:
                return {"pruned": True, "screen_val": v200}
            screen_out = v200
    model.eval()
    out = {"_v200": locals().get("screen_out")}
    with torch.no_grad():
        backup = [p.detach().clone() for p in model.parameters()]
        out[0.0] = float(np.mean([float(model(vx, vy).item()) for vx, vy in val]))
        for b, bufs in emas.items():
            corr = 1.0 - b ** steps
            for e, p in zip(bufs, model.parameters()):
                p.data.copy_((e / corr).to(p.dtype))
            out[b] = float(np.mean([float(model(vx, vy).item()) for vx, vy in val]))
            for bk, p in zip(backup, model.parameters()):
                p.data.copy_(bk)
    model.train()
    return out


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
    best200 = [None]                                 # running best step-200 val (disaster screen)

    def objs_of(g) -> tuple[float, float]:
        key = pg.canonical(g)
        if key not in seen:
            # v2 CPU surrogate (kill-only): prune provably-resonant recurrences before any GPU.
            ok_stab, z_star = is_temporally_stable(g)
            if not ok_stab:
                seen[key] = (float("inf"), g.extra_matmuls()); genomes[key] = g
                print(f"  PRUNED-CPU (stability envelope z*={z_star:.2f} too small for lr_scale "
                      f"{g.lr_scale:g})  {key}", flush=True)
                return seen[key]
            t0 = time.time()
            res = train_genome(ctx, g, args, screen=(200, best200[0], 0.35))
            c = g.extra_matmuls()                    # polar-matmul cost over Muon (cheaper < 0)
            if res.get("pruned"):
                seen[key] = (float("inf"), c); genomes[key] = g
                print(f"  PRUNED@200 (val {res['screen_val']:.3f} > best+0.35)  {key}", flush=True)
            else:
                if res.get("_v200") is not None:
                    best200[0] = res["_v200"] if best200[0] is None else min(best200[0], res["_v200"])
                # L3: one trained run prices ALL eval-EMA siblings — cache every protocol variant
                from dataclasses import replace as _dcr
                for b in [x for x in res if isinstance(x, float)]:
                    sib = _dcr(g, eval_ema_beta=b)
                    seen[pg.canonical(sib)] = (res[b], c); genomes[pg.canonical(sib)] = sib
                print(f"  val {seen[key][0]:.4f} (ema0 {res[0.0]:.4f} / .99 {res[0.99]:.4f} / "
                      f".999 {res[0.999]:.4f})  cost {c:+.3f}  ({time.time()-t0:.0f}s)  {key}", flush=True)
        return seen[key]

    def val_of(g):  return objs_of(g)[0]

    rng = np.random.default_rng(args.seed)
    # Seed ONLY with the standard solvers (owner decision 2026-07-03): knee0/wwd/ema are OUR
    # discoveries and belong in the REFERENCE table (must-beat bar), not the starting population —
    # a search that starts at the known frontier proves nothing about discovery. If the search
    # independently rediscovers knee0-like temporal structure, that is free replication evidence.
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
    def _bestproto_v(g):
        """Best val over the eval-EMA protocol siblings of g (all priced by one trained run)."""
        from dataclasses import replace as _dcr
        sibs = [seen[pg.canonical(_dcr(g, eval_ema_beta=b))][0]
                for b in pg.EVAL_EMA_MENU if pg.canonical(_dcr(g, eval_ema_beta=b)) in seen]
        return min(sibs) if sibs else seen[pg.canonical(g)][0]

    summary = {
        "config": {**vars(args), "polar_menu": [v.name for v in POLAR_MENU],
                   "gns_sha": _git_sha("/home/jonas/git/gns"),
                   "nanochat_sha": _git_sha("/home/jonas/git/nanochat"),
                   "n_evaluated": len(seen)},
        "muon_val": muon_v, "cheaper_muon_val": cheap_v, "lookahead_val": look_v,
        # EMA-matched references (best over eval protocols, via L3-cached siblings): candidates carry
        # an eval-EMA gene, so raw-reference margins overstate the win by the free Polyak-averaging
        # gain. THESE are the honest bars.
        "muon_val_bestproto": _bestproto_v(muon_g), "cheaper_muon_val_bestproto": _bestproto_v(cheaper_g),
        "lookahead_val_bestproto": _bestproto_v(look_g),
        "best_beats_muon_bestproto_by": _bestproto_v(muon_g) - best_v,
        "best_beats_lookahead_bestproto_by": _bestproto_v(look_g) - best_v,
        "best_val": best_v, "best_genome": best_key,
        "best_beats_muon_by": muon_v - best_v,             # raw-reference margin (inflated; kept for continuity)
        "best_beats_lookahead_by": look_v - best_v,
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
    print(f"  EMA-MATCHED (honest) margins: vs muon@bestproto {_bestproto_v(muon_g)-best_v:+.4f}, "
          f"vs lookahead@bestproto {_bestproto_v(look_g)-best_v:+.4f}")
    print(f"  of which cheaper-muon floor {muon_v-cheap_v:+.4f}, temporal residual {cheap_v-best_v:+.4f}")
    print(f"  Pareto knees (-> Stage-2): {pareto}")
    print(f"  best genome: {best_key}")


def _polar_by_name(name):
    from gns.program_grammar import _POLAR_BY_NAME
    return _POLAR_BY_NAME[name]


if __name__ == "__main__":
    main()
