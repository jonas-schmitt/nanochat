"""Program grammar search — the full matrix-primitive program search.

Searches the composed space (temporal grammar × phase-coupled polar schedule) for a
genome that DOMINATES both Muon and vanilla Lookahead at iso-FLOP. The coupling is
the novel part: on lookahead-sync steps the polar can be cheaper (the iterate is
already a smoothed average), and on intermediate steps it can be stronger (raw gradient).

Stage-1: d6, 1200 steps, 1-seed. Winners go to Stage-2 (d8 trend check + multi-seed).

  PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src \
  /home/jonas/git/tct-models/.venv/bin/python -u scripts/search_program.py \
    --depth 6 --steps 1200 --pop 8 --gens 3 --lr 0.02 --seed 0
"""
from __future__ import annotations

import argparse, json, math, time
import numpy as np
import torch

from train_compare_precond import (build_model, matrix_params, apply_norm_caution_update,
                                    polar_express_orth, _polar_with_coeffs, _build_other_adam)
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit
from gns import program_grammar as pg
from gns.program_grammar import POLAR_MENU, PolarVariant


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


# The polar_express coefficients (from nanochat.optim)
_PE = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933435376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]
_JOINTOPT_4 = ((5.5, -9.5, 5.0), (4.0, -3.5, 0.8), (3.9, -2.8, 0.5), (3.3, -2.4, 0.5))
_JORDAN = tuple((3.4445, -4.7750, 2.0315) for _ in range(5))


def _make_polar_fn():
    def polar_fn(variant: PolarVariant, g: torch.Tensor) -> torch.Tensor:
        g = g / (g.float().norm() + 1e-12)
        if variant.name == "polar5":
            return polar_express_orth(g, 5)
        elif variant.name == "polar4":
            return polar_express_orth(g, 4)
        elif variant.name == "polar3":
            return polar_express_orth(g, 3)
        elif variant.name == "jordan5":
            return _polar_with_coeffs(g, _JORDAN)
        elif variant.name == "jointopt4":
            return _polar_with_coeffs(g, _JOINTOPT_4)
        raise ValueError(f"unknown polar variant {variant.name}")
    return polar_fn


def main():
    args = _cfg()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    dev = "cuda"
    tok = get_tokenizer(); vocab = tok.get_vocab_size()
    model, _ = build_model(args, vocab, dev, args.seed)
    # torch.compile optional (can cause multiprocessing crashes on some platforms)
    if not args.no_compile:
        model = torch.compile(model, dynamic=False)
    init_params = [p.detach().clone() for p in model.parameters()]

    def materialise(split, n):
        ld = tokenizing_distributed_data_loader_with_state_bos_bestfit(tok, args.device_batch_size, args.max_seq_len, split=split, device=dev)
        return [tuple(t.clone() for t in next(ld)[:2]) for _ in range(n)]
    train = materialise("train", args.steps); val = materialise("val", args.n_val_batches)
    polar_fn = _make_polar_fn()

    def evaluate(genome: pg.ProgramGenome) -> float:
        with torch.no_grad():
            for p, init in zip(model.parameters(), init_params):
                p.copy_(init)
        mp = matrix_params(model); mp_set = {id(p) for p in mp}
        adam = _build_other_adam(model, mp_set)
        # Per-muon-param temporal state (momentum buffers, prev_D)
        st_mp = [pg.init_state(genome, p) for p in mp]
        # Per-ALL-param temporal state (for lookahead sync — all params, not just muon)
        from gns.temporal_grammar import init_state as tg_init, lookahead_sync as tg_la_sync
        st_all = [tg_init(genome.temporal, p) for p in model.parameters()]

        for step in range(1, args.steps + 1):
            x, y = train[step - 1]
            loss = model(x, y)
            model.zero_grad(set_to_none=True); adam.zero_grad(set_to_none=True); loss.backward()
            lrm = min(1.0, step / max(1, args.warmup_steps))
            cos_wd = args.weight_decay * 0.5 * (1 + math.cos(math.pi * step / args.steps))
            with torch.no_grad():
                for j, p in enumerate(mp):
                    if p.grad is None: continue
                    D = pg.program_step(genome, p.grad, st_mp[j], step, polar_fn)
                    if not torch.isfinite(D).all(): return float("inf")
                    apply_norm_caution_update(D, p, st_mp[j], args.lr * lrm, cos_wd, args.beta2)
            for gp in adam.param_groups: gp["lr"] = gp["base_lr"] * lrm
            adam.step()
            # Multilevel lookahead sync — ALL params (muon + adamw)
            for pi, p in enumerate(model.parameters()):
                with torch.no_grad():
                    tg_la_sync(genome.temporal, p, st_all[pi], step)
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([float(model(vx, vy).item()) for vx, vy in val]))
        model.train()
        return vl

    def desc(g): return pg.canonical(g)

    rng = np.random.default_rng(args.seed)
    pop = [pg.muon_program(), pg.lookahead_program(6, 0.5)] + [pg.random_program(rng) for _ in range(max(0, args.pop - 2))]
    seen, results = {}, []

    def score(g):
        key = pg.canonical(g)
        if key not in seen:
            t0 = time.time(); v = evaluate(g); seen[key] = v
            results.append({"genome": desc(g), "val": v, "cost": g.extra_matmuls(),
                            "is_muon": g == pg.muon_program(),
                            "is_lookahead": g == pg.lookahead_program(6, 0.5)})
            print(f"  val {v:.4f}  cost {g.extra_matmuls():+.3f}  ({time.time()-t0:.0f}s)  {desc(g)}", flush=True)
        return seen[key]

    print(f"=== PROGRAM grammar search (d{args.depth}, {args.steps} steps, lr {args.lr}, pop {args.pop}, gens {args.gens}) ===", flush=True)
    print(f"Polar menu: {[v.name for v in POLAR_MENU]}", flush=True)
    print("[gen 0] evaluating seed population (muon + lookahead + random):", flush=True)
    for g in pop:
        score(g)
    muon_v = seen[pg.canonical(pg.muon_program())]; la_v = seen[pg.canonical(pg.lookahead_program(6, 0.5))]
    anchors = {pg.canonical(pg.muon_program()): pg.muon_program(),
               pg.canonical(pg.lookahead_program(6, 0.5)): pg.lookahead_program(6, 0.5)}

    population = list(pop)
    for gen in range(1, args.gens + 1):
        population.sort(key=score)
        elite = population[: max(2, args.pop // 2)]
        children = []
        while len(children) < args.pop - len(elite):
            i, j = (int(x) for x in rng.integers(0, len(elite), 2))
            c = pg.crossover(elite[i], elite[j], rng)
            if rng.random() < 0.9:
                c = pg.mutate(c, rng)
            children.append(c)
        print(f"[gen {gen}] evaluating {len(children)} offspring:", flush=True)
        for c in children:
            score(c)
        pool = {pg.canonical(g): g for g in elite + children}
        pool.update(anchors)
        population = sorted(pool.values(), key=score)[: args.pop]

    results.sort(key=lambda r: r["val"])
    summary = {"muon_val": muon_v, "lookahead_val": la_v,
               "best_val": min(seen.values()),
               "best_beats_muon_by": muon_v - min(seen.values()),
               "best_beats_lookahead_by": la_v - min(seen.values()),
               "ranking": results[:10]}
    json.dump(summary, open(args.out, "w"), indent=1)
    print(f"\n=== RESULT ===\n  muon {muon_v:.4f} | lookahead {la_v:.4f} | best {min(seen.values()):.4f}")
    print(f"  best beats muon by {muon_v-min(seen.values()):+.4f}, beats lookahead by {la_v-min(seen.values()):+.4f}")
    print(f"  best genome: {results[0]['genome']}")


if __name__ == "__main__":
    main()