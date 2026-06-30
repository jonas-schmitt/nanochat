"""Grammar search over the TEMPORAL optimizer grammar (gns.temporal_grammar), real training in the loop.

Stage-1 of the post-campaign moonshot: search the ~free time-domain modifications to Muon for one that
DOMINATES both Muon and vanilla Lookahead at iso-FLOP. All temporal ops add zero matmuls, so iso-step ==
iso-FLOP and the only objective is final val loss (lower = better). This is the cheap d6 Stage-1; the
top genomes get a d8 (scale/trend) Stage-2 confirmation afterwards.

Efficiency: the model is built+compiled ONCE and reset to its init between genome evaluations (no per-genome
recompile). The optimizer is the verified gns.temporal_grammar (bit-exact vs the harness muon/muon_lookahead
arms). Muon and Lookahead are seeded into the population as reference points.

  PYTHONPATH=/home/jonas/git/nanochat:/home/jonas/git/gns/src python scripts/search_temporal.py --steps 600 --pop 8 --gens 3
"""
from __future__ import annotations

import argparse, json, time
import numpy as np
import torch

from train_compare_precond import (build_model, matrix_params, apply_norm_caution_update,
                                    polar_express_orth, _build_other_adam)
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit
from gns import temporal_grammar as tg


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
    p.add_argument("--seed", type=int, default=0); p.add_argument("--out", type=str, default="/home/jonas/git/gns/results/search_temporal.json")
    return p.parse_args()


def main():
    import math
    args = _cfg()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    dev = "cuda"
    tok = get_tokenizer(); vocab = tok.get_vocab_size()
    model, _ = build_model(args, vocab, dev, args.seed)
    model = torch.compile(model, dynamic=False)
    init_params = [p.detach().clone() for p in model.parameters()]

    def materialise(split, n):
        ld = tokenizing_distributed_data_loader_with_state_bos_bestfit(tok, args.device_batch_size, args.max_seq_len, split=split, device=dev)
        return [tuple(t.clone() for t in next(ld)[:2]) for _ in range(n)]
    train = materialise("train", args.steps); val = materialise("val", args.n_val_batches)
    # Pre-normalize the polar input to UNIT scale. The polar is scale-invariant in fp32 but its aggressive
    # 5-step iteration is bf16-scale-SENSITIVE (a 20x-larger input -> ~0.08 direction noise). Different
    # genomes produce gm at different scales (heavy-ball vs EMA, multi-timescale), so feeding a consistent
    # unit scale makes every genome's bf16 orthogonalization fair and makes muon_genome a well-conditioned
    # Muon. fp32-identical to the harness muon; removes a scale-dependent bf16 confound from the search.
    ORTH = lambda g: polar_express_orth(g / (g.float().norm() + 1e-12), args.ns_steps)

    def evaluate(genome: tg.TemporalGenome) -> float:
        with torch.no_grad():
            for p, init in zip(model.parameters(), init_params):
                p.copy_(init)
        mp = matrix_params(model); mp_set = {id(p) for p in mp}
        adam = _build_other_adam(model, mp_set)
        st = [tg.init_state(genome, p) for p in mp]
        slow = [p.detach().clone() for p in model.parameters()] if genome.lookahead_k > 0 else None
        for step in range(1, args.steps + 1):
            x, y = train[step - 1]
            loss = model(x, y)
            model.zero_grad(set_to_none=True); adam.zero_grad(set_to_none=True); loss.backward()
            lrm = min(1.0, step / max(1, args.warmup_steps))
            cos_wd = args.weight_decay * 0.5 * (1 + math.cos(math.pi * step / args.steps))
            with torch.no_grad():
                for j, p in enumerate(mp):
                    if p.grad is None: continue
                    D = tg.temporal_direction(genome, p.grad, st[j], ORTH)
                    if not torch.isfinite(D).all(): return float("inf")
                    apply_norm_caution_update(D, p, st[j], args.lr * lrm, cos_wd, args.beta2)
            for gp in adam.param_groups: gp["lr"] = gp["base_lr"] * lrm
            adam.step()
            if slow is not None and step % genome.lookahead_k == 0:
                with torch.no_grad():
                    for p, s in zip(model.parameters(), slow):
                        s.add_(p.detach() - s, alpha=genome.lookahead_alpha); p.copy_(s)
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([float(model(vx, vy).item()) for vx, vy in val]))
        model.train()
        return vl

    def desc(g): return (f"betas{g.momentum_betas} w{tuple(round(x,2) for x in g.norm_weights)} "
                         f"nest={int(g.nesterov)} extrap={g.extrap_coeff} la_k={g.lookahead_k} la_a={g.lookahead_alpha}")

    rng = np.random.default_rng(args.seed)
    # reference points + random init population
    pop = [tg.muon_genome(), tg.lookahead_genome(6, 0.5)] + [tg.random_genome(rng) for _ in range(max(0, args.pop - 2))]
    seen, results = {}, []
    def score(g):
        key = repr(g)
        if key not in seen:
            t0 = time.time(); v = evaluate(g); seen[key] = v
            results.append({"genome": desc(g), "val": v, "is_muon": g == tg.muon_genome(), "is_lookahead": g == tg.lookahead_genome(6, 0.5)})
            print(f"  val {v:.4f}  ({time.time()-t0:.0f}s)  {desc(g)}", flush=True)
        return seen[key]

    print(f"=== TEMPORAL grammar search (d{args.depth}, {args.steps} steps, lr {args.lr}, pop {args.pop}, gens {args.gens}) ===", flush=True)
    print("[gen 0] evaluating seed population (muon + lookahead + random):", flush=True)
    population = list(pop)
    for g in population:
        score(g)
    muon_v = seen[repr(tg.muon_genome())]; la_v = seen[repr(tg.lookahead_genome(6, 0.5))]
    anchors = {repr(tg.muon_genome()): tg.muon_genome(),
               repr(tg.lookahead_genome(6, 0.5)): tg.lookahead_genome(6, 0.5)}
    for gen in range(1, args.gens + 1):
        population.sort(key=score)
        elite = population[: max(2, args.pop // 2)]
        children = []
        while len(children) < args.pop - len(elite):
            i, j = (int(x) for x in rng.integers(0, len(elite), 2))
            c = tg.crossover(elite[i], elite[j], rng)
            if rng.random() < 0.9:
                c = tg.mutate(c, rng)
            children.append(c)
        print(f"[gen {gen}] evaluating {len(children)} offspring:", flush=True)
        for c in children:
            score(c)
        pool = {repr(g): g for g in elite + children}
        pool.update(anchors)  # muon + lookahead remain standing references across generations
        population = sorted(pool.values(), key=score)[: args.pop]
    results.sort(key=lambda r: r["val"])
    summary = {"muon_val": muon_v, "lookahead_val": la_v,
               "best_val": min(seen.values()),
               "best_beats_muon_by": muon_v - min(seen.values()),
               "best_beats_lookahead_by": la_v - min(seen.values()),
               "ranking": results}
    json.dump(summary, open(args.out, "w"), indent=1)
    print(f"\n=== RESULT ===\n  muon {muon_v:.4f} | lookahead {la_v:.4f} | best {min(seen.values()):.4f}")
    print(f"  best beats muon by {muon_v-min(seen.values()):+.4f}, beats lookahead by {la_v-min(seen.values()):+.4f}")
    print(f"  best genome: {results[0]['genome']}")


if __name__ == "__main__":
    main()
