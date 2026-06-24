"""Bulletproofing tests for the scalability-track checkpointing.

CPU unit tests (always run) cover the safety invariants — no silent destruction, atomic+durable
writes, exact resume serialization. GPU kill/resume integration tests (opt-in via CKPT_GPU_TESTS=1)
hard-kill a real tiny run mid-flight and assert completed units are *reused* (bit-identical), not
recomputed.

  pytest tests/test_checkpointing.py -q                      # CPU only
  CKPT_GPU_TESTS=1 pytest tests/test_checkpointing.py -q      # + GPU kill/resume
"""
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

NANO = Path("/home/jonas/git/nanochat")
RESULTS = Path("/home/jonas/git/gns/results")
ENV = {**os.environ, "PYTHONPATH": "/home/jonas/git/nanochat:/home/jonas/git/gns/src"}
UV = ["uv", "run", "--project", "/home/jonas/git/tct-models", "python", "-u"]
GPU = os.environ.get("CKPT_GPU_TESTS") == "1"
gpu_only = pytest.mark.skipif(not GPU, reason="set CKPT_GPU_TESTS=1 to run GPU kill/resume tests")


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, NANO / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


L = _mod("scaling_ladder")
S = _mod("search_optimizer")


# ----------------------------- CPU: no silent destruction -----------------------------
def _ladder_args(**ov):
    base = dict(restart=False, depths="6", arms="muon,ortho_shampoo", matrix_lr_grid="0.01",
                fixed_iters=20, opt_max_iters=100, opt_min_iters=10, device_batch_size=16,
                max_seq_len=1024)
    base.update(ov)
    return types.SimpleNamespace(**base)


def _search_args(**ov):
    base = dict(restart=False, depth_small=6, depth_large=12, eval_iters=10, n_steps=5, pop=3,
                lambda_slope=1.0, matrix_lr="0.02", seed=0)
    base.update(ov)
    return types.SimpleNamespace(**base)


def test_ladder_corrupt_and_mismatch_abort(tmp_path):
    p = tmp_path / "ck.json"
    L._save({"config": vars(_ladder_args()), "depths": [6], "baseline": "muon",
             "candidates": ["ortho_shampoo"], "pass_fixed": {"rungs": [{"depth": 6}]}}, p)
    # matching -> resumes
    got = L.load_or_init(p, _ladder_args(), [6], "muon", ["ortho_shampoo"])
    assert [r["depth"] for r in got["pass_fixed"]["rungs"]] == [6]
    # mismatch -> abort, file untouched
    with pytest.raises(SystemExit):
        L.load_or_init(p, _ladder_args(fixed_iters=999), [6], "muon", ["ortho_shampoo"])
    # corrupt -> abort, file untouched
    p.write_text("{not json")
    with pytest.raises(SystemExit):
        L.load_or_init(p, _ladder_args(), [6], "muon", ["ortho_shampoo"])
    assert p.read_text() == "{not json"
    # --restart -> fresh (intentional discard)
    fresh = L.load_or_init(p, _ladder_args(restart=True), [6], "muon", ["ortho_shampoo"])
    assert "pass_fixed" not in fresh


def test_search_corrupt_and_mismatch_abort(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "OUT_CKPT", tmp_path / "s.ckpt.json")
    rng = np.random.default_rng(0)
    pop = S.seed_population(5, 3, rng)
    S._save_ckpt({"config": vars(_search_args()), "gen_done": 0, "in_gen": None, "partial": [],
                  "population": pop, "history": [], "best": [0.5, pop[0]], "gates": {},
                  "muon_cache": {"6,10": 7.0}, "rng_state": rng.bit_generator.state})
    assert S.load_ckpt(_search_args()) is not None                      # matching resumes
    with pytest.raises(SystemExit):
        S.load_ckpt(_search_args(eval_iters=999))                       # mismatch aborts
    S.OUT_CKPT.write_text("{bad")
    with pytest.raises(SystemExit):
        S.load_ckpt(_search_args())                                     # corrupt aborts
    assert S.OUT_CKPT.read_text() == "{bad"                             # not overwritten
    assert S.load_ckpt(_search_args(restart=True)) is None             # restart discards


# ----------------------------- CPU: atomic + durable write -----------------------------
def test_atomic_write_leaves_no_partial(tmp_path):
    p = tmp_path / "a.json"
    L._save({"x": 1}, p)
    assert json.loads(p.read_text())["x"] == 1
    assert not p.with_suffix(".json.tmp").exists()  # tmp cleaned up by os.replace


def test_save_failure_keeps_previous_intact(tmp_path, monkeypatch):
    p = tmp_path / "a.json"
    L._save({"v": "good"}, p)
    monkeypatch.setattr(L.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    L._save({"v": "new"}, p)                       # must NOT raise (warns + continues)
    assert json.loads(p.read_text())["v"] == "good"  # previous checkpoint intact


# ----------------------------- CPU: exact resume serialization -----------------------------
def test_search_rng_roundtrip_is_bit_exact(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "OUT_CKPT", tmp_path / "s.ckpt.json")
    rng = np.random.default_rng(7)
    pop = S.seed_population(5, 3, rng)
    S._save_ckpt({"config": vars(_search_args(seed=7)), "gen_done": 0, "in_gen": 1,
                  "partial": [[0.3, pop[0]]], "population": pop, "history": [{"gen": 0, "best_F": 0.3}],
                  "best": [0.3, pop[0]], "gates": {}, "muon_cache": {"6,10": 7.0},
                  "rng_state": rng.bit_generator.state})
    s = S.load_ckpt(_search_args(seed=7))
    assert s["in_gen"] == 1 and len(s["partial"]) == 1
    rng2 = np.random.default_rng(0)
    rng2.bit_generator.state = s["rng_state"]
    assert rng2.random() == rng.random()           # restored stream is bit-exact


# ----------------------------- GPU: kill/resume reuses completed work -----------------------------
def _clean(tag):
    for f in RESULTS.glob(f"*{tag}*"):
        f.unlink()


def _run(cmd, **kw):
    return subprocess.run(cmd, env=ENV, cwd=NANO, timeout=1800, **kw)


@gpu_only
def test_ladder_kill_resume_reuses_d6(tmp_path):
    tag = "ckpttest_ladder"
    ck = RESULTS / f"scaling_ladder_{tag}.json"
    cmd = UV + [str(NANO / "scripts" / "scaling_ladder.py"), "--depths", "6,8", "--arms",
                "muon,ortho_shampoo", "--matrix-lr-grid", "0.02", "--mode", "fixed",
                "--fixed-iters", "15", "--batch-sweep", "", "--tag", tag]
    _clean(tag)
    # fresh run; kill the process group once d6 is checkpointed (d8 still pending)
    p = subprocess.Popen(cmd + ["--restart"], env=ENV, cwd=NANO, start_new_session=True)
    pre_kill_d6 = None
    for _ in range(1800):
        time.sleep(1)
        if ck.exists():
            try:
                d = json.loads(ck.read_text())
            except Exception:
                continue
            rungs = d.get("pass_fixed", {}).get("rungs", [])
            if any(r["depth"] == 6 for r in rungs):
                pre_kill_d6 = next(r for r in rungs if r["depth"] == 6)["candidates"]["ortho_shampoo"]["gap"]
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                break
        if p.poll() is not None:
            break
    assert pre_kill_d6 is not None, "never reached the d6 checkpoint to kill"
    p.wait()
    # resume: must skip d6 (reuse) and finish d8
    r = _run(cmd, capture_output=True, text=True)
    assert "[resume]" in r.stdout and "d6 already checkpointed" in r.stdout
    fin = json.loads(ck.read_text())
    gaps = {x["depth"]: x["candidates"]["ortho_shampoo"]["gap"] for x in fin["pass_fixed"]["rungs"]}
    assert set(gaps) == {6, 8}
    assert gaps[6] == pre_kill_d6           # d6 reused bit-identical, not recomputed
    _clean(tag)


@gpu_only
def test_harness_sub_run_resume_skips(tmp_path):
    out = tmp_path / "h.json"
    cmd = UV + [str(NANO / "scripts" / "train_compare_precond.py"), "--depth", "6",
                "--num-iterations", "8", "--arms", "muon,sgd", "--matrix-lr-grid", "0.02",
                "--eval-every", "8", "--out", str(out)]
    _run(cmd, check=True)
    first = json.loads(out.read_text())["arms"]
    r = _run(cmd, capture_output=True, text=True)            # re-run same --out
    assert r.stdout.count("[resume: skip]") == 2            # both (arm,lr) skipped
    second = json.loads(out.read_text())["arms"]
    for arm in ("muon", "sgd"):
        assert first[arm]["best_val"] == second[arm]["best_val"]  # reused identical
