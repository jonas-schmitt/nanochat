"""Registry of Muon orthogonalization arms for the GNS experiments.

Maps a short arm name to an orthogonalization *method* consumed by
``nanochat.optim.orthogonalize_eager``:

- a string method: ``"polar_express"`` (the nanochat stock map), ``"svd"`` (exact polar
  oracle), or ``"none"`` (SGD-momentum control); or
- a gns ``Schedule`` (tuple of ``gns.ir`` steps) parsed from
  ``gns/results/dgx_candidate_config.json`` — the GNS frontier + controls.

The gns config path defaults to the sibling checkout (``../gns`` next to this repo) and
can be overridden with the ``GNS_DGX_CONFIG`` environment variable. Resolving the schedule
set from gns's checked-in handoff artifact keeps the arm set in one source of truth
(``DGX_SPARK_PLAN.md``); no schedule canonicals are duplicated here.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# String-method arms (need only nanochat.optim, not gns).
STRING_METHODS = {
    "polar_express": "polar_express",  # nanochat stock baseline (run via the eager harness)
    "svd": "svd",                       # exact polar factor oracle
    "none": "none",                     # identity / SGD-momentum control
}


def _default_gns_config() -> Path:
    env = os.environ.get("GNS_DGX_CONFIG")
    if env:
        return Path(env)
    # sibling checkout: <parent>/gns/results/dgx_candidate_config.json
    return Path(__file__).resolve().parents[1].parent / "gns" / "results" / "dgx_candidate_config.json"


def gns_schedule_arms(config_path: Path | str | None = None) -> dict[str, tuple]:
    """Name -> gns Schedule for every ``kind == 'schedule'`` candidate in the config."""
    from gns.ir import parse_canonical  # local import: only needed for gns arms
    path = Path(config_path) if config_path is not None else _default_gns_config()
    payload = json.loads(path.read_text())
    arms: dict[str, tuple] = {}
    for cand in payload["candidates"]:
        if cand["kind"] == "schedule":
            arms[cand["name"]] = parse_canonical(cand["schedule"]["canonical"])
    return arms


def build_registry(config_path: Path | str | None = None) -> dict[str, object]:
    """Full arm name -> method registry (string methods + gns schedules)."""
    reg: dict[str, object] = dict(STRING_METHODS)
    reg.update(gns_schedule_arms(config_path))
    return reg


def get_orth_method(name: str, config_path: Path | str | None = None) -> object:
    reg = build_registry(config_path)
    if name not in reg:
        raise KeyError(f"unknown muon arm {name!r}; available: {sorted(reg)}")
    return reg[name]


def available_arms(config_path: Path | str | None = None) -> list[str]:
    return sorted(build_registry(config_path))


def set_fp8_real(real: bool) -> bool:
    """Select the gns fp8 execution path.

    Round 1 (steps-to-loss / quality) uses SIMULATED fp8 — per-tensor fp8e4m3 storage
    rounding of operands with bf16 compute (gns DESIGN convention 7). It is the faithful
    quality reference and, unlike the real cuBLAS kernel, has no matmul shape constraints.

    Round 2 (wall-clock) flips this to the real ``torch._scaled_mm`` kernel, which is what
    converts the modeled fp8 cost into measured wall-clock — but it requires the contracting
    dimension to be divisible by 16.

    Returns the value actually set (False if real fp8 is unavailable on this build)."""
    import gns.executor as ex
    ex.fp8_is_real = bool(real) and ex._detect_fp8()
    return ex.fp8_is_real
