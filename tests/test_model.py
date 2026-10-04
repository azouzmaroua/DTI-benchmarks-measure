from __future__ import annotations

import sys
import pathlib

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pharmunit.config import Config
from pharmunit.dataset import Batch
from pharmunit.model import PharmUnitDTI, total_loss

ATOM_DIM = 25
ESM_DIM = 64


def _cfg(**kw) -> Config:
    base = dict(esm_dim=ESM_DIM, hidden_dim=32, n_heads=2, n_cross_layers=1,
                gnn_layers=2, scorer_protein="esm", p2rank_dir="")
    base.update(kw)
    return Config(**base)


def _batch(B=4, N=10, U=5, R=32, seg=8, seed=0) -> Batch:
    g = torch.Generator().manual_seed(seed)
    S = R // seg
    unit_idx = torch.randint(0, U, (B, N), generator=g)
    unit_pharm = torch.rand(B, U, 7, generator=g)
    unit_pharm = unit_pharm / unit_pharm.sum(-1, keepdim=True)
    adj = torch.zeros(B, N, N)
    for b in range(B):
        adj[b] = torch.eye(N)
    return Batch(
        atom_x=torch.randn(B, N, ATOM_DIM, generator=g),
        atom_mask=torch.ones(B, N, dtype=torch.bool),
        adj=adj,
        unit_idx=unit_idx,
        unit_pharm=unit_pharm,
        unit_mask=torch.ones(B, U, dtype=torch.bool),
        unit_size=torch.ones(B, U),
        residue_emb=torch.randn(B, R, ESM_DIM, generator=g),
        residue_mask=torch.ones(B, R, dtype=torch.bool),
        pocket_prob=torch.zeros(B, R),
        seg_mask=torch.ones(B, S, dtype=torch.bool),
        seg_size=torch.full((B, S), float(seg)),
        y=torch.randint(0, 2, (B,), generator=g).float(),
    )


RUNGS = {
    "off":         dict(compress_enabled=False),
    "single_shot": dict(compress_enabled=True, compress_rounds=1,
                        compress_retention="fixed", compress_recompress="none"),
    "recursive":   dict(compress_enabled=True, compress_rounds=2,
                        compress_retention="learned", compress_recompress="context"),
}


@pytest.mark.parametrize("name", list(RUNGS))
def test_forward_runs(name):
    cfg = _cfg(**RUNGS[name])
    m = PharmUnitDTI(cfg, ATOM_DIM)
    out = m(_batch())
    assert out.shape == (4,)
    assert torch.isfinite(out).all(), "non-finite logit"


@pytest.mark.parametrize("name", list(RUNGS))
def test_no_dead_parameters(name):
    cfg = _cfg(**RUNGS[name])
    m = PharmUnitDTI(cfg, ATOM_DIM)
    logit, stats = m(_batch(), return_stats=True)
    loss, _ = total_loss(logit, _batch().y, stats)
    loss.backward()

    dead = [n for n, p in m.named_parameters()
            if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    assert not dead, f"[{name}] parameters received no gradient: {dead}"


def test_disabled_keeps_every_token():
    cfg = _cfg(**RUNGS["off"])
    m = PharmUnitDTI(cfg, ATOM_DIM)
    _, stats = m(_batch(), return_stats=True)
    assert stats["drug"].enabled is False and stats["protein"].enabled is False
    assert stats["drug"].n_rounds == 0
    assert list(m.compress_drug.parameters()) == []
    assert list(m.compress_protein.parameters()) == []


def test_recursive_actually_compresses():
    cfg = _cfg(**RUNGS["recursive"])
    m = PharmUnitDTI(cfg, ATOM_DIM)
    _, stats = m(_batch(U=8, R=64), return_stats=True)
    assert len(stats["drug"].keep_rates) == 2
    assert all(0.0 < r <= 1.0 for r in stats["drug"].keep_rates)
    assert stats["drug"].budget_loss is not None


def test_budget_loss_is_zero_when_retention_fixed():
    cfg = _cfg(**RUNGS["single_shot"])
    m = PharmUnitDTI(cfg, ATOM_DIM)
    b = _batch()
    logit, stats = m(b, return_stats=True)
    _, parts = total_loss(logit, b.y, stats)
    assert parts["budget"] == 0.0


def test_pharmacophore_ablation_changes_parameter_shape():
    with_p = PharmUnitDTI(_cfg(scorer_drug="pharm", **RUNGS["recursive"]), ATOM_DIM)
    without = PharmUnitDTI(_cfg(scorer_drug="learned", **RUNGS["recursive"]), ATOM_DIM)
    assert with_p.unit_encoder.proj.in_features == \
        without.unit_encoder.proj.in_features + 7


def test_pocket_without_p2rank_is_refused():
    with pytest.raises(ValueError, match="identically zero"):
        _cfg(scorer_protein="pocket", p2rank_dir="").validate()


def test_degenerate_recursion_is_refused():
    with pytest.raises(ValueError, match="degenerate"):
        _cfg(compress_enabled=True, compress_rounds=3,
             compress_recompress="none").validate()


def test_valid_configs_pass_validation():
    for name, kw in RUNGS.items():
        _cfg(**kw).validate()
