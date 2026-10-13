from __future__ import annotations

import torch
import torch.nn as nn

from .compression import RecursiveCompressor
from .config import Config
from .dataset import Batch
from .scorers import AtomEncoder, unit_mean_pool

class UnitEncoder(nn.Module):

    def __init__(self, d_hidden: int, use_pharm: bool, n_pharm: int = 7):
        super().__init__()
        self.use_pharm = use_pharm
        self.proj = nn.Linear(d_hidden + (n_pharm if use_pharm else 0), d_hidden)

    def forward(self, atom_h, unit_idx, unit_pharm, unit_mask):
        pooled = unit_mean_pool(atom_h, unit_idx, unit_pharm.shape[1])
        g = torch.cat([pooled, unit_pharm], dim=-1) if self.use_pharm else pooled
        return self.proj(g) * unit_mask.unsqueeze(-1).to(pooled.dtype)

class SegmentEncoder(nn.Module):

    def __init__(self, esm_dim: int, segment_len: int, d_hidden: int, use_pocket: bool):
        super().__init__()
        self.segment_len = segment_len
        self.use_pocket = use_pocket
        self.proj = nn.Linear(esm_dim + (1 if use_pocket else 0), d_hidden)

    def forward(self, residue_emb, pocket_prob, residue_mask, seg_mask):
        B, R, _ = residue_emb.shape
        S = R // self.segment_len
        valid = residue_mask.reshape(B, S, self.segment_len).to(residue_emb.dtype).unsqueeze(-1)
        denom = valid.sum(dim=2).clamp(min=1.0)
        e = (residue_emb.reshape(B, S, self.segment_len, -1) * valid).sum(dim=2) / denom
        if self.use_pocket:
            p = (pocket_prob.reshape(B, S, self.segment_len).unsqueeze(-1) * valid).sum(dim=2) / denom
            e = torch.cat([e, p], dim=-1)
        return self.proj(e) * seg_mask.unsqueeze(-1).to(e.dtype)

class CrossAttentionBlock(nn.Module):

    def __init__(self, d: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.a_dp = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.a_pd = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ln_d1, self.ln_p1 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ln_d2, self.ln_p2 = nn.LayerNorm(d), nn.LayerNorm(d)
        mk = lambda: nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.ff_d, self.ff_p = mk(), mk()

    def forward(self, zd, md, zp, mp):
        d_upd, _ = self.a_dp(zd, zp, zp, key_padding_mask=~mp, need_weights=False)
        p_upd, _ = self.a_pd(zp, zd, zd, key_padding_mask=~md, need_weights=False)
        zd = self.ln_d1(zd + torch.nan_to_num(d_upd))
        zp = self.ln_p1(zp + torch.nan_to_num(p_upd))
        return self.ln_d2(zd + self.ff_d(zd)), self.ln_p2(zp + self.ff_p(zp))

def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

class PharmUnitDTI(nn.Module):
    def __init__(self, cfg: Config, atom_dim: int):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        d = cfg.hidden_dim

        self.atom_encoder = AtomEncoder(atom_dim, d, n_layers=cfg.gnn_layers)
        self.unit_encoder = UnitEncoder(d, use_pharm=(cfg.scorer_drug == "pharm"))
        self.seg_encoder = SegmentEncoder(cfg.esm_dim, cfg.segment_len, d,
                                          use_pocket=(cfg.scorer_protein == "pocket"))

        mk_comp = lambda keep: RecursiveCompressor(
            d,
            enabled=cfg.compress_enabled,
            rounds=cfg.compress_rounds,
            keep_frac=keep,
            init_keep=cfg.compress_init_keep,
            retention=cfg.compress_retention,
            recompress=cfg.compress_recompress,
            budget_weight=cfg.compress_budget_weight,
            target_keep=cfg.compress_target_keep,
            temperature=cfg.compress_temperature,
            min_tokens=cfg.compress_min_tokens,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
        )
        self.compress_drug = mk_comp(cfg.keep_drug)
        self.compress_protein = mk_comp(cfg.keep_protein)

        self.blocks = nn.ModuleList(
            CrossAttentionBlock(d, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_cross_layers))
        self.classifier = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d, 1))

    def forward(self, batch: Batch, return_stats: bool = False):
        atom_h = self.atom_encoder(batch.atom_x, batch.adj, batch.atom_mask)
        zd = self.unit_encoder(atom_h, batch.unit_idx, batch.unit_pharm, batch.unit_mask)
        zp = self.seg_encoder(batch.residue_emb, batch.pocket_prob,
                              batch.residue_mask, batch.seg_mask)

        zd, md, sd = self.compress_drug(zd, batch.unit_mask)
        zp, mp, sp = self.compress_protein(zp, batch.seg_mask)

        for blk in self.blocks:
            zd, zp = blk(zd, md, zp, mp)

        pooled = torch.cat([masked_mean(zd, md), masked_mean(zp, mp)], dim=-1)
        logit = self.classifier(pooled).squeeze(-1)

        if return_stats:
            return logit, {"drug": sd, "protein": sp}
        return logit

def bce_loss(logit: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(logit, y)

def total_loss(logit, y, stats) -> tuple[torch.Tensor, dict]:
    task = bce_loss(logit, y)
    budget = stats["drug"].budget_loss + stats["protein"].budget_loss
    return task + budget, {"task": float(task.detach()), "budget": float(budget.detach())}
