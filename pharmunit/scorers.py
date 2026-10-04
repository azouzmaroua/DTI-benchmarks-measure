from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseGCNLayer(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.lin_msg = nn.Linear(d_in, d_out, bias=False)
        self.lin_self = nn.Linear(d_in, d_out, bias=True)

    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        h = h * mask.unsqueeze(-1).to(h.dtype)
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        agg = torch.bmm(adj, h) / deg
        out = F.relu(self.lin_msg(agg) + self.lin_self(h))
        return out * mask.unsqueeze(-1).to(out.dtype)


class AtomEncoder(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, n_layers: int = 3):
        super().__init__()
        self.proj_in = nn.Linear(d_in, d_hidden)
        self.layers = nn.ModuleList(
            [DenseGCNLayer(d_hidden, d_hidden) for _ in range(n_layers)]
        )

    def forward(self, atom_x: torch.Tensor, adj: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(atom_x)
        h = h * atom_mask.unsqueeze(-1).to(h.dtype)
        for layer in self.layers:
            h = layer(h, adj, atom_mask)
        return h


def unit_mean_pool(
    atom_h: torch.Tensor,
    unit_idx: torch.Tensor,
    U: int,
) -> torch.Tensor:
    B, N, D = atom_h.shape
    device = atom_h.device
    idx_safe = unit_idx.clamp(min=0)
    valid = (unit_idx >= 0).to(atom_h.dtype).unsqueeze(-1)
    atom_h_masked = atom_h * valid

    sums = atom_h.new_zeros(B, U, D)
    counts = atom_h.new_zeros(B, U, 1)
    sums.scatter_add_(1, idx_safe.unsqueeze(-1).expand(-1, -1, D), atom_h_masked)
    counts.scatter_add_(1, idx_safe.unsqueeze(-1), valid)
    return sums / counts.clamp(min=1.0)


class PWIRC(nn.Module):
    def __init__(
        self,
        d_hidden: int,
        scorer_drug: str = "pharm",
        d_scorer_hidden: int = 128,
        n_pharm: int = 7,
    ):
        super().__init__()
        assert scorer_drug in {"pharm", "learned"}
        self.scorer_drug = scorer_drug
        self.n_pharm = n_pharm

        d_in = d_hidden + (n_pharm if scorer_drug == "pharm" else 0)
        self.unit_proj = nn.Linear(d_in, d_hidden)
        self.scorer = nn.Sequential(
            nn.Linear(d_in, d_scorer_hidden),
            nn.GELU(),
            nn.Linear(d_scorer_hidden, 1),
        )

    def forward(
        self,
        atom_h: torch.Tensor,
        unit_idx: torch.Tensor,
        unit_pharm: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        U = unit_pharm.shape[1]
        pooled = unit_mean_pool(atom_h, unit_idx, U)
        if self.scorer_drug == "pharm":
            g = torch.cat([pooled, unit_pharm], dim=-1)
        else:
            g = pooled

        unit_vec = self.unit_proj(g)
        score = self.scorer(g).squeeze(-1)
        score = score.masked_fill(~unit_mask, float("-inf"))
        return unit_vec, score


class BSIRC(nn.Module):
    def __init__(
        self,
        esm_dim: int,
        segment_len: int,
        d_hidden: int,
        scorer_protein: str = "pocket",
        d_scorer_hidden: int = 128,
        max_residues: int = 2048,
    ):
        super().__init__()
        assert scorer_protein in {"pocket", "esm", "learned"}
        self.scorer_protein = scorer_protein
        self.segment_len = segment_len
        self.d_hidden = d_hidden

        if scorer_protein == "learned":
            self.res_emb = nn.Embedding(max_residues, d_hidden)
            d_in_seg = d_hidden + 0
        else:
            d_in_seg = esm_dim + (1 if scorer_protein == "pocket" else 0)

        self.seg_proj = nn.Linear(d_in_seg, d_hidden)
        self.scorer = nn.Sequential(
            nn.Linear(d_in_seg, d_scorer_hidden),
            nn.GELU(),
            nn.Linear(d_scorer_hidden, 1),
        )

    def forward(
        self,
        residue_emb: torch.Tensor,
        pocket_prob: torch.Tensor,
        residue_mask: torch.Tensor,
        seg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, R, _ = residue_emb.shape
        S = R // self.segment_len

        if self.scorer_protein == "learned":
            idx = torch.arange(R, device=residue_emb.device).clamp(max=self.res_emb.num_embeddings - 1)
            e = self.res_emb(idx).unsqueeze(0).expand(B, -1, -1)
            e = e * residue_mask.unsqueeze(-1).to(e.dtype)
            e_reshaped = e.reshape(B, S, self.segment_len, self.d_hidden)
            valid = residue_mask.reshape(B, S, self.segment_len).to(e.dtype).unsqueeze(-1)
            seg_vec = (e_reshaped * valid).sum(dim=2) / valid.sum(dim=2).clamp(min=1.0)
            score = self.scorer(seg_vec).squeeze(-1)
            score = score.masked_fill(~seg_mask, float("-inf"))
            return self.seg_proj(seg_vec), score

        esm_reshaped = residue_emb.reshape(B, S, self.segment_len, -1)
        valid = residue_mask.reshape(B, S, self.segment_len).to(residue_emb.dtype).unsqueeze(-1)
        esm_mean = (esm_reshaped * valid).sum(dim=2) / valid.sum(dim=2).clamp(min=1.0)

        if self.scorer_protein == "pocket":
            poc_reshaped = pocket_prob.reshape(B, S, self.segment_len).unsqueeze(-1)
            poc_mean = (poc_reshaped * valid).sum(dim=2) / valid.sum(dim=2).clamp(min=1.0)
            g = torch.cat([esm_mean, poc_mean], dim=-1)
        else:
            g = esm_mean

        seg_vec = self.seg_proj(g)
        score = self.scorer(g).squeeze(-1)
        score = score.masked_fill(~seg_mask, float("-inf"))
        return seg_vec, score


def apply_select_mode(
    scores: torch.Tensor,
    valid: torch.Tensor,
    mode: str = "top",
) -> torch.Tensor:
    if mode == "top":
        return scores
    if mode == "bottom":
        return (-scores).masked_fill(~valid, float("-inf"))
    if mode == "rand":
        return torch.rand(scores.shape, device=scores.device).masked_fill(
            ~valid, float("-inf"))
    raise ValueError(f"unknown select_mode: {mode!r}")


def topk_gather(
    values: torch.Tensor,
    scores: torch.Tensor,
    valid: torch.Tensor,
    keep_frac: float,
    straight_through: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, D = values.shape
    n_valid = valid.sum(dim=1)
    k_per = (n_valid.float() * keep_frac).ceil().long().clamp(min=1)
    K = int(k_per.max().item())

    kept_scores, idx = torch.topk(scores, k=K, dim=1)
    kept_values = torch.gather(values, 1, idx.unsqueeze(-1).expand(-1, -1, D))

    arange = torch.arange(K, device=values.device).unsqueeze(0)
    kept_mask = arange < k_per.unsqueeze(1)

    if straight_through:
        safe = torch.where(kept_mask, kept_scores, torch.zeros_like(kept_scores))

        m = kept_mask.to(safe.dtype)
        cnt = m.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (safe * m).sum(dim=1, keepdim=True) / cnt
        var = ((safe - mean) * m).pow(2).sum(dim=1, keepdim=True) / cnt
        z = (safe - mean) / (var + 1e-5).sqrt()

        gate = torch.sigmoid(z)
        gate_st = 1.0 + (gate - gate.detach())
        kept_values = kept_values * gate_st.unsqueeze(-1)

    kept_values = kept_values * kept_mask.unsqueeze(-1).to(kept_values.dtype)
    return kept_values, kept_mask


def token_topk_gather(
    values: torch.Tensor,
    valid: torch.Tensor,
    keep_frac: float,
    score_head: nn.Module,
    straight_through: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = score_head(values).squeeze(-1)
    scores = scores.masked_fill(~valid, float("-inf"))
    return topk_gather(values, scores, valid, keep_frac, straight_through)
