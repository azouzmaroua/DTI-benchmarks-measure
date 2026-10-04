from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class CompressionStats:
    keep_rates: list[float] = field(default_factory=list)
    budget_loss: torch.Tensor | None = None
    thresholds: list[float] = field(default_factory=list)
    n_rounds: int = 0
    enabled: bool = True


def _masked_moments(x: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mf = m.to(x.dtype)
    cnt = mf.sum(dim=1, keepdim=True).clamp(min=1.0)
    mean = (x * mf).sum(dim=1, keepdim=True) / cnt
    var = (((x - mean) * mf) ** 2).sum(dim=1, keepdim=True) / cnt
    return mean, var


class ContextRecompressor(nn.Module):
    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d),
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        mf = m.unsqueeze(-1).to(v.dtype)
        ctx = (v * mf).sum(dim=1, keepdim=True) / mf.sum(dim=1, keepdim=True).clamp(min=1.0)
        h = self.net(torch.cat([v, ctx.expand_as(v)], dim=-1))
        return self.norm(v + h) * mf


class AttnRecompressor(nn.Module):
    def __init__(self, d: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        upd, _ = self.attn(v, v, v, key_padding_mask=~m, need_weights=False)
        upd = torch.nan_to_num(upd, nan=0.0)
        return self.norm(v + upd) * m.unsqueeze(-1).to(v.dtype)


class RecursiveCompressor(nn.Module):
    def __init__(
        self,
        d: int,
        enabled: bool = True,
        rounds: int = 2,
        keep_frac: float = 0.70,
        init_keep: float | None = None,
        retention: str = "learned",
        recompress: str = "context",
        budget_weight: float = 0.05,
        target_keep: float = 0.50,
        temperature: float = 1.0,
        min_tokens: int = 1,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if retention not in {"fixed", "learned"}:
            raise ValueError(f"retention must be 'fixed' or 'learned', got {retention!r}")
        if recompress not in {"none", "context", "attn"}:
            raise ValueError(f"recompress must be 'none'|'context'|'attn', got {recompress!r}")
        if rounds < 1:
            raise ValueError(f"rounds must be >= 1, got {rounds}")

        self.enabled = enabled
        self.rounds = rounds
        self.keep_frac = keep_frac
        self.retention = retention
        self.recompress_mode = recompress
        self.budget_weight = budget_weight
        self.target_keep = target_keep
        self.temperature = temperature
        self.min_tokens = min_tokens

        if not enabled:
            return

        self.scorers = nn.ModuleList(
            nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 1, bias=False))
            for _ in range(rounds)
        )
        if recompress == "context":
            self.recompressors = nn.ModuleList(
                ContextRecompressor(d, dropout) for _ in range(rounds))
        elif recompress == "attn":
            self.recompressors = nn.ModuleList(
                AttnRecompressor(d, n_heads, dropout) for _ in range(rounds))
        else:
            self.recompressors = None

        if retention == "learned":
            ik = keep_frac if init_keep is None else init_keep
            if not 0.0 < ik <= 1.0:
                raise ValueError(f"init_keep must be in (0, 1], got {ik}")
            if ik > 0.99:
                raise ValueError(
                    f"init_keep={ik} saturates the straight-through gate: every "
                    f"gate -> 1, sigma'(.) -> 0, and the scorer receives ~4x less "
                    f"gradient. Use 0.95 for near-full initial retention.")
            self.tau = nn.Parameter(torch.full((rounds,), _z_for_keep(ik)))


    def forward(
        self,
        v: torch.Tensor,
        m: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, CompressionStats]:
        if not self.enabled:
            return v, m, CompressionStats(enabled=False, n_rounds=0,
                                          budget_loss=v.new_zeros(()))

        stats = CompressionStats(enabled=True, n_rounds=self.rounds,
                                 budget_loss=v.new_zeros(()))

        for r in range(self.rounds):
            if self.recompressors is not None:
                v = self.recompressors[r](v, m)

            score = self.scorers[r](v).squeeze(-1)
            score = score.masked_fill(~m, float("-inf"))

            safe = torch.where(m, score, torch.zeros_like(score))
            mean, var = _masked_moments(safe, m)
            z = (safe - mean) / (var + 1e-5).sqrt()
            z = z.masked_fill(~m, float("-inf"))

            n_valid = m.sum(dim=1)
            if self.retention == "fixed":
                k_per = (n_valid.float() * self.keep_frac).ceil().long()
                tau_r = None
            else:
                tau_r = self.tau[r]
                k_per = (z > tau_r).sum(dim=1)

                full_gate = torch.sigmoid((z - tau_r) / self.temperature)
                mf = m.to(full_gate.dtype)
                realised_keep_soft = (full_gate * mf).sum() / mf.sum().clamp(min=1.0)
            k_per = k_per.clamp(min=self.min_tokens).minimum(n_valid.clamp(min=1))
            K = int(k_per.max().item())

            kept_z, idx = torch.topk(z, k=K, dim=1)
            arange = torch.arange(K, device=v.device).unsqueeze(0)
            keep_mask = (arange < k_per.unsqueeze(1)) & (kept_z > float("-inf"))

            v = torch.gather(v, 1, idx.unsqueeze(-1).expand(-1, -1, v.shape[-1]))

            zc = torch.where(keep_mask, kept_z, torch.zeros_like(kept_z))
            shift = zc if tau_r is None else zc - tau_r
            gate = torch.sigmoid(shift / self.temperature)
            v = v * (1.0 + (gate - gate.detach())).unsqueeze(-1)

            m = keep_mask
            v = v * m.unsqueeze(-1).to(v.dtype)

            rate = (k_per.float() / n_valid.clamp(min=1).float()).mean()
            stats.keep_rates.append(float(rate.detach()))
            if tau_r is not None:
                stats.thresholds.append(float(tau_r.detach()))
                stats.budget_loss = stats.budget_loss + \
                    (realised_keep_soft - self.target_keep).clamp(min=0.0) ** 2

        stats.budget_loss = stats.budget_loss * self.budget_weight
        return v, m, stats

    def extra_repr(self) -> str:
        if not self.enabled:
            return "enabled=False (identity)"
        return (f"rounds={self.rounds}, retention={self.retention}, "
                f"recompress={self.recompress_mode}, budget_weight={self.budget_weight}")


def _z_for_keep(keep_frac: float) -> float:
    import math
    p = min(max(1.0 - keep_frac, 1e-4), 1 - 1e-4)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    rr = q * q
    return (((((a[0]*rr+a[1])*rr+a[2])*rr+a[3])*rr+a[4])*rr+a[5])*q / \
           (((((b[0]*rr+b[1])*rr+b[2])*rr+b[3])*rr+b[4])*rr+1)
