from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .config import Config

SELECTION_PREFIXES = (
    "compress_drug", "compress_protein",
)

def count_parameters(model: nn.Module) -> dict[str, int]:
    total = 0
    selection = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        total += n
        if name.split(".")[0] in SELECTION_PREFIXES:
            selection += n
    return {
        "total": total,
        "selection_overhead": selection,
        "shared_backbone": total - selection,
        "overhead_pct": round(100.0 * selection / max(total - selection, 1), 2),
    }

def measure_flops(model: nn.Module, batch, device: torch.device) -> Optional[int]:
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None

    model.eval()
    counter = FlopCounterMode(display=False)
    try:
        with counter, torch.no_grad():
            model(batch)
    except Exception:
        return None
    return int(counter.get_total_flops())

def measure_latency(
    model: nn.Module,
    batch,
    device: torch.device,
    warmup: int = 10,
    repeats: int = 50,
) -> dict[str, float]:
    model.eval()
    is_cuda = device.type == "cuda"

    with torch.no_grad():
        for _ in range(warmup):
            model(batch)
        if is_cuda:
            torch.cuda.synchronize()

        samples = []
        for _ in range(repeats):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(batch)
            if is_cuda:
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)

    samples.sort()
    n_pairs = int(batch.y.numel())
    median = statistics.median(samples)
    return {
        "latency_ms_median": round(median, 4),
        "latency_ms_p90": round(samples[int(0.9 * (len(samples) - 1))], 4),
        "latency_ms_per_pair": round(median / max(n_pairs, 1), 5),
        "batch_size": n_pairs,
        "repeats": repeats,
    }

def measure_peak_memory(model: nn.Module, batch, device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    model.eval()
    with torch.no_grad():
        model(batch)
    torch.cuda.synchronize()
    return round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 3)

@dataclass
class EfficiencyReport:
    tag: str
    dataset: str
    split_mode: str
    compress_enabled: bool
    compress_retention: str
    compress_recompress: str
    keep_drug: float
    keep_protein: float
    params_total: int
    params_selection_overhead: int
    params_overhead_pct: float
    flops: Optional[int]
    flops_measured: bool
    latency_ms_median: float
    latency_ms_p90: float
    latency_ms_per_pair: float
    peak_mem_mib: Optional[float]
    device: str
    batch_size: int

def profile(model: nn.Module, batch, cfg: Config, device: torch.device,
            tag: str = "") -> EfficiencyReport:
    params = count_parameters(model)
    flops = measure_flops(model, batch, device)
    lat = measure_latency(model, batch, device)
    mem = measure_peak_memory(model, batch, device)
    return EfficiencyReport(
        tag=tag or cfg.tag,
        dataset=cfg.dataset,
        split_mode=cfg.split_mode,
        compress_enabled=cfg.compress_enabled,
        compress_retention=cfg.compress_retention,
        compress_recompress=cfg.compress_recompress,
        keep_drug=cfg.keep_drug,
        keep_protein=cfg.keep_protein,
        params_total=params["total"],
        params_selection_overhead=params["selection_overhead"],
        params_overhead_pct=params["overhead_pct"],
        flops=flops,
        flops_measured=flops is not None,
        latency_ms_median=lat["latency_ms_median"],
        latency_ms_p90=lat["latency_ms_p90"],
        latency_ms_per_pair=lat["latency_ms_per_pair"],
        peak_mem_mib=mem,
        device=str(device),
        batch_size=lat["batch_size"],
    )

def compare(reports: list[EfficiencyReport], reference_tag: str) -> list[dict]:
    ref = next((r for r in reports if r.tag == reference_tag), None)
    if ref is None:
        raise ValueError(f"reference tag {reference_tag!r} not among {[r.tag for r in reports]}")

    rows = []
    for r in reports:
        flops_saved = None
        if r.flops is not None and ref.flops:
            flops_saved = round(100.0 * (1.0 - r.flops / ref.flops), 2)
        rows.append({
            **asdict(r),
            "flops_saved_pct_vs_ref": flops_saved,
            "latency_saved_pct_vs_ref": round(
                100.0 * (1.0 - r.latency_ms_median / ref.latency_ms_median), 2
            ),
            "speedup_vs_ref": round(ref.latency_ms_median / max(r.latency_ms_median, 1e-9), 3),
            "reference": reference_tag,
        })
    return rows

def write_report(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"[efficiency] wrote {out_path}")
