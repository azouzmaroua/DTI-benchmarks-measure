from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


@dataclass
class Config:
    dataset: str = "human"
    protocol: str = "random"
    train_file: str = "datasets/human/random/train.csv"
    val_file: str = "datasets/human/random/val.csv"
    test_file: str = "datasets/human/random/test.csv"

    esm_cache_dir: str = field(
        default_factory=lambda: os.environ.get("PHARMUNIT_ESM_CACHE", "cache/esm"))
    p2rank_dir: str = ""

    split_mode: str = "given"
    split_val_frac: float = 0.10
    split_test_frac: float = 0.10
    cold_pair_cold_val: bool = False
    deduplicate: bool = True
    drop_contradictory: bool = True

    max_residues: int = 1000
    esm_dim: int = 1280
    segment_len: int = 8

    scorer_drug: str = "pharm"
    scorer_protein: str = "esm"

    compress_enabled: bool = True
    compress_rounds: int = 2
    compress_retention: str = "learned"
    compress_recompress: str = "context"
    compress_budget_weight: float = 0.05
    compress_target_keep: float = 0.50
    compress_temperature: float = 1.0
    compress_min_tokens: int = 1

    compress_init_keep: float = 0.95

    keep_drug: float = 0.50
    keep_protein: float = 0.70

    hidden_dim: int = 256
    n_heads: int = 4
    n_cross_layers: int = 2
    dropout: float = 0.1
    gnn_layers: int = 3

    batch_size: int = 32
    epochs: int = 60
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    patience: int = 12
    num_workers: int = 0

    seed: int = 0
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    device: str = "cuda"
    log_dir: str = "runs"
    tag: str = "full"

    def validate(self) -> None:
        if self.compress_retention not in {"fixed", "learned"}:
            raise ValueError(f"compress_retention: {self.compress_retention!r}")
        if self.compress_recompress not in {"none", "context", "attn"}:
            raise ValueError(f"compress_recompress: {self.compress_recompress!r}")
        if self.compress_enabled and self.compress_rounds < 1:
            raise ValueError("compress_rounds must be >= 1 when compression is on")
        if self.compress_rounds > 1 and self.compress_recompress == "none":
            raise ValueError(
                "rounds>1 with recompress='none' is a degenerate configuration: "
                "with static scores, nested top-k equals a single top-k at the "
                "product ratio, so the extra rounds change nothing. Set "
                "recompress='context' or 'attn', or use rounds=1.")
        if self.scorer_protein == "pocket" and not self.p2rank_dir:
            raise ValueError(
                "scorer_protein='pocket' with no p2rank_dir: the pocket channel "
                "would be identically zero, making this indistinguishable from "
                "'esm'. Set p2rank_dir, or use scorer_protein='esm'.")
        if not 0.0 < self.compress_init_keep <= 0.99:
            raise ValueError(
                f"compress_init_keep must be in (0, 0.99]; {self.compress_init_keep} "
                f"saturates the gate and starves the scorer of gradient.")
        if self.split_mode not in {"given", "random", "cold_drug", "cold_target", "cold_pair"}:
            raise ValueError(f"split_mode: {self.split_mode!r}")

    def paths(self) -> dict[str, Path]:
        return {k: Path(v) for k, v in {
            "train": self.train_file, "val": self.val_file, "test": self.test_file,
            "esm_cache": self.esm_cache_dir, "log": self.log_dir,
        }.items()}
