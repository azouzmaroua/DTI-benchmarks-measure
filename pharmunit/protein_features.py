from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


def sequence_hash(seq: str) -> str:
    seq = seq.strip().upper().replace(" ", "").replace("\n", "")
    return hashlib.sha1(seq.encode("ascii", errors="replace")).hexdigest()


@dataclass
class ProteinFeatures:
    seq: str
    residue_emb: np.ndarray
    pocket_prob: np.ndarray
    length: int


def load_esm_embedding(seq: str, cache_dir: Path, esm_dim: int) -> np.ndarray:
    key = sequence_hash(seq)
    fp = Path(cache_dir) / f"{key}.npy"
    if not fp.exists():
        raise FileNotFoundError(
            f"ESM cache miss for sequence hash {key} at {fp}. "
            f"Run scripts/precompute_esm.py to build the cache."
        )
    arr = np.load(fp)
    if arr.ndim != 2 or arr.shape[1] != esm_dim:
        raise ValueError(
            f"ESM cache {fp} has shape {arr.shape}; expected (L, {esm_dim})."
        )
    return arr.astype(np.float32)


def load_pocket_probability(
    seq: str,
    cache_dir: Optional[Path],
) -> Optional[np.ndarray]:
    if cache_dir is None:
        return None
    fp = Path(cache_dir) / f"{sequence_hash(seq)}.npy"
    if not fp.exists():
        return None
    arr = np.load(fp).astype(np.float32).reshape(-1)
    return np.clip(arr, 0.0, 1.0)


def featurize_protein(
    seq: str,
    esm_cache_dir: Path,
    p2rank_cache_dir: Optional[Path],
    esm_dim: int,
    max_len: int,
) -> ProteinFeatures:
    seq = seq.strip().upper()
    emb = load_esm_embedding(seq, esm_cache_dir, esm_dim)
    pocket = load_pocket_probability(seq, p2rank_cache_dir)

    L = emb.shape[0]
    if pocket is None:
        pocket = np.zeros(L, dtype=np.float32)
    elif pocket.shape[0] != L:
        pocket = np.pad(pocket, (0, max(0, L - pocket.shape[0])))[:L]

    if L > max_len:
        emb = emb[:max_len]
        pocket = pocket[:max_len]
        L = max_len

    return ProteinFeatures(
        seq=seq,
        residue_emb=emb,
        pocket_prob=pocket,
        length=L,
    )
