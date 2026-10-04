from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .drug_features import DrugFeatures, featurize_drug, atom_feature_dim
from .protein_features import ProteinFeatures, featurize_protein


def load_pair_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    lookup = {c.lower(): c for c in df.columns}
    try:
        smiles_col, prot_col, label_col = lookup["smiles"], lookup["protein"], lookup["y"]
    except KeyError as e:
        raise ValueError(
            f"{path}: expected DrugBAN columns SMILES/Protein/Y, found {list(df.columns)}"
        ) from e
    out = df[[smiles_col, prot_col, label_col]].copy()
    out.columns = ["drug", "target", "label"]
    out["label"] = out["label"].astype(int)
    return out.reset_index(drop=True)


def load_pairs(path: str | Path) -> pd.DataFrame:
    return load_pair_csv(path) if str(path).lower().endswith(".csv") else load_pair_txt(path)


def load_pair_txt(path: str | Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            smiles, seq, label = parts[0], parts[1], parts[-1]
            try:
                y = int(label)
            except ValueError:
                continue
            rows.append((smiles, seq, y))
    df = pd.DataFrame(rows, columns=["drug", "target", "label"])
    return df


@dataclass
class Sample:
    drug: DrugFeatures
    protein: ProteinFeatures
    label: float


class DTIDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        esm_cache_dir: Path,
        p2rank_cache_dir: Optional[Path],
        esm_dim: int,
        max_len_protein: int,
        drug_cache: Optional[dict[str, DrugFeatures]] = None,
        protein_cache: Optional[dict[str, ProteinFeatures]] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.esm_cache_dir = Path(esm_cache_dir)
        self.p2rank_cache_dir = Path(p2rank_cache_dir) if p2rank_cache_dir else None
        self.esm_dim = esm_dim
        self.max_len_protein = max_len_protein
        self.drug_cache: dict[str, DrugFeatures] = drug_cache if drug_cache is not None else {}
        self.protein_cache: dict[str, ProteinFeatures] = protein_cache if protein_cache is not None else {}

        keep = []
        for i, s in enumerate(self.df["drug"].tolist()):
            if s in self.drug_cache:
                keep.append(i); continue
            df_feat = featurize_drug(s)
            if df_feat is not None:
                self.drug_cache[s] = df_feat
                keep.append(i)
        self.df = self.df.iloc[keep].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Sample:
        row = self.df.iloc[idx]
        smiles, seq, label = row["drug"], row["target"], float(row["label"])
        drug = self.drug_cache[smiles]
        prot = self.protein_cache.get(seq)
        if prot is None:
            prot = featurize_protein(
                seq,
                esm_cache_dir=self.esm_cache_dir,
                p2rank_cache_dir=self.p2rank_cache_dir,
                esm_dim=self.esm_dim,
                max_len=self.max_len_protein,
            )
            self.protein_cache[seq] = prot
        return Sample(drug=drug, protein=prot, label=label)


@dataclass
class Batch:
    atom_x: torch.Tensor
    atom_mask: torch.Tensor
    adj: torch.Tensor
    unit_idx: torch.Tensor
    unit_pharm: torch.Tensor
    unit_mask: torch.Tensor
    unit_size: torch.Tensor
    residue_emb: torch.Tensor
    residue_mask: torch.Tensor
    pocket_prob: torch.Tensor
    seg_mask: torch.Tensor
    seg_size: torch.Tensor
    y: torch.Tensor


def _pad_stack(arrays: list[np.ndarray], pad_to: int, pad_value=0) -> np.ndarray:
    B = len(arrays)
    tail = arrays[0].shape[1:]
    out = np.full((B, pad_to, *tail), pad_value, dtype=arrays[0].dtype)
    for i, a in enumerate(arrays):
        out[i, : a.shape[0]] = a
    return out


def _pad_adj(adjs: list[np.ndarray], pad_to: int) -> np.ndarray:
    B = len(adjs)
    out = np.zeros((B, pad_to, pad_to), dtype=np.float32)
    for i, a in enumerate(adjs):
        n = a.shape[0]
        out[i, :n, :n] = a
    return out


def make_collate(segment_len: int) -> Callable[[list[Sample]], Batch]:
    def collate(samples: list[Sample]) -> Batch:
        n_atoms = [s.drug.n_atoms for s in samples]
        n_units = [s.drug.n_units for s in samples]
        N = max(n_atoms)
        U = max(n_units)

        atom_x = _pad_stack([s.drug.atom_x for s in samples], N)
        adj = _pad_adj([s.drug.adj for s in samples], N)
        unit_idx = np.full((len(samples), N), -1, dtype=np.int64)
        unit_size = np.zeros((len(samples), U), dtype=np.float32)
        unit_pharm = np.zeros((len(samples), U, 7), dtype=np.float32)
        atom_mask = np.zeros((len(samples), N), dtype=bool)
        unit_mask = np.zeros((len(samples), U), dtype=bool)

        for i, s in enumerate(samples):
            nA, nU = s.drug.n_atoms, s.drug.n_units
            unit_idx[i, :nA] = s.drug.unit_idx.astype(np.int64)
            atom_mask[i, :nA] = True
            unit_mask[i, :nU] = True
            unit_pharm[i, :nU] = s.drug.unit_pharm
            counts = np.bincount(s.drug.unit_idx.astype(np.int64), minlength=nU)
            unit_size[i, :nU] = counts.astype(np.float32)

        R = max(s.protein.length for s in samples)
        R = ((R + segment_len - 1) // segment_len) * segment_len
        S = R // segment_len

        esm_dim = samples[0].protein.residue_emb.shape[1]
        residue_emb = np.zeros((len(samples), R, esm_dim), dtype=np.float32)
        pocket_prob = np.zeros((len(samples), R), dtype=np.float32)
        residue_mask = np.zeros((len(samples), R), dtype=bool)

        for i, s in enumerate(samples):
            L = s.protein.length
            residue_emb[i, :L] = s.protein.residue_emb
            pocket_prob[i, :L] = s.protein.pocket_prob
            residue_mask[i, :L] = True

        rm_reshaped = residue_mask.reshape(len(samples), S, segment_len)
        seg_mask = rm_reshaped.any(axis=-1)
        seg_size = rm_reshaped.astype(np.float32).sum(axis=-1)

        y = np.array([s.label for s in samples], dtype=np.float32)

        to_t = torch.from_numpy
        return Batch(
            atom_x=to_t(atom_x),
            atom_mask=to_t(atom_mask.astype(np.uint8)).bool(),
            adj=to_t(adj),
            unit_idx=to_t(unit_idx),
            unit_pharm=to_t(unit_pharm),
            unit_mask=to_t(unit_mask.astype(np.uint8)).bool(),
            unit_size=to_t(unit_size),
            residue_emb=to_t(residue_emb),
            residue_mask=to_t(residue_mask.astype(np.uint8)).bool(),
            pocket_prob=to_t(pocket_prob),
            seg_mask=to_t(seg_mask.astype(np.uint8)).bool(),
            seg_size=to_t(seg_size),
            y=to_t(y),
        )

    return collate


def infer_atom_dim() -> int:
    return atom_feature_dim()
