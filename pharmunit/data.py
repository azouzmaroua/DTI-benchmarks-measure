from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import Config
from .dataset import load_pairs
from .splits import make_split, audit_given_splits, summarize_split

@dataclass
class SplitReport:
    mode: str
    n_raw: int = 0
    n_duplicates_removed: int = 0
    n_contradictory_removed: int = 0
    n_after_clean: int = 0
    leakage_audit: dict = field(default_factory=dict)
    summary: pd.DataFrame | None = None

    def __str__(self) -> str:
        lines = [
            f"split_mode          {self.mode}",
            f"rows loaded         {self.n_raw}",
            f"duplicates removed  {self.n_duplicates_removed}",
            f"contradictory rows  {self.n_contradictory_removed}",
            f"rows after cleaning {self.n_after_clean}",
        ]
        if self.leakage_audit:
            a = self.leakage_audit
            lines.append(f"test pairs in train {a.get('test_pairs_in_train', 0)} "
                         f"({a.get('test_pairs_in_train_pct', 0)}%)")
        return "\n".join(lines)

def clean_pairs(df: pd.DataFrame, deduplicate: bool = True,
                drop_contradictory: bool = True) -> tuple[pd.DataFrame, int, int]:
    n_contra = 0
    if drop_contradictory:
        per_pair = df.groupby(["drug", "target"])["label"].nunique()
        bad = set(per_pair.index[per_pair > 1])
        if bad:
            keep = ~df.set_index(["drug", "target"]).index.isin(bad)
            n_contra = int((~keep).sum())
            df = df[keep]

    n_dup = 0
    if deduplicate:
        before = len(df)
        df = df.drop_duplicates(subset=["drug", "target"], keep="first")
        n_dup = before - len(df)

    return df.reset_index(drop=True), n_dup, n_contra

def prepare_splits(cfg: Config, seed: int | None = None
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitReport]:
    seed = cfg.seed if seed is None else seed
    p = cfg.paths()
    report = SplitReport(mode=cfg.split_mode)

    if cfg.split_mode == "given":
        tr, va, te = (load_pairs(p["train"]), load_pairs(p["val"]), load_pairs(p["test"]))
        report.n_raw = len(tr) + len(va) + len(te)
        report.n_after_clean = report.n_raw
        report.leakage_audit = audit_given_splits(tr, va, te)
        report.summary = summarize_split(
            pd.concat([tr, va, te], ignore_index=True),
            range(len(tr)),
            range(len(tr), len(tr) + len(va)),
            range(len(tr) + len(va), report.n_raw))
        return tr, va, te, report

    pool = pd.concat([load_pairs(p["train"]), load_pairs(p["val"]), load_pairs(p["test"])],
                     ignore_index=True)
    report.n_raw = len(pool)
    pool, n_dup, n_contra = clean_pairs(pool, cfg.deduplicate, cfg.drop_contradictory)
    report.n_duplicates_removed, report.n_contradictory_removed = n_dup, n_contra
    report.n_after_clean = len(pool)

    tr_i, va_i, te_i = make_split(
        pool, mode=cfg.split_mode, val_frac=cfg.split_val_frac,
        test_frac=cfg.split_test_frac, seed=seed,
        cold_pair_cold_val=cfg.cold_pair_cold_val)

    report.summary = summarize_split(pool, tr_i, va_i, te_i)
    tr, va, te = (pool.loc[list(tr_i)].reset_index(drop=True),
                  pool.loc[list(va_i)].reset_index(drop=True),
                  pool.loc[list(te_i)].reset_index(drop=True))
    report.leakage_audit = verify_coldness(tr, va, te, cfg.split_mode)
    return tr, va, te, report

def verify_coldness(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                    mode: str, strict: bool = True) -> dict:
    tr_d, tr_t = set(train["drug"]), set(train["target"])
    te_d, te_t = set(test["drug"]), set(test["target"])
    tr_p = set(map(tuple, train[["drug", "target"]].to_numpy()))
    te_p = set(map(tuple, test[["drug", "target"]].to_numpy()))

    audit = {
        "mode": mode,
        "test_pairs_in_train": len(te_p & tr_p),
        "test_pairs_in_train_pct": round(100.0 * len(te_p & tr_p) / max(len(te_p), 1), 2),
        "test_drugs_seen_pct": round(100.0 * len(te_d & tr_d) / max(len(te_d), 1), 2),
        "test_targets_seen_pct": round(100.0 * len(te_t & tr_t) / max(len(te_t), 1), 2),
    }

    violations = []
    if audit["test_pairs_in_train"]:
        violations.append(f"{audit['test_pairs_in_train']} test pairs appear in train")
    if mode in ("cold_drug", "cold_pair") and (te_d & tr_d):
        violations.append(f"{len(te_d & tr_d)} test drugs appear in train")
    if mode in ("cold_target", "cold_pair") and (te_t & tr_t):
        violations.append(f"{len(te_t & tr_t)} test targets appear in train")

    audit["ok"] = not violations
    if violations:
        msg = (f"\n{'!' * 74}\nCOLD SPLIT VIOLATION ({mode}): " + "; ".join(violations) +
               f"\nThis split does not measure what its name claims.\n{'!' * 74}\n")
        if strict:
            raise ValueError(msg)
        print(msg)
    return audit
