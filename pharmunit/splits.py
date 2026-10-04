from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _stratified_pair_split(
    df: pd.DataFrame,
    val_frac: float,
    test_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parts = {"train": [], "val": [], "test": []}
    for label_value in (0, 1):
        cls = df.index[df["label"] == label_value].to_numpy()
        cls = cls[rng.permutation(len(cls))]
        n = len(cls)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        parts["test"].append(cls[:n_test])
        parts["val"].append(cls[n_test:n_test + n_val])
        parts["train"].append(cls[n_test + n_val:])

    out = []
    for name in ("train", "val", "test"):
        joined = np.concatenate(parts[name]) if parts[name] else np.array([], dtype=int)
        out.append(joined[rng.permutation(len(joined))])
    return out[0], out[1], out[2]


def _by_entity(
    df: pd.DataFrame,
    key: str,
    val_frac: float,
    test_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stats = df.groupby(key)["label"].agg(["size", "sum"])
    entities = stats.index.tolist()
    rng.shuffle(entities)

    n_rows, n_pos = len(df), float(df["label"].sum())
    targets = {
        "test": (n_rows * test_frac, n_pos * test_frac),
        "val": (n_rows * val_frac, n_pos * val_frac),
        "train": (n_rows * (1 - test_frac - val_frac),
                  n_pos * (1 - test_frac - val_frac)),
    }
    cur = {f: [0.0, 0.0] for f in targets}
    assigned: dict[str, str] = {}

    for ent in entities:
        e_rows = float(stats.at[ent, "size"])
        e_pos = float(stats.at[ent, "sum"])
        best, best_need = None, -np.inf
        for fold, (t_rows, t_pos) in targets.items():
            need = (t_rows - cur[fold][0]) / max(t_rows, 1.0)
            if t_pos > 0:
                need += (t_pos - cur[fold][1]) / max(t_pos, 1.0)
            if need > best_need:
                best, best_need = fold, need
        assigned[ent] = best
        cur[best][0] += e_rows
        cur[best][1] += e_pos

    members = {f: {e for e, g in assigned.items() if g == f} for f in targets}
    for fold in ("test", "val"):
        if not members[fold] and len(members["train"]) > 1:
            members[fold].add(members["train"].pop())
    if not members["train"]:
        raise ValueError(
            f"cold split on {key!r} left no training entities "
            f"({len(entities)} entities, val_frac={val_frac}, test_frac={test_frac}). "
            f"Too few distinct entities for an entity-disjoint split.")

    out = []
    for fold in ("train", "val", "test"):
        out.append(df.index[df[key].isin(members[fold])].to_numpy())
    return out[0], out[1], out[2]


def check_fold_balance(
    df: pd.DataFrame,
    folds: dict[str, np.ndarray],
    tolerance: float = 0.15,
    strict: bool = False,
) -> dict:
    pool = float(df["label"].mean())
    report = {"pool_positive_rate": round(pool, 4), "tolerance": tolerance, "folds": {}}
    offenders = []
    for name, idx in folds.items():
        if len(idx) == 0:
            offenders.append(f"{name} is EMPTY")
            report["folds"][name] = {"n": 0, "positive_rate": None}
            continue
        rate = float(df.loc[list(idx), "label"].mean())
        drift = abs(rate - pool)
        report["folds"][name] = {"n": int(len(idx)), "positive_rate": round(rate, 4),
                                 "drift": round(drift, 4), "within_tolerance": drift <= tolerance}
        if drift > tolerance:
            offenders.append(
                f"{name}: positive rate {rate:.4f} vs pool {pool:.4f} (drift {drift:.4f})")

    if offenders:
        msg = (
            f"\n{'!' * 74}\n"
            f"SPLIT LABEL IMBALANCE: " + "; ".join(offenders) + "\n"
            f"Validation drives early stopping AND threshold tuning, so an imbalanced\n"
            f"val fold makes test F1/accuracy uninterpretable. Change seed, loosen the\n"
            f"entity constraint, or report this fold as degenerate — do not just run it.\n"
            f"{'!' * 74}\n"
        )
        if strict:
            raise ValueError(msg)
        print(msg)
    report["ok"] = not offenders
    return report


def _cold_pair(
    df: pd.DataFrame,
    val_frac: float,
    test_frac: float,
    rng: np.random.Generator,
    cold_val: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    drugs = df["drug"].unique().tolist()
    tgts = df["target"].unique().tolist()
    rng.shuffle(drugs)
    rng.shuffle(tgts)
    n_d, n_t = len(drugs), len(tgts)

    dt = max(1, int(round(n_d * np.sqrt(test_frac))))
    tt = max(1, int(round(n_t * np.sqrt(test_frac))))
    test_d, test_t = set(drugs[:dt]), set(tgts[:tt])
    is_test = df["drug"].isin(test_d) & df["target"].isin(test_t)

    if cold_val:
        dv = max(1, int(round(n_d * np.sqrt(val_frac))))
        tv = max(1, int(round(n_t * np.sqrt(val_frac))))
        val_d, val_t = set(drugs[dt:dt + dv]), set(tgts[tt:tt + tv])
        train_d, train_t = set(drugs[dt + dv:]), set(tgts[tt + tv:])
        is_val = df["drug"].isin(val_d) & df["target"].isin(val_t) & ~is_test
        is_train = (df["drug"].isin(train_d) & df["target"].isin(train_t)
                    & ~is_test & ~is_val)
        return (df.index[is_train].to_numpy(),
                df.index[is_val].to_numpy(),
                df.index[is_test].to_numpy())

    train_d, train_t = set(drugs[dt:]), set(tgts[tt:])
    is_train_block = df["drug"].isin(train_d) & df["target"].isin(train_t)
    train_pool = df.index[is_train_block].to_numpy()

    n_val = max(1, int(round(len(train_pool) * val_frac)))
    perm = rng.permutation(len(train_pool))
    val_idx = train_pool[perm[:n_val]]
    train_idx = train_pool[perm[n_val:]]
    return train_idx, val_idx, df.index[is_test].to_numpy()


def make_split(
    df: pd.DataFrame,
    mode: str,
    val_frac: float,
    test_frac: float,
    seed: int,
    cold_pair_cold_val: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)
    if mode == "random":
        tr, va, te = _stratified_pair_split(df, val_frac, test_frac, rng)
    elif mode == "cold_drug":
        tr, va, te = _by_entity(df, "drug", val_frac, test_frac, rng)
    elif mode == "cold_target":
        tr, va, te = _by_entity(df, "target", val_frac, test_frac, rng)
    elif mode == "cold_pair":
        tr, va, te = _cold_pair(df, val_frac, test_frac, rng, cold_val=cold_pair_cold_val)
    else:
        raise ValueError(f"unknown split mode: {mode}")

    check_fold_balance(df, {"train": tr, "val": va, "test": te})
    return tr, va, te


def audit_given_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    strict: bool = False,
) -> dict:
    tr = set(map(tuple, train[["drug", "target"]].to_numpy()))
    te = list(map(tuple, test[["drug", "target"]].to_numpy()))
    va = list(map(tuple, val[["drug", "target"]].to_numpy()))

    te_leak = sum(1 for p in te if p in tr)
    va_leak = sum(1 for p in va if p in tr)

    allrows = pd.concat([train, val, test], ignore_index=True)
    per_pair = allrows.groupby(["drug", "target"])["label"].nunique()
    contradictory = int((per_pair > 1).sum())

    audit = {
        "test_pairs_in_train": te_leak,
        "test_pairs_in_train_pct": round(100.0 * te_leak / max(len(te), 1), 2),
        "val_pairs_in_train": va_leak,
        "val_pairs_in_train_pct": round(100.0 * va_leak / max(len(va), 1), 2),
        "contradictory_label_pairs": contradictory,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
    }

    if te_leak or contradictory:
        msg = (
            f"\n{'!' * 74}\n"
            f"LEAKAGE IN SUPPLIED SPLITS: {te_leak} of {len(te)} test pairs "
            f"({audit['test_pairs_in_train_pct']}%) appear verbatim in train; "
            f"{va_leak} of {len(va)} val pairs likewise. "
            f"{contradictory} pair(s) carry contradictory labels.\n"
            f"Results from split_mode='given' are NOT a generalisation measurement.\n"
            f"Use split_mode in {{random, cold_drug, cold_target, cold_pair}}, which "
            f"deduplicate before splitting.\n{'!' * 74}\n"
        )
        if strict:
            raise ValueError(msg)
        print(msg)
    return audit


def summarize_split(
    df: pd.DataFrame,
    train_idx: Iterable[int],
    val_idx: Iterable[int],
    test_idx: Iterable[int],
) -> pd.DataFrame:
    rows = []
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        sub = df.loc[list(idx)]
        rows.append({
            "split": name,
            "n_pairs": len(sub),
            "n_pos": int((sub["label"] == 1).sum()),
            "n_neg": int((sub["label"] == 0).sum()),
            "n_drugs": sub["drug"].nunique(),
            "n_targets": sub["target"].nunique(),
        })
    return pd.DataFrame(rows)
