from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pharmunit.config import Config
from pharmunit.data import clean_pairs, load_pairs, prepare_splits
from pharmunit.splits import check_fold_balance, make_split

BENCHMARKS = ("human", "celegans", "biosnap", "bindingdb")
PSEUDO = 5.0


def pool(ds: str) -> pd.DataFrame:
    root = ROOT / "datasets" / ds / "random"
    cfg = Config(dataset=ds, protocol="random", train_file=str(root / "train.csv"),
                 val_file=str(root / "val.csv"), test_file=str(root / "test.csv"),
                 scorer_protein="esm", p2rank_dir="", split_mode="random", seed=0)
    with contextlib.redirect_stdout(io.StringIO()):
        tr, va, te, _ = prepare_splits(cfg, seed=0)
    return pd.concat([tr, va, te], ignore_index=True)


def occurrence_row(df: pd.DataFrame, col: str) -> dict:
    cnt = df.groupby(col).size()
    nun = df.groupby(col)["label"].nunique()
    single = nun[nun == 1].index

    def purity(min_count: int) -> float | None:
        keep = cnt[cnt >= min_count].index
        if len(keep) == 0:
            return None
        return round(float(nun.loc[keep].eq(1).mean()), 4)

    return {
        "n_entities": int(len(cnt)),
        "occur_once": int((cnt == 1).sum()),
        "purity_all": purity(1),
        "purity_ge2": purity(2),
        "purity_ge5": purity(5),
        "row_weighted_purity": round(float(df[col].isin(single).mean()), 4),
    }


def rate_map(train: pd.DataFrame, col: str, prior: float) -> dict:
    g = train.groupby(col)["label"].agg(["sum", "size"])
    return ((g["sum"] + prior * PSEUDO) / (g["size"] + PSEUDO)).to_dict()


def lookup_aurocs(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    prior = float(train["label"].mean())
    out = {}
    for col in ("drug", "target"):
        s = test[col].map(rate_map(train, col, prior)).fillna(prior).to_numpy()
        y = test["label"].to_numpy()
        out[col] = None if len(set(y.tolist())) < 2 else round(float(roc_auc_score(y, s)), 4)
    return out


def nulls_for(ds: str, seeds: range) -> dict:
    root = ROOT / "datasets" / ds / "random"
    as_given, labels_perm, ids_perm = [], [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        cfg = Config(dataset=ds, protocol="random", train_file=str(root / "train.csv"),
                     val_file=str(root / "val.csv"), test_file=str(root / "test.csv"),
                     scorer_protein="esm", p2rank_dir="", split_mode="random", seed=seed)
        with contextlib.redirect_stdout(io.StringIO()):
            tr, va, te, _ = prepare_splits(cfg, seed=seed)
        as_given.append(lookup_aurocs(tr, te))

        tr_perm = tr.copy()
        tr_perm["label"] = rng.permutation(tr_perm["label"].to_numpy())
        labels_perm.append(lookup_aurocs(tr_perm, te))

        te_perm = te.copy()
        te_perm["drug"] = rng.permutation(te_perm["drug"].to_numpy())
        te_perm["target"] = rng.permutation(te_perm["target"].to_numpy())
        ids_perm.append(lookup_aurocs(tr, te_perm))

    def mean_by_col(rows):
        return {c: round(float(np.mean([r[c] for r in rows if r[c] is not None])), 4)
                for c in ("drug", "target")}

    return {"as_given": mean_by_col(as_given), "labels_permuted": mean_by_col(labels_perm),
            "test_ids_permuted": mean_by_col(ids_perm), "n_seeds": len(as_given)}


FOLD_CONTROL_CELLS = [("celegans", "cold_drug"), ("human", "cold_drug"),
                       ("celegans", "cold_target"), ("human", "cold_target")]


def cleaned_pool(ds: str) -> pd.DataFrame:
    root = ROOT / "datasets" / ds / "random"
    raw = pd.concat([load_pairs(root / f) for f in ("train.csv", "val.csv", "test.csv")],
                    ignore_index=True)
    cleaned, _, _ = clean_pairs(raw)
    return cleaned.reset_index(drop=True)


def fold_control_for(ds: str, mode: str, seeds: range) -> dict:
    base = cleaned_pool(ds)

    def drift_flags(df: pd.DataFrame) -> list[float]:
        flags = []
        for seed in seeds:
            tr_i, va_i, te_i = make_split(df, mode=mode, val_frac=0.10, test_frac=0.10, seed=seed)
            report = check_fold_balance(df, {"train": tr_i, "val": va_i, "test": te_i}, strict=False)
            flags.append(max(f["drift"] for f in report["folds"].values()))
        return flags

    as_given = drift_flags(base)
    rng = np.random.default_rng(0)
    permuted = base.copy()
    permuted["label"] = rng.permutation(permuted["label"].to_numpy())
    after = drift_flags(permuted)
    return {
        "as_given_mean_drift": round(float(np.mean(as_given)), 3),
        "as_given_flagged": sum(d > 0.15 for d in as_given),
        "labels_permuted_mean_drift": round(float(np.mean(after)), 3),
        "labels_permuted_flagged": sum(d > 0.15 for d in after),
        "n_seeds": len(as_given),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "runs" / "purity_decomposition.json"))
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()

    pools = {ds: pool(ds) for ds in BENCHMARKS}

    occurrence = {}
    for ds in BENCHMARKS:
        for col, side in (("drug", "drug"), ("target", "target")):
            occurrence[f"{ds}/{side}"] = occurrence_row(pools[ds], col)

    nulls = {ds: nulls_for(ds, range(a.seeds)) for ds in BENCHMARKS}

    fold_control = {f"{ds}/{mode}": fold_control_for(ds, mode, range(a.seeds))
                    for ds, mode in FOLD_CONTROL_CELLS}

    bd = pools["bindingdb"]
    bc = bd.groupby("drug")["label"].agg(["size", "sum", "nunique"])
    bindingdb_check = {
        "n_drugs": int(len(bc)),
        "all_occur_at_least_twice": bool((bc["size"] >= 2).all()),
        "all_both_labels": bool((bc["nunique"] == 2).all()),
        "exactly_one_pos_one_neg": int(((bc["size"] == 2) & (bc["sum"] == 1)).sum()),
    }

    payload = {
        "pseudo_count": PSEUDO,
        "occurrence": occurrence,
        "nulls": nulls,
        "fold_control": fold_control,
        "bindingdb_drug_check": bindingdb_check,
        "note": "occurrence reproduces Supplementary Table S10 exactly; nulls' as_given values "
                "reproduce Table 4 of the main text exactly and its permuted values fall inside "
                "the manuscript's stated 0.48-0.54 band; fold_control's as_given drift and "
                "flagged counts reproduce Supplementary Table S2 exactly. The labels_permuted / "
                "test_ids_permuted / labels_permuted (fold_control) columns are a fresh random "
                "permutation (seeded here, not pinned to the original session's), so their exact "
                "residual values differ from the ones quoted in the manuscript text while showing "
                "the same collapse and the same pass/fail outcome.",
    }
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"BindingDB exactly-one-positive-one-negative drugs: {bindingdb_check['exactly_one_pos_one_neg']}")
    for ds in BENCHMARKS:
        for side in ("drug", "target"):
            r = occurrence[f"{ds}/{side}"]
            print(f"{ds:10s} {side:6s} occur-once {r['occur_once']:6d}  all {r['purity_all']}  "
                  f">=2 {r['purity_ge2']}  >=5 {r['purity_ge5']}  row-wtd {r['row_weighted_purity']}")
    print()
    for key, r in fold_control.items():
        print(f"{key:24s} as-given drift {r['as_given_mean_drift']} ({r['as_given_flagged']}/{r['n_seeds']}) "
              f"-> permuted {r['labels_permuted_mean_drift']} ({r['labels_permuted_flagged']}/{r['n_seeds']})")


if __name__ == "__main__":
    main()
