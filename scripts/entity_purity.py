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
from pharmunit.data import prepare_splits

PSEUDO = 5.0
FAMILIES = ("drug", "target", "product")

VAL_FRAC: float | None = None
TEST_FRAC: float | None = None


def _split_kw() -> dict:
    kw = {}
    if VAL_FRAC is not None:
        kw["split_val_frac"] = VAL_FRAC
    if TEST_FRAC is not None:
        kw["split_test_frac"] = TEST_FRAC
    return kw


def _rate_map(train: pd.DataFrame, col: str, prior: float) -> dict:
    g = train.groupby(col)["label"].agg(["sum", "size"])
    return ((g["sum"] + prior * PSEUDO) / (g["size"] + PSEUDO)).to_dict()


def _scores(train: pd.DataFrame, frame: pd.DataFrame, prior: float) -> dict:
    d = frame["drug"].map(_rate_map(train, "drug", prior)).fillna(prior).to_numpy()
    t = frame["target"].map(_rate_map(train, "target", prior)).fillna(prior).to_numpy()
    return {"drug": d, "target": t, "product": d * t}


def _auroc(y: np.ndarray, s: np.ndarray) -> float | None:
    if len(set(y.tolist())) < 2 or np.allclose(s, s[0]):
        return None
    return float(roc_auc_score(y, s))


def entity_class_purity(pool: pd.DataFrame) -> dict:
    out = {"n_rows": int(len(pool)), "pool_positive_rate": round(float(pool["label"].mean()), 4)}
    for col, name in (("drug", "drugs"), ("target", "targets")):
        nunique = pool.groupby(col)["label"].nunique()
        out[f"{name}_single_class"] = round(float((nunique == 1).mean()), 4)
        out[f"n_{name}"] = int(len(nunique))
    g = pool.groupby("drug")["label"].agg(["nunique", "max"])
    out["drugs_positive_only"] = int(((g["nunique"] == 1) & (g["max"] == 1)).sum())
    out["drugs_negative_only"] = int(((g["nunique"] == 1) & (g["max"] == 0)).sum())
    out["drugs_both"] = int((g["nunique"] > 1).sum())
    return out


def cell(dataset: str, split_mode: str, seed: int) -> dict:
    root = ROOT / "datasets" / dataset / "random"
    cfg = Config(dataset=dataset, protocol="random",
                 train_file=str(root / "train.csv"), val_file=str(root / "val.csv"),
                 test_file=str(root / "test.csv"), scorer_protein="esm",
                 p2rank_dir="", split_mode=split_mode, seed=seed, **_split_kw())
    with contextlib.redirect_stdout(io.StringIO()):
        train, val, test, _report = prepare_splits(cfg, seed=seed)

    prior = float(train["label"].mean())
    y_te = test["label"].to_numpy()
    y_va = val["label"].to_numpy()
    s_te = _scores(train, test, prior)
    s_va = _scores(train, val, prior)

    out = {"seed": seed, "n_test": int(len(test)), "n_val": int(len(val)),
           "train_positive_rate": round(prior, 4),
           "test_positive_rate": round(float(test["label"].mean()), 4), "raw": {}}

    for fam in FAMILIES:
        a = _auroc(y_te, s_te[fam])
        out["raw"][fam] = None if a is None else round(a, 4)

    sf = [max(a, 1.0 - a) for a in
          (_auroc(y_te, s_te[fam]) for fam in FAMILIES) if a is not None]
    out["sign_free"] = round(max(sf), 4) if sf else None

    best, choice = None, None
    for fam in FAMILIES:
        a_va = _auroc(y_va, s_va[fam])
        if a_va is None:
            continue
        sign = 1.0 if a_va >= 0.5 else -1.0
        score_va = max(a_va, 1.0 - a_va)
        if best is None or score_va > best:
            best, choice = score_va, (fam, sign)
    if choice is None:
        out["val_fixed"] = 0.5
        out["val_fixed_choice"] = "none (all families degenerate on val)"
    else:
        fam, sign = choice
        a_te = _auroc(y_te, sign * s_te[fam])
        out["val_fixed"] = 0.5 if a_te is None else round(a_te, 4)
        out["val_fixed_choice"] = f"{fam}{'+' if sign > 0 else '-'}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+",
                    default=["human", "celegans", "biosnap", "bindingdb"])
    ap.add_argument("--split-modes", nargs="+",
                    default=["random", "cold_drug", "cold_target", "cold_pair"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="runs/entity_purity.json")
    ap.add_argument("--pseudo-count", type=float, default=5.0,
                    help="Smoothing weight in rate(e) = (positives + k*pi)/(count + k). "
                         "The manuscript reports k=5 and a sensitivity check at k=1 and k=20; "
                         "write each run to its own --out path so they do not overwrite.")
    ap.add_argument("--val-frac", type=float, default=None,
                    help="Validation fraction. Set 0.1 together with --test-frac 0.2 to "
                         "reproduce the 7:1:2 split used by papers following the DrugBAN "
                         "setting, so the reference is comparable to their reported AUROCs.")
    ap.add_argument("--test-frac", type=float, default=None,
                    help="Test fraction; see --val-frac.")
    a = ap.parse_args()

    global PSEUDO, VAL_FRAC, TEST_FRAC
    PSEUDO = a.pseudo_count
    VAL_FRAC, TEST_FRAC = a.val_frac, a.test_frac

    payload: dict = {"pseudo_count": PSEUDO, "seeds": a.seeds, "purity": {}, "cells": {}}

    print("=" * 96)
    print("TABLE 2 -- entity-class purity, pooled over the CLEANED dataset (datasets/)")
    print("=" * 96)
    print(f"{'dataset':<12}{'rows':>8}{'pos rate':>10}{'drugs':>8}{'single':>9}"
          f"{'targets':>9}{'single':>9}   pos-only/neg-only/both drugs")
    for ds in a.datasets:
        root = ROOT / "datasets" / ds / "random"
        if not (root / "train.csv").exists():
            print(f"{ds:<12} SKIP -- {root} missing")
            continue
        cfg = Config(dataset=ds, protocol="random",
                     train_file=str(root / "train.csv"), val_file=str(root / "val.csv"),
                     test_file=str(root / "test.csv"), scorer_protein="esm",
                     p2rank_dir="", split_mode="random", seed=0)
        with contextlib.redirect_stdout(io.StringIO()):
            tr, va, te, _ = prepare_splits(cfg, seed=0)
        pool = pd.concat([tr, va, te], ignore_index=True)
        p = entity_class_purity(pool)
        payload["purity"][ds] = p
        print(f"{ds:<12}{p['n_rows']:>8}{p['pool_positive_rate']:>10.4f}"
              f"{p['n_drugs']:>8}{p['drugs_single_class']*100:>8.1f}%"
              f"{p['n_targets']:>9}{p['targets_single_class']*100:>8.1f}%"
              f"   {p['drugs_positive_only']}/{p['drugs_negative_only']}/{p['drugs_both']}")

    print()
    print("=" * 96)
    print("TABLE 3 / 4 -- identity lookups and references, per cell, mean over seeds")
    print("=" * 96)
    print(f"{'cell':<26}{'n':>3}{'drug':>9}{'target':>9}{'product':>9}"
          f"{'sign_free':>11}{'val_fixed':>11}{'|diff|':>9}")
    for ds in a.datasets:
        if ds not in payload["purity"]:
            continue
        for sm in a.split_modes:
            rows = []
            for s in a.seeds:
                try:
                    rows.append(cell(ds, sm, s))
                except Exception as exc:
                    print(f"{ds+'/'+sm:<26} seed {s} FAILED: {type(exc).__name__}: {exc}")
            if not rows:
                continue
            payload["cells"][f"{ds}/{sm}"] = rows

            def col(key, fam=None):
                vals = [(r["raw"][fam] if fam else r[key]) for r in rows]
                vals = [v for v in vals if v is not None]
                return float(np.mean(vals)) if vals else float("nan")

            sfv = np.array([r["sign_free"] for r in rows if r["sign_free"] is not None])
            vfv = np.array([r["val_fixed"] for r in rows if r["val_fixed"] is not None])
            diff = float(np.max(np.abs(sfv - vfv))) if len(sfv) == len(vfv) and len(sfv) else float("nan")
            flag = "  <-- CONSTRUCTIONS DIFFER" if diff > 0.0005 else ""
            print(f"{ds+'/'+sm:<26}{len(rows):>3}{col(None,'drug'):>9.4f}"
                  f"{col(None,'target'):>9.4f}{col(None,'product'):>9.4f}"
                  f"{col('sign_free'):>11.4f}{col('val_fixed'):>11.4f}{diff:>9.6f}{flag}")
            choices = sorted({r.get("val_fixed_choice", "?") for r in rows})
            print(f"{'':<26}   val-selected family/orientation across seeds: {choices}")

    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
