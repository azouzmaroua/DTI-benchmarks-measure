from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pharmunit.config import Config
from pharmunit.data import prepare_splits

import entity_purity as EP

FOLDS = ("train", "val", "test")
RAW_STEMS = {"human": "Human", "celegans": "Celegans"}

VARIANT = "random"


def _paths(dataset: str, source: str) -> dict[str, Path] | None:
    if source == "prepared":
        root = ROOT / "datasets" / dataset / VARIANT
        if VARIANT == "cluster":
            p = {"train": root / "source_train.csv",
                 "val": root / "target_train.csv",
                 "test": root / "target_test.csv"}
        else:
            p = {k: root / f"{k}.csv" for k in FOLDS}
    else:
        stem = RAW_STEMS.get(dataset)
        if stem is None:
            return None
        p = {k: ROOT / "data_raw" / f"{stem}_{k}.txt" for k in FOLDS}
    return p if all(v.exists() for v in p.values()) else None


def evaluate(dataset: str, source: str) -> dict | None:
    p = _paths(dataset, source)
    if p is None:
        return None
    cfg = Config(dataset=dataset, protocol="random",
                 train_file=str(p["train"]), val_file=str(p["val"]),
                 test_file=str(p["test"]), scorer_protein="esm",
                 p2rank_dir="", split_mode="given", seed=0)
    with contextlib.redirect_stdout(io.StringIO()):
        train, val, test, report = prepare_splits(cfg, seed=0)

    prior = float(train["label"].mean())
    y_te, y_va = test["label"].to_numpy(), val["label"].to_numpy()
    s_te, s_va = EP._scores(train, test, prior), EP._scores(train, val, prior)

    raw = {}
    for fam in EP.FAMILIES:
        a = EP._auroc(y_te, s_te[fam])
        raw[fam] = None if a is None else round(a, 4)

    best, choice = None, None
    for fam in EP.FAMILIES:
        a_va = EP._auroc(y_va, s_va[fam])
        if a_va is None:
            continue
        score = max(a_va, 1.0 - a_va)
        if best is None or score > best:
            best, choice = score, (fam, 1.0 if a_va >= 0.5 else -1.0)
    if choice is None:
        val_fixed, pick = 0.5, "none"
    else:
        fam, sign = choice
        a_te = EP._auroc(y_te, sign * s_te[fam])
        val_fixed = 0.5 if a_te is None else round(a_te, 4)
        pick = f"{fam}{'+' if sign > 0 else '-'}"

    sf = [max(a, 1.0 - a) for a in (EP._auroc(y_te, s_te[f]) for f in EP.FAMILIES)
          if a is not None]
    sign_free = round(max(sf), 4) if sf else None

    pool = pd.concat([train, val, test], ignore_index=True)
    pur = EP.entity_class_purity(pool)
    lk = report.leakage_audit or {}

    return {
        "dataset": dataset, "source": source,
        "variant": VARIANT if source == "prepared" else "distributed",
        "files": {k: str(v) for k, v in p.items()},
        "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
        "test_positive_rate": round(float(y_te.mean()), 4),
        "drug_lookup": raw["drug"], "target_lookup": raw["target"],
        "product_lookup": raw["product"],
        "reference_validation_fixed": val_fixed, "selected_family": pick,
        "reference_sign_free_upper_bound": sign_free,
        "drugs_single_class": pur["drugs_single_class"],
        "targets_single_class": pur["targets_single_class"],
        "test_pairs_in_train_pct": lk.get("test_pairs_in_train_pct"),
        "contradictory_label_pairs": lk.get("contradictory_label_pairs"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+",
                    default=["human", "celegans", "biosnap", "bindingdb"])
    ap.add_argument("--source", choices=["raw", "prepared", "both"], default="both")
    ap.add_argument("--variant", default="random",
                    help="Prepared split directory to read: 'random', or 'cold' for "
                         "DrugBAN's own cold-pair split (datasets/human/cold/).")
    ap.add_argument("--pseudo-count", type=float, default=5.0)
    ap.add_argument("--out", default="runs/reference_on_published_splits.json")
    a = ap.parse_args()

    global VARIANT
    VARIANT = a.variant
    EP.PSEUDO = a.pseudo_count
    sources = ["raw", "prepared"] if a.source == "both" else [a.source]
    rows = []

    print("=" * 112)
    print("Entity-frequency reference on splits USED AS SUPPLIED (no pooling, cleaning or re-splitting).")
    print("Directly comparable to any AUROC a paper reports on the same files.")
    print(f"Smoothing weight k = {EP.PSEUDO}")
    print("=" * 112)
    print(f"{'dataset':<11}{'source':<10}{'n_test':>7}{'drug':>8}{'target':>8}{'product':>9}"
          f"{'REFERENCE':>11}{'family':>9}{'purity d/t':>14}{'leak%':>7}")
    print("-" * 112)
    for src in sources:
        for ds in a.datasets:
            r = evaluate(ds, src)
            if r is None:
                continue
            rows.append(r)
            fmt = lambda v: "  n/a" if v is None else f"{v:.4f}"
            print(f"{ds:<11}{src:<10}{r['n_test']:>7}{fmt(r['drug_lookup']):>8}"
                  f"{fmt(r['target_lookup']):>8}{fmt(r['product_lookup']):>9}"
                  f"{r['reference_validation_fixed']:>11.4f}{r['selected_family']:>9}"
                  f"{r['drugs_single_class']*100:>8.1f}/{r['targets_single_class']*100:<5.1f}"
                  f"{(r['test_pairs_in_train_pct'] if r['test_pairs_in_train_pct'] is not None else 0):>7.1f}")

    print()
    print("HOW TO USE THIS TABLE")
    print("  Put the REFERENCE column beside published AUROCs reported on the same split files.")
    print("  A published result at or below the reference did not exceed a label-frequency")
    print("  lookup on its own evaluation data.")
    print()
    print("  The leak% column is the fraction of test pairs appearing verbatim in train in that")
    print("  source. Where it is non-zero, BOTH the reference and any model trained on those")
    print("  files are inflated by direct pair memorisation. The comparison stays fair; neither")
    print("  side is a clean generalisation estimate. Say so wherever this table is used.")

    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
