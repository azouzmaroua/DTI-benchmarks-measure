from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pharmunit.config import Config
from pharmunit.data import prepare_splits

FOLDS = ("train", "val", "test")
RAW_STEMS = {"human": "Human", "celegans": "Celegans"}

def _paths(dataset: str, source: str) -> dict[str, Path] | None:
    if source == "prepared":
        root = ROOT / "datasets" / dataset / "random"
        p = {k: root / f"{k}.csv" for k in FOLDS}
    else:
        stem = RAW_STEMS.get(dataset)
        if stem is None:
            return None
        p = {k: ROOT / "data_raw" / f"{stem}_{k}.txt" for k in FOLDS}
    return p if all(v.exists() for v in p.values()) else None

def audit(dataset: str, source: str) -> dict | None:
    p = _paths(dataset, source)
    if p is None:
        return None
    cfg = Config(dataset=dataset, protocol="random",
                 train_file=str(p["train"]), val_file=str(p["val"]),
                 test_file=str(p["test"]), scorer_protein="esm",
                 p2rank_dir="", split_mode="given", seed=0)
    with contextlib.redirect_stdout(io.StringIO()):
        train, _val, test, report = prepare_splits(cfg, seed=0)

    pos = train[train["label"] == 1]
    train_drugs, train_targets = set(pos["drug"]), set(pos["target"])

    scoreable = np.array([(d in train_drugs) and (t in train_targets)
                          for d, t in zip(test["drug"], test["target"])])
    y = test["label"].to_numpy()

    n_sc = int(scoreable.sum())
    n_pos_sc = int(y[scoreable].sum()) if n_sc else 0
    return {
        "dataset": dataset,
        "source": source,
        "files": {k: str(v) for k, v in p.items()},
        "n_train": int(len(train)), "n_test": int(len(test)),
        "n_train_positive_edges": int(len(pos)),
        "n_scoreable": n_sc,
        "frac_scoreable": round(n_sc / max(len(test), 1), 4),
        "n_positive_in_scoreable": n_pos_sc,
        "positive_rate_in_scoreable": round(n_pos_sc / n_sc, 4) if n_sc else None,
        "positive_rate_outside_scoreable":
            round(float(y[~scoreable].mean()), 4) if (~scoreable).any() else None,
        "test_positive_rate_overall": round(float(y.mean()), 4),
        "auroc_evaluable_on_scoreable": bool(n_sc and 0 < n_pos_sc < n_sc),
        "supplied_split_leakage": report.leakage_audit,
    }

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+",
                    default=["human", "celegans", "biosnap", "bindingdb"])
    ap.add_argument("--source", choices=["raw", "prepared", "both"], default="both")
    ap.add_argument("--out", default="runs/published_split_audit.json")
    a = ap.parse_args()

    sources = ["raw", "prepared"] if a.source == "both" else [a.source]
    rows = []
    for source in sources:
        print("=" * 110)
        label = ("data_raw/*.txt -- the benchmark authors' distributed files"
                 if source == "raw"
                 else "datasets/<ds>/random/*.csv -- this project's prepared files")
        print(f"Edge-prior domain, split_mode='given'   SOURCE: {label}")
        print("=" * 110)
        print(f"{'dataset':<12}{'n_test':>8}{'scoreable':>11}{'%':>8}"
              f"{'pos in scoreable':>19}{'pos rate':>10}{'AUROC?':>9}   pair leakage in source")
        for ds in a.datasets:
            r = audit(ds, source)
            if r is None:
                print(f"{ds:<12} not available from this source")
                continue
            rows.append(r)
            lk = r["supplied_split_leakage"] or {}
            leak = (f"{lk.get('test_pairs_in_train', '?')} of {r['n_test']} test pairs "
                    f"({lk.get('test_pairs_in_train_pct', '?')}%), "
                    f"{lk.get('contradictory_label_pairs', '?')} contradictory")
            pr = r["positive_rate_in_scoreable"]
            print(f"{ds:<12}{r['n_test']:>8}{r['n_scoreable']:>11}"
                  f"{r['frac_scoreable']*100:>7.1f}%"
                  f"{str(r['n_positive_in_scoreable']) + '/' + str(r['n_scoreable']):>19}"
                  f"{(pr if pr is not None else float('nan')):>10.4f}"
                  f"{('yes' if r['auroc_evaluable_on_scoreable'] else 'NO'):>9}   {leak}")
        print()

    print("Reading: where the positive rate inside the scoreable subset is 1.000, AUROC is")
    print("undefined there -- the region the edge prior can score is single-class. Compare it")
    print("with the positive rate OUTSIDE that subset; opposite signs mean the confound")
    print("reverses direction on that benchmark.")
    for r in rows:
        print(f"  {r['dataset']:<10} [{r['source']:<8}] inside {r['positive_rate_in_scoreable']}"
              f"   outside {r['positive_rate_outside_scoreable']}"
              f"   overall {r['test_positive_rate_overall']}")

    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
