from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pharmunit.config import Config
from pharmunit.data import prepare_splits

FOLDS = ("train", "val", "test")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["human", "celegans", "biosnap", "bindingdb"])
    ap.add_argument("--split-mode", required=True,
                    choices=["random", "cold_drug", "cold_target", "cold_pair"])
    ap.add_argument("--seed", type=int, default=0, help="Which draw of the split to write.")
    ap.add_argument("--source", default="random",
                    help="Split directory the pooled rows are read from (default: random).")
    ap.add_argument("--out-name", default=None,
                    help="Directory name under datasets/<dataset>/ (default: the split mode).")
    a = ap.parse_args()

    src = ROOT / "datasets" / a.dataset / a.source
    files = {k: src / f"{k}.csv" for k in FOLDS}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise SystemExit("missing source files:\n  " + "\n  ".join(missing))

    cfg = Config(dataset=a.dataset, protocol="random", seed=a.seed,
                 train_file=str(files["train"]), val_file=str(files["val"]),
                 test_file=str(files["test"]), scorer_protein="esm",
                 p2rank_dir="", split_mode=a.split_mode)
    with contextlib.redirect_stdout(io.StringIO()):
        train, val, test, report = prepare_splits(cfg, seed=a.seed)

    out = ROOT / "datasets" / a.dataset / (a.out_name or a.split_mode)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in (("train", train), ("val", val), ("test", test)):
        frame.rename(columns={"drug": "SMILES", "target": "Protein", "label": "Y"})[
            ["SMILES", "Protein", "Y"]].to_csv(out / f"{name}.csv", index=False)

    pool = pd.concat([train, val, test], ignore_index=True)
    te_d, te_t = set(test["drug"]), set(test["target"])
    tr_d, tr_t = set(train["drug"]), set(train["target"])
    print(f"{a.dataset} {a.split_mode}, seed {a.seed} -> {out.relative_to(ROOT)}")
    print(f"  rows      train {len(train)}  val {len(val)}  test {len(test)}  (pool {len(pool)})")
    print(f"  positives train {int(train['label'].sum())}  val {int(val['label'].sum())}  "
          f"test {int(test['label'].sum())}")
    print(f"  test drugs seen in train   {len(te_d & tr_d)} of {len(te_d)}")
    print(f"  test targets seen in train {len(te_t & tr_t)} of {len(te_t)}")
    if report.leakage_audit:
        print(f"  leakage audit {report.leakage_audit}")
    print("\nOne draw, not the manuscript's five-seed cell: compare a model run on these files")
    print("against the reference computed on these same files, and report the seed.")


if __name__ == "__main__":
    main()
