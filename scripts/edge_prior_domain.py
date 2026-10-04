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

PROTOCOLS = ("random", "cold_drug", "cold_target", "cold_pair")


def domain(dataset: str, split_mode: str, seed: int) -> dict:
    src = ROOT / "datasets" / dataset / "random"
    cfg = Config(dataset=dataset, protocol="random",
                 train_file=str(src / "train.csv"), val_file=str(src / "val.csv"),
                 test_file=str(src / "test.csv"), scorer_protein="esm",
                 p2rank_dir="", split_mode=split_mode, seed=seed)
    with contextlib.redirect_stdout(io.StringIO()):
        train, _val, test, _rep = prepare_splits(cfg, seed=seed)

    pos = train[train["label"] == 1]
    drugs, targets = set(pos["drug"]), set(pos["target"])
    sc = np.array([(d in drugs) and (t in targets) for d, t in zip(test["drug"], test["target"])])
    y = test["label"].to_numpy()
    n_sc = int(sc.sum())
    n_pos_sc = int(y[sc].sum()) if n_sc else 0
    return {
        "dataset": dataset, "protocol": split_mode, "seed": seed,
        "n_test": int(len(test)), "n_train_positive_edges": int(len(pos)),
        "n_scoreable": n_sc,
        "frac_scoreable": n_sc / max(len(test), 1),
        "n_positive_in_scoreable": n_pos_sc,
        "positive_rate_in_scoreable": (n_pos_sc / n_sc) if n_sc else None,
        "positive_rate_outside_scoreable": float(y[~sc].mean()) if (~sc).any() else None,
        "test_positive_rate": float(y.mean()),
        "evaluable_on_scoreable": bool(n_sc and 0 < n_pos_sc < n_sc),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["human", "celegans", "biosnap", "bindingdb"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default=str(ROOT / "runs" / "edge_prior_domain.json"))
    a = ap.parse_args()

    rows, summary = [], {}
    print(f"{'dataset':<10}{'protocol':<13}{'seed':>5}{'n_test':>8}{'scoreable':>11}"
          f"{'frac':>8}{'pos in sc':>12}{'rate in':>9}{'rate out':>10}")
    for ds in a.datasets:
        for pr in PROTOCOLS:
            cell = []
            for s in a.seeds:
                r = domain(ds, pr, s)
                rows.append(r)
                cell.append(r)
                rin = r["positive_rate_in_scoreable"]
                rout = r["positive_rate_outside_scoreable"]
                print(f"{ds:<10}{pr:<13}{s:>5}{r['n_test']:>8}{r['n_scoreable']:>11}"
                      f"{r['frac_scoreable']:>8.4f}"
                      f"{str(r['n_positive_in_scoreable']) + '/' + str(r['n_scoreable']):>12}"
                      f"{(rin if rin is not None else float('nan')):>9.4f}"
                      f"{(rout if rout is not None else float('nan')):>10.4f}")
            summary[f"{ds}/{pr}"] = {
                "seeds": a.seeds,
                "mean_frac_scoreable": float(np.mean([c["frac_scoreable"] for c in cell])),
                "seed0": next((c for c in cell if c["seed"] == 0), None),
            }

    out = Path(a.out)
    out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
