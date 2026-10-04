from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pharmunit.config import Config
from pharmunit.data import prepare_splits

PSEUDO = 5.0


def _entity_rate(train, col, prior):
    g = train.groupby(col)["label"].agg(["sum", "size"])
    return ((g["sum"] + prior * PSEUDO) / (g["size"] + PSEUDO)).to_dict()


def floors_for_split(train, test) -> dict:
    prior = float(train["label"].mean())
    y = test["label"].to_numpy()
    d = test["drug"].map(_entity_rate(train, "drug", prior)).fillna(prior).to_numpy()
    p = test["target"].map(_entity_rate(train, "target", prior)).fillna(prior).to_numpy()

    out = {"pool_positive_rate": round(prior, 4), "n_test": int(len(test))}
    if len(set(y)) < 2:
        out["degenerate"] = "test fold has a single class; AUROC undefined"
        return out
    for name, s in (("drug_prior", d), ("protein_prior", p), ("product_prior", d * p)):
        out[name] = {"auroc": round(float(roc_auc_score(y, s)), 4),
                     "auprc": round(float(average_precision_score(y, s)), 4)}

    aurocs = [out[k]["auroc"] for k in ("drug_prior", "protein_prior", "product_prior")]
    out["floor"] = max(max(a, round(1.0 - a, 4)) for a in aurocs)
    out["floor_positive_only"] = max(aurocs)
    out["anti_predictive"] = [k for k in ("drug_prior", "protein_prior", "product_prior")
                              if out[k]["auroc"] < 0.5]
    out["drug_collapsed"] = bool(np.allclose(d, d[0]))
    out["protein_collapsed"] = bool(np.allclose(p, p[0]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-seed trivial floors")
    ap.add_argument("--dataset", default="human",
                    choices=["human", "biosnap", "bindingdb", "celegans"])
    ap.add_argument("--protocol", default="random", choices=["random", "cluster", "cold"])
    ap.add_argument("--split-mode", default="cold_pair",
                    choices=["given", "random", "cold_drug", "cold_target", "cold_pair"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out-dir", default="runs/floors")
    a = ap.parse_args()

    root = Path("datasets") / a.dataset / a.protocol
    missing = [f for f in ("train.csv", "val.csv", "test.csv") if not (root / f).exists()]
    if missing:
        ap.error(f"{root} is missing {missing}")

    per_seed, floors = {}, []
    print(f"{'seed':>5}{'n_test':>8}{'drug':>9}{'protein':>9}{'product':>9}{'FLOOR':>9}")
    print("-" * 49)
    for seed in a.seeds:
        cfg = Config(dataset=a.dataset, protocol=a.protocol,
                     train_file=str(root / "train.csv"), val_file=str(root / "val.csv"),
                     test_file=str(root / "test.csv"), scorer_protein="esm",
                     p2rank_dir="", split_mode=a.split_mode, seed=seed)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            train, _val, test, report = prepare_splits(cfg, seed=seed)
        r = floors_for_split(train, test)
        r["split_warnings"] = buf.getvalue().strip() or None
        per_seed[seed] = r
        if "floor" in r:
            floors.append(r["floor"])
            print(f"{seed:>5}{r['n_test']:>8}{r['drug_prior']['auroc']:>9.4f}"
                  f"{r['protein_prior']['auroc']:>9.4f}{r['product_prior']['auroc']:>9.4f}"
                  f"{r['floor']:>9.4f}")
        else:
            print(f"{seed:>5}{r['n_test']:>8}   {r['degenerate']}")

    print("-" * 49)
    if floors:
        print(f"{'mean':>5}{'':>8}{'':>9}{'':>9}{'':>9}{np.mean(floors):>9.4f}"
              f"   (sd {np.std(floors, ddof=1) if len(floors) > 1 else 0:.4f})")
        c = per_seed[a.seeds[0]]
        if c.get("drug_collapsed") or c.get("protein_collapsed"):
            which = [n for n in ("drug", "protein") if c.get(f"{n}_collapsed")]
            print(f"\ncollapsed to chance: {', '.join(which)} prior "
                  f"(every test entity unseen -- score is constant)")

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    fp = out / f"{a.dataset}_{a.protocol}_{a.split_mode}.json"
    fp.write_text(json.dumps(
        {"dataset": a.dataset, "protocol": a.protocol, "split_mode": a.split_mode,
         "pseudo_count": PSEUDO, "per_seed": per_seed,
         "mean_floor": round(float(np.mean(floors)), 4) if floors else None},
        indent=2), encoding="utf-8")
    print(f"\nwrote {fp}")


if __name__ == "__main__":
    main()
