from __future__ import annotations

import argparse
import io
import contextlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pharmunit.config import Config
from pharmunit.data import prepare_splits

PSEUDO = 5.0


def xswap_edge_prior(edges, test_pairs, n_perm, swaps_per_edge, rng):
    edge_set = set(edges)
    counts = Counter()
    test_set = set(test_pairs)
    n_swaps = int(swaps_per_edge * len(edges))

    for _ in range(n_perm):
        cur = list(edge_set)
        cur_set = set(cur)
        for _ in range(n_swaps):
            i, j = rng.integers(0, len(cur), 2)
            if i == j:
                continue
            (d1, t1), (d2, t2) = cur[i], cur[j]
            if d1 == d2 or t1 == t2:
                continue
            a, b = (d1, t2), (d2, t1)
            if a in cur_set or b in cur_set:
                continue
            cur_set.discard((d1, t1)); cur_set.discard((d2, t2))
            cur_set.add(a); cur_set.add(b)
            cur[i], cur[j] = a, b
        for p in cur_set & test_set:
            counts[p] += 1

    return {p: counts[p] / n_perm for p in test_pairs}


def entity_frequency_reference(train, test):
    prior = float(train["label"].mean())
    y = test["label"].to_numpy()
    best = 0.5
    for col in ("drug", "target"):
        g = train.groupby(col)["label"].agg(["sum", "size"])
        rate = ((g["sum"] + prior * PSEUDO) / (g["size"] + PSEUDO)).to_dict()
        s = test[col].map(rate).fillna(prior).to_numpy()
        if len(set(y)) > 1 and not np.allclose(s, s[0]):
            a = roc_auc_score(y, s)
            best = max(best, a, 1 - a)
    return best


def run_cell(dataset, split_mode, seed, n_perm, swaps_per_edge):
    root = Path("datasets") / dataset / "random"
    cfg = Config(dataset=dataset, protocol="random",
                 train_file=str(root / "train.csv"), val_file=str(root / "val.csv"),
                 test_file=str(root / "test.csv"), scorer_protein="esm",
                 p2rank_dir="", split_mode=split_mode, seed=seed)
    with contextlib.redirect_stdout(io.StringIO()):
        train, _val, test, _rep = prepare_splits(cfg, seed=seed)

    pos = train[train["label"] == 1]
    edges = list(zip(pos["drug"], pos["target"]))
    train_drugs, train_targets = set(pos["drug"]), set(pos["target"])

    test_pairs = list(zip(test["drug"], test["target"]))
    y = test["label"].to_numpy()

    scoreable = np.array([(d in train_drugs) and (t in train_targets)
                          for d, t in test_pairs])

    rng = np.random.default_rng(seed)
    prior_map = xswap_edge_prior(edges, test_pairs, n_perm, swaps_per_edge, rng)
    ep = np.array([prior_map[p] for p in test_pairs])

    out = {"seed": seed, "n_test": len(test), "n_train_edges": len(edges),
           "frac_scoreable": float(scoreable.mean()),
           "edge_prior_nonzero": float((ep > 0).mean())}

    if len(set(y)) > 1:
        if np.allclose(ep, ep[0]):
            out["edge_prior_auroc"] = 0.5
            out["edge_prior_degenerate"] = True
        else:
            a = roc_auc_score(y, ep)
            out["edge_prior_auroc"] = round(float(max(a, 1 - a)), 4)
            out["edge_prior_degenerate"] = False
        out["entity_freq_auroc"] = round(entity_frequency_reference(train, test), 4)

    ys, eps = y[scoreable], ep[scoreable]
    out["n_scoreable"] = int(scoreable.sum())
    out["positives_in_scoreable"] = int(ys.sum())
    if len(ys) and len(set(ys)) > 1 and not np.allclose(eps, eps[0]):
        a_r = roc_auc_score(ys, eps)
        out["edge_prior_auroc_restricted"] = round(float(max(a_r, 1 - a_r)), 4)
    else:
        out["edge_prior_auroc_restricted"] = None
        out["restricted_undefined_reason"] = (
            "no scoreable pair" if not len(ys) else
            "single class in scoreable subset" if len(set(ys)) <= 1 else
            "constant prior on scoreable subset")
    return out


def self_test():
    rng = np.random.default_rng(0)
    edges = [(f"d{i%20}", f"t{(i*7)%13}") for i in range(120)]
    edges = list(set(edges))
    from collections import Counter as C
    d0, t0 = C(d for d, _ in edges), C(t for _, t in edges)
    cur = set(edges); cur_l = list(cur)
    for _ in range(5000):
        i, j = rng.integers(0, len(cur_l), 2)
        if i == j: continue
        (a1, b1), (a2, b2) = cur_l[i], cur_l[j]
        if a1 == a2 or b1 == b2: continue
        if (a1, b2) in cur or (a2, b1) in cur: continue
        cur.discard((a1, b1)); cur.discard((a2, b2))
        cur.add((a1, b2)); cur.add((a2, b1))
        cur_l[i], cur_l[j] = (a1, b2), (a2, b1)
    d1, t1 = C(d for d, _ in cur), C(t for _, t in cur)
    print(f"  drug degrees preserved:   {d0 == d1}")
    print(f"  target degrees preserved: {t0 == t1}")
    print(f"  edge count preserved:     {len(cur) == len(edges)}")
    return d0 == d1 and t0 == t1 and len(cur) == len(edges)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="human",
                    choices=["human", "biosnap", "bindingdb", "celegans"])
    ap.add_argument("--split-modes", nargs="+",
                    default=["random", "cold_drug", "cold_target", "cold_pair"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--permutations", type=int, default=50)
    ap.add_argument("--swaps-per-edge", type=float, default=10.0)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", default="runs/edge_prior_comparison.json")
    a = ap.parse_args()

    if a.self_test:
        print("XSwap self-test (degree preservation):")
        raise SystemExit(0 if self_test() else 1)

    print(f"dataset={a.dataset}  permutations={a.permutations}  "
          f"swaps/edge={a.swaps_per_edge}\n")
    print(f"{'protocol':<13}{'scoreable':>11}{'edge prior':>12}{'entity freq':>13}   note")
    print("-" * 72)

    results = {}
    for sm in a.split_modes:
        rows = [run_cell(a.dataset, sm, s, a.permutations, a.swaps_per_edge)
                for s in a.seeds]
        rows = [r for r in rows if "edge_prior_auroc" in r]
        if not rows:
            print(f"{sm:<13}  all seeds degenerate (single-class test fold)")
            continue
        sc = np.mean([r["frac_scoreable"] for r in rows])
        ep = np.mean([r["edge_prior_auroc"] for r in rows])
        ef = np.mean([r["entity_freq_auroc"] for r in rows])
        deg = all(r["edge_prior_degenerate"] for r in rows)
        note = "edge prior UNDEFINED (no test pair in training graph)" if sc == 0 else (
            "edge prior degenerate (constant score)" if deg else "")
        print(f"{sm:<13}{sc*100:>10.1f}%{ep:>12.4f}{ef:>13.4f}   {note}")
        restricted = [r["edge_prior_auroc_restricted"] for r in rows
                      if r.get("edge_prior_auroc_restricted") is not None]
        results[sm] = {"per_seed": rows, "frac_scoreable": round(float(sc), 4),
                       "edge_prior_auroc": round(float(ep), 4),
                       "entity_freq_auroc": round(float(ef), 4),
                       "edge_prior_auroc_restricted": (
                           round(float(np.mean(restricted)), 4) if restricted else None),
                       "n_seeds_restricted_defined": len(restricted)}
        if restricted:
            print(f"{'':<13}{'':>11}{'restricted':>12}"
                  f"{np.mean(restricted):>13.4f}   "
                  f"on the scoreable subset, {len(restricted)} of {len(rows)} seeds defined")

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    payload[a.dataset] = {"permutations": a.permutations,
                          "swaps_per_edge": a.swaps_per_edge, "protocols": results}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
