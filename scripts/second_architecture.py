from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pharmunit.config import Config
from pharmunit.data import prepare_splits
from pharmunit.protein_features import sequence_hash

RUNS = ROOT / "runs"
ESM_CACHE = ROOT / "cache" / "esm"
ECFP_BITS, ECFP_RADIUS = 1024, 2
AA = "ACDEFGHIKLMNPQRSTVWY"

CONTRASTS = [
    ("BioSNAP random vs cold-drug", "biosnap", "random", "cold_drug"),
    ("C. elegans random vs cold-target", "celegans", "random", "cold_target"),
    ("Human cold-drug vs cold-target", "human", "cold_drug", "cold_target"),
]


def ecfp_table(smiles: list[str]) -> dict[str, np.ndarray]:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=ECFP_RADIUS, fpSize=ECFP_BITS)
    out: dict[str, np.ndarray] = {}
    for s in set(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            out[s] = np.asarray(gen.GetFingerprintAsNumPy(mol), dtype=np.float32)
    return out


def composition_table(seqs: list[str]) -> dict[str, np.ndarray]:
    idx = {a: i for i, a in enumerate(AA)}
    out: dict[str, np.ndarray] = {}
    for s in set(seqs):
        q = "".join(c for c in s.strip().upper() if c in idx)
        v = np.zeros(420, dtype=np.float32)
        if q:
            for c in q:
                v[idx[c]] += 1.0
            for a, b in zip(q, q[1:]):
                v[20 + idx[a] * 20 + idx[b]] += 1.0
            v[:20] /= len(q)
            if len(q) > 1:
                v[20:] /= (len(q) - 1)
        out[s] = v
    return out


def esm_mean_table(seqs: list[str], dataset: str) -> dict[str, np.ndarray] | None:
    uniq = sorted(set(seqs))
    store = RUNS / f"esm_mean_{dataset}.npz"
    if store.exists():
        z = np.load(store, allow_pickle=False)
        keys = list(z["keys"])
        if set(keys) >= {sequence_hash(s) for s in uniq}:
            by_hash = {k: v for k, v in zip(keys, z["vecs"])}
            return {s: by_hash[sequence_hash(s)] for s in uniq}

    missing = [s for s in uniq if not (ESM_CACHE / f"{sequence_hash(s)}.npy").exists()]
    if missing:
        print(f"      esm_mlp unavailable for {dataset}: {len(missing)} of {len(uniq)} "
              f"sequences are absent from {ESM_CACHE}. Run scripts/precompute_esm.py "
              f"(see GPU_RUNBOOK.md); this is the only GPU-bound step.")
        return None

    print(f"      pooling ESM embeddings for {len(uniq)} sequences (one-time) ...")
    out, hashes, vecs = {}, [], []
    for s in uniq:
        h = sequence_hash(s)
        v = np.load(ESM_CACHE / f"{h}.npy").mean(axis=0).astype(np.float32)
        out[s], _ = v, hashes.append(h)
        vecs.append(v)
    store.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(store, keys=np.array(hashes), vecs=np.stack(vecs))
    print(f"      cached pooled means -> {store.name}")
    return out


def build_matrix(df: pd.DataFrame, drug_tab: dict, prot_tab: dict):
    d, t, y = df["drug"].to_numpy(), df["target"].to_numpy(), df["label"].to_numpy()
    keep = [i for i in range(len(df)) if d[i] in drug_tab and t[i] in prot_tab]
    X = np.stack([np.concatenate([drug_tab[d[i]], prot_tab[t[i]]]) for i in keep])
    return X, y[keep], len(df) - len(keep)


def _torch_mlp(Xtr, ytr, Xva, yva, Xte, seed: int) -> np.ndarray:
    import torch
    from torch import nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    T = lambda A: torch.tensor((A - mu) / sd, dtype=torch.float32, device=dev)
    xt, xv, xe = T(Xtr), T(Xva), T(Xte)
    yt = torch.tensor(ytr, dtype=torch.float32, device=dev)

    net = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    two_class_val = len(set(yva.tolist())) > 1
    best, best_state, bad = -1.0, None, 0

    for _ in range(300):
        net.train()
        perm = torch.randperm(len(xt), device=dev)
        for i in range(0, len(xt), 256):
            j = perm[i:i + 256]
            opt.zero_grad()
            lossf(net(xt[j]).squeeze(1), yt[j]).backward()
            opt.step()
        if not two_class_val:
            continue
        net.eval()
        with torch.no_grad():
            a = roc_auc_score(yva, net(xv).squeeze(1).cpu().numpy())
        if a > best:
            best, bad = a, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        import torch as _t
        return _t.sigmoid(net(xe).squeeze(1)).cpu().numpy()


def fit_predict(name, Xtr, ytr, Xva, yva, Xte, seed: int) -> np.ndarray:
    if name == "lr":
        sc = StandardScaler(with_mean=False).fit(Xtr)
        m = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear", random_state=seed)
        m.fit(sc.transform(Xtr), ytr)
        return m.predict_proba(sc.transform(Xte))[:, 1]
    if name == "gbt":
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                           early_stopping=False, random_state=seed)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]
    if name == "esm_mlp":
        return _torch_mlp(Xtr, ytr, Xva, yva, Xte, seed)
    raise ValueError(name)


def load_reference() -> dict:
    f = RUNS / "entity_purity.json"
    if not f.exists():
        raise SystemExit(f"missing {f}; run scripts/entity_purity.py first")
    payload = json.loads(f.read_text(encoding="utf-8"))
    return {cell: {int(r["seed"]): float(r["val_fixed"]) for r in rows
                   if r.get("val_fixed") is not None}
            for cell, rows in payload.get("cells", {}).items()}


def load_incumbent() -> dict:
    f = RUNS / "ladder_manifest.json"
    manifest = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    out = {}
    for cell, tag in manifest.items():
        if cell.startswith("_") or not tag:
            continue
        per = {}
        for d in sorted((RUNS / tag).glob("seed*")):
            m = d / "metrics.json"
            if m.exists():
                per[int(d.name[4:])] = float(
                    json.loads(m.read_text(encoding="utf-8"))["test"]["auroc"])
        if per:
            out[cell] = per
    return out


def paired(diffs: np.ndarray) -> dict:
    n = len(diffs)
    mean = float(diffs.mean())
    if n < 2:
        return {"n": n, "mean": mean, "lo": float("nan"), "hi": float("nan"), "p": float("nan")}
    se = float(diffs.std(ddof=1) / np.sqrt(n))
    t = float(stats.t.ppf(0.975, n - 1))
    _, p = stats.ttest_1samp(diffs, 0.0)
    return {"n": n, "mean": mean, "lo": mean - t * se, "hi": mean + t * se, "p": float(p)}


def _bootstrap_auroc(y, p, n_boot: int, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx].tolist())) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    if len(vals) < 2:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def run_cell(dataset: str, split_mode: str, seeds, models, given_dir=None) -> dict:
    if given_dir == "raw":
        stem = {"human": "Human", "celegans": "Celegans"}.get(dataset)
        if stem is None:
            raise SystemExit(f"no distributed files under data_raw/ for {dataset}")
        files = {k: ROOT / "data_raw" / f"{stem}_{k}.txt" for k in ("train", "val", "test")}
    else:
        root = ROOT / "datasets" / dataset / (given_dir or "random")
        files = {k: root / f"{k}.csv" for k in ("train", "val", "test")}
    kw = dict(dataset=dataset, protocol="random",
              train_file=str(files["train"]), val_file=str(files["val"]),
              test_file=str(files["test"]), scorer_protein="esm",
              p2rank_dir="", split_mode=("given" if given_dir else split_mode))

    with contextlib.redirect_stdout(io.StringIO()):
        tr, va, te, _ = prepare_splits(Config(seed=0, **kw), seed=0)
    pool = pd.concat([tr, va, te], ignore_index=True)
    drug_tab = ecfp_table(pool["drug"].tolist())
    comp_tab = composition_table(pool["target"].tolist())
    esm_tab = esm_mean_table(pool["target"].tolist(), dataset) if "esm_mlp" in models else None

    per_seed = {m: {} for m in models}
    preds: dict[str, tuple] = {}
    dropped_total = 0
    for s in seeds:
        with contextlib.redirect_stdout(io.StringIO()):
            tr, va, te, _ = prepare_splits(Config(seed=s, **kw), seed=s)
        for m in models:
            prot = esm_tab if m == "esm_mlp" else comp_tab
            if prot is None:
                continue
            Xtr, ytr, d1 = build_matrix(tr, drug_tab, prot)
            Xva, yva, d2 = build_matrix(va, drug_tab, prot)
            Xte, yte, d3 = build_matrix(te, drug_tab, prot)
            dropped_total += d1 + d2 + d3
            if len(set(yte.tolist())) < 2:
                continue
            p = fit_predict(m, Xtr, ytr, Xva, yva, Xte, s)
            per_seed[m][s] = float(roc_auc_score(yte, p))
            preds.setdefault(m, (np.asarray(yte), np.asarray(p)))
    if dropped_total:
        print(f"      WARNING: {dropped_total} row(s) unfeaturisable and dropped; the "
              f"model AUROC and the reference AUROC then describe different rows.")
    return per_seed, preds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["biosnap"])
    ap.add_argument("--split-modes", nargs="+", default=["random", "cold_drug"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--models", nargs="+", default=["lr", "gbt"],
                    choices=["lr", "gbt", "esm_mlp"])
    ap.add_argument("--given-dir", default=None,
                    help="Read a supplied split verbatim instead of re-splitting. "
                         "'cold' or 'random' read datasets/<ds>/<dir>/ (e.g. DrugBAN's own "
                         "cold-pair split); 'raw' reads the benchmark authors' distributed "
                         "files under data_raw/, available for human and celegans only. "
                         "Seeds then vary model initialisation, not the partition.")
    ap.add_argument("--fixed-reference", type=float, default=None,
                    help="Use this reference for every cell instead of the per-seed value "
                         "in entity_purity.json. Required with --given-dir, whose splits are "
                         "not in that file. On DrugBAN's Human cold split the reference is "
                         "0.5000 exactly -- verified: every scoring family degenerates there.")
    ap.add_argument("--bootstrap", type=int, default=2000,
                    help="Bootstrap resamples of the TEST SET used to interval a fixed "
                         "supplied split, where the across-seed interval is degenerate. "
                         "Set 0 to disable. Bootstrap intervals are marked * in the table.")
    ap.add_argument("--out", default="runs/second_architecture.json")
    a = ap.parse_args()

    reference, incumbent = load_reference(), load_incumbent()
    if a.fixed_reference is not None:
        for _ds in a.datasets:
            for _sm in a.split_modes:
                reference[f"{_ds}/{_sm}"] = {s: a.fixed_reference for s in a.seeds}
    results: dict[str, dict] = {}

    print("=" * 108)
    print("Margins over the entity-frequency reference, by model. 'headroom' is 1 - reference:")
    print("the largest positive margin any model could achieve on that cell.")
    print("=" * 108)
    print(f"{'cell':<24}{'model':<9}{'n':>3}{'AUROC':>9}{'ref':>9}{'headroom':>10}"
          f"{'margin':>10}{'95% CI':>22}{'cross-attn':>12}")

    for ds in a.datasets:
        if not (ROOT / "datasets" / ds / "random" / "train.csv").exists():
            print(f"{ds:<24} dataset missing, skipped")
            continue
        for sm in a.split_modes:
            cell = f"{ds}/{sm}"
            if cell not in reference:
                print(f"{cell:<24} no reference in entity_purity.json, skipped")
                continue
            print(f"{cell:<24} computing ...")
            per_seed, preds = run_cell(ds, sm, a.seeds, a.models, a.given_dir)
            results[cell] = {}
            for m in a.models:
                ok = sorted(set(per_seed.get(m, {})) & set(reference[cell]))
                if not ok:
                    continue
                mar = np.array([per_seed[m][s] - reference[cell][s] for s in ok])
                st = paired(mar)
                auroc = float(np.mean([per_seed[m][s] for s in ok]))
                ref = float(np.mean([reference[cell][s] for s in ok]))
                inc = incumbent.get(cell, {})
                iok = sorted(set(inc) & set(reference[cell]))
                inc_mar = (float(np.mean([inc[s] - reference[cell][s] for s in iok]))
                           if iok else float("nan"))
                lo, hi, kind = st["lo"], st["hi"], "across-seed"
                if (np.isnan(lo) or hi - lo < 1e-9) and a.bootstrap and m in preds:
                    b = _bootstrap_auroc(*preds[m], a.bootstrap)
                    if not np.isnan(b[0]):
                        lo, hi, kind = b[0] - ref, b[1] - ref, "bootstrap"
                results[cell][m] = {"seeds": ok, "auroc": auroc, "reference": ref,
                                    "headroom": 1.0 - ref, "margin": st["mean"],
                                    "ci": [lo, hi], "ci_kind": kind, "p": st["p"],
                                    "per_seed": per_seed[m], "cross_attention_margin": inc_mar}
                ci = f"[{lo:+.4f}, {hi:+.4f}]" + ("*" if kind == "bootstrap" else "")
                pm = "n/a" if np.isnan(inc_mar) else f"{inc_mar:+.4f}"
                print(f"{'':<24}{m:<9}{len(ok):>3}{auroc:>9.4f}{ref:>9.4f}"
                      f"{1.0 - ref:>10.4f}{st['mean']:>+10.4f}{ci:>22}{pm:>12}")

    print("\n" + "=" * 108)
    print("Matched-reference contrasts recomputed per model")
    print("=" * 108)
    print(f"{'contrast':<34}{'model':<9}{'n':>3}{'difference':>12}{'95% CI':>22}{'p':>10}")
    contrasts: dict[str, dict] = {}
    for label, ds, sa, sb in CONTRASTS:
        ca, cb = f"{ds}/{sa}", f"{ds}/{sb}"
        contrasts[label] = {}
        for m in a.models:
            ra, rb = results.get(ca, {}).get(m), results.get(cb, {}).get(m)
            if not ra or not rb:
                continue
            shared = sorted(set(ra["per_seed"]) & set(rb["per_seed"])
                            & set(reference[ca]) & set(reference[cb]))
            if len(shared) < 2:
                continue
            d = np.array([(ra["per_seed"][s] - reference[ca][s])
                          - (rb["per_seed"][s] - reference[cb][s]) for s in shared])
            st = paired(d)
            contrasts[label][m] = {"seeds": shared, **st}
            ci = f"[{st['lo']:+.4f}, {st['hi']:+.4f}]"
            print(f"{label:<34}{m:<9}{st['n']:>3}{st['mean']:>+12.4f}{ci:>22}{st['p']:>10.4f}")

    print("\n* = bootstrap interval over test pairs, used where the split is fixed and the")
    print("  across-seed interval is degenerate. It expresses test-set sampling uncertainty,")
    print("  not variation across split draws; the two are not interchangeable.")
    print("\nReading: these are reference instruments, not competitors. A lower AUROC than the")
    print("cross-attention model is expected. What matters is whether the SIGN of each margin")
    print("and the DIRECTION of each contrast survive a change of model family. On cells whose")
    print("headroom is small, agreement is close to arithmetic and carries little weight.")

    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cells": results, "contrasts": contrasts,
                               "models": a.models, "seeds": a.seeds}, indent=2),
                   encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
