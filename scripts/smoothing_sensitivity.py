from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

RUNS = Path(__file__).resolve().parent.parent / "runs"
K_FILES = {1.0: "entity_purity_k1.json", 5.0: "entity_purity.json", 20.0: "entity_purity_k20.json"}

CONTRASTS = [
    ("BioSNAP random vs cold-drug", "biosnap",
     "biosnap/random", "bs_random", "biosnap/cold_drug", "bs_cold_drug", [0, 1, 2, 3, 4]),
    ("C. elegans random vs cold-target", "celegans",
     "celegans/random", "ce_random", "celegans/cold_target", "ce_cold_target", [0, 1, 2, 3, 4]),
    ("Human cold-drug vs cold-target", "human",
     "human/cold_drug", "recursive_cd", "human/cold_target", "recursive_ct", [0, 1, 2]),
]
SELECTION_THRESHOLD = 0.03


def model_auroc(tag: str, seed: int) -> float | None:
    f = RUNS / tag / f"seed{seed}" / "metrics.json"
    if not f.exists():
        return None
    m = json.loads(f.read_text(encoding="utf-8"))
    return float(m["test"]["auroc"])


def references(payload: dict, cell: str) -> dict[int, float]:
    rows = payload.get("cells", {}).get(cell, [])
    return {int(r["seed"]): float(r["val_fixed"]) for r in rows if r.get("val_fixed") is not None}


def paired(diffs: np.ndarray) -> dict:
    n = len(diffs)
    mean = float(diffs.mean())
    if n < 2:
        return {"n": n, "mean": mean, "lo": float("nan"), "hi": float("nan"),
                "p": float("nan"), "wilcoxon": float("nan")}
    se = float(diffs.std(ddof=1) / np.sqrt(n))
    tcrit = float(stats.t.ppf(0.975, n - 1))
    t, p = stats.ttest_1samp(diffs, 0.0)
    try:
        w = float(stats.wilcoxon(diffs).pvalue)
    except Exception:
        w = float("nan")
    return {"n": n, "mean": mean, "lo": mean - tcrit * se, "hi": mean + tcrit * se,
            "p": float(p), "wilcoxon": w}


def welch(a: np.ndarray, b: np.ndarray) -> dict:
    if len(a) < 2 or len(b) < 2:
        return {"welch_lo": float("nan"), "welch_hi": float("nan"),
                "welch_p": float("nan"), "welch_df": float("nan")}
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = float(np.sqrt(va + vb))
    df = float((va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)))
    diff = float(a.mean() - b.mean())
    tcrit = float(stats.t.ppf(0.975, df))
    p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    return {"welch_lo": diff - tcrit * se, "welch_hi": diff + tcrit * se,
            "welch_p": p, "welch_df": df}


def holm(pvals: list[float]) -> list[float]:
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def main() -> None:
    payloads = {}
    for k, name in K_FILES.items():
        f = RUNS / name
        if not f.exists():
            print(f"MISSING {f} -- run entity_purity.py --pseudo-count {int(k)} first")
            continue
        payloads[k] = json.loads(f.read_text(encoding="utf-8"))
    if not payloads:
        return

    out: dict = {"selection_threshold": SELECTION_THRESHOLD, "contrasts": {}}

    for label, _bench, cell_a, tag_a, cell_b, tag_b, seeds in CONTRASTS:
        print("=" * 104)
        print(label)
        print("=" * 104)
        print(f"{'k':>4}{'ref A':>9}{'ref B':>9}{'gap':>8}{'rule':>7}"
              f"{'margin A':>10}{'margin B':>10}{'difference':>12}"
              f"{'95% CI':>22}{'p (paired)':>12}{'p (Wilcox)':>12}")
        rows = []
        for k in sorted(payloads):
            ref_a, ref_b = references(payloads[k], cell_a), references(payloads[k], cell_b)
            ma, mb, used = [], [], []
            for s in seeds:
                va, vb = model_auroc(tag_a, s), model_auroc(tag_b, s)
                if va is None or vb is None or s not in ref_a or s not in ref_b:
                    continue
                ma.append(va - ref_a[s])
                mb.append(vb - ref_b[s])
                used.append(s)
            if not used:
                print(f"{int(k):>4}  no usable seeds (missing metrics.json or reference)")
                continue
            ma, mb = np.array(ma), np.array(mb)
            st = {**paired(ma - mb), **welch(ma, mb),
                  "margins_a": ma.tolist(), "margins_b": mb.tolist()}
            gap = abs(float(np.mean([ref_a[s] for s in used])) - float(np.mean([ref_b[s] for s in used])))
            keeps = "yes" if gap < SELECTION_THRESHOLD else "NO"
            print(f"{int(k):>4}{np.mean([ref_a[s] for s in used]):>9.4f}"
                  f"{np.mean([ref_b[s] for s in used]):>9.4f}{gap:>8.4f}{keeps:>7}"
                  f"{ma.mean():>+10.4f}{mb.mean():>+10.4f}{st['mean']:>+12.4f}"
                  f"{'[' + format(st['lo'], '+.4f') + ', ' + format(st['hi'], '+.4f') + ']':>22}"
                  f"{st['p']:>12.4f}{st['wilcoxon']:>12.4f}")
            rows.append({"k": k, "seeds": used, "ref_a": float(np.mean([ref_a[s] for s in used])),
                         "ref_b": float(np.mean([ref_b[s] for s in used])), "reference_gap": gap,
                         "satisfies_selection_rule": gap < SELECTION_THRESHOLD,
                         "margin_a": float(ma.mean()), "margin_b": float(mb.mean()),
                         "per_seed_difference": (ma - mb).tolist(), **st})
        out["contrasts"][label] = rows
        print()

    print("Welch (primary) and Holm-adjusted p across the three contrasts, per k")
    labels = list(out["contrasts"])
    for k in sorted(payloads):
        rows_k = [next((r for r in out["contrasts"][lab] if r["k"] == k), None) for lab in labels]
        if any(r is None for r in rows_k):
            continue
        for key_p, key_adj in (("welch_p", "welch_p_holm"), ("p", "p_holm")):
            for r, a in zip(rows_k, holm([r[key_p] for r in rows_k])):
                r[key_adj] = a
        for lab, r in zip(labels, rows_k):
            print(f"  k={int(k):>2}  {lab:<34} diff {r['mean']:+.4f}  Welch "
                  f"[{r['welch_lo']:+.4f}, {r['welch_hi']:+.4f}] p={r['welch_p']:.4f} "
                  f"(Holm {r['welch_p_holm']:.4f}) | paired p={r['p']:.4f} (Holm {r['p_holm']:.4f})")
    print()

    print("Reading: 'rule' is whether the pair still satisfies the manuscript's own selection")
    print("criterion (same benchmark, reference heights within 0.03) at that smoothing weight.")
    print("A 'NO' means the contrast would not have been selected had k been set to that value.")

    f = RUNS / "smoothing_sensitivity.json"
    f.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {f}")


if __name__ == "__main__":
    main()
