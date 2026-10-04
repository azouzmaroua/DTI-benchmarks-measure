from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "runs" / "ladder_manifest.json"

DEFAULT_MANIFEST = {
    "_comment": "key = 'dataset/split_mode'; value = the train.py --tag for that "
               "cell, or null if not yet run. Add a tag here the moment a run "
               "completes, then rerun this script -- do not hand-type the result "
               "into the manuscript instead.",
    "biosnap/cold_pair": "bs_recursive",
    "biosnap/cold_target": None,
    "biosnap/cold_drug": None,
    "biosnap/random": "bs_random",
    "human/cold_pair": None,
    "human/cold_drug": None,
    "human/cold_target": None,
    "human/random": None,
    "bindingdb/cold_pair": None,
    "bindingdb/cold_target": None,
    "bindingdb/cold_drug": None,
    "bindingdb/random": None,
}


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(json.dumps(DEFAULT_MANIFEST, indent=2), encoding="utf-8")
        print(f"wrote default manifest to {MANIFEST_PATH} -- edit it to add tags, then rerun")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def load_floor(dataset: str, split_mode: str) -> dict[int, float] | None:
    fp = ROOT / "runs" / "floors" / f"{dataset}_random_{split_mode}.json"
    if not fp.exists():
        return None
    j = json.loads(fp.read_text(encoding="utf-8"))
    return {int(k): v["floor"] for k, v in j["per_seed"].items() if "floor" in v}


def load_model(tag: str) -> dict[int, float]:
    out: dict[int, float] = {}
    tag_dir = ROOT / "runs" / tag
    if not tag_dir.exists():
        return out
    for seed_dir in sorted(tag_dir.glob("seed*")):
        mp = seed_dir / "metrics.json"
        if not mp.exists():
            continue
        seed = int(seed_dir.name.replace("seed", ""))
        m = json.loads(mp.read_text(encoding="utf-8"))
        out[seed] = m["test"]["auroc"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "paper" / "table1_generated.md"))
    a = ap.parse_args()

    manifest = load_manifest()
    rows = []
    print(f"{'cell':<24}{'tag':<16}{'n':>3}{'floor':>9}{'model':>9}{'margin':>10}")
    print("-" * 71)
    for key, tag in manifest.items():
        if key.startswith("_") or tag is None:
            continue
        dataset, split_mode = key.split("/")
        floors = load_floor(dataset, split_mode)
        models = load_model(tag)
        if floors is None:
            print(f"{key:<24} SKIP -- no floor file for this dataset/split_mode")
            continue
        if not models:
            print(f"{key:<24} SKIP -- tag '{tag}' has no metrics.json under runs/{tag}/seed*/")
            continue
        common = sorted(set(floors) & set(models))
        if not common:
            print(f"{key:<24} SKIP -- floor seeds {sorted(floors)} and model seeds "
                  f"{sorted(models)} (tag '{tag}') share no seed")
            continue
        margins = np.array([models[s] - floors[s] for s in common])
        floor_mean = float(np.mean([floors[s] for s in common]))
        model_mean = float(np.mean([models[s] for s in common]))
        row = {"dataset": dataset, "protocol": split_mode, "tag": tag,
              "n_seeds": len(common), "seeds": common,
              "floor_mean": round(floor_mean, 4), "model_mean": round(model_mean, 4),
              "margin_mean": round(float(margins.mean()), 4)}
        if len(common) > 1:
            _, p = stats.ttest_1samp(margins, 0.0)
            ci = stats.t.interval(0.95, len(common) - 1, loc=margins.mean(),
                                  scale=stats.sem(margins))
            row.update(p_value=round(float(p), 5),
                      ci95=[round(float(ci[0]), 4), round(float(ci[1]), 4)])
        rows.append(row)
        print(f"{key:<24}{tag:<16}{len(common):>3}{floor_mean:>9.4f}{model_mean:>9.4f}"
              f"{margins.mean():>+10.4f}")

    if len(rows) >= 3:
        f = np.array([r["floor_mean"] for r in rows])
        m = np.array([r["model_mean"] for r in rows])
        s, i, r, p, se = stats.linregress(f, m)
        t_slope1 = (s - 1) / se if se else float("nan")
        p_slope1 = 2 * stats.t.sf(abs(t_slope1), len(rows) - 2) if se else float("nan")
        print(f"\nfit: model = {i:.4f} + {s:.4f}*floor   R2={r**2:.4f}   "
              f"se(slope)={se:.4f}   H0 slope=1 -> p={p_slope1:.2e}   n={len(rows)} cells")
    else:
        print(f"\nonly {len(rows)} complete cell(s) -- need >= 3 to fit the model-vs-floor line")

    md = ["| Dataset | Protocol | Floor | Model | Margin | 95% CI | n | tag |",
         "|---|---|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        ci = f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]" if "ci95" in r else "n/a (n=1)"
        md.append(f"| {r['dataset']} | {r['protocol']} | {r['floor_mean']:.4f} | "
                 f"{r['model_mean']:.4f} | {r['margin_mean']:+.4f} | {ci} | "
                 f"{r['n_seeds']} | `{r['tag']}` |")
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nwrote {out_path}")

    incomplete = [k for k, v in manifest.items() if not k.startswith("_") and v is None]
    if incomplete:
        print(f"\n{len(incomplete)} cell(s) still have no tag in the manifest: "
              f"{', '.join(incomplete)}")


if __name__ == "__main__":
    main()
