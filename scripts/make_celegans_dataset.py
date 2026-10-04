from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pharmunit.dataset import load_pairs

ROOT = Path(__file__).resolve().parent.parent
SRC = [ROOT / "data_raw" / f"Celegans_{p}.txt" for p in ("train", "val", "test")]
OUT = ROOT / "datasets" / "celegans" / "random"


def main() -> None:
    missing = [p for p in SRC if not p.exists()]
    if missing:
        raise SystemExit(f"missing source files: {[str(p) for p in missing]}")

    df = pd.concat([load_pairs(str(p)) for p in SRC], ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(subset=["drug", "target", "label"]).reset_index(drop=True)

    counts = df.groupby(["drug", "target"])["label"].nunique()
    bad = counts[counts > 1].index
    if len(bad):
        df = df[~df.set_index(["drug", "target"]).index.isin(bad)].reset_index(drop=True)

    rng = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    n = len(rng)
    n_test = int(round(0.10 * n))
    n_val = int(round(0.10 * n))
    parts = {
        "test": rng.iloc[:n_test],
        "val": rng.iloc[n_test:n_test + n_val],
        "train": rng.iloc[n_test + n_val:],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        out = part.rename(columns={"drug": "SMILES", "target": "Protein", "label": "Y"})
        out[["SMILES", "Protein", "Y"]].to_csv(OUT / f"{name}.csv", index=False)
        print(f"wrote {OUT / f'{name}.csv'}  ({len(out)} rows)")

    print(f"\npooled {before} rows -> {n} after dedup/contradiction removal")
    print(f"drugs {df['drug'].nunique()}  proteins {df['target'].nunique()}  "
          f"positive rate {df['label'].mean():.4f}")


if __name__ == "__main__":
    main()
