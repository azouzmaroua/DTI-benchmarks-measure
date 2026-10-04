# DTI-benchmarks-measure

**PharmUnit-DTI: a benchmark audit for drug–target interaction prediction**

This repository contains the code for an audit of common drug–target interaction
(DTI) benchmarks (Human, *C. elegans*, BioSNAP, BindingDB). It includes:

- **PharmUnit-DTI**, a reference model that encodes drugs as pharmacophore units and
  proteins as ESM-2 residue segments, joined by bidirectional cross-attention, with
  optional learned recursive token compression.
- **Split construction and leakage checks** for random, cold-drug, cold-target and
  cold-pair protocols. Every generated split is verified, and training stops if a
  "cold" split leaks test entities into train.
- **Trivial floors**: smoothed per-drug / per-target label-rate baselines that
  measure how much of a benchmark score memorised entity identity alone can explain.
- **Analysis scripts** for entity purity, edge-prior (degree-preserving permutation)
  baselines, a second non-neural architecture, and calibration against published
  DrugBAN numbers.

Results, data and trained weights are not included. The scripts regenerate them.

## Repository layout

```
pharmunit/          core package
  config.py         all hyper-parameters (dataclass Config)
  data.py           loading, de-duplication, split preparation, coldness audit
  data_sources.py   dataset/protocol resolution
  splits.py         random / cold_drug / cold_target / cold_pair split generation
  dataset.py        featurisation, Dataset and collate
  drug_features.py  RDKit atom graph + pharmacophore units
  protein_features.py  ESM-2 embedding cache lookup
  scorers.py        token scorers and top-k gathering
  compression.py    recursive token compressor
  model.py          PharmUnitDTI model and loss
  metrics.py, stats.py, efficiency.py
  train.py          training entry point (python -m pharmunit.train)
scripts/            analyses, baselines, tables and figures
tests/              pytest suite
```

## Installation

Python 3.10+ is required.

```bash
pip install -r requirements.txt
```

The test suite also needs `pytest`. `scripts/extract_drugban_published.py` also needs `pymupdf`.

## Data

Place the benchmark files as follows. These folders are git-ignored.

```
datasets/
  human/{random,cold}/{train,val,test}.csv
  biosnap/random/{train,val,test}.csv
  biosnap/cluster/{source_train,target_train,target_test}.csv
  bindingdb/random/{train,val,test}.csv
  bindingdb/cluster/{source_train,target_train,target_test}.csv
  celegans/random/{train,val,test}.csv      # built by scripts/make_celegans_dataset.py
data_raw/
  Human_{train,val,test}.txt
  Celegans_{train,val,test}.txt
```

- CSV files use the DrugBAN format, with columns `SMILES,Protein,Y`. The Human,
  BioSNAP and BindingDB splits are those distributed with DrugBAN
  (<https://github.com/peizhenbai/DrugBAN>).
- TXT files are whitespace-separated, one `smiles sequence label` per line.

Build the *C. elegans* CSV split from the raw files:

```bash
python scripts/make_celegans_dataset.py
```

### ESM-2 embeddings

Protein features are read from a precomputed ESM-2 cache. The default location is
`cache/esm`, or you can set `PHARMUNIT_ESM_CACHE`:

```bash
python scripts/precompute_esm.py --pair-file datasets/biosnap/random/*.csv --out-dir cache/esm
```

## Training

```bash
# shipped random split
python -m pharmunit.train --dataset biosnap --protocol random --tag bs_random --seeds 0 1 2 3 4

# split regenerated per seed from the pooled data, with a leakage check
python -m pharmunit.train --dataset biosnap --split-mode cold_drug --tag bs_cold_drug --seeds 0 1 2 3 4

# no-compression accuracy ceiling
python -m pharmunit.train --dataset biosnap --split-mode cold_pair --no-compress --tag bs_ceiling
```

Each run writes `config.json`, `env.json`, `history.csv`, `split_report.txt`,
`metrics.json` and `best.pt` to `runs/<tag>/seed<k>/`. Run
`python -m pharmunit.train -h` for all options, including compression rounds,
retention mode, budget weight and keep ratios.

## Analyses

All scripts run from the repository root and write to `runs/` by default.

| Script | Purpose |
|---|---|
| `floors.py` | Per-seed trivial floors (entity label-rate baselines) |
| `entity_purity.py` | Single-class entity fractions and purity-only AUROC |
| `purity_decomposition.py` | Decomposition of the purity signal across benchmarks |
| `smoothing_sensitivity.py` | Sensitivity of the floors to the pseudo-count |
| `edge_prior_comparison.py`, `edge_prior_domain.py` | Degree-preserving (XSwap) edge-prior baselines |
| `published_split_audit.py` | Leakage audit of the published splits |
| `reference_on_published_splits.py` | Floors evaluated on the published splits |
| `second_architecture.py` | Logistic regression / gradient boosting on ECFP + ESM features |
| `materialise_split.py` | Write a generated split to CSV |
| `extract_drugban_published.py`, `drugban_calibration.py` | Compare against published DrugBAN results |
| `build_ladder_table.py`, `make_supplement_tables.py`, `make_figure1.py`, `make_graphical_abstract.py` | Tables and figures from `runs/` |

The table and figure scripts read outputs of the training runs and the analysis
scripts above, so run those first.

## Tests

```bash
python -m pytest -q
```

Tests that need `datasets/human` are skipped when the data is not present.

## Reproducibility

Seeds are fixed and cuDNN runs in deterministic mode. GPU kernels are still not
guaranteed to be bitwise reproducible, so expect small run-to-run differences.
