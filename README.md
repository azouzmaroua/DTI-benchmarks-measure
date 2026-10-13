# PharmUnit-DTI

Code for a measurement study of four public drug–target interaction (DTI)
benchmarks — Human, *C. elegans*, BioSNAP, and BindingDB — and for
**PharmUnit-DTI**, a cross-attention DTI model with pharmacophore-informed
chemical-unit selection and learnable recursive compression.

**Status: research code, manuscript in preparation. Nothing here has been peer
reviewed.**

## What the study measures

Three things, the first two of which require no trained model:

1. **Whether the standard degree-based control can be computed at all.** A
   permutation-derived edge prior can score a test pair only when both of its
   entities appear in the training positive graph. Entity-disjoint evaluation
   removes that support.
2. **What a non-molecular predictor achieves.** An entity-frequency lookup
   that uses no molecular structure, no protein sequence, and no learned
   parameters.
3. **How the resulting reference changes the reading of a trained model's
   AUROC**, measured with the PharmUnit-DTI cross-attention model as the
   instrument.

## The model

```
SMILES  --RDKit-->  atoms + chemical-unit partition + pharmacophore histogram
        --GCN-->    atom vectors --pool--> unit vectors [mean(atoms) | pharm]
        --RecursiveCompressor--> kept units

sequence --ESM-2 (frozen, cached)--> residues --segment pool--> segment vectors
        --RecursiveCompressor--> kept segments

kept units <-> kept segments --cross-attention--> pool, concat, classify
```

The compression module applies R rounds of {re-encode → score → prune} with a
learned per-round retention threshold, so the keep rate is an **output** of
training rather than a hand-set constant. Three ablation rungs:
`compress_enabled=False` (no selection), `rounds=1, retention='fixed'`
(single-shot top-k, the baseline), and `rounds>1, retention='learned'`
(the proposed method).

## Data provenance, and one caution

`datasets/` and `data_raw/` contain third-party benchmark data, tracked in this
repository so every number is reproducible from a fresh clone. They carry
their original authors' terms. Layout:

```
datasets/<dataset>/<protocol>/{train,val,test}.csv   # DrugBAN format: SMILES,Protein,Y
data_raw/{Human,Celegans}_{train,val,test}.txt        # whitespace TXT, for the raw audit
```

**Caution for anyone reusing the distributed files:** the released Human and
*C. elegans* split files contain verbatim train–test pair overlap (13.7% and
22.2% of test pairs respectively, measured by
`scripts/published_split_audit.py --source raw`). Any result computed on those
files as supplied is affected by direct pair memorisation. The pipeline in
`pharmunit/data.py` pools, removes contradictory labels, deduplicates, and
only then re-splits, which removes it.

## Environment

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

`fair-esm` and a CUDA-capable GPU are needed only for the one-off ESM-2
embedding cache build (`scripts/precompute_esm.py`).

## Reproducing the analysis

Each script resolves paths from its own location, so it runs from any working
directory.

| Output | Command |
|---|---|
| Train the model (any ablation arm) | `python -m pharmunit.train --tag full --seeds 0 1 2` |
| Dataset descriptives, entity-class purity, identity lookups, the reference under both constructions, all benchmark–protocol cells | `python scripts/entity_purity.py` |
| Same, at other smoothing weights | `python scripts/entity_purity.py --pseudo-count 1 --out runs/entity_purity_k1.json` |
| Edge-prior domain and evaluability | `python scripts/edge_prior_comparison.py` |
| Edge-prior invariants (degree and edge-count preservation) | `python scripts/edge_prior_comparison.py --self-test` |
| Edge-prior domain on the benchmarks' own supplied split files | `python scripts/published_split_audit.py` |
| Per-seed trivial floors (no-molecule baselines) | `python scripts/floors.py --dataset human --split-mode cold_pair` |
| Model-versus-reference margins with per-cell intervals | `python scripts/build_ladder_table.py` |
| Smoothing-weight sensitivity and the matched contrasts | `python scripts/smoothing_sensitivity.py` |
| Second-architecture check (logistic regression, gradient-boosted trees, ESM-MLP) | `python scripts/second_architecture.py` |
| *C. elegans* dataset construction from the distributed files | `python scripts/make_celegans_dataset.py` |
| ESM-2 embedding cache (the only GPU-bound step) | `python scripts/precompute_esm.py --pair-file ... --out-dir cache/esm` |
| Figure 1: purity beside lookup AUROC | `python scripts/make_figure1.py` |
| Figure 2: model-reference margin forest plot | `python scripts/make_figure2.py` |
| Graphical abstract | `python scripts/make_graphical_abstract.py` |
| Supplementary LaTeX tables | `python scripts/make_supplement_tables.py` |

Tests: `python -m pytest tests/`

## Layout

```
pharmunit/    package: config, data loading, splits, features, model, training
scripts/      analysis and reproduction scripts (see table above)
tests/        pytest suite
```

## What is deliberately not in this repository

- **`cache/esm/`** — ~4.6 GB of per-residue ESM-2 embeddings, one `.npy` per
  protein sequence keyed by SHA-1 of the sequence. Rebuild with
  `scripts/precompute_esm.py`.
- **`runs/`** — per-seed run artifacts (`metrics.json`, `history.csv`,
  `best.pt`, `split_report.txt`). Every table in the manuscript is regenerated
  from these by the scripts above; no number is typed by hand.
- **`datasets/`, `data_raw/`** — third-party benchmark data, versioned in this
  repository; see "Data provenance" above.
