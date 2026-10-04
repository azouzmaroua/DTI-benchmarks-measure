from __future__ import annotations

import sys
import pathlib

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pharmunit.config import Config
from pharmunit.data import clean_pairs, prepare_splits, verify_coldness

ROOT = pathlib.Path(__file__).resolve().parents[1]
HUMAN = ROOT / "datasets" / "human" / "random"
HAS_DATA = (HUMAN / "train.csv").exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="datasets/human not present")


def _cfg(**kw) -> Config:
    base = dict(
        train_file=str(HUMAN / "train.csv"),
        val_file=str(HUMAN / "val.csv"),
        test_file=str(HUMAN / "test.csv"),
        scorer_protein="esm", p2rank_dir="", seed=0,
    )
    base.update(kw)
    return Config(**base)


def test_contradictions_dropped_before_deduplication():
    df = pd.DataFrame({
        "drug":   ["A", "A", "B", "B", "C"],
        "target": ["X", "X", "Y", "Y", "Z"],
        "label":  [1,   0,   1,   1,   0],
    })
    out, n_dup, n_contra = clean_pairs(df)
    assert n_contra == 2, "both rows of the contradictory pair must go"
    assert ("A", "X") not in set(map(tuple, out[["drug", "target"]].to_numpy()))
    assert n_dup == 1, "the duplicate B/Y row must be removed"
    assert len(out) == 2


def test_cleaning_can_be_disabled():
    df = pd.DataFrame({"drug": ["A", "A"], "target": ["X", "X"], "label": [1, 1]})
    out, n_dup, _ = clean_pairs(df, deduplicate=False, drop_contradictory=False)
    assert len(out) == 2 and n_dup == 0


@needs_data
@pytest.mark.parametrize("mode", ["cold_drug", "cold_target", "cold_pair"])
def test_cold_split_is_entity_disjoint(mode):
    tr, va, te, rep = prepare_splits(_cfg(split_mode=mode))
    assert len(tr) and len(va) and len(te), f"{mode} produced an empty fold"

    tr_d, tr_t = set(tr["drug"]), set(tr["target"])
    te_d, te_t = set(te["drug"]), set(te["target"])

    if mode in ("cold_drug", "cold_pair"):
        assert not (te_d & tr_d), f"{mode}: {len(te_d & tr_d)} test drugs seen in train"
    if mode in ("cold_target", "cold_pair"):
        assert not (te_t & tr_t), f"{mode}: {len(te_t & tr_t)} test targets seen in train"

    assert rep.leakage_audit["ok"], rep.leakage_audit


@needs_data
@pytest.mark.parametrize("mode", ["random", "cold_drug", "cold_target", "cold_pair"])
def test_no_verbatim_pair_leakage_in_any_mode(mode):
    tr, _, te, _ = prepare_splits(_cfg(split_mode=mode))
    tr_p = set(map(tuple, tr[["drug", "target"]].to_numpy()))
    te_p = set(map(tuple, te[["drug", "target"]].to_numpy()))
    assert not (tr_p & te_p), f"{mode}: {len(tr_p & te_p)} test pairs appear in train"


@needs_data
def test_cold_is_strictly_harder_than_random():
    _, _, te_r, _ = prepare_splits(_cfg(split_mode="random"))
    tr_r, _, _, _ = prepare_splits(_cfg(split_mode="random"))
    seen_random = len(set(te_r["drug"]) & set(tr_r["drug"])) / max(te_r["drug"].nunique(), 1)

    tr_c, _, te_c, _ = prepare_splits(_cfg(split_mode="cold_drug"))
    seen_cold = len(set(te_c["drug"]) & set(tr_c["drug"])) / max(te_c["drug"].nunique(), 1)

    assert seen_cold == 0.0
    assert seen_random > seen_cold, "random split should share drugs; cold must not"


@needs_data
def test_deduplication_is_reported_not_silent():
    _, _, _, rep = prepare_splits(_cfg(split_mode="random"))
    assert rep.n_raw > 0
    assert rep.n_after_clean == rep.n_raw - rep.n_duplicates_removed - rep.n_contradictory_removed


@needs_data
def test_splits_are_deterministic_given_seed():
    a = prepare_splits(_cfg(split_mode="cold_drug", seed=7))[2]
    b = prepare_splits(_cfg(split_mode="cold_drug", seed=7))[2]
    assert a.equals(b), "same seed must reproduce the same test fold"


@needs_data
def test_different_seeds_give_different_splits():
    a = prepare_splits(_cfg(split_mode="cold_drug", seed=1))[2]
    b = prepare_splits(_cfg(split_mode="cold_drug", seed=2))[2]
    assert not a.equals(b), "seed is not varying the split"


def test_verify_coldness_raises_on_a_leaking_cold_split():
    tr = pd.DataFrame({"drug": ["A"], "target": ["X"], "label": [1]})
    te = pd.DataFrame({"drug": ["A"], "target": ["Y"], "label": [0]})
    with pytest.raises(ValueError, match="COLD SPLIT VIOLATION"):
        verify_coldness(tr, tr, te, mode="cold_drug")


def test_verify_coldness_passes_a_clean_split():
    tr = pd.DataFrame({"drug": ["A"], "target": ["X"], "label": [1]})
    te = pd.DataFrame({"drug": ["B"], "target": ["Y"], "label": [0]})
    assert verify_coldness(tr, tr, te, mode="cold_pair")["ok"]
