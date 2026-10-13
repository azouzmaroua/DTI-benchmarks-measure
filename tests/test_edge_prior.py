from __future__ import annotations

import sys
import pathlib
from collections import Counter
from itertools import product

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from edge_prior_comparison import xswap_edge_prior

@pytest.fixture
def toy_graph():
    edges = sorted({(f"d{i % 15}", f"t{(i * 7) % 11}") for i in range(60)})
    drugs = sorted({d for d, _ in edges}) + ["d_unseen"]
    targets = sorted({t for _, t in edges}) + ["t_unseen"]
    pairs = list(product(drugs, targets))
    return edges, pairs

def test_unseen_entities_get_prior_zero(toy_graph):
    edges, pairs = toy_graph
    prior = xswap_edge_prior(edges, pairs, n_perm=30, swaps_per_edge=10,
                             rng=np.random.default_rng(0))
    unseen = [p for p in pairs if "unseen" in p[0] or "unseen" in p[1]]
    assert unseen, "fixture must contain unseen-entity pairs"
    assert all(prior[p] == 0.0 for p in unseen)

def test_edge_count_preserved_every_permutation(toy_graph):
    edges, pairs = toy_graph
    rng = np.random.default_rng(1)
    prior = xswap_edge_prior(edges, pairs, n_perm=25, swaps_per_edge=10, rng=rng)
    assert abs(sum(prior.values()) - len(edges)) < 1e-9

def test_degree_sequences_preserved(toy_graph):
    edges, pairs = toy_graph
    rng = np.random.default_rng(2)
    prior = xswap_edge_prior(edges, pairs, n_perm=25, swaps_per_edge=10, rng=rng)
    deg_d, deg_t = Counter(d for d, _ in edges), Counter(t for _, t in edges)
    exp_d, exp_t = Counter(), Counter()
    for (d, t), v in prior.items():
        exp_d[d] += v
        exp_t[t] += v
    assert all(abs(exp_d[d] - deg_d[d]) < 1e-9 for d in deg_d)
    assert all(abs(exp_t[t] - deg_t[t]) < 1e-9 for t in deg_t)

def test_swaps_actually_move_edges(toy_graph):
    edges, pairs = toy_graph
    rng = np.random.default_rng(3)
    prior = xswap_edge_prior(edges, pairs, n_perm=25, swaps_per_edge=10, rng=rng)
    moved = sum(1 for v in prior.values() if 0.0 < v < 1.0)
    assert moved > len(edges)
