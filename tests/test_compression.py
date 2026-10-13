from __future__ import annotations

import sys
import pathlib

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pharmunit.compression import RecursiveCompressor, _z_for_keep
from pharmunit.scorers import topk_gather

D = 32

def _batch(B=4, T=12, d=D, n_valid=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(B, T, d, generator=g)
    m = torch.zeros(B, T, dtype=torch.bool)
    for b in range(B):
        k = T if n_valid is None else n_valid[b]
        m[b, :k] = True
    return v * m.unsqueeze(-1), m

def test_disabled_is_exact_identity():
    v, m = _batch()
    c = RecursiveCompressor(D, enabled=False)
    out_v, out_m, st = c(v, m)
    assert out_v is v, "disabled compressor must not copy or transform V"
    assert out_m is m, "disabled compressor must not copy or transform M"
    assert st.enabled is False and st.n_rounds == 0
    assert list(c.parameters()) == [], "disabled compressor must own no parameters"

def test_disabled_adds_no_gradient_path():
    v, m = _batch()
    v = v.clone().requires_grad_(True)
    c = RecursiveCompressor(D, enabled=False)
    out_v, _, st = c(v, m)
    out_v.sum().backward()
    assert torch.equal(v.grad, torch.ones_like(v)), "identity must pass gradient unchanged"
    assert float(st.budget_loss) == 0.0

@pytest.mark.parametrize("rounds", [1, 2, 3])
@pytest.mark.parametrize("retention", ["fixed", "learned"])
def test_shapes_and_mask_consistency(rounds, retention):
    v, m = _batch(B=5, T=16, n_valid=[16, 12, 8, 3, 1])
    c = RecursiveCompressor(D, rounds=rounds, retention=retention, keep_frac=0.6)
    out_v, out_m, st = c(v, m)

    assert out_v.shape[0] == v.shape[0] and out_v.shape[2] == D
    assert out_v.shape[1] == out_m.shape[1]
    assert out_v.shape[1] <= v.shape[1], "compression must not grow the token set"
    assert out_m.sum(1).min() >= 1, "every sample must retain at least one token"
    assert (out_m.sum(1) <= m.sum(1)).all(), "kept count cannot exceed valid count"
    assert torch.equal(out_v[~out_m], torch.zeros_like(out_v[~out_m]))
    assert len(st.keep_rates) == rounds

def test_single_token_sample_survives():
    v, m = _batch(B=2, T=6, n_valid=[1, 1])
    c = RecursiveCompressor(D, rounds=3, retention="learned", keep_frac=0.1)
    out_v, out_m, _ = c(v, m)
    assert (out_m.sum(1) == 1).all(), "min_tokens floor must hold at every round"

def test_batch_of_one():
    v, m = _batch(B=1, T=9)
    c = RecursiveCompressor(D, rounds=2)
    out_v, out_m, _ = c(v, m)
    assert out_v.shape[0] == 1 and out_m.sum() >= 1

@pytest.mark.parametrize("recompress", ["none", "context", "attn"])
def test_every_round_scorer_receives_gradient(recompress):
    v, m = _batch(B=6, T=14)
    c = RecursiveCompressor(D, rounds=3, retention="learned", recompress=recompress)
    out_v, out_m, _ = c(v, m)
    out_v.sum().backward()

    for r, sc in enumerate(c.scorers):
        w = sc[0].weight
        assert w.grad is not None, f"round {r} scorer got grad=None"
        assert w.grad.abs().sum() > 0, f"round {r} scorer got an all-zero gradient"

def test_learned_threshold_receives_gradient():
    v, m = _batch(B=6, T=14)
    c = RecursiveCompressor(D, rounds=2, retention="learned")
    out_v, _, st = c(v, m)
    (out_v.sum() + st.budget_loss).backward()
    assert c.tau.grad is not None, "tau got grad=None -- retention is not learnable"
    assert c.tau.grad.abs().sum() > 0, "tau got an all-zero gradient"

def test_recompressor_receives_gradient():
    v, m = _batch(B=6, T=14)
    c = RecursiveCompressor(D, rounds=2, recompress="context")
    out_v, _, _ = c(v, m)
    out_v.sum().backward()
    for r, rc in enumerate(c.recompressors):
        g = rc.net[0].weight.grad
        assert g is not None and g.abs().sum() > 0, f"round {r} recompressor got no gradient"

def test_gate_is_invariant_to_scorer_output_scale():
    grads, keeps = [], []
    for scale in (1.0, 20.0, 200.0):
        torch.manual_seed(0)
        v, m = _batch(B=6, T=14, seed=1)
        c = RecursiveCompressor(D, rounds=1, retention="learned", recompress="none")
        with torch.no_grad():
            c.scorers[0][2].weight.mul_(scale)
        out_v, out_m, st = c(v, m)
        (out_v.sum() + st.budget_loss).backward()
        grads.append(float(c.tau.grad.abs().sum()))
        keeps.append(out_m.sum().item())

    assert all(g > 0 for g in grads), f"gradient to tau vanished: {grads}"
    assert len(set(keeps)) == 1, f"selection changed with score scale: {keeps}"
    assert max(grads) / min(grads) < 1.2, f"gradient to tau is scale-dependent: {grads}"

def test_standardisation_is_what_prevents_saturation():
    torch.manual_seed(0)
    s = torch.randn(64)
    raw, std = [], []
    for scale in (1.0, 20.0, 200.0):
        x = (s * scale).requires_grad_(True)
        torch.sigmoid(x).sum().backward()
        raw.append(float(x.grad.abs().mean()))

        y = (s * scale).requires_grad_(True)
        z = (y - y.mean()) / (y.var(unbiased=False) + 1e-5).sqrt()
        torch.sigmoid(z).sum().backward()
        std.append(float(y.grad.abs().mean() * scale))

    assert raw[0] / raw[-1] > 50, f"raw gate did not saturate as expected: {raw}"
    assert max(std) / min(std) < 2, f"standardised gate should be stable: {std}"

def test_straight_through_gate_is_exactly_one_forward():
    v, m = _batch(B=4, T=10)
    c = RecursiveCompressor(D, rounds=2, retention="fixed", recompress="none")
    c.eval()
    with torch.no_grad():
        a, ma, _ = c(v, m)
    b, mb, _ = c(v, m)
    assert torch.equal(ma, mb)
    assert torch.equal(a, b.detach()), "gate perturbed the forward value"

def test_single_round_fixed_matches_topk_gather():
    torch.manual_seed(0)
    v, m = _batch(B=4, T=12, n_valid=[12, 9, 6, 4], seed=3)
    c = RecursiveCompressor(D, rounds=1, retention="fixed", keep_frac=0.5,
                            recompress="none")
    c.eval()
    with torch.no_grad():
        got_v, got_m, _ = c(v, m)
        score = c.scorers[0](v).squeeze(-1).masked_fill(~m, float("-inf"))
        want_v, want_m = topk_gather(v, score, m, keep_frac=0.5, straight_through=True)

    assert torch.equal(got_m, want_m), "kept mask diverged from single-shot top-k"
    assert torch.allclose(got_v, want_v, atol=1e-6), "kept values diverged"

def test_recursion_differs_from_single_shot_at_matched_ratio():
    torch.manual_seed(0)
    v, m = _batch(B=8, T=20, seed=7)

    flat = RecursiveCompressor(D, rounds=1, retention="fixed", keep_frac=0.25,
                               recompress="none").eval()
    nested = RecursiveCompressor(D, rounds=2, retention="fixed", keep_frac=0.5,
                                 recompress="none").eval()
    nested.scorers[0].load_state_dict(flat.scorers[0].state_dict())
    nested.scorers[1].load_state_dict(flat.scorers[0].state_dict())
    with torch.no_grad():
        a, _, _ = flat(v, m)
        b, _, _ = nested(v, m)
    assert torch.allclose(a, b, atol=1e-5), \
        "without re-encoding, nested top-k must equal single-shot top-k"

    live = RecursiveCompressor(D, rounds=2, retention="fixed", keep_frac=0.5,
                               recompress="context").eval()
    live.scorers[0].load_state_dict(flat.scorers[0].state_dict())
    with torch.no_grad():
        cv, _, _ = live(v, m)
    assert not torch.allclose(a, cv, atol=1e-5), \
        "re-encoding changed nothing -- the recursion is decorative"

def test_budget_penalty_responds_to_target():
    v, m = _batch(B=6, T=16, seed=11)
    loose = RecursiveCompressor(D, rounds=1, retention="learned", target_keep=0.9)
    tight = RecursiveCompressor(D, rounds=1, retention="learned", target_keep=0.1)
    tight.tau.data.copy_(loose.tau.data)
    tight.scorers[0].load_state_dict(loose.scorers[0].state_dict())
    _, _, sl = loose(v, m)
    _, _, stg = tight(v, m)
    assert float(stg.budget_loss.detach()) >= float(sl.budget_loss.detach())

def test_penalty_has_a_real_ceiling():
    v, m = _batch(B=8, T=200, seed=23)
    c = RecursiveCompressor(D, rounds=1, retention="learned",
                            target_keep=0.5, budget_weight=1.0)
    with torch.no_grad():
        c.tau.fill_(0.7)
    _, _, stats = c(v, m)

    realised = stats.keep_rates[0]
    assert realised < 0.4, f"test setup must land clearly below target; got {realised}"
    assert float(stats.budget_loss.detach()) < 1e-4, (
        f"realised retention {realised:.3f} is already under the 0.5 target; "
        f"the penalty should be ~0, not {float(stats.budget_loss.detach()):.4f}")

def test_budget_weight_zero_disables_the_penalty():
    v, m = _batch(B=6, T=24, seed=17)

    def budget(bw: float) -> float:
        torch.manual_seed(0)
        c = RecursiveCompressor(D, rounds=2, retention="learned",
                                recompress="context", budget_weight=bw).eval()
        with torch.no_grad():
            return float(c(v, m)[2].budget_loss)

    assert budget(0.0) == 0.0, "zero weight must zero the penalty"
    raw = budget(1.0)
    assert raw > 0.0, "the test is vacuous unless the unweighted penalty is nonzero"
    assert budget(0.0) != raw, "0.0 and 1.0 must not be the same configuration"
    for bw in (0.01, 0.05, 0.20, 2.0):
        assert abs(budget(bw) - bw * raw) < 1e-6, bw

def test_deterministic_in_eval():
    v, m = _batch(B=4, T=12)
    c = RecursiveCompressor(D, rounds=3, retention="learned").eval()
    with torch.no_grad():
        a, ma, _ = c(v, m)
        b, mb, _ = c(v, m)
    assert torch.equal(a, b) and torch.equal(ma, mb)

def test_z_for_keep_is_monotone():
    qs = [_z_for_keep(f) for f in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert all(qs[i] > qs[i + 1] for i in range(len(qs) - 1)), qs
    assert abs(_z_for_keep(0.5)) < 1e-6, "median keep must sit at z=0"

def test_init_keep_is_independent_of_keep_frac():
    a = RecursiveCompressor(D, rounds=1, retention="learned", keep_frac=0.50, init_keep=0.95)
    b = RecursiveCompressor(D, rounds=1, retention="learned", keep_frac=0.90, init_keep=0.95)
    assert torch.equal(a.tau.data, b.tau.data), "tau must follow init_keep, not keep_frac"
    assert abs(float(a.tau.data[0]) - _z_for_keep(0.95)) < 1e-6

def test_full_initial_retention_is_refused():
    with pytest.raises(ValueError, match="saturates"):
        RecursiveCompressor(D, retention="learned", init_keep=1.0)
    with pytest.raises(ValueError, match="init_keep must be in"):
        RecursiveCompressor(D, retention="learned", init_keep=0.0)

def test_near_full_init_keeps_the_scorer_trainable():
    grads = {}
    for ik in (0.50, 0.95):
        torch.manual_seed(0)
        c = RecursiveCompressor(D, rounds=1, retention="learned", recompress="none",
                                keep_frac=0.50, init_keep=ik)
        v, m = _batch(B=8, T=20, seed=1)
        out, _, st = c(v, m)
        (out.sum() + st.budget_loss).backward()
        grads[ik] = float(c.scorers[0][0].weight.grad.abs().mean())
    assert grads[0.95] > 0, "no gradient at the default init"
    assert 0.3 < grads[0.95] / grads[0.50] < 3.0, grads

def test_init_keep_defaults_to_keep_frac_when_unset():
    c = RecursiveCompressor(D, rounds=1, retention="learned", keep_frac=0.70)
    assert abs(float(c.tau.data[0]) - _z_for_keep(0.70)) < 1e-6
