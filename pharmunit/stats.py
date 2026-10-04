from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

METRICS = {
    "auroc": roc_auc_score,
    "auprc": average_precision_score,
}


@dataclass
class PairedResult:
    metric: str
    name_a: str
    name_b: str
    point_a: float
    point_b: float
    diff: float
    ci95_low: float
    ci95_high: float
    p_value: float
    p_holm: float | None
    n_boot: int
    n_pairs: int

    @property
    def significant(self) -> bool:
        p = self.p_holm if self.p_holm is not None else self.p_value
        return p < 0.05

    def sentence(self) -> str:
        verb = "beats" if self.diff > 0 else "trails"
        sig = "significant" if self.significant else "NOT significant"
        return (
            f"{self.name_a} {verb} {self.name_b} on {self.metric} by "
            f"{abs(self.diff):.4f} (95% CI [{self.ci95_low:.4f}, {self.ci95_high:.4f}], "
            f"p={self.p_value:.4g}) — {sig}"
        )


def paired_bootstrap(
    y_true: Sequence[int],
    prob_a: Sequence[float],
    prob_b: Sequence[float],
    metric: str = "auroc",
    n_boot: int = 5000,
    seed: int = 0,
    name_a: str = "A",
    name_b: str = "B",
) -> PairedResult:
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {sorted(METRICS)}")
    fn = METRICS[metric]

    y = np.asarray(list(y_true), dtype=int)
    a = np.asarray(list(prob_a), dtype=float)
    b = np.asarray(list(prob_b), dtype=float)
    if not (len(y) == len(a) == len(b)):
        raise ValueError(
            f"unpaired inputs: len(y)={len(y)}, len(a)={len(a)}, len(b)={len(b)}. "
            "Both models must be scored on the same test rows in the same order."
        )

    point_a, point_b = float(fn(y, a)), float(fn(y, b))
    observed = point_a - point_b

    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y[idx]
        if yt.min() == yt.max():
            continue
        diffs.append(fn(yt, a[idx]) - fn(yt, b[idx]))

    diffs = np.asarray(diffs)
    if diffs.size == 0:
        raise RuntimeError("every bootstrap resample was single-class; test set too small")

    centred = diffs - diffs.mean()
    p = 2.0 * min(
        float((centred >= abs(observed)).mean()),
        float((centred <= -abs(observed)).mean()),
    )
    p = max(p, 1.0 / diffs.size)
    return PairedResult(
        metric=metric,
        name_a=name_a,
        name_b=name_b,
        point_a=point_a,
        point_b=point_b,
        diff=observed,
        ci95_low=float(np.percentile(diffs, 2.5)),
        ci95_high=float(np.percentile(diffs, 97.5)),
        p_value=float(min(p, 1.0)),
        p_holm=None,
        n_boot=int(diffs.size),
        n_pairs=n,
    )


def holm_correct(results: list[PairedResult]) -> list[PairedResult]:
    order = sorted(range(len(results)), key=lambda i: results[i].p_value)
    m = len(results)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * results[i].p_value)
        running = max(running, adj)
        results[i].p_holm = running
    return results


@dataclass
class UnpairedResult:
    metric: str
    name_a: str
    name_b: str
    point_a: float
    point_b: float
    diff: float
    ci95_low: float
    ci95_high: float
    n_boot: int

    @property
    def significant(self) -> bool:
        return not (self.ci95_low <= 0.0 <= self.ci95_high)

    def sentence(self) -> str:
        verdict = "excludes zero" if self.significant else "INCLUDES ZERO"
        return (
            f"{self.name_a} - {self.name_b} on {self.metric}: {self.diff:+.4f} "
            f"(95% CI [{self.ci95_low:+.4f}, {self.ci95_high:+.4f}], {verdict})"
        )


def unpaired_bootstrap_diff(
    y_a: Sequence[int], prob_a: Sequence[float],
    y_b: Sequence[int], prob_b: Sequence[float],
    metric: str = "auroc",
    n_boot: int = 2000,
    seed: int = 0,
    name_a: str = "A",
    name_b: str = "B",
) -> UnpairedResult:
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {sorted(METRICS)}")
    fn = METRICS[metric]

    ya = np.asarray(list(y_a), dtype=int); pa = np.asarray(list(prob_a), dtype=float)
    yb = np.asarray(list(y_b), dtype=int); pb = np.asarray(list(prob_b), dtype=float)

    point_a, point_b = float(fn(ya, pa)), float(fn(yb, pb))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        ia = rng.integers(0, len(ya), size=len(ya))
        ib = rng.integers(0, len(yb), size=len(yb))
        if ya[ia].min() == ya[ia].max() or yb[ib].min() == yb[ib].max():
            continue
        diffs.append(fn(ya[ia], pa[ia]) - fn(yb[ib], pb[ib]))

    diffs = np.asarray(diffs)
    if diffs.size == 0:
        raise RuntimeError("every bootstrap resample was single-class; test sets too small")
    return UnpairedResult(
        metric=metric, name_a=name_a, name_b=name_b,
        point_a=point_a, point_b=point_b, diff=point_a - point_b,
        ci95_low=float(np.percentile(diffs, 2.5)),
        ci95_high=float(np.percentile(diffs, 97.5)),
        n_boot=int(diffs.size),
    )


@dataclass
class SotaAudit:
    dataset: str
    metric: str
    ours: float
    beaten_by: dict[str, float]
    ok: bool

    def message(self) -> str:
        if self.ok:
            return f"[{self.dataset}/{self.metric}] SOTA claim OK: ours={self.ours:.4f} is highest."
        listed = ", ".join(f"{k}={v:.4f}" for k, v in
                           sorted(self.beaten_by.items(), key=lambda kv: -kv[1]))
        return (
            f"[{self.dataset}/{self.metric}] SOTA CLAIM REFUSED: ours={self.ours:.4f} is beaten "
            f"by {len(self.beaten_by)} baseline(s): {listed}. "
            "Do not bold this row and do not describe the gap as a margin in our favour."
        )


def audit_sota_claim(dataset: str, metric: str, ours: float,
                     baselines: dict[str, float]) -> SotaAudit:
    beaten = {k: v for k, v in baselines.items() if v > ours}
    return SotaAudit(dataset=dataset, metric=metric, ours=ours,
                     beaten_by=beaten, ok=not beaten)


def assert_claims(audits: list[SotaAudit], strict: bool = True) -> None:
    failed = [a for a in audits if not a.ok]
    for a in audits:
        print(a.message())
    if failed and strict:
        raise AssertionError(
            f"{len(failed)} unsupported SOTA claim(s). Fix the claim, not the check."
        )
