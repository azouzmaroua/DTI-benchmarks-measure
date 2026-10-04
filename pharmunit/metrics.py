from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_curve, roc_auc_score,
    average_precision_score,
)


@dataclass
class MetricScores:
    auroc: float
    auprc: float
    f1: float
    accuracy: float
    threshold: float


def _tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    denom = np.where((prec + rec) > 0, prec + rec, 1.0)
    f1 = 2 * prec * rec / denom
    if len(thr) == 0:
        return 0.5
    return float(thr[int(np.argmax(f1[:-1]))])


def compute(y_true: Iterable[int], y_prob: Iterable[float],
            threshold: float | None = None) -> MetricScores:
    y_true = np.asarray(list(y_true), dtype=int)
    y_prob = np.asarray(list(y_prob), dtype=float)
    if threshold is None:
        threshold = _tune_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return MetricScores(
        auroc=float(roc_auc_score(y_true, y_prob)),
        auprc=float(average_precision_score(y_true, y_prob)),
        f1=float(f1_score(y_true, y_pred)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        threshold=float(threshold),
    )


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, threshold: float,
                 n_boot: int = 1000, seed: int = 0) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aurocs, auprcs, f1s, accs = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        if yt.min() == yt.max():
            continue
        aurocs.append(roc_auc_score(yt, yp))
        auprcs.append(average_precision_score(yt, yp))
        pred = (yp >= threshold).astype(int)
        f1s.append(f1_score(yt, pred, zero_division=0))
        accs.append(accuracy_score(yt, pred))

    def ci(vals: list[float]) -> tuple[float, float]:
        if not vals:
            return (float("nan"), float("nan"))
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

    return {"auroc": ci(aurocs), "auprc": ci(auprcs), "f1": ci(f1s), "accuracy": ci(accs)}


def as_dict(m: MetricScores) -> dict:
    return asdict(m)
