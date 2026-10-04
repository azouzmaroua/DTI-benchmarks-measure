from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import prepare_splits
from .dataset import DTIDataset, make_collate, infer_atom_dim
from .metrics import compute as compute_metrics, as_dict
from .model import PharmUnitDTI, total_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def env_record() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "bitwise_reproducible": False,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _loader(df, cfg: Config, shuffle: bool, caches: tuple[dict, dict]) -> DataLoader:
    ds = DTIDataset(df, esm_cache_dir=Path(cfg.esm_cache_dir),
                    p2rank_cache_dir=Path(cfg.p2rank_dir) if cfg.p2rank_dir else None,
                    esm_dim=cfg.esm_dim, max_len_protein=cfg.max_residues,
                    drug_cache=caches[0], protein_cache=caches[1])
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.num_workers, collate_fn=make_collate(cfg.segment_len))


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray, dict]:
    model.eval()
    probs, ys, keep_d, keep_p = [], [], [], []
    for batch in loader:
        batch = _to(batch, device)
        logit, stats = model(batch, return_stats=True)
        probs.append(torch.sigmoid(logit).cpu().numpy())
        ys.append(batch.y.cpu().numpy())
        if stats["drug"].enabled:
            keep_d.append(float(np.prod(stats["drug"].keep_rates)))
            keep_p.append(float(np.prod(stats["protein"].keep_rates)))
    aux = {"keep_drug": float(np.mean(keep_d)) if keep_d else 1.0,
           "keep_protein": float(np.mean(keep_p)) if keep_p else 1.0}
    return np.concatenate(ys), np.concatenate(probs), aux


def _to(batch, device):
    for f in batch.__dataclass_fields__:
        setattr(batch, f, getattr(batch, f).to(device))
    return batch


def run(cfg: Config, seed: int) -> dict:
    cfg.validate()
    set_seed(seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    train_df, val_df, test_df, report = prepare_splits(cfg, seed=seed)
    out = Path(cfg.log_dir) / cfg.tag / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "split_report.txt").write_text(
        str(report) + "\n\n" + str(report.summary), encoding="utf-8")

    caches: tuple[dict, dict] = ({}, {})
    tr = _loader(train_df, cfg, True, caches)
    va = _loader(val_df, cfg, False, caches)
    te = _loader(test_df, cfg, False, caches)

    model = PharmUnitDTI(cfg, infer_atom_dim()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")
    (out / "env.json").write_text(json.dumps(env_record(), indent=2), encoding="utf-8")

    hist_path = out / "history.csv"
    hist = open(hist_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(hist)
    writer.writerow(["epoch", "train_loss", "task_loss", "budget_loss",
                     "val_auroc", "keep_drug", "keep_protein"])

    best_auroc, best_epoch, patience = -1.0, -1, 0
    for epoch in range(cfg.epochs):
        model.train()
        tot = task_s = budget_s = 0.0
        n = 0
        for batch in tr:
            batch = _to(batch, device)
            logit, stats = model(batch, return_stats=True)
            loss, parts = total_loss(logit, batch.y, stats)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tot += float(loss.detach()); task_s += parts["task"]; budget_s += parts["budget"]; n += 1

        y, p, aux = evaluate(model, va, device)
        m = compute_metrics(y, p)
        writer.writerow([epoch, tot / max(n, 1), task_s / max(n, 1), budget_s / max(n, 1),
                         m.auroc, aux["keep_drug"], aux["keep_protein"]])
        hist.flush()
        print(f"epoch {epoch:3d}  loss {tot/max(n,1):.4f}  val_auroc {m.auroc:.4f}  "
              f"keep d/p {aux['keep_drug']:.3f}/{aux['keep_protein']:.3f}")

        if m.auroc > best_auroc:
            best_auroc, best_epoch, patience = m.auroc, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_auroc": m.auroc}, out / "best.pt")
        else:
            patience += 1
            if patience >= cfg.patience:
                print(f"early stop at epoch {epoch} (best {best_epoch}, {best_auroc:.4f})")
                break
    hist.close()

    model.load_state_dict(torch.load(out / "best.pt", map_location=device)["model"])
    y, p, aux = evaluate(model, te, device)
    test_m = compute_metrics(y, p)
    result = {"seed": seed, "tag": cfg.tag, "best_epoch": best_epoch,
              "val_auroc": best_auroc, "test": as_dict(test_m),
              "realised_keep": aux, "split": report.leakage_audit}
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["test"], indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Train PharmUnit-DTI")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--split-mode", default=None,
                    choices=["given", "random", "cold_drug", "cold_target", "cold_pair"])
    ap.add_argument("--no-compress", action="store_true",
                    help="disable compression entirely (accuracy ceiling arm)")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--retention", default=None, choices=["fixed", "learned"])
    ap.add_argument("--recompress", default=None, choices=["none", "context", "attn"])
    ap.add_argument("--budget-weight", type=float, default=None)
    ap.add_argument("--target-keep", type=float, default=None)
    ap.add_argument("--init-keep", type=float, default=None,
                    help="initial retention for the learned arm (default 0.95; must be <= 0.99)")
    ap.add_argument("--keep-drug", type=float, default=None)
    ap.add_argument("--keep-protein", type=float, default=None)
    ap.add_argument("--scorer-drug", default=None, choices=["pharm", "learned"])
    ap.add_argument("--dataset", default=None,
                    choices=["human", "biosnap", "bindingdb", "celegans"])
    ap.add_argument("--protocol", default="random", choices=["random", "cluster", "cold"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    a = ap.parse_args()

    cfg = Config(tag=a.tag)
    if a.dataset:
        root = Path("datasets") / a.dataset / a.protocol
        missing = [f for f in ("train.csv", "val.csv", "test.csv") if not (root / f).exists()]
        if missing:
            ap.error(f"{root} is missing {missing}")
        cfg.dataset, cfg.protocol = a.dataset, a.protocol
        cfg.train_file, cfg.val_file, cfg.test_file = (
            str(root / "train.csv"), str(root / "val.csv"), str(root / "test.csv"))
    if a.keep_drug is not None:    cfg.keep_drug = a.keep_drug
    if a.keep_protein is not None: cfg.keep_protein = a.keep_protein
    if a.scorer_drug:     cfg.scorer_drug = a.scorer_drug
    if a.split_mode:      cfg.split_mode = a.split_mode
    if a.no_compress:     cfg.compress_enabled = False
    if a.rounds:          cfg.compress_rounds = a.rounds
    if a.retention:       cfg.compress_retention = a.retention
    if a.recompress:      cfg.compress_recompress = a.recompress
    if a.budget_weight is not None: cfg.compress_budget_weight = a.budget_weight
    if a.target_keep is not None:   cfg.compress_target_keep = a.target_keep
    if a.init_keep is not None:     cfg.compress_init_keep = a.init_keep
    if a.epochs:          cfg.epochs = a.epochs
    if a.device:          cfg.device = a.device
    if a.batch_size:      cfg.batch_size = a.batch_size

    results = [run(cfg, s) for s in a.seeds]
    aurocs = [r["test"]["auroc"] for r in results]
    print(f"\n{a.tag}: test AUROC {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f} "
          f"over {len(aurocs)} seed(s)")


if __name__ == "__main__":
    main()
