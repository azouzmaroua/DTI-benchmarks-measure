from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DATASETS_ROOT = "datasets"

SHIPPED = {
    "human":     ("random", "cold"),
    "biosnap":   ("random", "cluster"),
    "bindingdb": ("random", "cluster"),
}
GENERATED = ("cold_drug", "cold_target", "cold_pair")

ALL_PROTOCOLS = ("random", "cold", "cluster") + GENERATED

@dataclass(frozen=True)
class Resolved:
    dataset: str
    protocol: str
    train_file: str
    val_file: str
    test_file: str
    split_mode: str
    note: str = ""

    def as_config_kwargs(self) -> dict:
        return {
            "train_file": self.train_file,
            "val_file": self.val_file,
            "test_file": self.test_file,
            "split_mode": self.split_mode,
        }

def available(dataset: str) -> tuple[str, ...]:
    if dataset not in SHIPPED:
        raise KeyError(f"unknown dataset {dataset!r}; expected one of {sorted(SHIPPED)}")
    return SHIPPED[dataset] + GENERATED

def resolve(dataset: str, protocol: str, root: str = DATASETS_ROOT) -> Resolved:
    dataset = dataset.lower()
    protocol = protocol.lower()
    if dataset not in SHIPPED:
        raise KeyError(f"unknown dataset {dataset!r}; expected one of {sorted(SHIPPED)}")
    if protocol not in ALL_PROTOCOLS:
        raise KeyError(f"unknown protocol {protocol!r}; expected one of {ALL_PROTOCOLS}")

    base = Path(root) / dataset
    full = base / "full.csv"

    if protocol in GENERATED:
        if not full.exists():
            raise FileNotFoundError(f"{full} is required for protocol {protocol!r}")
        return Resolved(dataset, protocol, str(full), "", "", protocol,
                        note=f"generated per seed from {full}")

    if protocol not in SHIPPED[dataset]:
        raise KeyError(
            f"protocol {protocol!r} is not shipped for {dataset!r}; "
            f"available: {available(dataset)}"
        )

    if protocol == "cluster":
        src = base / "cluster" / "source_train.csv"
        test = base / "cluster" / "target_test.csv"
        for p in (src, test):
            if not p.exists():
                raise FileNotFoundError(p)
        return Resolved(dataset, protocol, str(src), "", str(test), "random",
                        note="carve val out of source_train; target_train unused "
                             "(no domain adaptation in this model)")

    sub = base / protocol
    tr, va, te = sub / "train.csv", sub / "val.csv", sub / "test.csv"
    for p in (tr, va, te):
        if not p.exists():
            raise FileNotFoundError(p)
    return Resolved(dataset, protocol, str(tr), str(va), str(te), "given")

def describe_grid() -> str:
    lines = [f"{'dataset':<11} {'protocol':<12} {'source':<10} split_mode"]
    lines.append("-" * 56)
    for ds in SHIPPED:
        for proto in available(ds):
            try:
                r = resolve(ds, proto)
            except (FileNotFoundError, KeyError) as e:
                lines.append(f"{ds:<11} {proto:<12} {'MISSING':<10} {type(e).__name__}")
                continue
            src = "shipped" if proto in SHIPPED[ds] else "full.csv"
            lines.append(f"{ds:<11} {proto:<12} {src:<10} {r.split_mode}")
    return "\n".join(lines)
