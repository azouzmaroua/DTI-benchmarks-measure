from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

NUM = r"[-+0-9.eE]+|nan"
TEST_RE = re.compile(
    rf"Test at Best Model of Epoch (?P<epoch>\d+) with test loss (?:{NUM})\s+"
    rf"AUROC (?P<auroc>{NUM})\s+AUPRC (?P<auprc>{NUM})\s+"
    rf"Sensitivity (?P<printed_sens>{NUM})\s+Specificity (?P<printed_spec>{NUM})\s+"
    rf"Accuracy (?P<acc>{NUM})")
VAL_RE = re.compile(
    rf"Validation at Epoch (?P<epoch>\d+) with validation loss (?:{NUM})\s+"
    rf"AUROC (?P<auroc>{NUM})\s+AUPRC (?P<auprc>{NUM})")
SEED_RE = re.compile(r"'SEED': (\d+)")
CFG_RE = re.compile(r"Config yaml: (\S+)")
DEFAULT_SEED = 42

DATASETS = ("human", "biosnap", "bindingdb")
SPLITS = ("random", "cold", "cluster")
CROSS_DOMAIN = {"cluster"}

SETTINGS = {
    ("human", "random"): ("reference_on_published_splits.json", "test.csv"),
    ("human", "cold"): ("reference_drugban_cold.json", "test.csv"),
    ("biosnap", "random"): ("reference_on_published_splits.json", "test.csv"),
    ("bindingdb", "random"): ("reference_on_published_splits.json", "test.csv"),
    ("biosnap", "cluster"): ("reference_drugban_cluster.json", "target_test.csv"),
    ("bindingdb", "cluster"): ("reference_drugban_cluster.json", "target_test.csv"),
}
PUBLISHED = {
    ("human", "random"): ("figure2", "random"),
    ("human", "cold"): ("figure2", "cold"),
    ("biosnap", "random"): ("table2", "biosnap_random"),
    ("bindingdb", "random"): ("table2", "bindingdb_random"),
    ("biosnap", "cluster"): ("table2", "biosnap_cluster"),
    ("bindingdb", "cluster"): ("table2", "bindingdb_cluster"),
}


def _display(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def parse_log_text(text: str) -> dict | None:
    tests = list(TEST_RE.finditer(text))
    val = {int(m["epoch"]): float(m["auroc"]) for m in VAL_RE.finditer(text)}
    if not tests and not val:
        return None
    seed, cfg = SEED_RE.search(text), CFG_RE.search(text)
    return {"test": tests[-1].groupdict() if tests else None,
            "val_auroc_by_epoch": val,
            "seed": int(seed.group(1)) if seed else None,
            "config": cfg.group(1) if cfg else None}


def load_runs(log_dir: Path, transcription: Path) -> dict:
    runs: dict[tuple[str, str, int], dict] = {}

    def place(key: tuple[str, str, int], payload: dict) -> None:
        old = runs.get(key)
        if old is not None:
            raise SystemExit(
                f"\nTwo runs both claim {key[0]} {key[1]} seed {key[2]}:\n"
                f"  {old['source']}  (seed from {old['seed_from']})\n"
                f"  {payload['source']}  (seed from {payload['seed_from']})\n"
                "Refusing to guess which to keep, because dropping one would corrupt the\n"
                "seed mean and SD. Give each run an explicit seed: put _s<seed> in the log\n"
                "file name, or a \"seed\" field in its transcription entry.")
        runs[key] = payload

    if log_dir.exists():
        for f in sorted(log_dir.glob("*.log")):
            tokens = re.split(r"[_\-.]", f.stem.lower())
            ds = next((t for t in tokens if t in DATASETS), None)
            split = next((t for t in tokens if t in SPLITS), None)
            parsed = parse_log_text(f.read_text(encoding="utf-8", errors="ignore"))
            if not (ds and split and parsed):
                continue
            if parsed["seed"] is not None:
                seed, how = parsed["seed"], "log header"
            else:
                tok = next((t for t in tokens if re.fullmatch(r"s\d+", t)), None)
                seed, how = (int(tok[1:]), "file name") if tok else (DEFAULT_SEED, "assumed (config default)")
            place((ds, split, seed), {**parsed, "seed": seed, "seed_from": how, "source": _display(f)})
    if transcription.exists():
        payload = json.loads(transcription.read_text(encoding="utf-8"))
        for key, entry in payload.get("runs", {}).items():
            parts = key.split("/")
            ds, split = parts[0], parts[1]
            seed = int(entry.get("seed", DEFAULT_SEED))
            parsed = parse_log_text("\n".join([entry["test_line"], *entry.get("validation_lines", [])]))
            if not parsed:
                continue
            payload = {**parsed, "seed": seed, "seed_from": "transcription",
                       "source": f"{transcription.name} (transcribed)"}
            existing = runs.get((ds, split, seed))
            if existing is None:
                place((ds, split, seed), payload)
            elif existing["test"] is None and parsed["test"] is not None:
                print(f"note: {existing['source']} is still running ({len(existing['val_auroc_by_epoch'])} "
                      f"epochs); keeping the finished {ds} {split} seed {seed} result from {payload['source']}")
                runs[(ds, split, seed)] = payload
            elif existing["seed_from"] == "assumed (config default)":
                raise SystemExit(
                    f"\n{existing['source']} has no seed in its header or file name, so it was assumed\n"
                    f"to be seed {DEFAULT_SEED}, which collides with the transcribed {ds} {split} seed {seed}\n"
                    f"({payload['source']}). Refusing to guess. Rename the log with _s<seed>.")
    return runs


def load_reference(artifact: str, dataset: str) -> dict | None:
    f = RUNS / artifact
    if not f.exists():
        return None
    for row in json.loads(f.read_text(encoding="utf-8")):
        if row.get("dataset") == dataset and row.get("source") == "prepared":
            return row
    return None


def load_published() -> dict:
    f = RUNS / "drugban_published_values.json"
    if not f.exists():
        return {}
    p = json.loads(f.read_text(encoding="utf-8"))
    out = {}
    for key, (where, col) in PUBLISHED.items():
        if where == "figure2":
            v = p["figure2_human_mean_sd_5_runs"]["AUROC"][col]["DrugBAN"]
            out[key] = (v["mean"], v["sd"])
        else:
            v = p["table2_auroc_mean_sd_5_runs"]["DrugBAN"][col]
            out[key] = tuple(v) if v else None
    return out


def test_file_counts(ds: str, split: str, fname: str, ref: dict) -> tuple[int, int, str]:
    for p in (ROOT / "datasets" / ds / split / fname, Path(ref["files"]["test"])):
        if p.exists():
            df = pd.read_csv(p)
            y = df["Y" if "Y" in df.columns else "label"].astype(float)
            return int(len(df)), int((y == 1).sum()), "file"
    n = int(ref["n_test"])
    return n, round(float(ref["test_positive_rate"]) * n), "reference artifact"


def class_size_of_first_rate(r1: float, r2: float, acc: float, n: int) -> int | None:
    if not all(math.isfinite(v) for v in (r1, r2, acc)) or abs(r1 - r2) < 1e-12:
        return None
    a = round(n * (acc - r2) / (r1 - r2))
    b = n - a
    if a <= 0 or b <= 0:
        return None
    k1, k2 = round(r1 * a), round(r2 * b)
    exact = (abs(k1 / a - r1) < 1e-9 and abs(k2 / b - r2) < 1e-9
             and abs((k1 + k2) / n - acc) < 1e-9)
    return a if exact else None


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (auc * (1.0 - auc) + (n_pos - 1) * (q1 - auc * auc)
           + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg)
    return math.sqrt(max(var, 0.0))


def margin_ci(auc: float, ref: float, constant: bool, n_pos: int, n_neg: int
              ) -> tuple[float, float]:
    se_d = hanley_mcneil_se(auc, n_pos, n_neg)
    if constant:
        return auc - 1.96 * se_d - ref, auc + 1.96 * se_d - ref
    se = math.sqrt(se_d ** 2 + hanley_mcneil_se(ref, n_pos, n_neg) ** 2)
    return auc - ref - 1.96 * se, auc - ref + 1.96 * se


def identity(run: dict, ref: dict, n: int, n_pos: int) -> tuple[bool, str]:
    n_neg = n - n_pos
    if int(ref["n_test"]) != n or abs(n_pos / n - float(ref["test_positive_rate"])) > 5e-5 + 1e-12:
        return False, (f"reference was computed on {ref['n_test']} rows at positive rate "
                       f"{ref['test_positive_rate']}, file has {n} rows / {n_pos} positives")
    t = run["test"]
    size = class_size_of_first_rate(float(t["printed_sens"]), float(t["printed_spec"]),
                                    float(t["acc"]), n)
    if size is None:
        return False, "DrugBAN's printed rates do not reproduce this file's size"
    if n_pos == n_neg and size == n_pos:
        return True, "passed (balanced file: either rate convention)"
    if size == n_neg:
        return True, "passed; printed 'Sensitivity' is the true-negative rate (DrugBAN code)"
    if size == n_pos:
        return True, "passed; printed rates follow the standard convention"
    return False, f"DrugBAN's rates fit classes of {size}/{n - size}, file has {n_pos}/{n_neg}"


def welch(m1: float, s1: float, n1: int, m2: float, s2: float, n2: int) -> tuple[float, float, float | None]:
    v1, v2 = s1 * s1 / n1, s2 * s2 / n2
    t = (m1 - m2) / math.sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / (v1 * v1 / (n1 - 1) + v2 * v2 / (n2 - 1))
    try:
        from scipy.stats import t as student_t
        return t, df, float(2 * student_t.sf(abs(t), df))
    except ImportError:
        return t, df, None


def evaluate_run(run: dict, ds: str, split: str, ref: dict, n: int, n_pos: int, counted_from: str,
                 label: str) -> dict:
    n_neg = n - n_pos
    auc = float(run["test"]["auroc"])
    ok, note = identity(run, ref, n, n_pos)
    base = {"dataset": ds, "split": split, "seed": run["seed"], "seed_from": run["seed_from"],
            "config": run.get("config"), "drugban_auroc": auc, "drugban_source": run["source"],
            "identity_check": note}
    if not ok:
        print(f"{label:<19}{run['seed']:>5}{n:>7}   IDENTITY CHECK FAILED -- {note}; no margin reported")
        return {**base, "identity_passed": False}

    r = float(ref["reference_validation_fixed"])
    constant = ref.get("selected_family") == "none"
    lo, hi = margin_ci(auc, r, constant, n_pos, n_neg)
    print(f"{label:<19}{run['seed']:>5}{n:>7}{f'{n_pos}/{n_neg}':>10}{r:>10.4f}{auc:>9.4f}{auc - r:>+9.4f}"
          f"{f'[{lo:+.4f}, {hi:+.4f}]':>24}")
    row = {**base, "identity_passed": True, "n_test": n, "n_pos": n_pos, "n_neg": n_neg,
           "counts_from": counted_from, "reference": r, "reference_constant": constant,
           "drugban_auprc": float(run["test"]["auprc"]), "drugban_best_epoch": int(run["test"]["epoch"]),
           "margin": auc - r, "margin_ci95": [lo, hi],
           "interval": "exact shift of DrugBAN's interval" if constant else "unpaired, conservative",
           "checkpoint_selected_on": "test fold" if split in CROSS_DOMAIN else "val.csv"}

    if split in CROSS_DOMAIN:
        val = run["val_auroc_by_epoch"]
        if val:
            last = max(val)
            best_epoch, best = None, -1.0
            for e in sorted(val):
                if val[e] >= best:
                    best_epoch, best = e, val[e]
            f_auc = val[last]
            f_lo, f_hi = margin_ci(f_auc, r, constant, n_pos, n_neg)
            equal = abs(best - auc) < 1e-6
            print(f"{'':<24}checkpoint chosen on the TEST fold: reported = per-epoch max "
                  f"{'(confirmed)' if equal else f'(NOT equal: max {best:.4f})'}, epoch {best_epoch}")
            print(f"{'  final epoch ' + str(last):<24}{'':>17}{r:>10.4f}{f_auc:>9.4f}{f_auc - r:>+9.4f}"
                  f"{f'[{f_lo:+.4f}, {f_hi:+.4f}]':>24}   no test-label selection")
            row.update({"per_epoch_max_equals_reported": equal, "per_epoch_max_epoch": best_epoch,
                        "final_epoch": last, "final_epoch_auroc": f_auc,
                        "final_epoch_margin": f_auc - r, "final_epoch_margin_ci95": [f_lo, f_hi],
                        "selection_gain": auc - f_auc})
        else:
            print(f"{'':<24}checkpoint chosen on the TEST fold; no per-epoch lines here, "
                  f"so no selection-free figure -- supply the full log")
    return row


def summarise(rows: list[dict], pub: tuple | None, ds: str, split: str) -> dict:
    aucs = [r["drugban_auroc"] for r in rows]
    k = len(aucs)
    mean, sd = statistics.mean(aucs), statistics.stdev(aucs)
    out = {"dataset": ds, "split": split, "seeds": [r["seed"] for r in rows], "n_seeds": k,
           "auroc_mean": mean, "auroc_sd": sd, "margin_mean": mean - rows[0]["reference"]}
    line = f"{'':<24}{k} seeds: DrugBAN {mean:.4f} +/- {sd:.4f} (sample SD)"
    if pub:
        t, df, p = welch(mean, sd, k, pub[0], pub[1], 5)
        out.update({"published_mean": pub[0], "published_sd": pub[1], "difference_vs_published": mean - pub[0],
                    "welch_t": t, "welch_df": df, "welch_p": p})
        line += (f" | published {pub[0]:.3f} +/- {pub[1]:.3f} (5 runs): difference {mean - pub[0]:+.4f}, "
                 f"Welch t={t:.2f}" + (f", p={p:.3f}" if p is not None else ""))
    print(line)
    withlog = [r for r in rows if "final_epoch_auroc" in r]
    if split in CROSS_DOMAIN and len(withlog) > 1:
        finals = [r["final_epoch_auroc"] for r in withlog]
        gains = [r["selection_gain"] for r in withlog]
        reported = [r["drugban_auroc"] for r in withlog]
        out.update({"final_epoch_n_seeds": len(withlog),
                    "final_epoch_seeds": [r["seed"] for r in withlog],
                    "final_epoch_mean": statistics.mean(finals), "final_epoch_sd": statistics.stdev(finals),
                    "reported_mean_same_seeds": statistics.mean(reported),
                    "selection_gain_mean": statistics.mean(gains), "selection_gain_sd": statistics.stdev(gains),
                    "best_epochs": [r["per_epoch_max_epoch"] for r in withlog]})
        note = "" if len(withlog) == k else f" (the {len(withlog)} of {k} seeds whose full log is present)"
        print(f"{'':<24}final epoch {out['final_epoch_mean']:.4f} +/- {out['final_epoch_sd']:.4f} vs reported "
              f"{out['reported_mean_same_seeds']:.4f}{note}: test-fold selection adds "
              f"{out['selection_gain_mean']:+.4f} +/- {out['selection_gain_sd']:.4f}; best epochs {out['best_epochs']}")
    elif split in CROSS_DOMAIN:
        print(f"{'':<24}final-epoch average needs at least two full logs ({len(withlog)} of {k} available)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-dir", default="runs/drugban_logs",
                    help="Directory holding DrugBAN logs named like drugban_<dataset>_<split>[_s<seed>].log.")
    ap.add_argument("--transcription", default="runs/drugban_test_lines.json",
                    help="Fallback: log lines transcribed from terminal output.")
    ap.add_argument("--out", default="runs/drugban_calibration.json")
    a = ap.parse_args()

    resolve = lambda s: Path(s) if Path(s).is_absolute() else ROOT / s
    runs = load_runs(resolve(a.log_dir), resolve(a.transcription))
    published = load_published()

    rule = "=" * 112
    print(rule)
    print("DrugBAN (first-hand) versus the entity-frequency reference, on DrugBAN's own split files")
    print(rule)
    print(f"{'setting':<19}{'seed':>5}{'n_test':>7}{'pos/neg':>10}{'reference':>10}{'DrugBAN':>9}"
          f"{'margin':>9}{'95% CI (Hanley-McNeil)':>24}")
    print("-" * 112)

    rows, summaries = [], []
    for (ds, split), (artifact, fname) in SETTINGS.items():
        label = f"{ds} {split}"
        ref = load_reference(artifact, ds)
        pub = published.get((ds, split))
        pub_s = f"published {pub[0]:.3f} +/- {pub[1]:.3f}" if pub else "published value not loaded"
        if ref is None:
            print(f"{label:<19}reference artifact runs/{artifact} missing")
            continue
        seeds = sorted(s for (d, sp, s) in runs if d == ds and sp == split)
        if not seeds:
            print(f"{label:<19}DrugBAN not run yet  (reference {ref['reference_validation_fixed']:.4f}; {pub_s})")
            continue
        n, n_pos, counted_from = test_file_counts(ds, split, fname, ref)
        done = []
        for seed in seeds:
            run = runs[(ds, split, seed)]
            if run["test"] is None:
                print(f"{label:<19}{seed:>5}  in progress: {len(run['val_auroc_by_epoch'])} epochs logged")
                continue
            row = evaluate_run(run, ds, split, ref, n, n_pos, counted_from, label)
            rows.append(row)
            if row["identity_passed"]:
                done.append(row)
        if done:
            print(f"{'':<24}identity {done[0]['identity_check']}; counts from {counted_from}; {pub_s}")
        if len(done) > 1:
            summaries.append(summarise(done, pub, ds, split))

    print()
    print("Per-run intervals: Hanley-McNeil approximate, from class counts (test-set sampling")
    print("uncertainty). Seed-to-seed variation is the SD across seeds. Unpaired intervals are conservative.")

    out = resolve(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "seed_summaries": summaries}, indent=2), encoding="utf-8")
    print(f"\nwrote {_display(out)}")


if __name__ == "__main__":
    main()
