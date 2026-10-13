from __future__ import annotations

import contextlib
import csv
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = ROOT / "paper" / "supp_tables"
sys.path.insert(0, str(ROOT))

BENCH = {"human": "Human", "celegans": r"\textit{C.\ elegans}", "biosnap": "BioSNAP",
         "bindingdb": "BindingDB"}
PROTO = {"random": "random", "cold_drug": "cold-drug", "cold_target": "cold-target",
         "cold_pair": "cold-pair"}
DATASETS = ("human", "celegans", "biosnap", "bindingdb")
PROTOCOLS = ("random", "cold_drug", "cold_target", "cold_pair")
TOLERANCE = 0.15

TABLE6 = {
    "human/random": (0.9893, 0.9599, -0.0294, (-0.0375, -0.0214), 5),
    "human/cold_drug": (0.9228, 0.9143, -0.0085, (-0.0379, +0.0209), 5),
    "human/cold_target": (0.9435, 0.8485, -0.0951, (-0.2257, +0.0356), 5),
    "human/cold_pair": (0.5000, 0.7046, +0.2046, (+0.0350, +0.3743), 5),
    "celegans/random": (0.9943, 0.9619, -0.0324, (-0.0368, -0.0280), 5),
    "celegans/cold_drug": (0.8864, 0.8064, -0.0800, (-0.3723, +0.2124), 5),
    "celegans/cold_target": (0.9715, 0.8854, -0.0861, (-0.1512, -0.0210), 5),
    "celegans/cold_pair": (0.5000, 0.5662, +0.0662, (-0.0421, +0.1745), 5),
    "biosnap/random": (0.8276, 0.9162, +0.0885, (+0.0812, +0.0959), 5),
    "biosnap/cold_drug": (0.8197, 0.8577, +0.0380, (+0.0176, +0.0583), 5),
    "biosnap/cold_target": (0.6684, 0.8275, +0.1591, (+0.1219, +0.1962), 5),
    "biosnap/cold_pair": (0.5000, 0.7831, +0.2831, (+0.2526, +0.3136), 5),
    "bindingdb/random": (0.9022, 0.9029, +0.0007, (-0.0075, +0.0089), 5),
    "bindingdb/cold_drug": (0.9017, 0.9109, +0.0092, (-0.0185, +0.0369), 5),
    "bindingdb/cold_target": (0.5420, 0.6627, +0.1208, (-0.0523, +0.2938), 5),
    "bindingdb/cold_pair": (0.5000, 0.6468, +0.1468, (+0.1148, +0.1788), 5),
}
PAIRED_SEEDS = range(5)
LOG_ONLY: dict = {}

def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def f4(x: float) -> str:
    return f"{x:.4f}"

def sgn4(x: float) -> str:
    return f"${x:+.4f}$"

def ci(lo: float, hi: float) -> str:
    return f"$[{lo:+.4f}, {hi:+.4f}]$"

def pfmt(p: float) -> str:
    if p != p:
        return "---"
    return f"{p:.4f}" if p < 0.01 else f"{p:.3f}"

def write(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(body, encoding="utf-8")
    print(f"wrote {(OUT / name).relative_to(ROOT)}")

def tabular(caption: str, label: str, spec: str, header: str, rows: list[str],
            number: int, size: str = r"\small") -> str:
    lines = [r"\setcounter{table}{%d}" % (number - 1), r"\begin{table}[p]",
             r"\caption{" + caption + r"}\label{" + label + "}", r"\centering", size,
             r"\begin{tabular}{@{}" + spec + r"@{}}", r"\toprule", header + r" \\", r"\midrule"]
    lines += rows
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)

def paired(d: np.ndarray) -> tuple[float, float, float, float]:
    m = float(d.mean())
    if len(d) < 2:
        return m, float("nan"), float("nan"), float("nan")
    lo, hi = stats.t.interval(0.95, len(d) - 1, loc=m, scale=stats.sem(d))
    return m, float(lo), float(hi), float(stats.ttest_1samp(d, 0.0).pvalue)

def logged_runs(log: str, tag: str) -> list[list[float]]:
    runs, cur = [], None
    for line in (RUNS / "autodl_logs" / log).read_text(encoding="utf-8", errors="replace").splitlines():
        for m in re.finditer(r"pharmunit\.train(.*?)--tag (\S+)", line):
            cur = [] if m.group(2) == tag else None
            if cur is not None:
                runs.append(cur)
        m = re.match(r'\s*"auroc": ([0-9.]+),', line)
        if m and cur is not None:
            cur.append(float(m.group(1)))
    return [r for r in runs if r]

def model_runs(tag: str) -> dict[int, dict]:
    out = {}
    for m in sorted((RUNS / tag).glob("seed*/metrics.json")):
        out[int(m.parent.name[4:])] = jload(m)
    return out

def references(payload: dict, cell: str) -> dict[int, dict]:
    return {int(r["seed"]): r for r in payload["cells"][cell]}

def split_report_counts(tag: str, seed: int) -> dict[str, tuple[int, int | None]] | None:
    f = RUNS / tag / f"seed{seed}" / "split_report.txt"
    if not f.exists():
        return None
    out = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*\d+\s+(train|val|test)\s+(\d+)\s+(\d+|\.\.\.)\s+", line)
        if m:
            out[m.group(1)] = (int(m.group(2)), None if m.group(3) == "..." else int(m.group(3)))
    return out

def split_report_targets(tag: str, seed: int) -> dict[str, int]:
    f = RUNS / tag / f"seed{seed}" / "split_report.txt"
    if not f.exists():
        return {}
    text = f.read_text(encoding="utf-8")
    if not re.search(r"split\s+\.\.\.\s+n_targets", text):
        return {}
    return {m.group(1): int(m.group(2))
            for m in re.finditer(r"\s*\d+\s+(train|val|test)\s+\.\.\.\s+(\d+)\s*$", text, re.M)}

def draw_split(dataset: str, mode: str, seed: int):
    from pharmunit.config import Config
    from pharmunit.data import prepare_splits
    src = ROOT / "datasets" / dataset / "random"
    cfg = Config(dataset=dataset, protocol="random", train_file=str(src / "train.csv"),
                 val_file=str(src / "val.csv"), test_file=str(src / "test.csv"),
                 scorer_protein="esm", p2rank_dir="", split_mode=mode, seed=seed)
    with contextlib.redirect_stdout(io.StringIO()):
        tr, va, te, _ = prepare_splits(cfg, seed=seed)
    return tr, va, te

def split_balance(purity: dict, manifest: dict) -> dict:
    cache = RUNS / "split_balance.json"
    if cache.exists():
        return jload(cache)
    out = {}
    for ds in DATASETS:
        pool = purity["purity"][ds]["pool_positive_rate"]
        for pr in PROTOCOLS:
            cell = f"{ds}/{pr}"
            tag = manifest.get(cell)
            seeds = set(range(5)) | (set(model_runs(tag)) if tag else set())
            rows = {}
            for s in sorted(seeds):
                tr, va, te = draw_split(ds, pr, s)
                folds = {}
                for name, fr in (("train", tr), ("val", va), ("test", te)):
                    rate = float(fr["label"].mean())
                    folds[name] = {"n": int(len(fr)), "n_pos": int(fr["label"].sum()),
                                   "rate": rate, "drift": abs(rate - pool)}
                rows[str(s)] = folds
                print(f"  split {cell} seed {s}: max drift "
                      f"{max(v['drift'] for v in folds.values()):.3f}")
            out[cell] = {"pool_positive_rate": pool, "seeds": rows}
    cache.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {cache.relative_to(ROOT)}")
    return out

def s2(balance: dict, manifest: dict) -> None:
    rows, mismatches, checked, sizes_only, extra_note = [], [], 0, set(), {}
    targets_only = set()
    for ds in DATASETS:
        for pr in PROTOCOLS:
            cell = f"{ds}/{pr}"
            b = balance[cell]
            tag = manifest.get(cell)
            model_seeds = sorted(model_runs(tag)) if tag else []
            for s in model_seeds:
                rep = split_report_counts(tag, s)
                got = b["seeds"].get(str(s))
                if rep is None or got is None:
                    continue
                if not rep:
                    tg = split_report_targets(tag, s)
                    if not tg:
                        continue
                    tr, va, te = draw_split(ds, pr, s)
                    red = {"train": tr["target"].nunique(), "val": va["target"].nunique(),
                           "test": te["target"].nunique()}
                    checked += 1
                    targets_only.add(f"{cell} seed {s}")
                    for fold, n_t in red.items():
                        if tg.get(fold) != n_t:
                            mismatches.append(f"{cell} seed {s} {fold}: report {tg.get(fold)} targets, "
                                              f"redrawn {n_t}")
                    continue
                checked += 1
                for fold in ("train", "val", "test"):
                    n, npos = rep.get(fold, (None, None))
                    if n != got[fold]["n"] or (npos is not None and npos != got[fold]["n_pos"]):
                        mismatches.append(f"{cell} seed {s} {fold}: report {rep.get(fold)}, "
                                          f"redrawn {(got[fold]['n'], got[fold]['n_pos'])}")
                    if npos is None:
                        sizes_only.add(f"{cell} seed {s}")
            cells = []
            for s in range(5):
                folds = b["seeds"][str(s)]
                worst = max(folds, key=lambda k: folds[k]["drift"])
                d = folds[worst]["drift"]
                mark = r"$^{\dagger}$" if d > TOLERANCE else ""
                cells.append(f"{d:.3f}{mark}")
            if tag is None:
                used = "---"
            else:
                used = ", ".join(str(s) for s in model_seeds if s in PAIRED_SEEDS)
                extra = [s for s in model_seeds if s not in PAIRED_SEEDS]
                if extra:
                    flagged = sum(max(f["drift"] for f in b["seeds"][str(s)].values()) > TOLERANCE
                                  for s in extra)
                    extra_note[cell] = (min(extra), max(extra), flagged)
            rows.append(f"{BENCH[ds]} & {PROTO[pr]} & {b['pool_positive_rate']:.4f} & "
                        + " & ".join(cells) + f" & {used} \\\\")
        rows.append(r"\addlinespace")
    rows = rows[:-1]
    assert checked == 85, checked
    print(f"S2: {checked} retained runs checked against their split reports, "
          f"{len(mismatches)} mismatch(es); fold sizes only for {sorted(sizes_only)}")
    for m in mismatches:
        print("  MISMATCH", m)
    if mismatches:
        raise SystemExit("a retained run was not trained on the split redrawn for its seed")
    extra_txt = "".join(
        f" {BENCH[c.split('/')[0]]} {PROTO[c.split('/')[1]]} was also run at seeds {lo}--{hi} for "
        f"Supplementary Table~S7; {'none' if n == 0 else n} of those {hi - lo + 1} seeds "
        f"{'is' if n == 1 else 'are'} outside the tolerance."
        for c, (lo, hi, n) in extra_note.items())
    cap = (r"Per-fold label drift by cell and seed: for each seed, the largest absolute "
           r"difference between a fold's positive rate and the pooled positive rate, over "
           r"the training, validation and test folds. $^{\dagger}$ marks a seed whose drift "
           r"exceeds the 0.15 tolerance at which the pipeline warns but continues "
           r"(\texttt{pharmunit/splits.py}, \texttt{check\_fold\_balance}). ``Model seeds'' "
           r"are the seeds paired in Table~6 of the main text." + extra_txt + " Every "
           f"split report held for a model run ({checked} runs: seeds 0--4 of each of the 16 cells and "
           r"seeds 5--9 of Human cold-pair) describes the same fold sizes "
           r"and positive counts as the split redrawn here for its seed"
           + (f" ({len(sizes_only)} report{'s' if len(sizes_only) > 1 else ''} printed "
              r"fold sizes only, which match" if sizes_only else "")
           + (("; " if sizes_only else " (") + f"{len(targets_only)} printed only the number of "
              r"distinct targets per fold, which also matches" if targets_only else "")
           + (")" if sizes_only or targets_only else "")
           + r". Source: \texttt{runs/split\_balance.json}, redrawn by "
           r"\texttt{scripts/make\_supplement\_tables.py}.")
    write("s2.tex", tabular(cap, "tbl:s2", "LLRRRRRRL",
                            r"Benchmark & Protocol & Pool & s0 & s1 & s2 & s3 & s4 & Model seeds",
                            rows, 2, r"\footnotesize"))

def s3_and_table6(purity: dict, manifest: dict, balance: dict) -> dict:
    rows, recomputed = [], {}
    cd = balance["celegans/cold_drug"]["seeds"]
    n_out = sum(max(f["drift"] for f in cd[str(s)].values()) > TOLERANCE for s in range(5))
    n_word = "all five of its" if n_out == 5 else \
        {4: "four", 3: "three", 2: "two", 1: "one", 0: "none"}[n_out] + " of its five"
    for cell in TABLE6:
        ds, pr = cell.split("/")
        tag = manifest[cell]
        runs = {s: r for s, r in model_runs(tag).items() if s in PAIRED_SEEDS}
        refs = references(purity, cell)
        margins = []
        first = True
        for s in sorted(runs):
            model = float(runs[s]["test"]["auroc"])
            if s in refs:
                ref = 0.5 if refs[s]["val_fixed"] is None else float(refs[s]["val_fixed"])
            elif pr == "cold_pair":
                ref = 0.5
            else:
                continue
            margins.append(model - ref)
            lead = f"{BENCH[ds]} & {PROTO[pr]} & \\texttt{{{tag.replace('_', chr(92) + '_')}}}" \
                if first else " & & "
            first = False
            rows.append(f"{lead} & {s} & {f4(ref)} & {f4(model)} & {sgn4(model - ref)} \\\\")
        d = np.array(margins)
        m, lo, hi, _p = paired(d)
        ref_mean = float(np.mean([float(refs[s]["val_fixed"])
                                  if s in refs and refs[s]["val_fixed"] is not None else 0.5
                                  for s in sorted(runs)]))
        mod_mean = float(np.mean([float(runs[s]["test"]["auroc"]) for s in sorted(runs)]))
        recomputed[cell] = (ref_mean, mod_mean, m, (lo, hi), len(d))
        rows.append(r"\addlinespace")
    for cell, (mean, sd, n) in LOG_ONLY.items():
        ds, pr = cell.split("/")
        ref = float(np.mean([r["val_fixed"] for r in purity["cells"][cell]]))
        rows.append(f"{BENCH[ds]} & {PROTO[pr]} & \\textit{{log}} & mean of {n} & {f4(ref)} & "
                    f"{mean:.4f} $\\pm$ {sd:.4f} & {sgn4(mean - ref)} \\\\")
    first_runs = {t: ", ".join(f"{float(r['test']['auroc']):.4f}" for _, r in sorted(model_runs(t).items()))
                  for t in ("recursive_cd", "recursive_ct")}
    bs = [r for r in logged_runs("biosnap.txt", "bs_recursive") if len(r) == 5]
    bs_files = [float(model_runs("bs_recursive")[s]["test"]["auroc"]) for s in range(5)]
    bs_other = [r for r in bs if not np.allclose(r, bs_files)]
    assert len(bs_other) == 1 and any(np.allclose(r, bs_files) for r in bs), "bs_recursive log/file mismatch"
    ce_joined = [x for r in logged_runs("celegans.txt", "celegans_random") for x in r]
    assert len(ce_joined) == 5
    fmt5 = lambda xs: ", ".join(f"{x:.4f}" for x in xs)
    other_runs = (f"Two other cells of Table 6 were also run twice: BioSNAP cold-pair, whose first run gave "
                  f"{fmt5(bs_other[0])} (the reported second run is in \\texttt{{runs/bs\\_recursive}}), and "
                  f"\\textit{{C.\\ elegans}} random, which gave {fmt5(ce_joined)} on the machine of the "
                  f"first \\textit{{C.\\ elegans}} cold-pair run (the reported laptop run is in "
                  f"\\texttt{{runs/ce\\_random}}); logs are in \\texttt{{runs/autodl\\_logs}}. ")
    hcp = model_runs(manifest["human/cold_pair"])
    hcp_extra = [float(hcp[s]["test"]["auroc"]) for s in sorted(hcp) if s not in PAIRED_SEEDS]
    hcp_m, hcp_lo, hcp_hi, _ = paired(np.array([float(hcp[s]["test"]["auroc"]) - 0.5 for s in sorted(hcp)]))
    hu0 = float(jload(RUNS / "hu_random_seed0_original_metrics.json")["test"]["auroc"])
    hu0_new = float(model_runs(manifest["human/random"])[0]["test"]["auroc"])
    v2_0 = float(model_runs("ce_cold_drug_v2")[0]["test"]["auroc"])
    v3_0 = float(model_runs(manifest["celegans/cold_drug"])[0]["test"]["auroc"])
    cap = (r"Per-seed model AUROC, reference and margin for every cell of Table~6 of the main "
           r"text, of which this is the per-seed form (seeds 0--4). The reference is the "
           r"validation-fixed construction at $k = 5$ (\texttt{runs/entity\_purity.json}); under "
           r"cold-pair it is 0.5000 at every seed by construction. Model values are "
           r"\texttt{test.auroc} in \texttt{runs/<tag>/seed<k>/metrics.json}; each run directory's "
           r"\texttt{PROVENANCE.md} records how and where it was run. "
           f"Human cold-pair was also run at seeds 5--9 for Supplementary Table~S7 "
           f"({fmt5(hcp_extra)}); over all ten seeds its margin is ${hcp_m:+.4f}$ "
           f"$[{hcp_lo:+.4f}, {hcp_hi:+.4f}]$. "
           f"Human random seed 0 was rerun after a duplicate launch rewrote its files; the first run "
           f"gave {hu0:.4f} against the rerun's {hu0_new:.4f} (\\texttt{{runs/hu\\_random\\_seed0\\_original\\_metrics.json}}). "
           r"\textit{C.\ elegans} cold-pair is the rerun described in Section~4.5 of the main text; "
           r"the first run of that cell, kept only as a training log "
           r"(\texttt{runs/ladder\_manifest.json}), gave 0.6426, 0.6358, 0.7062, 0.5601 and 0.4857 "
           r"at seeds 0--4, mean 0.6061 (margin $+0.1061$ $[+0.0007, +0.2115]$), and an earlier rerun of "
           r"seeds 3 and 4 on that machine gave "
           r"0.5705 and 0.4781 (\texttt{runs/autodl\_logs/celegans.txt}); its seed-0 configuration "
           r"and history files were overwritten by a restart that crashed before training, and its "
           r"seed-0 \texttt{metrics.json} is the original. \textit{C.\ elegans} cold-drug "
           f"(\\texttt{{runs/ce\\_cold\\_drug\\_v3}}) has {n_word} splits outside the label-balance "
           r"tolerance (Supplementary Table~S2); without seed 1, where the model scores below 0.5 in both "
           r"runs, its margin is $+0.0233$ $[-0.0610, +0.1076]$, still spanning zero. Its first run survives only as a training log "
           r"(0.9012, 0.4494, 0.9762, 0.8682 and 0.9135 at seeds 0--4, mean 0.8217; "
           r"\texttt{runs/autodl\_logs/celegans.txt}), and an interrupted second attempt "
           f"(\\texttt{{runs/ce\\_cold\\_drug\\_v2}}) gave {v2_0:.4f} at seed 0 against {v3_0:.4f} in the "
           r"reported run. "
           f"The first runs of Human cold-drug and cold-target (three seeds, retention initialised at the "
           f"fixed-arm keep fractions; \\texttt{{runs/recursive\\_cd}}, \\texttt{{runs/recursive\\_ct}}) gave "
           f"AUROC {first_runs['recursive_cd']} and {first_runs['recursive_ct']}; both cells were rerun at "
           f"five seeds with retention initialisation 0.95 and only the rerun is used. "
           + other_runs)
    body = [r"\setcounter{table}{2}",
            r"\begin{longtable}{@{}LLLRRRR@{}}",
            r"\caption{" + cap + r"}\label{tbl:s3}\\",
            r"\toprule", r"Benchmark & Protocol & Run & Seed & Reference & Model & Margin \\",
            r"\midrule", r"\endfirsthead",
            r"\toprule", r"Benchmark & Protocol & Run & Seed & Reference & Model & Margin \\",
            r"\midrule", r"\endhead"] + rows + [r"\bottomrule", r"\end{longtable}", ""]
    write("s3.tex", "\n".join(body))

    print("\nTable 6 recomputed from artifacts (paper value in brackets where different):")
    bad = 0
    for cell, got in recomputed.items():
        want = TABLE6[cell]
        vals = [round(got[0], 4), round(got[1], 4), round(got[2], 4),
                round(got[3][0], 4), round(got[3][1], 4), got[4]]
        exp = [want[0], want[1], want[2], want[3][0], want[3][1], want[4]]
        diff = [f"{v} [{e}]" if abs(v - e) > 0.00011 else f"{v}" for v, e in zip(vals, exp)]
        bad += any(abs(v - e) > 0.00011 for v, e in zip(vals, exp))
        print(f"  {cell:<22} ref {diff[0]}  model {diff[1]}  margin {diff[2]}  "
              f"CI {diff[3]}, {diff[4]}  n {diff[5]}")
    print(f"  {bad} cell(s) differ from the manuscript beyond rounding")
    return recomputed

EDGE_PERMUTATIONS = 1000

def s4() -> None:
    dom = jload(RUNS / "edge_prior_domain.json")
    cmp_ = jload(RUNS / "edge_prior_comparison.json")
    cold = jload(RUNS / "edge_prior_cold_rows.json")
    audit = jload(RUNS / "published_split_audit.json")
    given = jload(RUNS / f"edge_prior_given_files_{EDGE_PERMUTATIONS}.json")
    assert given["permutations"] == EDGE_PERMUTATIONS
    given_by = {(r["dataset"], r["source"]): r for r in given["rows"]}

    def verdict(n_pos: int, n: int) -> str:
        n_min = min(n_pos, n - n_pos)
        return "single class" if n_min == 0 else ("degenerate" if n_min <= 7 else "yes")

    by = {}
    for r in dom["rows"]:
        by.setdefault((r["dataset"], r["protocol"]), []).append(r)
    rows = []
    for ds in DATASETS:
        for pr in PROTOCOLS:
            cell = sorted(by[(ds, pr)], key=lambda r: r["seed"])
            frac = np.mean([r["frac_scoreable"] for r in cell]) * 100
            counts = "; ".join(f"{r['n_positive_in_scoreable']}/{r['n_scoreable']}" for r in cell)
            if all(r["n_scoreable"] == 0 for r in cell):
                ev = "empty domain"
            else:
                mins = [min(r["n_positive_in_scoreable"],
                            r["n_scoreable"] - r["n_positive_in_scoreable"]) for r in cell]
                ev = ("single class at " + ", ".join(f"s{r['seed']}" for r, m in zip(cell, mins) if m == 0)
                      + "; degenerate otherwise") if 0 in mins else \
                    ("degenerate" if max(mins) <= 7 else "yes")
            if pr == "random":
                assert cmp_[ds]["permutations"] == EDGE_PERMUTATIONS, (ds, cmp_[ds]["permutations"])
                pc = cmp_[ds]["protocols"]["random"]
                full = f4(pc["edge_prior_auroc_tiebroken_raw"])
                if ev == "yes":
                    rr = [r["edge_prior_auroc_restricted_raw"] for r in pc["per_seed"]
                          if r.get("edge_prior_auroc_restricted") is not None]
                    ev = f"yes; {f4(float(np.mean(rr)))}"
            else:
                src = cmp_ if pr in cmp_.get(ds, {}).get("protocols", {}) else cold
                full = "---"
            rows.append(f"{BENCH[ds]} & {PROTO[pr]} & {frac:.1f}\\% & {counts} & {full} & {ev} \\\\")
        rows.append(r"\addlinespace")
    for r in audit:
        src = "distributed" if r["source"] == "raw" else "prepared"
        ev = verdict(r["n_positive_in_scoreable"], r["n_scoreable"])
        g = given_by[(r["dataset"], r["source"])]
        assert g["n_scoreable"] == r["n_scoreable"], (r["dataset"], r["source"])
        if ev == "yes":
            ev = f"yes; {f4(g['restricted_auroc']['mean_raw'])}"
        rows.append(f"{BENCH[r['dataset']]} & {src} & {r['frac_scoreable'] * 100:.1f}\\% & "
                    f"{r['n_positive_in_scoreable']}/{r['n_scoreable']} & "
                    f"{f4(g['tiebroken_auroc']['mean_raw'])} & {ev} \\\\")
    cap = (r"Edge-prior domain by benchmark and protocol. Scoreable is the fraction of test "
           r"pairs with both entities in the training positive graph, mean over seeds 0--2; "
           r"the next column gives positives within the scoreable subset per seed. "
           f"Full-test AUROC is the three-seed mean at {EDGE_PERMUTATIONS} permutations; unscoreable "
           r"pairs score 0, and scoreable pairs whose estimate is still 0 are ranked just above them, "
           r"the order the estimate converges to (Section~S4.2). No value is reoriented. "
           r"A dash in the AUROC column marks an empty domain: no pair is scoreable, the prior is "
           r"constant and no AUROC is defined. ``Degenerate'' means both labels occur but the minority label has "
           r"at most seven pairs; where the subset is evaluable, its restricted AUROC follows the "
           r"verdict. The last six rows use files as supplied: the benchmark authors' distributed "
           r"files, and the prepared files of Table~2 of the main text, whose split is fixed, so "
           r"their three seeds vary only the permutations. "
           r"Sources: \texttt{runs/edge\_prior\_domain.json}, "
           r"\texttt{runs/edge\_prior\_comparison.json}, \texttt{runs/edge\_prior\_cold\_rows.json}, "
           r"\texttt{runs/published\_split\_audit.json}, "
           f"\\texttt{{runs/edge\\_prior\\_given\\_files\\_{EDGE_PERMUTATIONS}.json}}.")
    write("s4.tex", tabular(cap, "tbl:s4", "LLRLRL",
                            r"Benchmark & Protocol & Scoreable & Positives in scoreable & "
                            r"Full-test AUROC & Evaluable?", rows, 4, r"\footnotesize"))

def s4_convergence(purity: dict) -> None:
    conv = jload(RUNS / "edge_prior_convergence.json")
    assert conv["chosen_permutations"] == EDGE_PERMUTATIONS
    dp = jload(RUNS / "edge_prior_degree_product.json")["cells"]
    ref_sub = {"biosnap": jload(RUNS / "reference_on_scoreable_subset.json"),
               "bindingdb": jload(RUNS / "reference_on_scoreable_subset_bindingdb.json")}
    counts = (25, 100, 400, 1000)
    counts += tuple(n for n in (10000,) if any(str(n) in conv["series"][ds] for ds in DATASETS))
    rows = []
    for ds in DATASETS:
        ser = conv["series"][ds]
        seeds = [r["seed"] for r in dp[ds]["per_seed"]]
        ref_full = float(np.mean([r["val_fixed"] for r in purity["cells"][f"{ds}/random"]
                                  if int(r["seed"]) in seeds]))
        meas = [("full test", "tiebroken", dp[ds]["mean"]["product_full"], ref_full)]
        if ds in ref_sub:
            assert ref_sub[ds]["dataset"] == ds and [r["seed"] for r in ref_sub[ds]["rows"]] == seeds
            meas.append(("scoreable subset", "restricted", dp[ds]["mean"]["product_restricted"],
                         ref_sub[ds]["mean"]["reference_on_scoreable"]))
        for i, (name, key, lim, ref) in enumerate(meas):
            vals = [f4(ser[str(n)][key]) if str(n) in ser and ser[str(n)][key] is not None else "---"
                    for n in counts]
            lead = BENCH[ds] if i == 0 else ""
            rows.append(f"{lead} & {name} & " + " & ".join(vals) + f" & {f4(lim)} & {f4(ref)} \\\\")
    cap = (r"Random-split edge-prior AUROC by permutation count (10 swaps per edge, three-seed means, "
           r"seeds 0--2), as computed. ``Full test'' scores unscoreable pairs 0 and ranks scoreable pairs "
           r"with a zero estimate just above them (Section~3.4 of the main text); ``scoreable subset'' is "
           r"the restricted AUROC, shown where that subset holds both classes in non-degenerate numbers. "
           r"``Degree product'' scores each pair by the product of its drug's and its target's degree in "
           r"the training positive graph, which needs no permutation; the permutation estimate approaches "
           r"it. ``Reference'' is the validation-fixed entity-frequency reference on the same pairs and "
           r"seeds. A dash is a count not run (BindingDB was first run at 25 permutations by a script "
           r"version that did not record the tie-broken value"
           + (r"; only BioSNAP, still rising at 1,000, was run at 10,000" if 10000 in counts else "")
           + r"). Sources: "
           r"\texttt{runs/edge\_prior\_convergence.json} (\texttt{scripts/edge\_prior\_convergence.py}), "
           r"\texttt{runs/edge\_prior\_degree\_product.json}, "
           r"\texttt{runs/reference\_on\_scoreable\_subset\{,\_bindingdb\}.json}.")
    body = tabular(cap, "tbl:s4b", "LL" + "R" * (len(counts) + 2),
                   r"Benchmark & AUROC on & " + " & ".join(f"{n:,}" for n in counts)
                   + r" & Degree product & Reference", rows, 4, r"\footnotesize")
    write("s4b.tex", r"{\renewcommand{\thetable}{S4b}" + "\n" + body.rstrip("\n") + "}\n")

def s5(recomputed: dict, purity: dict) -> None:
    pts = [(c, *v) for c, v in recomputed.items()]
    pts.sort(key=lambda t: t[1])
    x = np.array([p[1] for p in pts])
    y = np.array([p[2] for p in pts])
    slope, icpt, r, _p, _se = stats.linregress(x, y)
    rho = stats.spearmanr(x, y).correlation
    rows = []
    for c, ref, mod, mar, _ci, n in pts:
        ds, pr = c.split("/")
        rows.append(f"{BENCH[ds]} & {PROTO[pr]} & {f4(ref)} & {f4(mod)} & {sgn4(mar)} & {n} \\\\")
    words = {16: "sixteen", 10: "ten", 4: "four", 3: "three"}
    n_cells = len(pts)
    n_pinned = sum(c.endswith("cold_pair") for c, *_ in pts)
    n_bench = len({c.split("/")[0] for c, *_ in pts})
    cap = (f"Model against reference across the {words.get(n_cells, n_cells)} cells of Table~6 of "
           r"the main text, ordered by reference height. Descriptively, the least-squares line is "
           f"model $= {icpt:.3f} + {slope:.3f}\\times$ reference ($R^2 = {r ** 2:.3f}$) and "
           f"the rank correlation is {rho:.3f}. No interval or test is given: the cells come from "
           f"{words.get(n_bench, n_bench)} benchmarks and share splits and a model, so they are not "
           f"independent observations, and {words.get(n_pinned, n_pinned)} are pinned at 0.5000 by "
           r"construction. Recomputed from the same artifacts as Supplementary Table~S3.")
    write("s5.tex", tabular(cap, "tbl:s5", "LLRRRR",
                            r"Benchmark & Protocol & Reference & Model & Margin & $n$", rows, 5))

def s6() -> None:
    ks = {1: jload(RUNS / "entity_purity_k1.json"), 5: jload(RUNS / "entity_purity.json"),
          20: jload(RUNS / "entity_purity_k20.json")}
    rows, agree = [], {k: 0 for k in ks}
    for ds in DATASETS:
        for pr in PROTOCOLS:
            cell = f"{ds}/{pr}"
            parts = []
            for k, pay in ks.items():
                rs = [{**r, "sign_free": 0.5 if r["sign_free"] is None else r["sign_free"]}
                      for r in pay["cells"][cell]]
                vf = float(np.mean([r["val_fixed"] for r in rs]))
                sf = float(np.mean([r["sign_free"] for r in rs]))
                gap = max(abs(r["sign_free"] - r["val_fixed"]) for r in rs)
                ok = gap < 0.00005
                agree[k] += ok
                parts.append(f"{f4(vf)}" + ("" if ok else f" ({f4(sf)})"))
            rows.append(f"{BENCH[ds]} & {PROTO[pr]} & " + " & ".join(parts) + r" \\")
        rows.append(r"\addlinespace")
    rows = rows[:-1]
    mk = jload(RUNS / "margin_k_sensitivity.json")["cells"]
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{2}{@{}L}{PharmUnit-DTI margin (Table~6)} & $k = 1$ & $k = 5$ & "
                r"$k = 20$ \\")
    rows.append(r"\midrule")
    arrow = {"above": r"$^{\uparrow}$", "below": r"$^{\downarrow}$", "spans 0": ""}
    n_status = n_sign = 0
    sign_cells = []
    for ds in DATASETS:
        for pr in PROTOCOLS:
            c = mk[f"{ds}/{pr}"]
            n_status += c["status_changes"]
            n_sign += c["sign_changes"]
            if c["sign_changes"]:
                sign_cells.append(f"{BENCH[ds]} {PROTO[pr]}")
            rows.append(f"{BENCH[ds]} & {PROTO[pr]} & " + " & ".join(
                f"{sgn4(c[f'k{k}']['margin'])}{arrow[c[f'k{k}']['status']]}" for k in (1, 5, 20)) + r" \\")
        rows.append(r"\addlinespace")
    rows = rows[:-1]
    sens = jload(RUNS / "smoothing_sensitivity.json")["contrasts"]
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{2}{@{}L}{Matched contrast, Wilcoxon $p$} & $k = 1$ & $k = 5$ & "
                r"$k = 20$ \\")
    rows.append(r"\midrule")
    for label, rs in sens.items():
        byk = {int(r["k"]): r for r in rs}
        n = len(rs[0]["seeds"])
        floor = 2 / 2 ** n
        rows.append(f"\\multicolumn{{2}}{{@{{}}L}}{{{label.replace('C. elegans', 'C.~elegans')} "
                    f"($n = {n}$, minimum {floor:.4f})}} & "
                    + " & ".join(pfmt(byk[k]["wilcoxon"]) for k in (1, 5, 20)) + r" \\")
    print("S6: cells where the two constructions agree at every seed:",
          {k: f"{v} of 16" for k, v in agree.items()})
    cap = (r"Reference by cell at smoothing weights $k = 1$, 5 and 20: five-seed mean of the "
           r"validation-fixed construction, with the oracle upper bound in brackets wherever "
           r"the two differ at any seed. The constructions agree at every seed in "
           f"{agree[1]}, {agree[5]} and {agree[20]} of the 16 cells at $k = 1$, 5 and 20. "
           r"Cold-pair is exactly 0.5000 at every $k$ on every benchmark. The middle block repeats "
           r"each paired margin of Table~6 of the main text against the reference at that $k$ "
           r"(seeds 0--4); $^{\uparrow}$ and $^{\downarrow}$ mark a 95\% interval entirely above or "
           f"below zero. The interval keeps its status across $k$ in {16 - n_status} of the 16 cells, "
           f"and {n_sign} margin{'s' if n_sign != 1 else ''} change{'' if n_sign != 1 else 's'} sign"
           + (f" ({' and '.join(sign_cells)}, whose intervals include zero at every $k$). "
              if sign_cells else ". ") +
           f"The lower block gives the Wilcoxon signed-rank $p$ for the {len(sens)} matched-reference "
           r"contrasts of Supplementary Section~S9, with the smallest two-sided $p$ attainable at that "
           r"number of pairs: at these sample sizes a non-significant Wilcoxon $p$ is "
           r"uninformative rather than evidence of no difference. Sources: "
           r"\texttt{runs/entity\_purity\_k1.json}, \texttt{runs/entity\_purity.json}, "
           r"\texttt{runs/entity\_purity\_k20.json}, \texttt{runs/margin\_k\_sensitivity.json}, "
           r"\texttt{runs/smoothing\_sensitivity.json}.")
    write("s6.tex", tabular(cap, "tbl:s6", "LLRRR",
                            r"Benchmark & Protocol & $k = 1$ & $k = 5$ & $k = 20$", rows, 6,
                            r"\footnotesize"))

def s7() -> None:
    def keep(runs: dict) -> tuple[float, float]:
        kd = [r["realised_keep"]["keep_drug"] for r in runs.values() if r.get("realised_keep")]
        kp = [r["realised_keep"]["keep_protein"] for r in runs.values() if r.get("realised_keep")]
        return (float(np.mean(kd)), float(np.mean(kp))) if kd else (1.0, 1.0)

    arms = [
        ("Pharmacophore scorer", "Human cold-pair", "recursive_ik95", "recursive_nopharm",
         "learned scorer"),
        ("Pharmacophore scorer", "BioSNAP cold-pair", "bs_recursive", "bs_nopharm",
         "learned scorer"),
        ("Compression", "BioSNAP cold-pair", "bs_recursive", "bs_ceiling", "none"),
        ("Budget weight 0.05", "BioSNAP cold-pair", "bs_recursive", "bs_sweep_bw0.0",
         r"``0.0''$^{\ddagger}$"),
    ]
    nopharm_mean = float(np.mean([r["test"]["auroc"] for r in model_runs("bs_nopharm").values()]))
    recursive_mean = float(np.mean([r["test"]["auroc"] for r in model_runs("bs_recursive").values()]))
    logged_nopharm = 0.7707
    logged_epoch0 = 0.7662
    nop0 = RUNS / "bs_nopharm" / "seed0"
    nop_started = jload(nop0 / "env.json")["timestamp"][:10]
    with open(nop0 / "history.csv", encoding="utf-8") as fh:
        nop_epoch0 = float(next(csv.DictReader(fh))["val_auroc"])
    comp = model_runs("bs_recursive")
    cells_frac = sorted(r["realised_keep"]["keep_drug"] * r["realised_keep"]["keep_protein"]
                        for r in comp.values())
    low, high = cells_frac[:-1], cells_frac[-1]
    rows = []
    for claim, cell, a, b, against in arms:
        ra, rb = model_runs(a), model_runs(b)
        common = sorted(set(ra) & set(rb))
        d = np.array([ra[s]["test"]["auroc"] - rb[s]["test"]["auroc"] for s in common])
        m, lo, hi, p = paired(d)
        ka = keep({s: ra[s] for s in common})
        kb = keep({s: rb[s] for s in common}) if b != "bs_ceiling" else (1.0, 1.0)
        rows.append(f"{claim} & {against} & {cell} & {len(common)} & {sgn4(m)} & {ci(lo, hi)} & "
                    f"{pfmt(p)} & {ka[0]:.2f}/{ka[1]:.2f} & {kb[0]:.2f}/{kb[1]:.2f} \\\\")
    cap = (r"The matched ablations on which the architectural claims for PharmUnit-DTI were "
           r"withdrawn. Differences are paired per seed (first arm minus second), with "
           r"$t$-intervals. ``Keep'' is the realised fraction of drug/protein tokens retained, "
           r"mean over seeds, from \texttt{realised\_keep} in each run's "
           r"\texttt{metrics.json}; attention cost scales with their product. On Human the two "
           r"scorer arms retain alike and differ by nothing detectable. On BioSNAP they retain "
           r"very differently, so the scorer and the amount of compression are confounded and "
           r"the difference cannot be attributed to the feature. Compression is not shown to "
           r"cost accuracy, but no non-inferiority margin was specified in advance, so none is "
           f"claimed; {len(low)} of the {len(cells_frac)} compressed seeds kept "
           f"{min(low) * 100:.2f}--{max(low) * 100:.2f}\\% of attention cells and one "
           f"{high * 100:.1f}\\%. " + r"$^{\ddagger}$A budget weight of 0.0 was not applied as zero: "
           r"the guard \texttt{if self.budget\_weight:} treats 0.0 as false and returns the "
           r"unscaled penalty, equivalent to weight 1.0, twenty times the default. The Human "
           r"control run with that setting (\texttt{human\_bw0}), on which a claim about the "
           r"compression mechanism rested, is therefore invalid, and the claim is withdrawn "
           r"rather than resolved. The defect is fixed in the deposited code "
           r"(\texttt{pharmunit/compression.py}), with a regression test "
           r"(\texttt{tests/test\_compression.py}). Runs: \texttt{runs/<tag>/seed<k>/}. "
           f"The deposited \\texttt{{bs\\_nopharm}} files (mean {nopharm_mean:.4f}; run started {nop_started}, "
           r"per \texttt{env.json}) are not the run whose training log is retained "
           r"(\texttt{runs/autodl\_logs/biosnap.txt}: 0.7707 $\pm$ 0.0202 under the same command; its seed-0 "
           f"validation AUROC at epoch 0 is {logged_epoch0:.3f}, against {nop_epoch0:.3f} in the deposited run), so that "
           r"arm was run at least twice; the table uses the deposited files. With the logged mean instead, the BioSNAP "
           f"scorer difference would be {recursive_mean - logged_nopharm:+.4f} rather than "
           f"{recursive_mean - nopharm_mean:+.4f}; either way the two arms retain very differently, so the difference "
           r"cannot be attributed to the feature, and no conclusion of the paper changes.")
    write("s7.tex", tabular(cap, "tbl:s7", "LLLRRLRRR",
                            r"Claim & Against & Cell & $n$ & Difference & 95\% CI & $p$ & "
                            r"Keep 1 & Keep 2", rows, 7, r"\scriptsize"))

def s9() -> None:
    sens = jload(RUNS / "smoothing_sensitivity.json")["contrasts"]
    names = {"BioSNAP random vs cold-drug": ("BioSNAP random", "vs cold-drug"),
             "C. elegans random vs cold-target": (r"\textit{C.\ elegans} random", "vs cold-target"),
             "Human cold-drug vs cold-target": ("Human cold-drug", "vs cold-target"),
             "BindingDB random vs cold-drug": ("BindingDB random", "vs cold-drug")}
    rows = []
    for label, rs in sens.items():
        a, b = names[label]
        for i, r in enumerate(sorted(rs, key=lambda r: r["k"])):
            lead = a if i == 0 else (r"\quad " + b if i == 1 else "")
            rows.append(f"{lead} & {int(r['k'])} & {r['reference_gap']:.4f} & "
                        f"{'yes' if r['satisfies_selection_rule'] else 'no'} & {sgn4(r['mean'])} & "
                        f"{ci(r['welch_lo'], r['welch_hi'])} & {pfmt(r['welch_p'])} & "
                        f"{pfmt(r['welch_p_holm'])} & {pfmt(r['p'])} \\\\")
        rows.append(r"\addlinespace")
    rows = rows[:-1]
    cap = (r"Matched-reference contrasts and their sensitivity to the smoothing weight $k$ "
           r"of Eq.~2 of the main text. ``Gap'' is the difference in reference height and "
           r"``Rule'' whether the pair still satisfies the selection criterion at that $k$. "
           r"The primary analysis is an unpaired Welch test on per-seed margins, since the two "
           r"arms draw different partitions and share only a seed index; ``Holm'' adjusts its "
           f"$p$ across the {len(sens)} contrasts at each $k$. ``Paired'' is the $p$ of a paired "
           r"$t$-test on within-seed margin differences, the sensitivity analysis. Every arm "
           r"has five seeds. Source: "
           r"\texttt{runs/smoothing\_sensitivity.json} "
           r"(\texttt{scripts/smoothing\_sensitivity.py}).")
    write("s9.tex", tabular(cap, "tbl:contrasts", "LRRLRLRRR",
                            r"Contrast & $k$ & Gap & Rule & Difference & Welch 95\% CI & $p$ & "
                            r"Holm $p$ & Paired $p$", rows, 9, r"\footnotesize"))

def s12(purity: dict) -> None:
    sm = jload(RUNS / "size_matched_check.json")
    full = {(m["cell"], m["family"]): m for m in jload(RUNS / "margin_tally.json")["margins"]}
    rows = []
    for ds in ("biosnap", "bindingdb"):
        for pr in ("random", "cold_drug", "cold_target"):
            cell = f"{ds}/{pr}"
            ps, sx = sm["cells"][cell]["per_seed"], sm["cells"][cell]["summary"]
            ref_full = float(np.mean([r["val_fixed"] for r in purity["cells"][cell]]))
            lead = BENCH[ds] if pr == "random" else ""
            row = (f"{lead} & {PROTO[pr]} & {round(np.mean(ps['full_train_n'])):,} & "
                   f"{round(np.mean(ps['matched_n'])):,} & {f4(ref_full)} & {f4(sx['reference_mean'])}")
            for fam, key in (("LR", "lr"), ("GBT", "gbt")):
                row += (f" & {sgn4(full[(cell, fam)]['margin'])} & {sgn4(sx[key]['margin'])} "
                        f"{ci(sx[key]['lo'], sx[key]['hi'])}")
            rows.append(row + r" \\")
    cap = (r"Size-matched check. Each BioSNAP and BindingDB training fold is shrunk, by uniform sampling "
           r"(seed $1000 + s$), to the Human training fold of the same protocol and seed $s$; validation and "
           r"test folds are unchanged. The validation-fixed reference is recomputed on the shrunk fold and "
           r"logistic regression (LR) and gradient-boosted trees (GBT) are refitted on it, with the features "
           r"and settings of Section~S8. ``Full'' columns are the margins of Table~S8b; ``matched'' margins are "
           r"paired per-seed means over seeds 0--4 with 95\% $t$ intervals. Training-pair counts are "
           r"five-seed means. The ESM-MLP was not refitted because its protein embeddings for these two "
           r"benchmarks are not cached on the machine used. Source: \texttt{runs/size\_matched\_check.json} "
           r"(\texttt{scripts/size\_matched\_check.py}).")
    body = tabular(cap, "tbl:s12", "LLRRRRRRRR",
                   r"& & \multicolumn{2}{c}{Training pairs} & \multicolumn{2}{c}{Reference} & "
                   r"\multicolumn{2}{c}{LR margin} & \multicolumn{2}{c}{GBT margin} \\" + "\n"
                   r"Benchmark & Protocol & full & matched & full & matched & full & matched [95\% CI] & "
                   r"full & matched [95\% CI]", rows, 12, r"\scriptsize\setlength{\tabcolsep}{3.5pt}")
    write("s12.tex", body.replace(r"\begin{table}[p]", r"\begin{table}[htbp]"))

def main() -> None:
    manifest = {k: v for k, v in jload(RUNS / "ladder_manifest.json").items()
                if not k.startswith("_")}
    purity = jload(RUNS / "entity_purity.json")
    balance = split_balance(purity, manifest)
    s2(balance, manifest)
    recomputed = s3_and_table6(purity, manifest, balance)
    s4()
    s4_convergence(purity)
    s5(recomputed, purity)
    s6()
    s7()
    s9()
    s12(purity)

if __name__ == "__main__":
    main()
