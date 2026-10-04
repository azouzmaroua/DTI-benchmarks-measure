from __future__ import annotations

import contextlib
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
    "biosnap/cold_pair": (0.5000, 0.7831, +0.2831, (+0.2526, +0.3136), 5),
    "human/cold_pair": (0.5000, 0.7269, +0.2269, (+0.1565, +0.2974), 10),
    "biosnap/cold_target": (0.6684, 0.8275, +0.1591, (+0.1219, +0.1962), 5),
    "biosnap/random": (0.8276, 0.9162, +0.0885, (+0.0812, +0.0959), 5),
    "biosnap/cold_drug": (0.8197, 0.8577, +0.0380, (+0.0176, +0.0583), 5),
    "human/cold_drug": (0.9326, 0.9310, -0.0016, (-0.0347, +0.0314), 3),
    "celegans/random": (0.9943, 0.9619, -0.0324, (-0.0368, -0.0280), 5),
    "human/cold_target": (0.9457, 0.8846, -0.0610, (-0.2349, +0.1128), 3),
    "celegans/cold_target": (0.9715, 0.8854, -0.0861, (-0.1512, -0.0210), 5),
}
LOG_ONLY = {"celegans/cold_pair": (0.6061, 0.0759, 5), "celegans/cold_drug": (0.8217, 0.1894, 5)}


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
    rows, mismatches, checked, sizes_only = [], [], 0, set()
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
                used = r"\textit{log}" if cell in LOG_ONLY else "---"
            elif len(model_seeds) > 5:
                extra = [b["seeds"][str(s)] for s in model_seeds if s >= 5]
                n_out = sum(max(f["drift"] for f in e.values()) > TOLERANCE for e in extra)
                used = f"0--{max(model_seeds)} ({n_out} of seeds 5--9 flagged)"
            else:
                used = ", ".join(str(s) for s in model_seeds)
            rows.append(f"{BENCH[ds]} & {PROTO[pr]} & {b['pool_positive_rate']:.4f} & "
                        + " & ".join(cells) + f" & {used} \\\\")
        rows.append(r"\addlinespace")
    rows = rows[:-1]
    print(f"S2: {checked} retained runs checked against their split reports, "
          f"{len(mismatches)} mismatch(es); fold sizes only for {sorted(sizes_only)}")
    for m in mismatches:
        print("  MISMATCH", m)
    if mismatches:
        raise SystemExit("a retained run was not trained on the split redrawn for its seed")
    cap = (r"Per-fold label drift by cell and seed: for each seed, the largest absolute "
           r"difference between a fold's positive rate and the pooled positive rate, over "
           r"the training, validation and test folds. $^{\dagger}$ marks a seed whose drift "
           r"exceeds the 0.15 tolerance at which the pipeline warns but continues "
           r"(\texttt{pharmunit/splits.py}, \texttt{check\_fold\_balance}). ``Model seeds'' "
           r"are the seeds paired in Table~6 of the main text; \textit{log} marks a cell "
           r"whose model value comes from its training log (Supplementary Table~S3). Every "
           f"retained run's own split report ({checked} runs) describes the same fold sizes "
           r"and positive counts as the split redrawn here for its seed"
           + (f" ({len(sizes_only)} report{'s' if len(sizes_only) > 1 else ''} printed "
              r"fold sizes only, which match)" if sizes_only else "") + r". Source: "
           r"\texttt{runs/split\_balance.json}, redrawn by "
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
    order = ["biosnap/random", "biosnap/cold_drug", "biosnap/cold_target", "biosnap/cold_pair",
             "human/cold_drug", "human/cold_target", "human/cold_pair",
             "celegans/random", "celegans/cold_target"]
    for cell in order:
        ds, pr = cell.split("/")
        tag = manifest[cell]
        runs = model_runs(tag)
        refs = references(purity, cell)
        margins = []
        first = True
        for s in sorted(runs):
            model = float(runs[s]["test"]["auroc"])
            if s in refs:
                ref = float(refs[s]["val_fixed"])
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
        ref_mean = float(np.mean([float(refs[s]["val_fixed"]) if s in refs else 0.5
                                  for s in sorted(runs)]))
        mod_mean = float(np.mean([float(runs[s]["test"]["auroc"]) for s in sorted(runs)]))
        recomputed[cell] = (ref_mean, mod_mean, m, (lo, hi), len(d))
        rows.append(r"\addlinespace")
    for cell, (mean, sd, n) in LOG_ONLY.items():
        ds, pr = cell.split("/")
        ref = float(np.mean([r["val_fixed"] for r in purity["cells"][cell]]))
        rows.append(f"{BENCH[ds]} & {PROTO[pr]} & \\textit{{log}} & mean of {n} & {f4(ref)} & "
                    f"{mean:.4f} $\\pm$ {sd:.4f} & {sgn4(mean - ref)} \\\\")
    cap = (r"Per-seed model AUROC, reference and margin for every cell with retained "
           r"per-seed artifacts, the per-seed form of Table~6 of the main text. The "
           r"reference is the validation-fixed construction at $k = 5$ "
           r"(\texttt{runs/entity\_purity.json}); under cold-pair it is 0.5000 at every seed "
           r"by construction, which is also the value used for Human cold-pair seeds 5--9. "
           r"Model values are \texttt{test.auroc} in \texttt{runs/<tag>/seed<k>/metrics.json}. "
           r"The last two rows come from training logs whose per-seed files were not "
           r"retained (\texttt{runs/ladder\_manifest.json}): their model entry is a mean "
           r"$\pm$ SD over five seeds and their margin is the difference of means, so no "
           r"per-seed pairing is available; \textit{C.\ elegans} cold-drug additionally has "
           f"{n_word} splits outside the label-balance tolerance "
           r"(Supplementary Table~S2) and is used in no inference.")
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


def s4() -> None:
    dom = jload(RUNS / "edge_prior_domain.json")
    cmp_ = jload(RUNS / "edge_prior_comparison.json")
    audit = jload(RUNS / "published_split_audit.json")
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
            full = cmp_.get(ds, {}).get("protocols", {}).get(pr, {}).get("edge_prior_auroc")
            full = f4(full) if (full is not None and pr == "random") else \
                ("0.5000$^{*}$" if full is not None else "---")
            rows.append(f"{BENCH[ds]} & {PROTO[pr]} & {frac:.1f}\\% & {counts} & {full} & {ev} \\\\")
        rows.append(r"\addlinespace")
    for r in audit:
        src = "distributed" if r["source"] == "raw" else "prepared"
        n_min = min(r["n_positive_in_scoreable"], r["n_scoreable"] - r["n_positive_in_scoreable"])
        ev = "single class" if n_min == 0 else ("degenerate" if n_min <= 7 else "yes")
        rows.append(f"{BENCH[r['dataset']]} & {src} & {r['frac_scoreable'] * 100:.1f}\\% & "
                    f"{r['n_positive_in_scoreable']}/{r['n_scoreable']} & --- & {ev} \\\\")
    cap = (r"Edge-prior domain by benchmark and protocol. Scoreable is the fraction of test "
           r"pairs with both entities in the training positive graph, mean over seeds 0--2; "
           r"the next column gives positives within the scoreable subset per seed. "
           r"Full-test AUROC assigns unscoreable pairs prior 0 and is the three-seed mean; "
           r"$^{*}$ where no pair is scoreable the prior is constant and that value is not a "
           r"measurement. ``Degenerate'' means both labels occur but the minority label has "
           r"at most seven pairs. The last six rows use files as supplied: the benchmark "
           r"authors' distributed files, and the prepared files of Table~2 of the main text. "
           r"Sources: \texttt{runs/edge\_prior\_domain.json} "
           r"(\texttt{scripts/edge\_prior\_domain.py}), "
           r"\texttt{runs/edge\_prior\_comparison.json}, "
           r"\texttt{runs/published\_split\_audit.json}.")
    write("s4.tex", tabular(cap, "tbl:s4", "LLRLRL",
                            r"Benchmark & Protocol & Scoreable & Positives in scoreable & "
                            r"Full-test AUROC & Evaluable?", rows, 4, r"\footnotesize"))


def s5(recomputed: dict, purity: dict) -> None:
    pts = [(c, *v) for c, v in recomputed.items()]
    ce = "celegans/cold_pair"
    pts.append((ce, 0.5, LOG_ONLY[ce][0], LOG_ONLY[ce][0] - 0.5, (float("nan"),) * 2, 5))
    pts.sort(key=lambda t: t[1])
    x = np.array([p[1] for p in pts])
    y = np.array([p[2] for p in pts])
    slope, icpt, r, _p, _se = stats.linregress(x, y)
    rho = stats.spearmanr(x, y).correlation
    rows = []
    for c, ref, mod, mar, _ci, n in pts:
        ds, pr = c.split("/")
        tag = r" \textit{(log)}" if c == ce else ""
        rows.append(f"{BENCH[ds]} & {PROTO[pr]}{tag} & {f4(ref)} & {f4(mod)} & {sgn4(mar)} & {n} \\\\")
    cap = (r"Model against reference across the ten cells of Table~6 of the main text, "
           r"ordered by reference height. Descriptively, the least-squares line is "
           f"model $= {icpt:.3f} + {slope:.3f}\\times$ reference ($R^2 = {r ** 2:.3f}$) and "
           f"the rank correlation is {rho:.3f}. No interval or test is given: the ten cells "
           r"come from three benchmarks and share splits and a model, so they are not "
           r"independent observations, and three are pinned at 0.5000 by construction. "
           r"Recomputed from the same artifacts as Supplementary Table~S3.")
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
           r"Cold-pair is exactly 0.5000 at every $k$ on every benchmark. The lower block gives "
           r"the Wilcoxon signed-rank $p$ for the three matched-reference contrasts of "
           r"Section~4.8 of the main text, with the smallest two-sided $p$ attainable at that "
           r"number of pairs: at these sample sizes a non-significant Wilcoxon $p$ is "
           r"uninformative rather than evidence of no difference. Sources: "
           r"\texttt{runs/entity\_purity\_k1.json}, \texttt{runs/entity\_purity.json}, "
           r"\texttt{runs/entity\_purity\_k20.json}, \texttt{runs/smoothing\_sensitivity.json}.")
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
           r"(\texttt{tests/test\_compression.py}). Runs: \texttt{runs/<tag>/seed<k>/}.")
    write("s7.tex", tabular(cap, "tbl:s7", "LLLRRLRRR",
                            r"Claim & Against & Cell & $n$ & Difference & 95\% CI & $p$ & "
                            r"Keep, arm 1 & Keep, arm 2", rows, 7, r"\footnotesize"))


def s9() -> None:
    sens = jload(RUNS / "smoothing_sensitivity.json")["contrasts"]
    names = {"BioSNAP random vs cold-drug": ("BioSNAP random", "vs cold-drug"),
             "C. elegans random vs cold-target": (r"\textit{C.\ elegans} random", "vs cold-target"),
             "Human cold-drug vs cold-target": ("Human cold-drug", "vs cold-target")}
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
           r"$p$ across the three contrasts at each $k$. ``Paired'' is the $p$ of a paired "
           r"$t$-test on within-seed margin differences, the sensitivity analysis. Human arms "
           r"have three seeds, the others five. Source: "
           r"\texttt{runs/smoothing\_sensitivity.json} "
           r"(\texttt{scripts/smoothing\_sensitivity.py}).")
    write("s9.tex", tabular(cap, "tbl:contrasts", "LRRLRLRRR",
                            r"Contrast & $k$ & Gap & Rule & Difference & Welch 95\% CI & $p$ & "
                            r"Holm $p$ & Paired $p$", rows, 9, r"\footnotesize"))


def main() -> None:
    manifest = {k: v for k, v in jload(RUNS / "ladder_manifest.json").items()
                if not k.startswith("_")}
    purity = jload(RUNS / "entity_purity.json")
    balance = split_balance(purity, manifest)
    s2(balance, manifest)
    recomputed = s3_and_table6(purity, manifest, balance)
    s4()
    s5(recomputed, purity)
    s6()
    s7()
    s9()


if __name__ == "__main__":
    main()
