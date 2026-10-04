from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
ORDER = ["human", "celegans", "biosnap", "bindingdb"]
LABELS = {"human": "Human", "celegans": "C. elegans", "biosnap": "BioSNAP", "bindingdb": "BindingDB"}

INK, MUTED, LIGHT = "#222222", "#6b6b6b", "#d9d9d9"
ORANGE, VERMILION, BLUE = "#E69F00", "#D55E00", "#0072B2"


def load(name: str):
    return json.loads((RUNS / name).read_text(encoding="utf-8"))


def paired_margin(tag: str, dataset: str, split_mode: str):
    floors = load(f"floors/{dataset}_random_{split_mode}.json")["per_seed"]
    diffs = []
    for d in sorted((RUNS / tag).glob("seed*")):
        met = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        auroc = met["test"]["auroc"] if "test" in met else met["auroc"]
        diffs.append(auroc - floors[d.name[4:]]["floor"])
    m = statistics.mean(diffs)
    lo, hi = stats.t.interval(0.95, len(diffs) - 1, loc=m, scale=stats.sem(diffs))
    return m, lo, hi


def main() -> None:
    domain = load("edge_prior_domain.json")
    dom = domain["summary"]
    pos_rates = [r["positive_rate_in_scoreable"] for r in domain["rows"]
                 if r["dataset"] == "human" and r["protocol"] == "random"]
    ep = load("entity_purity.json")
    cal = {(r["dataset"], r["split"], r["seed"]): r for r in load("drugban_calibration.json")["rows"]}

    scoreable = dom["human/random"]["mean_frac_scoreable"]
    pos_in_scoreable = dom["human/random"]["seed0"]["positive_rate_in_scoreable"]
    cold = [dom[f"human/{p}"]["mean_frac_scoreable"] for p in ("cold_drug", "cold_target", "cold_pair")]
    purity = {ds: ep["purity"][ds]["drugs_single_class"] for ds in ORDER}
    reference = {ds: statistics.mean(s["val_fixed"] for s in ep["cells"][f"{ds}/random"]) for ds in ORDER}
    db = cal[("human", "random", 42)]
    rows = [
        ("C. elegans\nPharmUnit-DTI", *paired_margin("ce_random", "celegans", "random")),
        ("Human\nDrugBAN", db["margin"], *db["margin_ci95"]),
        ("BioSNAP\nPharmUnit-DTI", *paired_margin("bs_random", "biosnap", "random")),
    ]

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
    fig = plt.figure(figsize=(10.24, 3.94))

    titles = [(0.025, "1  The degree baseline\n    cannot score cold splits"),
              (0.355, "2  Labels follow\n    drug identity"),
              (0.690, "3  Models against a\n    name-only reference")]
    for x, t in titles:
        fig.text(x, 0.955, t, fontsize=15, weight="bold", va="top", linespacing=1.15)
    for x in (0.318, 0.672):
        fig.text(x, 0.47, "❯", fontsize=26, color=MUTED, ha="center", va="center")

    a1 = fig.add_axes([0.03, 0.22, 0.26, 0.50])
    bars = [("Random split", scoreable), ("Cold splits", max(cold))]
    for i, (lab, frac) in enumerate(bars):
        y = 1 - i
        a1.barh(y, 1, color=LIGHT, height=0.42)
        if frac > 0:
            a1.barh(y, frac, color=ORANGE, height=0.42)
            lo_pos, hi_pos = min(pos_rates), max(pos_rates)
            a1.text(frac + 0.03, y, f"{frac:.0%} scoreable,\n{100 * lo_pos:.0f}–{100 * hi_pos:.0f}% positive",
                    fontsize=12.5, va="center", linespacing=1.1)
        else:
            a1.text(0.03, y, "0% scoreable", fontsize=12.5, va="center")
        a1.text(0, y + 0.36, lab, fontsize=13.5, weight="bold", va="bottom")
    a1.set_xlim(0, 1)
    a1.set_ylim(-0.4, 1.75)
    a1.axis("off")
    fig.text(0.03, 0.14, "Human test pairs, edge prior", fontsize=12, color=MUTED, ha="left")

    a2 = fig.add_axes([0.445, 0.20, 0.19, 0.52])
    h = 0.36
    ys = [len(ORDER) - 1 - i for i in range(len(ORDER))]
    a2.barh([y + h / 2 for y in ys], [purity[d] for d in ORDER], h, color=VERMILION,
            label="drugs in one class")
    a2.barh([y - h / 2 for y in ys], [reference[d] for d in ORDER], h, color=BLUE,
            label="name-only AUROC")
    for y, d in zip(ys, ORDER):
        a2.text(purity[d] + 0.03, y + h / 2, f"{purity[d]:.0%}", fontsize=11.5,
                va="center", color=VERMILION)
        a2.text(reference[d] + 0.03, y - h / 2, f"{reference[d]:.2f}", fontsize=11.5,
                va="center", color=BLUE)
    a2.set_yticks(ys)
    a2.set_yticklabels([LABELS[d] for d in ORDER], fontsize=13)
    for tick, d in zip(a2.get_yticklabels(), ORDER):
        if d == "celegans":
            tick.set_fontstyle("italic")
    a2.tick_params(axis="y", length=0)
    a2.set_xlim(0, 1.22)
    a2.set_xticks([0, 0.5, 1])
    a2.set_xticklabels(["0", "0.5", "1"], fontsize=11.5)
    for s in ("top", "right", "left"):
        a2.spines[s].set_visible(False)
    a2.legend(fontsize=11.5, frameon=False, loc="lower center", bbox_to_anchor=(0.22, 1.0), ncol=2,
              handlelength=0.9, columnspacing=0.7, handletextpad=0.4)

    a3 = fig.add_axes([0.825, 0.20, 0.15, 0.52])
    for i, (lab, m, lo, hi) in enumerate(rows):
        y = len(rows) - 1 - i
        col = VERMILION if hi < 0 else BLUE if lo > 0 else MUTED
        a3.plot([lo, hi], [y, y], color=col, lw=3, solid_capstyle="round")
        a3.plot(m, y, "o", color=col, ms=9)
        bench, model = lab.split("\n")
        a3.text(-0.07, y + 0.17, bench, fontsize=12, ha="right", va="center",
                fontstyle="italic" if bench == "C. elegans" else "normal")
        a3.text(-0.07, y - 0.17, model, fontsize=12, ha="right", va="center")
    a3.axvline(0, color=INK, lw=1)
    a3.set_xlim(-0.06, 0.12)
    a3.set_ylim(-0.6, len(rows) - 0.4)
    a3.set_xticks([-0.05, 0, 0.05, 0.10])
    a3.set_xticklabels(["−0.05", "0", "0.05", "0.10"], fontsize=11)
    a3.set_yticks([])
    for s in ("top", "right", "left"):
        a3.spines[s].set_visible(False)
    fig.text(0.90, 0.10, "model − reference", fontsize=12, color=MUTED, ha="center")

    fig.text(0.5, 0.035, "Report entity-class purity and a paired name-only reference with every DTI AUROC.",
             fontsize=14, ha="center", style="italic")
    fig.show()
    out = ROOT / "paper" / "figures" / "graphical_abstract.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".tif"), dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    print(f"wrote {out.relative_to(ROOT)} (.pdf, .png, .tif)")
    print(f"  edge prior: {scoreable:.3f} scoreable, {pos_in_scoreable:.0%} positive; cold max {max(cold)}")
    for d in ORDER:
        print(f"  {LABELS[d]:<11} purity {purity[d]:.3f}  reference {reference[d]:.4f}")
    for lab, m, lo, hi in rows:
        print(f"  {lab.replace(chr(10), ' / '):<26} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
