from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from scipy import stats

print(matplotlib.get_backend())
ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
GROUPS = [
    ("BioSNAP", "biosnap", ["random", "cold_drug", "cold_target", "cold_pair"]),
    ("Human", "human", ["cold_drug", "cold_target", "cold_pair"]),
    ("C. elegans", "celegans", ["random", "cold_target", "cold_pair"]),
]
PROTOCOL = {"random": "random", "cold_drug": "cold-drug", "cold_target": "cold-target",
            "cold_pair": "cold-pair"}
ABOVE, SPANS, BELOW = "#0072B2", "#6b6b6b", "#D55E00"

def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def cell_values(ds: str, mode: str, manifest: dict):
    tag = manifest.get(f"{ds}/{mode}")
    if isinstance(tag, str):
        floors = load(RUNS / "floors" / f"{ds}_random_{mode}.json")["per_seed"]
        ref, mod = [], []
        for d in sorted((RUNS / tag).glob("seed*"), key=lambda p: int(p.name[4:])):
            met = load(d / "metrics.json")
            mod.append(met["test"]["auroc"] if "test" in met else met["auroc"])
            ref.append(floors[d.name[4:]]["floor"])
        diffs = [m - r for m, r in zip(mod, ref)]
        mean = statistics.mean(diffs)
        lo, hi = stats.t.interval(0.95, len(diffs) - 1, loc=mean, scale=stats.sem(diffs))
        return dict(ref=statistics.mean(ref), model=statistics.mean(mod), margin=mean,
                    lo=lo, hi=hi, n=len(diffs), log_only=False)
    note = manifest.get("_null_reasons", {}).get(f"{ds}/{mode}", "")
    m = re.search(r"log records ([0-9.]+)", note)
    if m is None:
        return None
    model = float(m.group(1))
    return dict(ref=0.5, model=model, margin=model - 0.5, lo=None, hi=None, n=5, log_only=True)

def main() -> None:
    manifest = load(RUNS / "ladder_manifest.json")
    rows, y, group_rows = [], 0.0, []
    for label, ds, modes in GROUPS:
        first = y
        for mode in modes:
            v = cell_values(ds, mode, manifest)
            if v is None:
                continue
            rows.append((y, label, ds, mode, v))
            y -= 1
        group_rows.append((label, first, y + 1))
        y -= 0.6

    if not rows:
        print("no completed cells in runs/; run the ladder first")
        return

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig = plt.figure(figsize=(9.2, 4.3))
    ax = fig.add_axes([0.245, 0.15, 0.305, 0.74])
    to_cols = blended_transform_factory(fig.transFigure, ax.transData)

    for (label, top, bottom) in group_rows:
        ax.axhspan(bottom - 0.5, top + 0.5, color="0.94" if label != "Human" else "white", zorder=0, lw=0)
        ax.text(-0.235 / 0.305 - 0.02, top, label, transform=blended_transform_factory(ax.transAxes, ax.transData),
                fontsize=8.5, weight="bold", va="center",
                fontstyle="italic" if label == "C. elegans" else "normal")

    for yy, label, ds, mode, v in rows:
        pinned = mode == "cold_pair"
        if v["log_only"]:
            colour = SPANS
        elif v["hi"] < 0:
            colour = BELOW
        elif v["lo"] > 0:
            colour = ABOVE
        else:
            colour = SPANS
        if v["lo"] is not None:
            ax.plot([v["lo"], v["hi"]], [yy, yy], color=colour, lw=1.8, solid_capstyle="butt", zorder=2)
        ax.plot(v["margin"], yy, "o", ms=6.5, zorder=3, markeredgewidth=1.4,
                color=colour, markerfacecolor="white" if pinned else colour)
        ci = "not available" if v["lo"] is None else f"[{v['lo']:+.4f}, {v['hi']:+.4f}]"
        for x, text, ha in ((0.625, f"{v['ref']:.4f}", "right"), (0.705, f"{v['model']:.4f}", "right"),
                            (0.73, f"{v['margin']:+.4f}  {ci}", "left"), (0.99, str(v["n"]), "right")):
            ax.text(x, yy, text.replace("-", "−"), transform=to_cols, ha=ha, va="center", fontsize=7.5)

    head_y = rows[0][0] + 1.0
    for x, text, ha in ((0.625, "Ref.", "right"), (0.705, "Model", "right"),
                        (0.73, "Margin  [95% CI]", "left"), (0.99, "n", "right")):
        ax.text(x, head_y, text, transform=to_cols, ha=ha, va="center", fontsize=7.5, weight="bold")

    ax.vlines(0, rows[-1][0] - 0.8, rows[0][0] + 0.5, color="0.15", lw=0.9, zorder=1)
    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels([PROTOCOL[r[3]] for r in rows], fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(rows[-1][0] - 0.8, head_y + 0.4)
    ax.set_xlim(-0.27, 0.36)
    ax.set_xticks([-0.2, -0.1, 0, 0.1, 0.2, 0.3])
    ax.set_xticklabels(["−0.2", "−0.1", "0", "0.1", "0.2", "0.3"])
    ax.set_xlabel("Margin: model AUROC − reference AUROC", fontsize=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    handles = [Line2D([], [], color=ABOVE, marker="o", lw=1.8, label="interval above 0"),
               Line2D([], [], color=SPANS, marker="o", lw=1.8, label="interval includes 0"),
               Line2D([], [], color=BELOW, marker="o", lw=1.8, label="interval below 0"),
               Line2D([], [], color="0.3", marker="o", lw=0, markerfacecolor="white",
                      label="cold-pair: reference fixed at 0.5")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.2,
               bbox_to_anchor=(0.5, -0.005), handlelength=1.6, columnspacing=1.2)
    plt.show()
    out = ROOT / "paper" / "figures" / "fig2_margins.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"wrote {out.relative_to(ROOT)} and .png")
    for yy, label, ds, mode, v in rows:
        ci = "n/a" if v["lo"] is None else f"[{v['lo']:+.4f}, {v['hi']:+.4f}]"
        print(f"  {label:<10} {PROTOCOL[mode]:<11} ref {v['ref']:.4f} model {v['model']:.4f} "
              f"margin {v['margin']:+.4f} {ci} n={v['n']}")

    
if __name__ == "__main__":
    main()
