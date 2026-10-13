from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
LABELS = {"human": "Human", "celegans": "C. elegans", "biosnap": "BioSNAP", "bindingdb": "BindingDB"}
ORDER = ["human", "celegans", "biosnap", "bindingdb"]
PURITY_COLOUR, LOOKUP_COLOUR = "#D55E00", "#0072B2"

def main() -> None:
    d = json.loads((ROOT / "runs" / "entity_purity.json").read_text(encoding="utf-8"))
    purity, cells = d["purity"], d["cells"]

    data = {}
    for ds in ORDER:
        seeds = cells[f"{ds}/random"]
        data[ds] = {
            "drug": (purity[ds]["drugs_single_class"],
                     statistics.mean(s["raw"]["drug"] for s in seeds)),
            "target": (purity[ds]["targets_single_class"],
                       statistics.mean(s["raw"]["target"] for s in seeds)),
        }

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    width, x = 0.36, range(len(ORDER))
    for ax, side in zip(axes, ("drug", "target")):
        pur = [data[ds][side][0] for ds in ORDER]
        look = [data[ds][side][1] for ds in ORDER]
        ax.bar([i - width / 2 for i in x], pur, width, label="single-class fraction",
               color=PURITY_COLOUR, edgecolor="white", hatch="///", linewidth=0.4)
        ax.bar([i + width / 2 for i in x], look, width, label="lookup AUROC",
               color=LOOKUP_COLOUR, edgecolor=LOOKUP_COLOUR, linewidth=0.6)
        ax.axhline(0.5, color="0.1", linestyle=(0, (4, 3)), linewidth=0.8)
        ax.set_title(f"{side} side", fontsize=9)
        ax.set_xticks(list(x))
        ax.set_xticklabels([LABELS[ds] for ds in ORDER], fontsize=8, rotation=20, ha="right")
        for tick, ds in zip(ax.get_xticklabels(), ORDER):
            if ds == "celegans":
                tick.set_fontstyle("italic")
        ax.set_ylim(0, 1.05)
        ax.tick_params(labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("fraction / AUROC", fontsize=9)
    bd = data["bindingdb"]["drug"][1]
    axes[0].text(3 + width + 0.10, 0.54, f"{bd:.3f}\n(anti-predictive)",
                 fontsize=6.5, ha="right", va="bottom", color="0.1", linespacing=1.1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, 1.06))

    fig.tight_layout()
    out = ROOT / "paper" / "figures" / "fig1_purity_lookup.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.show()
    print(f"wrote {out.relative_to(ROOT)} and .png")
    for ds in ORDER:
        dp, dl = data[ds]["drug"]
        tp, tl = data[ds]["target"]
        print(f"  {LABELS[ds]:<11} drug purity {dp:.4f} lookup {dl:.4f} | "
              f"target purity {tp:.4f} lookup {tl:.4f}")

if __name__ == "__main__":
    main()
