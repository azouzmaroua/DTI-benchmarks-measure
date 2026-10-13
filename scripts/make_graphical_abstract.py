from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (Arc, Circle, Ellipse, FancyBboxPatch,
                                Polygon, Rectangle, Wedge)
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
ORDER = ["human", "celegans", "biosnap", "bindingdb"]

INK, MUTED, GREY, LIGHT, PANEL = "#222222", "#5f5f5f", "#9a9a9a", "#d6d6d6", "#f3f3f3"
VERMILION, BLUE = "#D55E00", "#0072B2"
VERM_TINT, BLUE_TINT = "#fbe6da", "#dde9f4"
CEL = r"$\it{C.\ elegans}$"

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

def molecule(ax, x, y, s, color=INK, lw=1.5):
    ang = np.deg2rad(np.arange(30, 391, 60))
    px, py = x + s * np.cos(ang), y + s * np.sin(ang)
    ax.add_patch(Polygon(np.c_[px, py], closed=True, fc="white", ec=color, lw=lw,
                         joinstyle="round", zorder=5))
    ax.add_patch(Circle((x, y), 0.55 * s, fill=False, ec=color, lw=lw * 0.8, zorder=6))
    v = np.array([px[2], py[2]])
    chain = [v, v + s * np.array([-0.85, 0.5]), v + s * np.array([-1.7, 0.0])]
    ax.plot(*np.array(chain).T, color=color, lw=lw, solid_capstyle="round", zorder=5)
    ax.add_patch(Circle(chain[-1], 0.2 * s, color=color, zorder=5))
    w = np.array([px[4], py[4]])
    ax.plot([w[0], w[0]], [w[1], w[1] - 0.8 * s], color=color, lw=lw,
            solid_capstyle="round", zorder=5)

def helix(ax, x0, y, length, r, turns=3.0, width=0.75, tilt=1.1, color="#333333", back="#c4c4c4"):
    t = np.linspace(0, turns * 2 * np.pi, 600)
    p = length / (turns * 2 * np.pi)
    cx, cy = x0 + p * t + tilt * p * np.sin(t), y + r * np.cos(t)
    half = 0.5 * width * np.abs(np.sin(t)) + 0.08
    front = np.sin(t) > 0
    edges = np.flatnonzero(np.diff(front.astype(int))) + 1
    for seg in np.split(np.arange(len(t)), edges):
        if seg[-1] + 1 < len(t):
            seg = np.r_[seg, seg[-1] + 1]
        f = front[seg[1]]
        pts = np.r_[np.c_[cx[seg] - half[seg], cy[seg]], np.c_[cx[seg] + half[seg], cy[seg]][::-1]]
        ax.add_patch(Polygon(pts, closed=True, fc=color if f else back, ec=color if f else GREY,
                             lw=0.7, joinstyle="round", zorder=7 if f else 5))

def worm(ax, x, y, length, color=INK):
    u = np.linspace(0, 1, 80)
    xs = x - length / 2 + length * u
    ys = y + 0.55 * np.sin(2 * np.pi * 1.15 * u + 0.6)
    width = 0.8 + 4.2 * np.sin(np.pi * (0.06 + 0.8 * u)) ** 0.7
    for i in range(len(u) - 1):
        ax.plot(xs[i:i + 2], ys[i:i + 2], color=color, lw=width[i], solid_capstyle="round", zorder=5)
    ax.add_patch(Circle((xs[-6], ys[-6]), 0.09, color="white", zorder=6))

def person(ax, x, y, s, color=INK):
    ax.add_patch(Circle((x, y + 0.5 * s), 0.34 * s, color=color, zorder=5))
    ax.add_patch(Wedge((x, y - 0.75 * s), 0.78 * s, 0, 180, color=color, zorder=5))

def database(ax, x, y, w, h, color=INK):
    ry = 0.17 * w
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, color=color, zorder=5))
    ax.add_patch(Ellipse((x, y - h / 2), w, 2 * ry, color=color, zorder=5))
    ax.add_patch(Ellipse((x, y + h / 2), w, 2 * ry, fc=GREY, ec=color, lw=1.2, zorder=6))
    for f in (1 / 6, -1 / 6):
        ax.add_patch(Arc((x, y + f * h), w, 2 * ry, theta1=180, theta2=360, color="white", lw=1.3, zorder=6))

def hub(ax, x, y, s, color=VERMILION):
    for a in np.deg2rad([90, 162, 234, 306, 18]):
        ex, ey = x + s * np.cos(a), y + s * np.sin(a)
        ax.plot([x, ex], [y, ey], color=GREY, lw=1.4, zorder=4)
        ax.add_patch(Circle((ex, ey), 0.22 * s, color=GREY, zorder=5))
    ax.add_patch(Circle((x, y), 0.38 * s, color=color, zorder=6))

def one_class_drug(ax, x, y, s):
    for dy in (-1.1, 0, 1.1):
        ax.plot([x, x + 2.1 * s], [y, y + dy * s], color=VERMILION, lw=1.6, zorder=4)
        ax.add_patch(Circle((x + 2.1 * s, y + dy * s), 0.24 * s, color=GREY, zorder=5))
    ax.add_patch(Polygon(np.c_[x + 0.75 * s * np.cos(np.deg2rad(np.arange(30, 391, 60))),
                               y + 0.75 * s * np.sin(np.deg2rad(np.arange(30, 391, 60)))],
                         fc="white", ec=INK, lw=1.4, zorder=6))

def frequency_bars(ax, x, y, s):
    for i, h in enumerate((1.6, 0.7, 1.15, 0.4)):
        ax.add_patch(Rectangle((x - 1.1 * s + 0.6 * s * i, y - 0.8 * s), 0.42 * s, h * s,
                               color=BLUE, zorder=5))
    ax.plot([x - 1.25 * s, x + 1.25 * s], [y - 0.8 * s] * 2, color=INK, lw=1.2, zorder=6)

def split_grid(ax, x, y, size, test, drop=()):
    c = size / 5
    for i in range(5):
        for j in range(5):
            fc = INK if (i, j) in test else "white" if (i, j) in drop else LIGHT
            ax.add_patch(Rectangle((x - size / 2 + j * c, y + size / 2 - (i + 1) * c), c * 0.86, c * 0.86,
                                   color=fc, zorder=5))

def panel(ax, x0, x1, title):
    ax.add_patch(FancyBboxPatch((x0, 6.3), x1 - x0, 32.9, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=PANEL, ec="none", zorder=0))
    ax.text(x0 + 1.2, 37.4, title, fontsize=16, weight="bold", va="center", zorder=8)

def main() -> None:
    dom = load("edge_prior_domain.json")["summary"]
    ep = load("entity_purity.json")
    cal = {(r["dataset"], r["split"], r["seed"]): r for r in load("drugban_calibration.json")["rows"]}

    cold_scoreable = max(v["mean_frac_scoreable"] for k, v in dom.items() if "/cold" in k)
    purity = {ds: ep["purity"][ds]["drugs_single_class"] for ds in ORDER}
    reference = {ds: statistics.mean(s["val_fixed"] for s in ep["cells"][f"{ds}/random"]) for ds in ORDER}
    rows = [
        (CEL + ", cold-target", "P", *paired_margin("ce_cold_target", "celegans", "cold_target")),
        (CEL + ", random", "P", *paired_margin("ce_random", "celegans", "random")),
        ("Human, random", "D", cal[("human", "random", 42)]["margin"], *cal[("human", "random", 42)]["margin_ci95"]),
        ("BioSNAP, random", "D", cal[("biosnap", "random", 42)]["margin"],
         *cal[("biosnap", "random", 42)]["margin_ci95"]),
        ("BioSNAP, random", "P", *paired_margin("bs_random", "biosnap", "random")),
    ]

    assert cold_scoreable == 0.0
    assert [round(100 * purity[d], 1) for d in ORDER] == [99.4, 96.7, 5.1, 0.0], purity
    assert [round(reference[d], 2) for d in ("human", "celegans", "biosnap")] == [0.99, 0.99, 0.83], reference
    table = [(-0.0861, -0.1512, -0.0210), (-0.0324, -0.0368, -0.0280), (0.0004, -0.0089, 0.0097),
             (0.0717, 0.0583, 0.0851), (0.0885, 0.0812, 0.0959)]
    for (lab, _, *v), want in zip(rows, table):
        assert all(abs(a - b) < 5e-5 for a, b in zip(v, want)), (lab, v, want)

    plt.rcParams.update({"font.family": "Arial", "text.color": INK, "mathtext.fontset": "custom", "pdf.fonttype": 42,
                         "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"})
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.set_aspect("equal")
    ax.axis("off")

    panel(ax, 0.8, 32.4, "1  What we tested")
    panel(ax, 34.6, 65.4, "2  What we measured")
    panel(ax, 67.6, 99.2, "3  What we found")
    for x in (33.5, 66.5):
        ax.add_patch(Polygon([[x - 0.55, 24.2], [x + 0.65, 22.6], [x - 0.55, 21.0]], color=GREY, zorder=8))

    ax.text(4.6, 34.3, "drugs", fontsize=11, color=MUTED, ha="center", va="center")
    ax.text(26.4, 34.3, "protein targets", fontsize=11, color=MUTED, ha="center", va="center")
    dy, ty = [31.4, 27.3, 23.2], [31.4, 27.3, 23.2]
    pos = [(0, 0), (0, 1), (1, 1), (2, 0), (2, 2)]
    neg = [(1, 2), (2, 1), (0, 2)]
    for i, j in neg:
        ax.plot([7.0, 22.6], [dy[i], ty[j]], color=GREY, lw=1.2, ls=(0, (2.2, 2.0)), zorder=3)
    for i, j in pos:
        ax.plot([7.0, 22.6], [dy[i], ty[j]], color=INK, lw=1.5, zorder=3)
    for y in dy:
        molecule(ax, 5.6, y, 1.25)
    for y in ty:
        helix(ax, 23.0, y, 7.0, 1.15)
    ax.plot([7.6, 10.4], [20.3, 20.3], color=INK, lw=1.5)
    ax.text(10.9, 20.3, "interacts", fontsize=10.5, color=MUTED, va="center")
    ax.plot([18.0, 20.8], [20.3, 20.3], color=GREY, lw=1.2, ls=(0, (2.2, 2.0)))
    ax.text(21.3, 20.3, "does not", fontsize=10.5, color=MUTED, va="center")

    xs = [5.0, 12.6, 20.2, 27.8]
    person(ax, xs[0], 16.4, 1.5)
    worm(ax, xs[1], 16.0, 5.2)
    database(ax, xs[2], 16.0, 2.6, 2.6)
    database(ax, xs[3], 16.0, 2.6, 2.6)
    for x, lab in zip(xs, ["Human", CEL, "BioSNAP", "BindingDB"]):
        ax.text(x, 12.7, lab, fontsize=11, ha="center", va="center")

    cells = [(i, j) for i in range(5) for j in range(5)]
    grids = [
        ("random", {(0, 3), (1, 1), (2, 4), (3, 0), (4, 2)}, ()),
        ("cold-drug", {(4, j) for j in range(5)}, ()),
        ("cold-target", {(i, 4) for i in range(5)}, ()),
        ("cold-pair", {(i, j) for i in (3, 4) for j in (3, 4)},
         {(i, j) for i, j in cells if (i >= 3) != (j >= 3)}),
    ]
    for x, (lab, test, drop) in zip(xs, grids):
        split_grid(ax, x, 9.5, 3.0, test, drop)
        ax.text(x, 7.2, lab, fontsize=10, ha="center", va="center", color=MUTED)

    steps = [
        (hub, 31.0, f"{100 * cold_scoreable:.0f}%", VERMILION,
         "of cold-split pairs the degree\nbaseline can score"),
        (one_class_drug, 23.0, f"{100 * purity['human']:.0f}% / {100 * purity['celegans']:.0f}%", VERMILION,
         f"of drugs carry one label\n(Human / {CEL}; BioSNAP {100 * purity['biosnap']:.0f}%)"),
        (frequency_bars, 15.0, f"{reference['human']:.2f}", BLUE,
         "AUROC of the entity-frequency\nreference: identity alone"),
    ]
    ax.plot([38.6, 38.6], [15.0, 31.0], color=LIGHT, lw=1.2, zorder=1)
    for icon, y, big, col, cap in steps:
        ax.add_patch(Circle((38.6, y), 2.25, fc="white", ec=LIGHT, lw=1.0, zorder=3))
        if icon is one_class_drug:
            icon(ax, 37.6, y, 0.9)
        else:
            icon(ax, 38.6, y, 1.3)
        ax.text(42.2, y + 2.2, big, fontsize=24, weight="bold", color=col, va="center")
        ax.text(42.2, y - 1.7, cap, fontsize=11, color=INK, va="center", linespacing=1.15)
    ax.text(50.0, 8.2, "4 benchmarks × 4 splits\n5 model families, paired per seed",
            fontsize=10, color=MUTED, ha="center", va="center", linespacing=1.2)

    lo_x, hi_x, v0, v1 = 69.6, 97.8, -0.17, 0.11
    to_x = lambda v: lo_x + (v - v0) / (v1 - v0) * (hi_x - lo_x)
    top, step, y_lo, y_hi = 29.6, 4.25, 10.6, 32.0
    zero = to_x(0)
    ax.add_patch(Rectangle((lo_x, y_lo), zero - lo_x, y_hi - y_lo, color=VERM_TINT, zorder=1))
    ax.add_patch(Rectangle((zero, y_lo), hi_x - zero, y_hi - y_lo, color=BLUE_TINT, zorder=1))
    ax.plot([zero, zero], [y_lo, y_hi], color=INK, lw=1.1, zorder=2)

    def marker(x, y, kind, col, s=0.6):
        if kind == "P":
            ax.add_patch(Circle((x, y), s, color=col, zorder=5))
        else:
            ax.add_patch(Polygon([[x, y + 1.35 * s], [x + 1.15 * s, y], [x, y - 1.35 * s], [x - 1.15 * s, y]],
                                 color=col, zorder=5))

    for k, (lab, kind, m, lo, hi) in enumerate(rows):
        y = top - k * step
        col = VERMILION if hi < 0 else BLUE if lo > 0 else MUTED
        ax.text(lo_x + 0.5, y + 1.2, lab, fontsize=10.5, va="center", zorder=4)
        ax.text(hi_x - 0.4, y + 1.2, f"{m:+.4f}".replace("-", "−"), fontsize=10.5, weight="bold",
                color=col, ha="right", va="center", zorder=4)
        ax.plot([to_x(lo), to_x(hi)], [y - 0.5] * 2, color=col, lw=2.6, solid_capstyle="round", zorder=4)
        marker(to_x(m), y - 0.5, kind, col)
    for v, s in ((-0.1, "−0.1"), (0, "0"), (0.1, "+0.1")):
        ax.plot([to_x(v)] * 2, [y_lo, y_lo - 0.6], color=INK, lw=1)
        ax.text(to_x(v), 9.1, s, fontsize=10, ha="center", va="center")
    ax.text((lo_x + zero) / 2, 7.5, "below reference", fontsize=10.5, color=VERMILION,
            ha="center", va="center", weight="bold")
    ax.text((zero + hi_x) / 2, 7.5, "above reference", fontsize=10.5, color=BLUE,
            ha="center", va="center", weight="bold")
    ax.text(lo_x, 34.6, "model − reference AUROC, 95% CI", fontsize=10.5, color=MUTED, va="center")
    marker(lo_x + 0.6, 33.0, "P", INK, 0.45)
    ax.text(lo_x + 1.5, 33.0, "PharmUnit-DTI", fontsize=10, va="center")
    marker(lo_x + 12.6, 33.0, "D", INK, 0.45)
    ax.text(lo_x + 13.5, 33.0, "DrugBAN", fontsize=10, va="center")

    ax.add_patch(FancyBboxPatch((0.8, 0.7), 98.4, 4.6, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=BLUE_TINT, ec="none", zorder=0))
    ax.text(50, 3.0, "Report entity-class purity and a paired entity-frequency reference with every DTI AUROC.",
            fontsize=14, weight="bold", ha="center", va="center")

    out = ROOT / "paper" / "figures" / "graphical_abstract.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".tif"), dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    print(f"wrote {out.relative_to(ROOT)} (.pdf, .png, .tif)")
    print(f"  edge prior, cold splits: {cold_scoreable:.1%} scoreable")
    for d in ORDER:
        print(f"  {d:<10} purity {purity[d]:.3f}  reference {reference[d]:.4f}")
    for lab, kind, m, lo, hi in rows:
        print(f"  {lab.replace(CEL, 'C. elegans'):<24} {'PharmUnit-DTI' if kind == 'P' else 'DrugBAN':<14}"
              f" {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")

if __name__ == "__main__":
    main()
