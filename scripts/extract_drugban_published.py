from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:
    import fitz

ROOT = Path(__file__).resolve().parent.parent
METHODS = ["SVM", "RF", "DeepConv-DTI", "GraphDTA", "MolTrans", "DrugBAN"]
TICKS = {"0.60", "0.65", "0.70", "0.75", "0.80", "0.85", "0.90", "0.95", "1.00"}
GREY = (0.502, 0.502, 0.502)
DPI = 600
TABLE2_ROWS = {"DrugBAN": "DrugBAN", "MolTransCDAN": "MolTrans_CDAN",
               "DrugBANDANN": "DrugBAN_DANN", "DrugBANCDAN": "DrugBAN_CDAN"}
TABLE2_COLS = ["bindingdb_random", "biosnap_random", "bindingdb_cluster", "biosnap_cluster"]


def _pm(s: str) -> list[float] | None:
    if s.strip() == "-":
        return None
    mean, sd = s.split("±")
    return [float(mean), float(sd)]


def table2(doc) -> dict:
    text = "\n".join(p.get_text() for p in doc)
    start = text.index("Table 2. Ablation study in AUROC")
    lines = [l.strip() for l in text[start:start + 2000].splitlines()]
    out = {}
    for label, key in TABLE2_ROWS.items():
        i = lines.index(label)
        out[key] = dict(zip(TABLE2_COLS, (_pm(v) for v in lines[i + 1:i + 5])))
    return out


def _dist(a, b) -> float:
    return sum((u - v) ** 2 for u, v in zip(a, b))


def figure2(page) -> dict:
    words = page.get_text("words")
    ticks = [(w[0], (w[1] + w[3]) / 2, float(w[4])) for w in words if w[4] in TICKS]
    lines = [(it[1], it[2], d.get("color")) for d in page.get_drawings() for it in d["items"] if it[0] == "l"]
    panels = sorted((d["rect"] for d in page.get_drawings()
                     if d.get("fill") and _dist(d["fill"], (0.918, 0.918, 0.949)) < 1e-4
                     and d["rect"].width > 100 and d["rect"].height > 100), key=lambda r: r.x0)
    pix = page.get_pixmap(dpi=DPI)
    s = DPI / 72.0
    rgb = lambda x, y: tuple(c / 255 for c in pix.pixel(int(round(x * s)), int(round(y * s)))[:3])

    if len(panels) != 2:
        raise SystemExit(f"expected 2 panel backgrounds in Figure 2, found {len(panels)}")

    out, offsets, pending = {}, [], []
    for rect, metric in zip(panels, ("AUROC", "AUPRC")):
        grid = sorted({round(p.y, 4) for p, q, c in lines if c and min(c) > 0.99
                       and abs(p.y - q.y) < 1e-3 and rect.x0 - 1 <= min(p.x, q.x) <= rect.x0 + 5})
        pts = [(min(grid, key=lambda g: abs(g - yc)), v) for tx, yc, v in ticks if rect.x0 - 40 < tx < rect.x0]
        n = len(pts)
        my, mv = sum(y for y, _ in pts) / n, sum(v for _, v in pts) / n
        a = sum((y - my) * (v - mv) for y, v in pts) / sum((y - my) ** 2 for y, _ in pts)
        b = mv - a * my
        resid = max(abs(a * y + b - v) for y, v in pts)
        if len(pts) != len(TICKS) or len({y for y, _ in pts}) != len(pts):
            raise SystemExit(f"{metric}: {len(pts)} tick labels matched to "
                             f"{len({y for y, _ in pts})} distinct gridlines, expected {len(TICKS)} of each")
        y_of = lambda v: (v - b) / a
        vert = sorted({round(p.x, 3): (p.x, min(p.y, q.y), max(p.y, q.y)) for p, q, c in lines
                       if c and _dist(c, GREY) < 1e-3 and abs(p.x - q.x) < 1e-3
                       and rect.x0 < p.x < rect.x1 and abs(p.y - q.y) > 0.3}.values())
        mid = (rect.x0 + rect.x1) / 2
        out[metric] = {"axis_fit_max_residual": resid}
        for split, eb in (("random", [v for v in vert if v[0] < mid]), ("cold", [v for v in vert if v[0] >= mid])):
            if len(eb) != 4:
                raise SystemExit(f"{metric} {split}: found {len(eb)} error bars, expected 4 "
                                 f"(method identity here is positional and cannot be checked otherwise)")
            spacing = (eb[-1][0] - eb[0][0]) / (len(eb) - 1)
            centres = [eb[0][0] - 2 * spacing, eb[0][0] - spacing] + [v[0] for v in eb]
            cell = {}
            for m, cx, e in zip(METHODS, centres, [None, None] + eb):
                xs, y = cx + 0.35 * spacing, y_of(0.605)
                body, top = rgb(xs, y), None
                while y > y_of(1.0) - 3:
                    c = rgb(xs, y)
                    if _dist(c, body) > 0.004 and _dist(c, GREY) > 0.02:
                        top = y + 0.5 / s
                        break
                    y -= 1.0 / s
                raster = a * top + b
                if e:
                    mean = a * (e[1] + e[2]) / 2 + b
                    offsets.append(mean - raster)
                    cell[m] = {"mean": round(mean, 4), "sd": round((a * e[1] - a * e[2]) / 2, 4),
                               "read_from": "vector error bar"}
                else:
                    cell[m] = {"raster_top": raster, "sd": None}
                    pending.append(cell[m])
            out[metric][split] = cell
    offset = sum(offsets) / len(offsets)
    for c in pending:
        c["mean"] = round(c.pop("raster_top") + offset, 3)
        c["read_from"] = f"600-dpi raster bar top + {offset:.4f} offset (about +/-0.001); no error bar drawn"
    out["raster_vs_vector_max_abs_offset"] = max(abs(o) for o in offsets)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", required=True, help="PDF of arXiv:2208.02194v2.")
    ap.add_argument("--out", default="runs/drugban_published_values.json")
    a = ap.parse_args()

    pdf = Path(a.pdf)
    doc = fitz.open(pdf)
    first = doc[0].get_text()
    if "arXiv:2208.02194v2" not in first:
        raise SystemExit("this is not arXiv:2208.02194v2 -- page 1 does not carry that identifier")
    fig_page = next(p for p in doc if "Figure 2. In-domain performance comparison on the Human" in p.get_text())

    result = {
        "source": {
            "citation": "Bai P, Miljkovic F, John B, Lu H. Interpretable bilinear attention network with "
                        "domain adaptation improves drug-target prediction. Nat Mach Intell 5, 126-136 (2023).",
            "file_read": "arXiv:2208.02194v2 (19 Jan 2023)",
            "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
            "figure2_page": fig_page.number + 1,
        },
        "table2_auroc_mean_sd_5_runs": table2(doc),
        "figure2_human_mean_sd_5_runs": figure2(fig_page),
    }
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    f2 = result["figure2_human_mean_sd_5_runs"]
    print(f"Figure 2 (Human), axis residual {f2['AUROC']['axis_fit_max_residual']:.1e}, "
          f"raster vs vector max offset {f2['raster_vs_vector_max_abs_offset']:.4f}")
    for metric in ("AUROC", "AUPRC"):
        for split in ("random", "cold"):
            row = "  ".join(f"{m} {v['mean']:.3f}" + (f"+/-{v['sd']:.3f}" if v["sd"] is not None else "*")
                            for m, v in f2[metric][split].items())
            print(f"  {metric} {split:<7} {row}")
    print("  * raster-read, no error bar")
    print("Table 2 (AUROC):")
    for k, row in result["table2_auroc_mean_sd_5_runs"].items():
        print(f"  {k:<14} " + "  ".join(f"{c}={'-' if v is None else f'{v[0]:.3f}+/-{v[1]:.3f}'}"
                                         for c, v in row.items()))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
