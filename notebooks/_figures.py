"""Redraw the data panels of the ModFormer paper's Figures 2, 3 and 4 from the released figure tables.

Every ``figNx`` function draws one panel onto an ``Axes`` and returns a small dict of what it plotted.
``figure2``, ``figure3`` and ``figure4`` lay the panels out on one figure, add the panel letters and return
``(figure, digest)``, where ``digest`` is a JSON-serializable summary: the panels drawn, the number of marks in
each, and the numbers the figure caption states.

Only matplotlib, numpy and pandas are needed to draw. When ``modformer`` is importable it also locates the cache
folder, checks each cached table against its published checksum and fetches missing tables. The tables are read from
the folder in ``MODFORMER_FIGURE_DATA`` or, failing that, from ``<modformer home>/figure-data`` (``$MODFORMER_HOME`` or
``~/.cache/modformer``).
Figure 4 panel a combines a drawing of the mutated duplexes with two data tracks and is not reproduced here.
"""
from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, PathPatch, Rectangle
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D, blended_transform_factory

# ---------------------------------------------------------------------------------------------- style
INK, LABEL = "#231F20", "#5A5A5B"
RED, PINK, BLUE, BROWN = "#EF4444", "#F7BEBD", "#0072B2", "#9C534D"
SLATE = "#618095"
BAND, BAND_EDGE = "#E6E6E7", "#9A9A9B"
FAMILY_COLOR = {"ModFormer": RED, "Training ablations": BROWN, "Alternative architectures": SLATE}
FS = 8.0
MM = 1 / 25.4

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
    "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
    "legend.fontsize": FS, "axes.titlesize": FS, "axes.titleweight": "normal",
    "text.color": INK, "axes.labelcolor": LABEL, "xtick.color": INK, "ytick.color": INK,
    "xtick.labelcolor": LABEL, "ytick.labelcolor": LABEL, "axes.edgecolor": INK, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "lines.linewidth": 0.8, "pdf.fonttype": 42, "figure.facecolor": "white", "savefig.facecolor": "white",
    "mathtext.default": "regular",
}
plt.rcParams.update(STYLE)
try:
    findfont(FontProperties(family=STYLE["font.sans-serif"][:-1]), fallback_to_default=False)
except ValueError:
    warnings.warn("None of Liberation Sans, Arial, Helvetica or Nimbus Sans is installed, so text is set in DejaVu "
                  "Sans, which is wider. Labels may overlap or run past the page edge. Install fonts-liberation to "
                  "draw the figures as published.", stacklevel=2)


# ---------------------------------------------------------------------------------------------- tables
DATA_DOI = "10.5281/zenodo.24300820"


def _weights():
    """The ``modformer.weights`` module, or None when ``modformer`` is not importable."""
    try:
        from modformer import weights
    except ImportError:
        return None
    return weights


def tables_folder() -> Path:
    """Folder holding the figure tables."""
    env = os.environ.get("MODFORMER_FIGURE_DATA")
    if env:
        return Path(env).expanduser()
    weights = _weights()
    if weights is not None:
        return weights.figure_data_dir()
    home = os.environ.get("MODFORMER_HOME")
    return (Path(home).expanduser() if home else Path.home() / ".cache" / "modformer") / "figure-data"


def _find_in(folder: Path, name: str) -> Path:
    """``name`` in ``folder``, or in the ``main`` or ``extended_data`` folder of the unpacked
    ``figure_source_data.zip`` (``fig*`` tables in the first, ``ed*`` in the second)."""
    sub = "extended_data" if name.startswith("ed") else "main"
    for path in (folder / name, folder / sub / name, folder / "figure_source_data" / sub / name):
        if path.exists():
            return path
    return folder / name


def load_table(name: str) -> pd.DataFrame:
    """Read one figure table (a CSV file name).

    With ``MODFORMER_FIGURE_DATA`` set, the table is read from that folder as it is. Otherwise ModFormer checks the
    cached table against its published checksum and fetches it when it is missing.
    """
    folder = tables_folder()
    if os.environ.get("MODFORMER_FIGURE_DATA"):
        path = _find_in(folder, name)
        if not path.exists():
            raise FileNotFoundError(
                f"figure table {name!r} not found in {folder}, which MODFORMER_FIGURE_DATA names. Put the CSV file "
                "there, or unset MODFORMER_FIGURE_DATA so that ModFormer fetches the tables into its cache."
            )
        return pd.read_csv(path)
    path = folder / name
    weights = _weights()
    if weights is not None and name in weights.FIGURE_DATA:
        weights.ensure_figure_data(names=[name])
    if not path.exists():
        if weights is None:
            raise FileNotFoundError(
                f"figure table {name!r} not found in {folder}. Install ModFormer (pip install '.[notebooks]' from the "
                f"repository) to fetch it automatically, or copy the CSV file from figure_source_data.zip (doi "
                f"{DATA_DOI}) into that folder, or set MODFORMER_FIGURE_DATA to a folder that holds it."
            )
        raise FileNotFoundError(
            f"figure table {name!r} is not one of the published figure tables. Put the CSV file in {folder}, or set "
            "MODFORMER_FIGURE_DATA to a folder that holds it."
        )
    return pd.read_csv(path)


# ---------------------------------------------------------------------------------------------- helpers
def _mm_size(ax) -> tuple[float, float]:
    """Width and height of an axes in millimeters."""
    box = ax.get_position()
    fig = ax.get_figure()
    return box.width * fig.get_figwidth() / MM, box.height * fig.get_figheight() / MM


def _add_axes_mm(fig, left, top, width, height, page_h):
    """An axes placed in millimeters from the page's top-left corner."""
    page_w = fig.get_figwidth() / MM
    return fig.add_axes([left / page_w, 1 - (top + height) / page_h, width / page_w, height / page_h])


def _letter(fig, letter, x_mm, y_mm, page_h):
    fig.text(x_mm * MM / fig.get_figwidth(), 1 - y_mm / page_h, letter, fontsize=9, fontweight="bold",
             va="top", ha="left", color=INK)


def _round(v, nd=6):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), nd)


def _one_line_to_two(label: str) -> str:
    """'HEK293T + IFN' -> 'HEK293T' newline '+ IFN' for a row label."""
    return label.replace(" + ", "\n+ ")


def _clean_axis(ax, left=True):
    ax.spines["bottom"].set_color(LABEL)
    ax.spines["left"].set_visible(left)
    if left:
        ax.spines["left"].set_color(LABEL)
    ax.tick_params(colors=LABEL, labelcolor=LABEL)


# =================================================================================== Figure 2
TEST_SETS = ["REDIportal", "Neuro-2a", "Mouse brain", "Alu duplexes", "Invertebrates"]
MODEL_ROWS = ["ModFormer", "No language model", "Trained without structure", "Adjacent partner only",
              "No triangle multiplication", "One-hot BiLSTM", "CNN on sequence only", "Language model only"]
ROW_Y = dict(zip(MODEL_ROWS, [0.0, 1.6, 2.6, 3.6, 4.6, 6.2, 7.2, 8.2]))   # a gap between model groups
_COL_STEP = 1.0 + 3.5 / 25.9      # one column plus its gutter, in column widths
_YLIM = (9.49, -0.6)
_F2 = {"AUC": dict(lim=(0.5, 1.0), ticks=(0.5, 0.75, 1.0), xlabel="AUC"),
       "Spearman": dict(lim=(0.0, 1.0), ticks=(0.0, 0.5, 1.0), xlabel="Spearman correlation")}


def _fig2_panel(ax, metric: str, header: bool, table: pd.DataFrame | None):
    t = load_table("fig2_cells.csv") if table is None else table
    t = t[t.metric == metric]
    spec = _F2[metric]
    lo, hi = spec["lim"]
    family = dict(zip(t.model, t.family))
    ax.set_ylim(*_YLIM)
    ax.set_xlim(0, len(TEST_SETS) * _COL_STEP - (_COL_STEP - 1))
    ticks, ticklabels, values, intervals, n_bars, n_sq = [], [], {}, {}, 0, 0
    for ci, ts in enumerate(TEST_SETS):
        x0 = ci * _COL_STEP
        for v in spec["ticks"]:
            ticks.append(x0 + (v - lo) / (hi - lo))
            ticklabels.append(f"{v:g}")
        ax.plot([x0, x0 + 1], [_YLIM[0], _YLIM[0]], color=LABEL, lw=0.5, clip_on=False)
        g = t[t.test_set == ts]
        if g.empty:  # a test set without measured rates
            ax.add_patch(Rectangle((x0, _YLIM[1]), 1, _YLIM[0] - _YLIM[1], fc="#F3F3F4", ec="#B4B4B5", lw=0.5,
                                   zorder=0))
            ax.text(x0 + 0.5, (_YLIM[0] + _YLIM[1]) / 2, "No measured rate", ha="center", va="center", color=LABEL)
            continue
        def sc(v, x0=x0):
            return x0 + (v - lo) / (hi - lo)

        for model in MODEL_ROWS:
            y = ROW_Y[model]
            for state in ("zero-shot", "fine-tuned"):
                r = g[(g.model == model) & (g.state == state)].iloc[0]
                if state == "zero-shot":
                    ax.barh(y, sc(r.value) - x0, left=x0, height=0.53, color=FAMILY_COLOR[family[model]], lw=0,
                            zorder=1)
                    ax.plot([sc(r.ci_low), sc(r.ci_high)], [y, y], color=INK, lw=0.6, zorder=3,
                            solid_capstyle="butt")
                    n_bars += 1
                else:
                    h = 0.8 / 4.29
                    ax.add_patch(Rectangle((sc(r.ci_low), y - h / 2), sc(r.ci_high) - sc(r.ci_low), h, fc="#D7D6D7",
                                           ec="#B2B2B3", lw=0.4, zorder=2))
                    ax.plot([sc(r.value)], [y], marker="s", ms=4.0, mfc="white", mec=LABEL, mew=0.6, ls="",
                            zorder=4)
                    n_sq += 1
                values[f"{ts}|{model}|{state}"] = _round(r.value)
                intervals[f"{ts}|{model}|{state}"] = [_round(r.ci_low), _round(r.ci_high)]
        ref_value = float(g[(g.model == "ModFormer") & (g.state == "zero-shot")].value.iloc[0])
        xr = sc(ref_value)
        ax.plot([xr] * 2, _YLIM, color=LABEL, lw=0.5, ls=(0, (1.2, 2.9)), zorder=1.5)
        if ci == 0 and metric == "AUC":      # names the dotted line: ModFormer's zero-shot value
            ax.text(xr - 0.8 / 25.9, _YLIM[0] - 0.45, "ModFormer", ha="right", va="center", color=LABEL)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.tick_params(axis="x", length=2.0, width=0.4, pad=2.5, colors=LABEL, labelcolor=LABEL)
    ax.set_yticks([ROW_Y[m] for m in MODEL_ROWS])
    ax.set_yticklabels(MODEL_ROWS, color=INK)
    ax.tick_params(axis="y", length=0, pad=4)
    for lab in ax.get_yticklabels():
        if lab.get_text() == "ModFormer":
            lab.set_fontweight("bold")
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel(spec["xlabel"], labelpad=3)
    if header:
        _fig2_header(ax)
    return {"n_marks": n_bars + n_sq, "n_bars": n_bars, "n_squares": n_sq, "values": values,
            "intervals": intervals}


def _fig2_header(ax):
    """Titles over the five columns and the two-level bracket above them."""
    _, h_mm = _mm_size(ax)
    tf = blended_transform_factory(ax.transData, ax.transAxes)

    def up(mm):
        return 1 + mm / h_mm

    for ci, ts in enumerate(TEST_SETS):
        ax.text(ci * _COL_STEP + 0.5, up(0.0), ts, transform=tf, ha="center", va="bottom", fontweight="bold",
                clip_on=False)
    groups = [(0, [("Transcriptome", 0, 2), ("Known dsRNA duplexes", 3, 4)], True),
              (1, [("Mouse", 0, 2), ("Human", 3, 3), ("Other animals", 4, 4)], False)]
    for level, items, bracket in groups:
        y = up(5.75 + 5.4 * (1 - level))
        for name, c0, c1 in items:
            xa, xb = c0 * _COL_STEP, c1 * _COL_STEP + 1
            xs, ys = ([xa, xa, xb, xb], [y - 0.9 / h_mm, y, y, y - 0.9 / h_mm]) if bracket else ([xa, xb], [y, y])
            ax.add_line(Line2D(xs, ys, transform=tf, color=LABEL, lw=0.5, clip_on=False))
            ax.text((xa + xb) / 2, y + 0.5 / h_mm, name, transform=tf, ha="center", va="bottom", clip_on=False,
                    color=LABEL if not bracket else INK, style="normal" if bracket else "italic")


def fig2a(ax, table=None):
    """Panel a: editing-site classification AUC on five test sets, zero-shot (bar) and fine-tuned (square)."""
    return _fig2_panel(ax, "AUC", True, table)


def fig2b(ax, table=None):
    """Panel b: Spearman correlation between predicted and measured editing rates."""
    return _fig2_panel(ax, "Spearman", False, table)


def figure2(table=None):
    """Figure 2, panels a and b. Returns ``(figure, digest)``."""
    t = load_table("fig2_cells.csv") if table is None else table
    w, h = 180.0, 129.6
    fig = plt.figure(figsize=(w * MM, h * MM))
    ax_h = (_YLIM[0] - _YLIM[1]) * 4.29
    out = {}
    for letter, top, fn in (("a", 16.08, fig2a), ("b", 72.11, fig2b)):
        ax = _add_axes_mm(fig, 33.5, top, 143.5, ax_h, h)
        out[letter] = fn(ax, t)
        ax.patch.set_alpha(0)
        _letter(fig, letter, 0.5, top - 15.5 if letter == "a" else top - 4.5, h)
    handles = [Rectangle((0, 0), 1, 1, color=c, lw=0) for c in FAMILY_COLOR.values()]
    handles.append((Line2D([], [], color="#D7D6D7", lw=3.0, solid_capstyle="butt"),
                    Line2D([], [], marker="s", ms=4.0, mfc="white", mec=LABEL, mew=0.6, ls="")))
    fig.legend(handles, [*FAMILY_COLOR, "Fine-tuned"], loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.4, handletextpad=0.5, columnspacing=1.8)
    zero_shot_auc = {k.split("|")[0]: v for k, v in out["a"]["values"].items() if k.endswith("ModFormer|zero-shot")}
    digest = {"figure": "2", "panels": {k: {"marks": v["n_marks"], "values": v["values"], "intervals": v["intervals"]}
                                        for k, v in out.items()},
              "headline": {"modformer_zero_shot_auc": zero_shot_auc}}
    return fig, digest


# =================================================================================== Figure 3
_FIG3A_XLABEL = "Top-ranked adenosines (%)"
BASE_COLOR = {"A": "#4F7E54", "C": "#035A6F", "G": "#B9901D", "U": "#BD74B1", "-": "#D1D5DB", "edit": "#991B1B",
              "backbone": "#8A8A8C", "bond": "#B9B9BA", "loop": "#E5E5E6"}
PAIRS = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}
MIN_SELECTED = 30            # a base needs this many selected sites to be drawn
MIN_LETTER_MM = 0.35         # smaller letters are left out
RED_BAND, SLATE_BAND, OVERLAP_BAND = "#F7BEBD", "#D8E1E8", "#D1A8AC"


def fig3a(ax, curves=None, points=None):
    """Panel a: recovery of highly edited Alu sites versus the fraction of candidates screened."""
    c = load_table("fig3a_recall_curves.csv") if curves is None else curves
    p = load_table("fig3a_recall_points.csv") if points is None else points
    prev = float(p.prevalence.iloc[0])
    lm = "Language model only"
    col = {"ModFormer": RED, lm: SLATE}
    band = {"ModFormer": RED_BAND, lm: SLATE_BAND}
    dash = (0, (3.0, 1.6))
    data = {}
    for m in col:
        zs = c[(c.model == m) & (c.state == "zero-shot")].sort_values("kept")
        ft = c[(c.model == m) & (c.state == "fine-tuned")].sort_values("kept")
        data[m] = (zs.kept.to_numpy() * 100, zs.recall.to_numpy() * 100, ft.recall.to_numpy() * 100)
    x = data["ModFormer"][0]
    lo = {m: np.minimum(d[1], d[2]) for m, d in data.items()}
    hi = {m: np.maximum(d[1], d[2]) for m, d in data.items()}
    for m in (lm, "ModFormer"):
        ax.fill_between(x, lo[m], hi[m], color=band[m], lw=0, zorder=1)
    olo, ohi = np.maximum(lo[lm], lo["ModFormer"]), np.minimum(hi[lm], hi["ModFormer"])
    ax.fill_between(x, olo, ohi, where=ohi > olo, interpolate=True, color=OVERLAP_BAND, lw=0, zorder=1.5)
    n_curves = 0
    for m, z in ((lm, 3), ("ModFormer", 4)):
        ax.plot(x, data[m][1], color=col[m], lw=0.8, ls=dash, zorder=z)
        ax.plot(x, data[m][2], color=col[m], lw=1.0, zorder=z + 0.5)
        n_curves += 2
    dot = (0, (1.0, 1.6))
    ax.plot(x, x, color=LABEL, lw=0.5, ls=dot, zorder=0)
    ax.plot(x, np.minimum(100.0, x / prev), color=LABEL, lw=0.5, ls=dot, zorder=0)
    ax.axvline(prev * 100, color=LABEL, lw=0.5, ls=dot, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(1, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([1, 10, 100])
    ax.set_xticklabels(["1", "10", "100"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel(_FIG3A_XLABEL, labelpad=1.5)
    ax.set_ylabel("Sites ≥10% edited found (%)", labelpad=1.5)
    t = dict(color=LABEL, linespacing=1.0)
    ax.text(prev * 100 * 1.07, 98, f"Top\n{prev * 100:.0f}%", ha="left", va="top", **t)
    ax.text(3.6, 50, "Perfect\nranking", ha="center", va="center", **t)
    ax.text(97, 14, "Random\nranking", ha="right", va="center", **t)
    for i, (lab, ls, lw) in enumerate((("Fine-tuned", "-", 1.0), ("Zero-shot", dash, 0.8))):
        yk = 95 - i * 7.5
        ax.add_line(Line2D([1.12, 1.75], [yk, yk], color=LABEL, lw=lw, ls=ls))
        ax.text(1.9, yk, lab, ha="left", va="center", **t)
    handles = [Rectangle((0, 0), 1, 1, fc=band[m], ec=col[m], lw=0.6) for m in col]
    ax.legend(handles, list(col), loc="lower left", bbox_to_anchor=(-0.19, 1.0), ncol=2, frameon=False,
              handlelength=1.4, handleheight=0.9, handletextpad=0.3, columnspacing=0.6, borderaxespad=0.2,
              labelcolor=INK)
    at = {}
    for m in col:
        for j, state in ((1, "zero-shot"), (2, "fine-tuned")):
            for kept in (0.01, 0.1):
                at[f"{m}|{state}|{kept:g}"] = _round(np.interp(np.log10(kept), np.log10(x / 100), data[m][j]) / 100)
    return {"n_marks": n_curves, "prevalence": _round(prev), "recall_at_kept": at}


def _enrichment(counts: pd.DataFrame, n_selected: int) -> pd.DataFrame:
    """Letter heights: log2 enrichment of a base in the selected sites over all sites, kept when positive and
    supported by enough selected sites."""
    t = counts.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        t["log2_fold"] = np.log2(t.fg / t.bg)
    t["h"] = np.where((t.fg * n_selected >= MIN_SELECTED) & (t.log2_fold > 0), t.log2_fold, 0.0)
    return t


def _consensus(t, strand, k):
    g = t[(t.strand == strand) & (t.offset == k) & (t.base != "-") & (t.h > 0)]
    return "" if g.empty else g.loc[g.h.idxmax(), "base"]


def fig3b(ax, counts=None, meta=None):
    """Panel b: sequence and opposing-base enrichment among high-scoring adenosines, drawn as two logos around the
    edited adenosine (edited strand above, opposing strand below) with the duplex between them."""
    cnt = load_table("fig3b_logo_counts.csv") if counts is None else counts
    m = load_table("fig3b_logo_meta.csv") if meta is None else meta
    meta_v = dict(zip(m.quantity, m.value))
    w = int(meta_v["window_nt"])
    offs = list(range(-w, w + 1))
    t = _enrichment(cnt[cnt.offset.abs() <= w], int(meta_v["n_selected"]))
    col_h = t.groupby(["strand", "offset"]).h.sum()
    a_h = float(col_h["own"].drop(0, errors="ignore").max())          # the edited A is as tall as the tallest column
    own_max, pair_max = a_h, float(col_h["paired"].max())
    w_mm, h_mm = _mm_size(ax)
    duplex = 19.0
    s = (h_mm - duplex) / (own_max + pair_max)                         # mm per log2 unit
    low_h, up_h = s * pair_max, s * own_max
    xl = (-w - 0.6, w + 0.6)
    mm_x = w_mm / (xl[1] - xl[0])
    ax.set_xlim(*xl)
    ax.set_ylim(0, h_mm)                                               # y in millimeters, from the bottom
    ax.axis("off")
    prop = FontProperties(family="sans-serif", weight="bold")

    def glyph(ch, x, y0, height_mm, color, gw=0.84):
        path = TextPath((0, 0), ch, size=1, prop=prop)
        bb = path.get_extents()
        patch = PathPatch(path, fc=color, ec="none")
        patch.set_transform(Affine2D().translate(-bb.x0, -bb.y0).scale(gw / bb.width, height_mm / bb.height)
                            .translate(x - gw / 2, y0) + ax.transData)
        ax.add_patch(patch)

    letters, values = 0, {}
    for strand, down in (("own", False), ("paired", True)):
        base_y = low_h + duplex if not down else low_h                  # the edge facing the duplex
        for k in offs:
            g = t[(t.strand == strand) & (t.offset == k)].sort_values("h")
            used = 0.0
            for b, h, lf in zip(g.base, g.h, g.log2_fold):
                if h <= 0 or not np.isfinite(h):
                    continue
                hm = h * s
                y0 = base_y - used - hm if down else base_y + used
                values[f"{strand}|{k}|{b}"] = _round(h)
                if b == "-":
                    ax.add_patch(Rectangle((k - 0.42, y0), 0.84, hm, fc=BASE_COLOR["-"], ec="none"))
                elif hm >= MIN_LETTER_MM:
                    glyph(b, k, y0, hm, BASE_COLOR[b])
                    letters += 1
                used += hm
    glyph("A", 0, low_h + duplex, a_h * s, BASE_COLOR["edit"])
    letters += 1
    # the enrichment scale: 1- to 2-fold on the upper logo
    xb, tick = xl[0] + 0.02, 0.9 / mm_x
    y_top0 = low_h + duplex
    ax.plot([xb + tick, xb, xb, xb + tick], [y_top0, y_top0, y_top0 + s, y_top0 + s], color=LABEL, lw=0.5,
            clip_on=False, solid_joinstyle="miter")
    for v, lab in ((0, "1"), (1, "2")):
        ax.text(xb - 0.5 / mm_x, y_top0 + v * s, lab, ha="right", va="center", color=LABEL)
    ax.text(xb - 6.2 / mm_x, h_mm, "Enrichment (fold)", rotation=90, ha="center", va="top", color=LABEL)
    ax.text(xl[1], h_mm, "Edited strand", ha="right", va="top", color=LABEL)
    # the duplex between the logos
    r, bulge, lw_bb = 1.75, 0.8, 0.8
    yu = low_h + duplex - 1.0 - 2.9 - 0.4 - bulge - r
    yl = yu - 8.4
    top = {k: ("A" if k == 0 else _consensus(t, "own", k)) for k in offs}
    bot = {k: _consensus(t, "paired", k) for k in offs}
    ytop = {k: yu + (bulge if k == 0 else 0) for k in offs}
    ybot = {k: yl - (bulge if k == 0 else 0) for k in offs}
    xe, xr = xl[0] + 0.05, xl[1] - 0.05
    for y in (yu, yl):
        for xs in ([xe, offs[0]], [offs[-1], xr]):
            ax.plot(xs, [y, y], color=BASE_COLOR["backbone"], lw=lw_bb, ls=(0, (1.5, 1.2)), zorder=1)
    ax.plot(offs, [ytop[k] for k in offs], color=BASE_COLOR["backbone"], lw=lw_bb, zorder=1)
    ax.plot(offs, [ybot[k] for k in offs], color=BASE_COLOR["backbone"], lw=lw_bb, zorder=1)
    for k in offs:
        if k != 0 and (top[k], bot[k]) in PAIRS:
            ax.plot([k, k], [ybot[k] + r, ytop[k] - r], color=BASE_COLOR["bond"], lw=0.6, zorder=2,
                    solid_capstyle="butt")
    for k in offs:
        for y, base, edit in ((ytop[k], top[k], k == 0), (ybot[k], bot[k], False)):
            fc = BASE_COLOR["loop"] if not base else (BASE_COLOR["edit"] if edit else BASE_COLOR[base])
            ax.add_patch(Ellipse((k, y), 2 * r / mm_x, 2 * r, fc=fc, ec="white", lw=0.4, zorder=3))
            if base:
                ax.text(k, y, base, ha="center", va="center_baseline", color="white", fontweight="bold", zorder=4)
        if k != 0:
            ax.text(k, ytop[k] + r + 0.4, f"{k:+d}".replace("-", "−"), ha="center", va="bottom", color=LABEL)
    for x, ha, labs in ((xe - 0.05, "right", ("5′", "3′")), (xr + 0.05, "left", ("3′", "5′"))):
        ax.text(x, yu, labs[0], ha=ha, va="center", color=LABEL)
        ax.text(x, yl, labs[1], ha=ha, va="center", color=LABEL)
    return {"n_marks": letters, "n_selected": int(meta_v["n_selected"]), "n_evaluated": int(meta_v["n_evaluated"]),
            "log2_enrichment": values}


def fig3c(ax, table=None):
    """Panel c: predicted change in editing log-odds after creating or removing mismatches opposite the adenosine."""
    v = load_table("fig3c_paired_edits.csv") if table is None else table
    w_mm, h_mm = _mm_size(ax)
    lim = 0.72
    ax.set_xlim(-lim, lim)
    mismatches = list(dict.fromkeys(v.mismatch))
    ypos, y = {}, 0.0
    for mm_ in mismatches:
        for j, edit in enumerate(("Repair", "Create", "Control")):
            ypos[(mm_, edit)] = y + j
        y += 3.6
    ax.set_ylim(y - 3.6 + 2.5, -0.5)
    ax.axvline(0, color=LABEL, lw=0.5, ls=(0, (1.0, 1.6)), zorder=1)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    _clean_axis(ax, left=False)
    tf = blended_transform_factory(ax.transAxes, ax.transData)
    ax_left_mm = 35.0

    def fx(x_mm):                       # position in mm from the slot's left edge -> axes fraction
        return (x_mm - ax_left_mm) / w_mm

    means, intervals, n_points = {}, {}, 0
    for r in v.itertuples():
        yy = ypos[(r.mismatch, r.edit)]
        control = r.edit == "Control"
        if control:
            ax.add_patch(Rectangle((-r.control_margin, yy - 0.4), 2 * r.control_margin, 0.8, fc="#E6E6E6",
                                   ec="#B2B2B2", lw=0.5, zorder=0.5))
        ax.plot([r.ci_low, r.ci_high], [yy, yy], color=INK, lw=0.6, zorder=3, solid_capstyle="butt")
        ax.plot(r.mean, yy, "o", ms=3.2, mfc="white" if control else INK, mec=INK, mew=0.6, zorder=4)
        n_points += 1
        start, end = r.label.split(" → ")
        if control:
            ax.text(fx(0.5), yy, "Control", transform=tf, ha="left", va="center", color=INK, clip_on=False)
        ax.text(fx(17.6), yy, start, transform=tf, ha="right", va="center", color=INK, clip_on=False)
        ax.text(fx(22.6), yy, end, transform=tf, ha="left", va="center", color=INK, clip_on=False)
        ax.annotate("", xy=(fx(22.0), yy), xytext=(fx(18.3), yy), xycoords=tf, textcoords=tf, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-|>", lw=0.5, color=INK, mutation_scale=4.0, shrinkA=0, shrinkB=0))
        ax.text(fx(33.5), yy, f"{int(r.n_families)}", transform=tf, ha="right", va="center", color=LABEL,
                clip_on=False)
        means[f"{r.mismatch}|{r.edit}"] = _round(r.mean)
        intervals[f"{r.mismatch}|{r.edit}"] = [_round(r.ci_low), _round(r.ci_high), int(r.n_families),
                                                _round(r.control_margin)]
    ax.text(fx(33.5), 1.0 + 2.6 / h_mm, "n", transform=ax.transAxes, ha="right", va="center", color=LABEL,
            style="italic", clip_on=False)
    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_xticklabels(["−0.5", "0", "0.5"])
    ax.set_xlabel("Change in editing logit", labelpad=1.5)
    ky = -4.2 / h_mm
    ax.add_patch(Rectangle((fx(0.5), ky - 1.3 / h_mm), 3.6 / w_mm, 2.6 / h_mm, transform=ax.transAxes, fc="#E6E6E6",
                           ec="#B2B2B2", lw=0.5, clip_on=False))
    ax.text(fx(4.9), ky, "Control margin", transform=ax.transAxes, ha="left", va="center", color=INK, clip_on=False)
    return {"n_marks": n_points, "mean_change": means, "intervals": intervals}


def figure3(curves=None, points=None, counts=None, meta=None, edits=None):
    """Figure 3, panels a-c. Returns ``(figure, digest)``."""
    w, h = 180.0, 62.0
    fig = plt.figure(figsize=(w * MM, h * MM))
    top, bot = 8.5, 53.0
    out = {}
    out["a"] = fig3a(_add_axes_mm(fig, 10.5, top, 44.0, bot - top, h), curves, points)
    out["b"] = fig3b(_add_axes_mm(fig, 58 + 10.0, top, 41.0, 59.5 - top, h), counts, meta)
    out["c"] = fig3c(_add_axes_mm(fig, 114 + 35.0, top, 27.5, bot - top, h), edits)
    for letter, x in (("a", 0.5), ("b", 58.5), ("c", 114.5)):
        _letter(fig, letter, x, 0.3, h)
    p = load_table("fig3a_recall_points.csv") if points is None else points
    p10 = p[np.isclose(p.kept, 0.1)]
    headline = {
        "create_a_c_log_odds": out["c"]["mean_change"]["A:C|Create"],
        "repair_a_c_log_odds": out["c"]["mean_change"]["A:C|Repair"],
        "max_abs_control_change": _round(max(abs(v) for k, v in out["c"]["mean_change"].items()
                                             if k.endswith("|Control"))),
        "n_selected": out["b"]["n_selected"], "n_evaluated": out["b"]["n_evaluated"],
        "alu_adenosines": int(p.n_sites.iloc[0]), "alu_highly_edited": int(p.n_high.iloc[0]),
        "highly_edited_fraction": _round(p.prevalence.iloc[0]),
        "recall_at_10pct": {f"{r.model}|{r.state}": _round(r.recall) for r in p10.itertuples()},
    }
    digest = {"figure": "3", "panels": {
        "a": {"marks": out["a"]["n_marks"], "values": out["a"]["recall_at_kept"]},
        "b": {"marks": out["b"]["n_marks"], "values": out["b"]["log2_enrichment"]},
        "c": {"marks": out["c"]["n_marks"], "values": out["c"]["mean_change"], "intervals": out["c"]["intervals"]}},
        "headline": headline}
    return fig, digest


# =================================================================================== Figure 4
E_COLOR = {"HEK293T": "#56B4E9", "MCF7": "#009E73", "ADAR1-KO + ADAR2-OE": "#A6611A"}
ZS = "ModFormer (zero-shot)"
C_MODELS = (ZS, "Trained without structure", "Joint fine-tune (129 designs)", "Library-trained (129 designs)")
C_DY = (-0.325, -0.108, 0.108, 0.325)
COND_Y = (0.0, 1.0, 2.0, 3.35, 4.35)
D_DY, D_BAR_H, D_BOX_PAD = 0.215, 0.35, 0.08
F_DY = 0.19
_KEY_PITCH = 1.25 * FS / 72 / MM          # line pitch of a two-line key, mm
_F4_PAGE_H = 117.4


def _key(ax, handles, labels, dx_mm, dy_mm, **kw):
    """A legend whose top-left corner sits ``dx_mm`` right of and ``dy_mm`` below the axes' top-left corner
    (negative values go left and up)."""
    fig = ax.get_figure()
    box = ax.get_position()
    x = box.x0 + dx_mm * MM / fig.get_figwidth()
    y = box.y1 - dy_mm * MM / fig.get_figheight()
    kw = {"ncol": 1, "columnspacing": 0.8, **kw}
    return fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(x, y), frameon=False, handlelength=1.0,
                      handletextpad=0.35, labelspacing=0.25, borderpad=0, borderaxespad=0, labelcolor=INK, **kw)


def _cond_axis(ax, labels):
    ax.set_ylim(COND_Y[-1] + 0.6, COND_Y[0] - 0.6)
    ax.set_yticks(COND_Y)
    ax.set_yticklabels([_one_line_to_two(c) for c in labels], linespacing=1.0, color=INK)
    ax.tick_params(axis="y", length=0, pad=2)


def _library_marks(c: pd.DataFrame) -> int:
    marks = {int(re.search(r"\((\d+) designs\)", m).group(1)) for m in c.model if "Library-trained" in m}
    assert len(marks) == 1, marks
    return marks.pop()


def fig4b(ax, budget=None, conditions=None):
    """Panel b: editing-change Spearman correlation versus the number of library constructs used for training."""
    b = load_table("fig4b_budget.csv") if budget is None else budget
    c = load_table("fig4c_conditions.csv") if conditions is None else conditions
    r = b[b.series.str.startswith("Library-trained")].sort_values("n_train_constructs")
    zero_shot = float(b[b.series == ZS].spearman.iloc[0])
    lib = (r.n_train_constructs / r.pct_of_library * 100).round().astype(int)
    assert lib.nunique() == 1, "the budgets imply different library sizes"
    library, mark = int(lib.iloc[0]), _library_marks(c)
    mark_row = r[r.n_train_constructs == mark]
    assert len(mark_row) == 1, "the library-trained models of panel c use a budget that panel b does not draw"
    ax.axhline(zero_shot, color=RED, lw=0.8, zorder=1)
    ax.plot(r.n_train_constructs, r.spearman, color=BLUE, lw=1.1, zorder=2)
    filled, n_points = [], 0
    for q in r.itertuples():
        f = str(q.paired_above_release) == "True"
        ax.plot(q.n_train_constructs, q.spearman, "o", ms=3.6, mec=BLUE, mew=0.9, mfc=BLUE if f else "white",
                zorder=3)
        n_points += 1
        if f:
            filled.append(int(q.n_train_constructs))
    ax.axvline(mark, color=LABEL, lw=0.7, ls=(0, (1, 1.5)), zorder=0)
    ax.text(mark / 1.04, 0.885, f"{mark} ({mark_row.pct_of_library.iloc[0]:.0f}%)", ha="right", va="top",
            color=LABEL)
    ax.set_xscale("log")
    ticks = [int(v) for v in r.n_train_constructs]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) if t in (13, 26, 39, 65, 129, 258) else "" for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(10.9, 291.0)
    ax.set_ylim(0.3, 0.9)
    ax.set_yticks([0.3, 0.5, 0.7, 0.9])
    ax.set_xlabel(f"Library constructs trained on\n(n of {library:,})", labelpad=2, linespacing=1.1)
    ax.set_ylabel("Predicted vs measured\nediting change (Spearman)", labelpad=2, linespacing=1.1)
    _key(ax, [Line2D([], [], color=BLUE, lw=1.1, marker="o", ms=3.6), Line2D([], [], color=RED, lw=0.8)],
         ["Library-trained, HEK293T", ZS.replace(" (", ", ").rstrip(")")], -10.5, -8.7)
    vals = {f"n={int(q.n_train_constructs)}": _round(q.spearman) for q in r.itertuples()}
    vals["ModFormer (zero-shot)"] = _round(zero_shot)
    return {"n_marks": n_points + 1, "values": vals, "filled": filled, "mark": mark,
            "mark_pct": int(round(mark_row.pct_of_library.iloc[0])), "library": library}


def _c_style():
    return {ZS: dict(marker="o", ms=3.4, mfc="white", mec=RED, mew=0.8, color=RED),
            C_MODELS[1]: dict(marker="D", ms=2.8, mfc=BROWN, mec=BROWN, color=BROWN),
            C_MODELS[2]: dict(marker="s", ms=3.0, mfc="white", mec=LABEL, mew=0.8, color=LABEL),
            C_MODELS[3]: dict(marker="*", ms=5.0, mfc=BLUE, mec=BLUE, mew=0.3, color=BLUE)}


def fig4c(ax, table=None, order=None):
    """Panel c: editing-change Spearman correlation for four models in five cellular conditions."""
    t = load_table("fig4c_conditions.csv") if table is None else table
    conds = list(dict.fromkeys(load_table("fig4d_top_pick.csv").condition)) if order is None else list(order)
    style = _c_style()
    ax.spines["left"].set_visible(False)
    ax.axvline(0, color=LABEL, lw=0.7, ls=(0, (1, 1.5)), zorder=0)
    vals, intervals, n_points = {}, {}, 0
    for r in t.itertuples():
        st = dict(style[r.model])
        col = st.pop("color")
        y = COND_Y[conds.index(r.condition)] + C_DY[C_MODELS.index(r.model)]
        ax.plot([r.ci_low, r.ci_high], [y, y], color=col, lw=0.8, zorder=2)
        ax.plot(r.spearman, y, ls="none", zorder=3, **st)
        n_points += 1
        vals[f"{r.condition}|{r.model}"] = _round(r.spearman)
        intervals[f"{r.condition}|{r.model}"] = [_round(r.ci_low), _round(r.ci_high)]
    unscored = [c for c in conds if "Library-trained (129 designs)" not in set(t[t.condition == c].model)]
    for c in unscored:
        ax.text(0.07, COND_Y[conds.index(c)] + 0.3, "not scored", color=LABEL, va="center")
    ax.set_xlim(-0.045, 0.81)
    ax.set_xticks([0, 0.4, 0.8])
    ax.set_xticklabels(["0", "0.4", "0.8"])
    _cond_axis(ax, conds)
    ax.set_xlabel("Predicted vs measured\nediting change (Spearman)", labelpad=2, linespacing=1.1)
    names = ["ModFormer, zero-shot", "Without structure", "Joint fine-tune", "Library-trained"]
    handles = [Line2D([], [], ls="none", **{k: v for k, v in style[m].items() if k != "color"}) for m in C_MODELS]
    _key(ax, handles, names, -16.5, -8.7, ncol=2)
    return {"n_marks": n_points, "values": vals, "intervals": intervals, "not_scored": unscored}


def fig4d(ax, table=None):
    """Panel d: share of constructs whose predicted highest-gain (or highest-loss) site changed as predicted."""
    d = load_table("fig4d_top_pick.csv") if table is None else table
    conds = list(d.condition)
    vals, intervals, n_bars = {}, {}, 0
    for i, r in enumerate(d.itertuples()):
        y = COND_Y[i]
        for dy, v, lo, hi, color, tag in ((-D_DY, r.up_rate, r.up_random_lo, r.up_random_hi, RED, "up"),
                                          (D_DY, r.down_rate, r.down_random_lo, r.down_random_hi, PINK, "down")):
            ax.barh(y + dy, 100 * v, height=D_BAR_H, color=color, lw=0, zorder=2)
            n_bars += 1
            box = (100 * lo, y + dy - D_BAR_H / 2 - D_BOX_PAD)
            bw, bh = 100 * (hi - lo), D_BAR_H + 2 * D_BOX_PAD
            ax.add_patch(Rectangle(box, bw, bh, fc=BAND, ec="none", zorder=1))
            ax.add_patch(Rectangle(box, bw, bh, fc="none", ec=BAND_EDGE, lw=0.6, alpha=0.85, zorder=3))
            vals[f"{r.condition}|{tag}"] = _round(100 * v, 4)
            intervals[f"{r.condition}|{tag}"] = [_round(100 * lo, 4), _round(100 * hi, 4)]
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    _cond_axis(ax, conds)
    ax.set_xlabel("Constructs where the zero-shot\npick changed as predicted (%)", labelpad=2, linespacing=1.1)
    h = [Rectangle((0, 0), 1, 1, fc=RED, lw=0), Rectangle((0, 0), 1, 1, fc=BAND, ec=BAND_EDGE, lw=0.6),
         Rectangle((0, 0), 1, 1, fc=PINK, lw=0)]
    _key(ax, h[:2], ["Top-ranked: increased", "Random site"], -16.5, -8.7, ncol=2)
    _key(ax, h[2:], ["Bottom-ranked: decreased"], -16.5, -8.7 + _KEY_PITCH)
    return {"n_marks": n_bars, "values": vals, "intervals": intervals}


def fig4e(ax, table=None):
    """Panel e: fraction of the top-ranked sites with measured editing gains versus the number tested."""
    e = load_table("fig4e_topn.csv") if table is None else table
    labels = list(dict.fromkeys(e.condition))
    env = e.groupby("budget", sort=True).agg(lo=("shuffled_lo", "min"), hi=("shuffled_hi", "max")).reset_index()
    ax.fill_between(env.budget, env.lo, env.hi, facecolor=BAND, edgecolor=BAND_EDGE, lw=0.5, zorder=0)
    last = env.iloc[-1]
    ax.text(float(last.budget) * 0.88, 0.5 * (last.lo + last.hi), "shuffled predictions", ha="right", va="center",
            color=LABEL)
    vals, n_points = {}, 0
    for lab in labels:
        g = e[e.condition == lab].sort_values("budget")
        col = E_COLOR[lab]
        ax.plot(g.budget, g.share_increased, color=col, lw=1.0, marker="o", ms=3.0, mfc="white", mec=col, mew=0.8,
                zorder=2)
        for r in g.itertuples():
            vals[f"{lab}|{int(r.budget)}"] = _round(r.share_increased)
            n_points += 1
    ax.set_xscale("log")
    budgets = sorted(int(b) for b in env.budget)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) if b in (10, 50, 100, 500, 2000) else "" for b in budgets])
    ax.minorticks_off()
    ax.set_xlim(9.0, 2790.0)
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_xlabel("Top-ranked sites tested", labelpad=2)
    ax.set_ylabel("ModFormer zero-shot: share of\ntop-ranked sites that increased", labelpad=2, linespacing=1.1)
    h = {lab: Line2D([], [], color=E_COLOR[lab], lw=1.2) for lab in labels}
    order = [lab for lab in E_COLOR if lab in labels]
    _key(ax, [h[lab] for lab in order[:2]], order[:2], -10.5, -8.7, ncol=2)
    _key(ax, [h[lab] for lab in order[2:]], order[2:], -10.5, -8.7 + _KEY_PITCH)
    return {"n_marks": n_points, "values": vals,
            "band": {f"{int(r.budget)}": [_round(r.lo), _round(r.hi)] for r in env.itertuples()}}


_F_XLIM = (-0.815, 0.97)
_F_STEP = (_F_XLIM[1] - _F_XLIM[0]) * (1 + 8.0 / 47.25)      # two heads side by side with an 8 mm gutter


def fig4f(ax, table=None):
    """Panel f: editing-change Spearman correlation on three endogenous substrates, for two models and two heads."""
    t = load_table("fig4f_endogenous.csv") if table is None else table
    groups = list(dict.fromkeys(t.substrate))
    heads = list(dict.fromkeys(t["head"]))
    n = t.groupby("substrate", sort=False).n_variants.first()
    ax.set_ylim(len(groups) - 0.45, -0.6)
    ax.set_xlim(_F_XLIM[0], _F_XLIM[0] + _F_STEP * len(heads) - (_F_STEP - (_F_XLIM[1] - _F_XLIM[0])))
    tf = blended_transform_factory(ax.transData, ax.transAxes)
    ticks, tick_labels, vals, intervals, n_points = [], [], {}, {}, 0
    for j, head in enumerate(heads):
        off = j * _F_STEP
        ax.plot([off, off], [-0.6, len(groups) - 0.45], color=LABEL, lw=0.7, ls=(0, (1, 1.5)), zorder=0)
        for r in t[t["head"] == head].itertuples():
            zero_shot = r.model == ZS
            y = groups.index(r.substrate) + (-F_DY if zero_shot else F_DY)
            color = RED if zero_shot else BLUE
            ax.plot([off + r.ci_low, off + r.ci_high], [y, y], color=color, lw=0.8, zorder=2)
            if zero_shot:
                ax.plot(off + r.spearman, y, "o", ms=3.4, mfc="white", mec=RED, mew=0.8, zorder=3)
            else:
                ax.plot(off + r.spearman, y, "*", ms=5.0, mew=0.3, color=BLUE, zorder=3)
            vals[f"{r.substrate}|{head}|{r.model}"] = _round(r.spearman)
            intervals[f"{r.substrate}|{head}|{r.model}"] = [_round(r.ci_low), _round(r.ci_high)]
            n_points += 1
        ax.plot([off + _F_XLIM[0], off + _F_XLIM[1]], [len(groups) - 0.45] * 2, color=LABEL, lw=0.6, clip_on=False)
        ticks += [off - 0.5, off, off + 0.5]
        tick_labels += ["−0.5", "0", "0.5"]
        ax.text(off, 1.0 + 0.6 / 36, head, transform=tf, ha="center", va="bottom", color=INK)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([f"{g}\nn = {int(n[g])}" for g in groups], linespacing=1.0, color=INK)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlabel("Predicted vs measured\nediting change (Spearman)", labelpad=2, linespacing=1.1)
    h = [Line2D([], [], ls="none", marker="o", ms=3.4, mfc="white", mec=RED, mew=0.8),
         Line2D([], [], ls="none", marker="*", ms=5.0, mew=0.3, color=BLUE)]
    _key(ax, h, ["ModFormer, zero-shot", "Library-trained"], -10.5, -8.7, ncol=2)
    return {"n_marks": n_points, "values": vals, "intervals": intervals, "n_variants": {g: int(n[g]) for g in groups}}


def figure4(budget=None, conditions=None, top_pick=None, topn=None, endogenous=None):
    """Figure 4, data panels b-f (panel a is not reproduced). Returns ``(figure, digest)``."""
    w, h = 180.0, _F4_PAGE_H
    fig = plt.figure(figsize=(w * MM, h * MM))
    row1, row2, plot_h = 10.2, 68.9, 36.0
    ax = {"b": _add_axes_mm(fig, 13.5, row1, 44.5, plot_h, h), "c": _add_axes_mm(fig, 79.5, row1, 38.0, plot_h, h),
          "d": _add_axes_mm(fig, 139.0, row1, 38.0, plot_h, h), "e": _add_axes_mm(fig, 13.5, row2, 44.5, plot_h, h),
          "f": _add_axes_mm(fig, 75.5, row2, 102.5, plot_h, h)}
    top_pick = load_table("fig4d_top_pick.csv") if top_pick is None else top_pick
    out = {"b": fig4b(ax["b"], budget, conditions),
           "c": fig4c(ax["c"], conditions, list(top_pick.condition)),
           "d": fig4d(ax["d"], top_pick), "e": fig4e(ax["e"], topn), "f": fig4f(ax["f"], endogenous)}
    for letter, x, y in (("b", 0.3, 1.5), ("c", 60.3, 1.5), ("d", 119.8, 1.5), ("e", 0.3, 60.2), ("f", 62.3, 60.2)):
        _letter(fig, letter, x, y, h)
    digest = {"figure": "4",
              "panels": {k: {"marks": v["n_marks"], "values": v["values"],
                             **({"intervals": v["intervals"]} if "intervals" in v else {})} for k, v in out.items()},
              "headline": {"budget_mark": out["b"]["mark"], "budget_mark_pct": out["b"]["mark_pct"],
                           "library_designs": out["b"]["library"], "filled_budgets": out["b"]["filled"],
                           "not_scored": out["c"]["not_scored"], "n_variants": out["f"]["n_variants"],
                           "shuffled_band": out["e"]["band"]},
              "panel_a": "drawing, not reproduced"}
    return fig, digest
