"""Redraw the data panels of the ModFormer paper's Extended Data Figures 1-3 from their source tables.

``ed1()``: Extended Data Figure 1, benchmarks against published predictors (panels b-d).
``ed2()``: Extended Data Figure 2, training data and held-out predictions (panels a-f).
``ed3()``: Extended Data Figure 3, the library screen (panels a-f).

Each returns ``(figure, digest)``, in the shape of the digests of ``_figures.py``. The digest lists, per panel, the number of
marks and the plotted values. Extended Data Figure 1 lists AUCs and Spearman correlations to four decimals. Extended Data
Figure 2 lists counts and flags (panel a), percentages (panels b and e) and AUCs and Spearman correlations to four decimals
(panels c, d and f). Extended Data Figure 3 lists percentages to two
decimals and, under ``headline``, the differences quoted in its caption. Tables are read with ``_figures.load_table``: from
``MODFORMER_FIGURE_DATA`` when it is set, else from the ``figure-data`` folder of the ModFormer cache, where every pinned
figure table is fetched on first use.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from _figures import (BAND, BAND_EDGE, BLUE, FAMILY_COLOR, INK, LABEL, MM, PINK, RED, _add_axes_mm, _letter,
                      load_table)

ED3_TABLES = ["ed3a_pick_distance.csv", "ed3b_conditions_pick.csv", "ed3c_distance_bands.csv", "ed3d_topk.csv",
              "ed3e_topn_share.csv", "ed3f_largest_gains.csv", "ed3_top_pick_difference.csv"]
ZS, LIB = "ModFormer, zero-shot", "Library-trained, 129 designs"
LIGHT_BLUE, NAVY = "#9CC3E0", "#08306B"
CONDS = ("HEK293T", "MCF7")
LINE = {ZS: dict(color=RED, marker="o", ms=2.6, mfc="white", mec=RED), LIB: dict(color=BLUE, marker="*", ms=4.2, mfc=BLUE, mec=BLUE)}
H, FACET_W, GAP = 190.0, 31.0, 5.5
BOX = dict(fc=BAND, ec=BAND_EDGE, lw=0.5)


def _p(v):
    return round(100 * float(v), 2)


def _box(ax, x0, width, lo, hi):
    """The gray interval box of a random or shuffled baseline, in percent."""
    ax.add_patch(Rectangle((x0, 100 * lo), width, max(100 * (hi - lo), 0.8), zorder=4, **BOX))


def _facets(fig, x, top, h, ylabel, xlabel, labels):
    """Two facets (HEK293T, MCF7) with a shared y label and one centered x label."""
    axs = [_add_axes_mm(fig, x + i * (FACET_W + GAP), top, FACET_W, h, H) for i in range(2)]
    for ax, cond in zip(axs, CONDS):
        ax.set_title(cond, pad=1.5, color=INK)
    axs[0].set_ylabel(ylabel, labelpad=2, linespacing=1.1)
    axs[1].tick_params(labelleft=False)
    labels.append((top, axs, (x + FACET_W + GAP / 2) * MM / fig.get_figwidth(), xlabel))
    return axs


def _key(fig, x, y, handles, labels):
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(x * MM / fig.get_figwidth(), 1 - y / H), frameon=False,
               handlelength=1.0, handletextpad=0.4, borderpad=0, borderaxespad=0, labelcolor=INK)


def _percent_axis(ax, lim=100.0):
    ax.set_ylim(0, lim)
    ax.set_yticks([0, 50, 100])


def _bars(ax, g, cond, xc, val, lo, hi, labels, vals, ivs, rotate=0):
    """Red and blue bars per category, with the baseline box over both."""
    n = 0
    cats = list(dict.fromkeys(g[xc]))
    for i, c in enumerate(cats):
        e = g[g[xc] == c]
        _box(ax, i - 0.4, 0.8, e[lo].min(), e[hi].max())
        ivs[f"{cond}|{c}"] = [_p(e[lo].min()), _p(e[hi].max())]
        for dx, m in ((-0.2, ZS), (0.2, LIB)):
            v = float(e[e.model == m][val].iloc[0])
            ax.bar(i + dx, 100 * v, 0.36, color=LINE[m]["color"], lw=0, zorder=2)
            vals[f"{cond}|{c}|{m}"] = _p(v)
            n += 1
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([labels(c) for c in cats], rotation=rotate, ha="right" if rotate else "center", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(cats) - 0.4)
    _percent_axis(ax)
    return n


def _panel_a(ax, g, cond, vals, ivs):
    z = g[g.model == ZS].sort_values("bin_start_nt")
    edges = np.append(z.bin_start_nt, z.bin_end_nt.iloc[-1])
    ax.stairs(100 * z.share_of_all_sites, edges, fill=True, color=BAND, zorder=1)
    vals[f"{cond}|All sites"] = [_p(v) for v in z.share_of_all_sites]
    for m in (ZS, LIB):
        s = g[g.model == m].sort_values("bin_start_nt")
        ax.stairs(100 * s.share_of_top_picks, edges, color=LINE[m]["color"], lw=1.0, zorder=3)
        vals[f"{cond}|{m}"] = [_p(v) for v in s.share_of_top_picks]
    ax.set_xlim(0, 144)
    ax.set_xticks([0, 40, 80, 120])
    ax.set_ylim(0, 42)
    ax.set_yticks([0, 20, 40])
    if cond == CONDS[0]:
        ax.text(46, 35, "Top pick", ha="left", va="center", color=LABEL)
        ax.text(140, 12, "All sites", ha="right", va="center", color=LABEL)
    return 3


def _panel_d(ax, g, cond, vals, ivs):
    ks = sorted(g.picks_per_construct.unique())
    ax.axvspan(100 * g.random_lo.iloc[0], 100 * g.random_hi.iloc[0], zorder=1, **BOX)
    ivs[f"{cond}|random"] = [_p(g.random_lo.iloc[0]), _p(g.random_hi.iloc[0])]
    n = 0
    for i, k in enumerate(ks):
        for dy, m in ((-0.15, ZS), (0.15, LIB)):
            r = g[(g.model == m) & (g.picks_per_construct == k)].iloc[0]
            kw = LINE[m]
            ax.plot([100 * r.ci_low, 100 * r.ci_high], [i + dy] * 2, color=kw["color"], lw=0.6, zorder=3)
            ax.plot([100 * r.share_increased], [i + dy], ls="", zorder=4, **kw)
            vals[f"{cond}|{m}|{k}"] = _p(r.share_increased)
            ivs[f"{cond}|{m}|{k}"] = [_p(r.ci_low), _p(r.ci_high)]
            n += 1
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels(ks)
    ax.set_ylim(len(ks) - 0.5, -0.5)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.tick_params(axis="y", length=2)
    return n


def _panel_f(ax, g, cond, vals, ivs):
    env = g.groupby("sites_tested").agg(lo=("shuffled_lo", "min"), hi=("shuffled_hi", "max")).reset_index()
    ax.fill_between(env.sites_tested, 100 * env.lo, 100 * env.hi, facecolor=BAND, lw=0, zorder=1)
    ivs[cond] = {str(int(r.sites_tested)): [_p(r.lo), _p(r.hi)] for r in env.itertuples()}
    n = 0
    for m in (ZS, LIB):
        s = g[g.model == m].sort_values("sites_tested")
        ax.plot(s.sites_tested, 100 * s.share_found, lw=0.9, zorder=3, **LINE[m])
        for r in s.itertuples():
            vals[f"{cond}|{m}|{int(r.sites_tested)}"] = _p(r.share_found)
            n += 1
    ax.set_xscale("log")
    ax.set_xticks([10, 100, 1000])
    ax.set_xticklabels(["10", "100", "1000"])
    ax.minorticks_off()
    ax.set_xlim(8, 2500)
    if cond == CONDS[0]:
        ax.text(2200, 14, "Shuffled\npredictions", ha="right", va="bottom", color=LABEL, linespacing=1.1)
    return n


def _panel_b(ax, t, vals, ivs):
    """Paired bars per condition: top-ranked pick increased (dark) and bottom-ranked pick decreased (light)."""
    xs = (0, 1, 2, 3.3, 4.3)
    for x, r in zip(xs, t.itertuples()):
        dark, light = (RED, PINK) if r.model == ZS else (NAVY if r.up_rate < r.up_random else BLUE, LIGHT_BLUE)
        for dx, tag, color in ((-0.2, "up", dark), (0.2, "down", light)):
            v, rnd, lo, hi = (getattr(r, f"{tag}_{s}") for s in ("rate", "random", "random_lo", "random_hi"))
            ax.bar(x + dx, 100 * v, 0.36, color=color, lw=0, zorder=2)
            _box(ax, x + dx - 0.22, 0.44, lo, hi)
            vals[f"{r.condition}|{tag}"], vals[f"{r.condition}|{tag}_random"] = _p(v), _p(rnd)
            ivs[f"{r.condition}|{tag}"] = [_p(lo), _p(hi)]
    ax.set_xticks(xs)
    ax.set_xticklabels([c.replace("ADAR1-KO ", "").replace("HEK293T + IFN", "HEK293T\n+ IFN") for c in t.condition],
                       rotation=45, ha="right", rotation_mode="anchor", linespacing=1.0)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, 4.9)
    _percent_axis(ax)
    ax.set_ylabel("Constructs where the pick\nwent as predicted (%)", labelpad=2, linespacing=1.1)
    ax.plot([2.85, 2.85, 4.75, 4.75], [102, 105, 105, 102], color=LABEL, lw=0.6, clip_on=False)
    ax.text(3.8, 106.5, "ADAR1-KO", ha="center", va="bottom", color=INK)
    return 2 * len(t)


def ed3(tables=None):
    """Extended Data Figure 3, panels a-f. ``tables`` maps file names to DataFrames (read with ``load_table`` when omitted)."""
    T = {n: load_table(n) for n in ED3_TABLES} if tables is None else tables
    fig = plt.figure(figsize=(180 * MM, H * MM))
    tops, h = (10.0, 73.0, 136.0), 33.0
    out = {}
    xlabels = []

    def facet_panel(letter, x, row, table, ylabel, xlabel, draw):
        axs = _facets(fig, x, tops[row], h, ylabel, xlabel, xlabels)
        vals, ivs, n = {}, {}, 0
        for ax, cond in zip(axs, CONDS):
            n += draw(ax, T[table][T[table].condition == cond], cond, vals, ivs)
        out[letter] = {"marks": n, "values": vals, "intervals": ivs}
        return axs

    facet_panel("a", 12.0, 0, ED3_TABLES[0], "Share per 8-nt bin (%)", "Distance from mutation's paired position (nt)", _panel_a)
    ax_b = _add_axes_mm(fig, 106.0, tops[0], 36.0, h, H)
    vals, ivs = {}, {}
    n = _panel_b(ax_b, T[ED3_TABLES[1]], vals, ivs)
    out["b"] = {"marks": n, "values": vals, "intervals": ivs}
    box = [Rectangle((0, 0), 1, 1, fc=BLUE, lw=0), Rectangle((0, 0), 1, 1, fc=LIGHT_BLUE, lw=0),
           Rectangle((0, 0), 1, 1, **BOX)]
    _key(fig, 146.0, tops[0] + 10.0, box, ["Top-ranked:\nincreased", "Bottom-ranked:\ndecreased", "Random site"])

    def bars(val, lo, hi, xc, labels, rotate):
        return lambda ax, g, cond, v, iv: _bars(ax, g, cond, xc, val, lo, hi, labels, v, iv, rotate)

    facet_panel("c", 12.0, 1, ED3_TABLES[2], "Top picks that increased (%)", "Distance from mutation's paired position (nt)",
                bars("up_rate", "up_random_lo", "up_random_hi", "band_nt", lambda c: c.replace("-", "–"), 45))
    facet_panel("d", 104.0, 1, ED3_TABLES[3], "Picks per construct", "Top picks that increased (%)", _panel_d)
    ax_e = facet_panel("e", 12.0, 2, ED3_TABLES[4], "Top-ranked sites\nthat increased (%)", "Sites tested",
                bars("share_increased", "shuffled_lo", "shuffled_hi", "sites_tested", lambda c: str(int(c)), 0))
    ax_f = facet_panel("f", 104.0, 2, ED3_TABLES[5], "Largest increases\nfound (%)", "Sites tested", _panel_f)
    for axs in (ax_e, ax_f):
        for ax in axs:
            _percent_axis(ax)
    gray = Rectangle((0, 0), 1, 1, **BOX)
    _key(fig, 5.5, tops[1] - 9.5, [gray], ["Random site, same band"])
    _key(fig, 93.5, tops[1] - 9.5, [gray], ["Random picks, same construct"])
    _key(fig, 5.5, tops[2] - 9.5, [gray], ["Shuffled predictions"])
    for letter, x, row in (("a", 0.5, 0), ("b", 88.5, 0), ("c", 0.5, 1), ("d", 88.5, 1), ("e", 0.5, 2), ("f", 88.5, 2)):
        _letter(fig, letter, x, tops[row] - 9.5, H)
    models = [(Patch(fc=LINE[m]["color"], lw=0), Line2D([], [], **LINE[m])) for m in (ZS, LIB)]
    fig.legend(models, [ZS, LIB], loc="lower center", bbox_to_anchor=(0.5, 0.004), ncol=2, frameon=False, labelcolor=INK,
               handlelength=2.4, handletextpad=0.5, columnspacing=2.5, handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)})
    fig.canvas.draw()                  # the shared x labels of a row sit on one line, just below its lowest tick label
    low = {}
    for row, axs, _, _ in xlabels:
        ticks = [t.get_window_extent().y0 for ax in axs for t in ax.get_xticklabels()]
        low[row] = min(ticks + [low.get(row, ticks[0])])
    for row, _, x, text in xlabels:
        fig.text(x, (low[row] - 0.5 * fig.dpi / 72 * 8) / fig.bbox.height, text, ha="center", va="top", color=LABEL)
    diff = T[ED3_TABLES[6]]
    headline = {"top_pick_difference": {r.condition: [round(float(x), 3) for x in (r.difference, r.ci_low, r.ci_high)]
                                        for r in diff.itertuples()}}
    return fig, {"figure": "ED3", "panels": out, "headline": headline}


# ================================================================================ Extended Data Figure 1
ED1_TABLES = ["ed1b_auc_by_benchmark.csv", "ed1c_auc_summary.csv", "ed1d_rate_spearman.csv"]
RULE = "#D7D6D7"
# method -> color, marker, marker size (pt), pale bar color of panel c
METHOD = {"ModFormer": ("#EF4444", "o", 3.9, "#F7BEBD"), "AdarEdit": ("#334E61", "^", 3.4, "#99A6B0"),
          "InoDREAM": ("#618095", "<", 3.4, "#B0C0CA"), "Neighbor rule": ("#9A9A9B", "x", 3.4, "#C4C4C5"),
          "EditPredict": ("#49677C", "v", 3.1, "#A4B3BE"), "PreAIS": ("#7997AB", ">", 3.1, "#BCCBD5"),
          "ATTIC": ("#92ACBD", "p", 3.3, "#C8D6DE")}
DX = dict(zip(METHOD, [-0.14, 0.0, 0.14, *np.linspace(-0.3, 0.3, 4)]))
DOT = (0, (1.0, 1.8))
CD_TOP, CD_BOTTOM = 96.85, 165.9                          # panels c and d, in page millimeters


def _page(h, top=0.0):
    """A page 180 mm wide and ``h`` mm tall; ``ov`` takes text in page millimeters (``top`` is the page's first row)."""
    fig = plt.figure(figsize=(180 * MM, h * MM))
    ov = fig.add_axes([0, 0, 1, 1], zorder=10)
    ov.set_xlim(0, 180)
    ov.set_ylim(top + h, top)
    ov.axis("off")
    return fig, ov, lambda x0, x1, y0, y1: _add_axes_mm(fig, x0, y0 - top, x1 - x0, y1 - y0, h)


def _mark(ax, x, y, m, **kw):
    color, shape, ms, _ = METHOD[m]
    return ax.plot(x, y, ls="", marker=shape, ms=ms, mew=0.7, mfc="white" if m == "ModFormer" else color, mec=color, **kw)


def _lbl(ov, x, y, s, m=None, **kw):
    ov.text(x, y, s, va="center", color=kw.pop("color", INK), fontweight="bold" if m == "ModFormer" else "normal", **kw)


def ed1(tables=None):
    """Extended Data Figure 1, panels b-d (panel a, the table of method capabilities, is not drawn)."""
    T = {n: load_table(n) for n in ED1_TABLES} if tables is None else tables
    b, c, d = (T[n] for n in ED1_TABLES)
    top, h = 89.0, 92.0
    fig, ov, axmm = _page(h, top)
    sets = list(dict.fromkeys(b.benchmark))
    order = list(c.method)
    out = {k: {"marks": 0, "values": {}, "intervals": {}} for k in "bcd"}
    for x, letter in ((0.5, "b"), (71.0, "c"), (133.0, "d")):
        ov.text(x, 93.2, letter, fontsize=9, fontweight="bold", color=INK)

    # b: AUC on four benchmark sets
    ax = axmm(10, 65, 104.9, 165.9)
    ax.set_xlim(-0.4, 3.4)
    ax.set_ylim(0.45, 1.075)
    ax.spines["left"].set_bounds(0.45, 1.0)
    ax.spines["bottom"].set_bounds(-0.3, 3.3)
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels(["Inverte-\nbrates" if s == "Invertebrates" else s for s in sets], linespacing=1.0)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])
    for i, s in enumerate(sets):
        if b[b.benchmark == s].other_species.iloc[0]:
            ax.axvspan(i - 0.5, i + 0.5 if i < len(sets) - 1 else 3.4, color=BAND, lw=0, zorder=0)
    out["b"]["other_species"] = [s for s in sets if b[b.benchmark == s].other_species.iloc[0]]
    ax.axhline(0.5, color=LABEL, lw=0.6, ls=DOT, zorder=1)
    ov.text(65, 163.65, "Chance", ha="right", color=LABEL)
    ov.text(51.25, 107.53, "Transfer to", ha="center", color=LABEL)
    ov.text(51.25, 110.26, "other species", ha="center", color=LABEL)
    ov.text(3.55, 135.4, "AUC", ha="center", va="center", color=LABEL, rotation=90)
    rings = []
    for m in reversed(order):
        t = b[b.method == m].set_index("benchmark").loc[sets]
        x = np.arange(len(sets)) + DX[m]
        if t.ci_low.notna().all():                        # a line and 95% intervals where the table has them
            lw = {"ModFormer": 1.1}.get(m, 0.9)
            ax.plot(x, t.auc, color=METHOD[m][0], lw=lw, zorder=2)
            for xi, r in zip(x, t.itertuples()):
                ax.plot([xi, xi], [r.ci_low, r.ci_high], color=METHOD[m][0], lw=0.7, solid_capstyle="butt", zorder=2)
                out["b"]["intervals"][f"{m}|{r.Index}"] = [round(r.ci_low, 4), round(r.ci_high, 4)]
        _mark(ax, x, t.auc, m, zorder=3, clip_on=False)
        for xi, r in zip(x, t.itertuples()):
            out["b"]["values"][f"{m}|{r.Index}"] = round(r.auc, 4)
            out["b"]["marks"] += 1
            if r.tested_on_own_training_source:
                ax.plot(xi, r.auc, ls="", marker="o", ms=7.0, mfc="none", mec=METHOD[m][0], mew=0.7, zorder=4)
                rings.append(f"{m}|{r.Index}")
    out["b"]["rings"] = sorted(rings)
    for m, y in (("ModFormer", 92.65), ("InoDREAM", 95.7), ("AdarEdit", 98.75)):
        ov.plot([11.55, 16.07], [y - 0.28] * 2, color=METHOD[m][0], lw=1.1 if m == "ModFormer" else 0.9, solid_capstyle="butt")
        _mark(ov, 13.81, y - 0.28, m)
        _lbl(ov, 16.91, y, m, m)
    ov.plot([35.25], [97.25], ls="", marker="o", ms=7.0, mfc="none", mec=LABEL, mew=0.7)
    _lbl(ov, 37.79, 95.68, "Tested on its own")
    _lbl(ov, 37.79, 98.71, "training source")
    _lbl(ov, 11.55, 103.1, "Other methods: markers as in c", color=LABEL)

    # c: worst to best set, and the mean of four
    ax = axmm(92, 123, CD_TOP, CD_BOTTOM)
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([0.5, 0.75, 1.0])
    ax.set_xticklabels(["0.5", "0.75", "1"])
    ax.axvline(0.5, color=LABEL, lw=0.6, ls=DOT, zorder=1)
    for i, r in enumerate(c.itertuples()):
        ax.plot([r.worst_auc, r.best_auc], [i, i], color=METHOD[r.method][3], lw=3.0, solid_capstyle="round", zorder=2)
        _mark(ax, r.worst_auc, i, r.method, zorder=3)
        y = CD_TOP + (i + 0.5) * (CD_BOTTOM - CD_TOP) / len(order)
        _lbl(ov, 91.3, y, r.method, r.method, ha="right")
        _lbl(ov, 129, y, f"{r.mean_auc:.2f}", r.method, ha="center")
        out["c"]["values"].update({f"{r.method}|mean": round(r.mean_auc, 4), f"{r.method}|worst": round(r.worst_auc, 4),
                                   f"{r.method}|best": round(r.best_auc, 4)})
        out["c"]["marks"] += 1
    ov.text(129, 95.5, "Mean", ha="center", color=LABEL)
    ov.text(107.5, 172.51, "AUC, worst to best set", ha="center", color=LABEL)

    # d: Spearman correlation with the editing rate, four sets per method
    ax = axmm(137, 176, CD_TOP, CD_BOTTOM)
    ax.set_xlim(-0.1, 0.6)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6"])
    ax.axvline(0, color=LABEL, lw=0.6, ls=DOT, zorder=1)
    for i in range(1, len(order)):
        ax.axhline(i - 0.5, color=RULE, lw=0.5, zorder=0)
    rows = list(dict.fromkeys(d.benchmark))
    marks = []                                            # the Neighbor rule markers and interval ends, labeled by benchmark set
    for i, m in enumerate(order):
        for k, s in enumerate(rows):
            r = d[(d.method == m) & (d.benchmark == s)].iloc[0]
            y = i + (k - (len(rows) - 1) / 2) * 0.22
            ax.plot([r.ci_low, r.ci_high], [y, y], color=METHOD[m][0], lw=0.6, solid_capstyle="butt", zorder=2)
            _mark(ax, r.spearman, y, m, zorder=3)
            if m == "Neighbor rule":
                marks.append((s, 137 + (r.spearman + 0.1) / 0.7 * 39, 137 + (r.ci_high + 0.1) / 0.7 * 39, CD_TOP + (y + 0.5) * (CD_BOTTOM - CD_TOP) / len(order)))
            out["d"]["values"][f"{m}|{s}"] = round(r.spearman, 4)
            out["d"]["intervals"][f"{m}|{s}"] = [round(r.ci_low, 4), round(r.ci_high, 4)]
            out["d"]["marks"] += 1
    ov.text(156.5, 172.51, "Spearman correlation with rate", ha="center", color=LABEL)
    fig.canvas.draw()
    for s, x, xhi, y in marks:
        t = ov.text(176, y, s, ha="right", va="center", color=LABEL)
        left = ov.transData.inverted().transform((t.get_window_extent().x0, 0))[0]
        ov.plot([max(x + 1.4, xhi), left - 0.8], [y, y], color=LABEL, lw=0.4, solid_capstyle="butt")
    return fig, {"figure": "ED1", "panels": out}


# ================================================================================ Extended Data Figure 2
ED2_TABLES = ["ed2a_datasets.csv", "ed2b_finetune_mix.csv", "ed2cd_pretraining_sets.csv", "ed2e_rate_scale.csv",
              "ed2f_alu_overlap.csv"]
SPLIT = (("train", "Train", "#B9B9BA"), ("validation", "Validation", "#E6E6E7"), ("test", "Test", INK))
FINE = "#4B5563"
CD_ROWS = {"ModFormer": 0.0, "No language model": 1.6, "Trained without structure": 2.6, "Adjacent partner only": 3.6,
           "No triangle multiplication": 4.6, "One-hot BiLSTM": 6.2, "CNN on sequence only": 7.2, "Language model only": 8.2}


def _count(v):
    return f"{v / 1e6:.0f} M" if v >= 1e6 else f"{v / 1e3:.0f} k"


def _key_square(ov, x, y, label, color=None):
    if color:
        ov.add_patch(Rectangle((x, y - 1.0), 2.4, 2.0, fc=color, lw=0))
    else:
        ov.plot([x + 1.2], [y], ls="", marker="s", ms=3.0, mfc="white", mec=INK, mew=0.6)
    ov.text(x + 3.4, y, label, va="center", color=INK)


def _a(ov, axmm, a, out):
    """Panel a: labeled adenosines per dataset (log scale), the train/validation/test split, and where each is used."""
    n = len(a)
    ys = lambda i: 14.0 + i * 4.6
    ax = axmm(36, 86, 11.7, 43.9)
    ax.set_xscale("log")
    ax.set_xlim(1e3, 1.5e9)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([1e3, 1e5, 1e7, 1e9])
    ax.set_xticklabels(["1 k", "100 k", "10 M", "1 B"])
    ax.minorticks_off()
    ax.set_xlabel("Labeled adenosines", color=LABEL, labelpad=1)
    sp = axmm(106, 134, 11.7, 43.9)
    sp.set_xlim(0, 1)
    sp.set_ylim(n - 0.5, -0.5)
    sp.set_yticks([])
    sp.spines["left"].set_visible(False)
    sp.set_xticks([0, 1])
    sp.set_xticklabels(["0", "100%"])
    vals, marks = {}, 0
    for i, r in enumerate(a.itertuples()):
        key = f"{r.group}|{r.dataset}"
        ov.text(9, ys(i), r.dataset, va="center", color=INK)
        ax.barh(i, r.labeled_total - 1e3, left=1e3, height=0.62, fc="white", ec=INK, lw=0.6)
        ax.text(r.labeled_total * 1.3, i, _count(r.labeled_total), va="center", color=INK)
        left = 0.0
        for col, _, color in SPLIT:
            v = getattr(r, f"labeled_{col}")
            sp.barh(i, v / r.labeled_total, left=left, height=0.62, fc=color, ec=INK, lw=0.5)
            left += v / r.labeled_total
            vals[f"{key}|{col}"] = int(v)
        vals[f"{key}|total"] = int(r.labeled_total)
        for x, used, tag in ((150, r.used_in_pretraining, "pretraining"), (166, r.used_in_fine_tuning, "fine_tuning")):
            ov.plot([x], [ys(i)], ls="", marker="o", ms=4, mfc=INK if used else "white", mec=INK, mew=0.6)
            vals[f"{key}|{tag}"] = bool(used)
        marks += 1 + len(SPLIT) + 2
    for g in dict.fromkeys(a.group):                            # a bracket and a rotated name per group of rows
        idx = [i for i, v in enumerate(a.group) if v == g]
        y0, y1 = ys(idx[0]) - 1.8, ys(idx[-1]) + 1.8
        ov.plot([6.5, 6.5], [y0, y1], color=LABEL, lw=0.6)
        ov.text(3.5, (y0 + y1) / 2, g, rotation=90, ha="center", va="center", color=LABEL)
    for x, s in ((150, "Pretrain"), (166, "Fine-tune")):
        ov.text(x, 9.2, s, ha="center", va="center", color=INK)
    ov.text(120, 9.2, "Split", ha="center", va="center", color=INK)
    _key_square(ov, 96, 4.2, "Train", SPLIT[0][2])
    _key_square(ov, 109.5, 4.2, "Validation", SPLIT[1][2])
    _key_square(ov, 128, 4.2, "Test", SPLIT[2][2])
    ov.plot([147], [4.2], ls="", marker="o", ms=4, mfc=INK, mec=INK, mew=0.6)
    ov.text(149.5, 4.2, "Used", va="center", color=INK)
    ov.plot([160], [4.2], ls="", marker="o", ms=4, mfc="white", mec=INK, mew=0.6)
    ov.text(162.5, 4.2, "Not used", va="center", color=INK)
    out["a"] = {"marks": marks, "values": vals, "intervals": {}}


def _b(ov, axmm, b, out):
    """Panel b: share of the joint fine-tune's draws from each source, and the heads each source trains."""
    n, y0 = len(b), 64.1
    ax = axmm(50, 110, y0, y0 + n * 3.8)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 0.25, 0.5])
    ax.set_xticklabels(["0", "25", "50%"])
    vals = {}
    for i, r in enumerate(b.itertuples()):
        y = y0 + (i + 0.5) * 3.8
        ov.text(47, y, r.source, ha="right", va="center", color=INK)
        ax.barh(i, r.share_of_draws, height=0.62, fc="#C4C4C5", lw=0)
        ax.text(r.share_of_draws + 0.012, i, f"{100 * r.share_of_draws:.1f}%", va="center", color=INK)
        vals[f"{r.source}|share"] = _p(r.share_of_draws)
        for x, flag, tag in ((130, r.trains_classification_head, "classification"), (160, r.trains_rate_head, "rate")):
            ov.plot([x], [y], ls="", marker="o", ms=4, mfc=INK if flag else "white", mec=INK, mew=0.6)
            vals[f"{r.source}|{tag}_head"] = bool(flag)
    ov.text(47, 59.5, "Training sources", ha="right", va="center", color=LABEL)
    ov.text(80, 59.5, "Share of draws", ha="center", va="center", color=LABEL)
    ov.text(130, 59.5, "Classification head", ha="center", va="center", color=LABEL)
    ov.text(160, 59.5, "Rate head", ha="center", va="center", color=LABEL)
    out["b"] = {"marks": 3 * n, "values": vals, "intervals": {}}


def _cd(ov, axmm, cd, out):
    """Panels c and d: AUC and rate Spearman of every model on the two human pretraining sets, zero-shot and fine-tuned."""
    top, h, x_first, right = 106.0, 32.0, 46.0, 178.0
    colw = (right - x_first - 9.0 - 2 * 6.0) / 4
    sets = list(dict.fromkeys(cd.test_set))
    fam = dict(zip(cd.model, cd.family))
    for k, (metric, letter, xlab, lim, ticks) in enumerate((("AUC", "c", "AUC", (0.5, 1.0), (0.5, 0.75, 1.0)),
                                                         ("Spearman", "d", "Spearman correlation", (0.0, 1.0), (0.0, 0.5, 1.0)))):
        x0 = x_first + k * (2 * colw + 15.0)
        ov.text(x0 - 6.0 if k else 0.5, 101.5, letter, fontsize=9, fontweight="bold", color=INK, va="center")
        vals, ivs, n = {}, {}, 0
        for j, ts in enumerate(sets):
            ax = axmm(x0 + j * (colw + 6.0), x0 + j * (colw + 6.0) + colw, top, top + h)
            ax.set_ylim(8.8, -0.6)
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.set_title(ts, pad=2, color=INK, fontweight="bold")
            ax.set_xlim(*lim)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:g}" for t in ticks])
            g = cd[(cd.metric == metric) & (cd.test_set == ts)]
            if g.empty:
                ax.text(0.5, 0.5, "No measured\nrate", transform=ax.transAxes, ha="center", va="center", color=LABEL)
                continue
            for r in g.itertuples():
                y = CD_ROWS[r.model]
                key = f"{ts}|{r.model}|{r.state}"
                vals[key], ivs[key] = round(r.value, 4), [round(r.ci_low, 4), round(r.ci_high, 4)]
                n += 1
                if r.state == "zero-shot":
                    ax.barh(y, r.value - lim[0], left=lim[0], height=0.62, color=FAMILY_COLOR[r.family], lw=0)
                    ax.plot([r.ci_low, r.ci_high], [y, y], color=INK, lw=0.4)
                else:
                    ax.plot([r.ci_low, r.ci_high], [y, y], color=INK, lw=0.4, zorder=3)
                    ax.plot(r.value, y, marker="s", ms=3.0, mfc="white", mec=INK, mew=0.6, ls="", zorder=4)
            ref = g[(g.model == "ModFormer") & (g.state == "zero-shot")].value.iloc[0]
            ax.axvline(ref, color=LABEL, lw=0.4, ls=(0, (1.2, 1.2)), zorder=1)
        ov.text(x0 + colw + 3.0, 145.5, xlab, ha="center", va="center", color=INK)
        out[letter] = {"marks": n, "values": vals, "intervals": ivs}
    for name, y in CD_ROWS.items():
        ov.text(44.0, top + (y + 0.6) * h / 9.4, name, ha="right", va="center", color=INK,
                fontweight="bold" if name == "ModFormer" else "normal")
    for x, label, color in ((56, "ModFormer", RED), (82, "Training ablations", FAMILY_COLOR["Training ablations"]),
                            (118, "Alternative architectures", FAMILY_COLOR["Alternative architectures"]), (157, "Fine-tuned", None)):
        _key_square(ov, x, 152.5, label, color)


def _e(ov, axmm, e, out):
    """Panel e: mean measured against mean predicted editing rate in bins of predicted rate, with 95% intervals."""
    ax = axmm(12, 84, 163, 191)
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 50)
    ax.set_xticks(range(0, 26, 5))
    ax.set_yticks(range(0, 51, 10))
    ax.set_xlabel("Predicted rate (%)", labelpad=1)
    ax.set_ylabel("Measured rate (%)", labelpad=1)
    ax.plot([0, 25], [0, 25], color="#B9B9BA", lw=0.5, zorder=0)
    ax.text(20.5, 19.0, "measured = predicted", color=LABEL, ha="center", va="top", rotation_mode="anchor",
            rotation=float(np.degrees(np.arctan((28 / 50) / (72 / 25)))))
    vals, ivs, n = {}, {}, 0
    for k, (state, color, name) in enumerate((("zero-shot", RED, "ModFormer, zero-shot"), ("fine-tuned", FINE, "ModFormer, fine-tuned"))):
        g = e[e.state == state]
        x, y = 100 * g.predicted_mean.to_numpy(), 100 * g.measured_mean.to_numpy()
        for r in g.itertuples():
            ax.plot([100 * r.predicted_mean] * 2, [100 * r.measured_low, 100 * r.measured_high], color="#8C8C8D", lw=0.4, zorder=1)
            ax.plot([100 * r.predicted_low, 100 * r.predicted_high], [100 * r.measured_mean] * 2, color="#8C8C8D", lw=0.4, zorder=1)
            key = f"{state}|{r.bin}"
            vals[key] = [round(100 * r.predicted_mean, 4), round(100 * r.measured_mean, 4)]
            ivs[key] = [round(100 * v, 4) for v in (r.predicted_low, r.predicted_high, r.measured_low, r.measured_high)]
            n += 1
        ax.plot(x, y, color=color, lw=0.9, zorder=3)
        ax.plot(x, y, ls="", marker="o", ms=1.9, mfc=color, mec=color, mew=0.4, zorder=4)
        kx = 12 + k * 40
        ov.plot([kx, kx + 4], [158.6, 158.6], color=color, lw=0.9)
        ov.plot([kx + 2], [158.6], ls="", marker="o", ms=1.9, mfc=color, mec=color, mew=0.4)
        ov.text(kx + 5, 158.6, name, va="center", color=INK)
    ov.text(0.5, 157.0, "e", fontsize=9, fontweight="bold", color=INK, va="center")
    out["e"] = {"marks": n, "values": vals, "intervals": ivs}


def _f(ov, axmm, f, out):
    """Panel f: AUC and rate Spearman on all Alu test duplexes and on those that do and do not overlap pretraining."""
    groups = list(dict.fromkeys(f.group))
    ypos = (0.0, 1.15, 2.15)
    vals, ivs, n = {}, {}, 0
    for k, (metric, xlab, lim, ticks) in enumerate((("AUC", "Ranking AUC", (0.75, 0.95), (0.8, 0.9)),
                                                    ("Spearman", "Rate Spearman", (0.4, 0.8), (0.4, 0.5, 0.6, 0.7, 0.8)))):
        x0 = 130 + k * 26
        ax = axmm(x0, x0 + 20, 163, 191)
        ax.set_xlim(*lim)
        ax.set_ylim(2.65, -0.55)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.set_xlabel(xlab, labelpad=1)
        ax.axhline((ypos[0] + ypos[1]) / 2, color=LABEL, lw=0.3, ls=(0, (1.2, 1.2)), zorder=1)
        for g, y in zip(groups, ypos):
            for state in ("zero-shot", "fine-tuned"):
                r = f[(f.group == g) & (f.metric == metric) & (f.state == state)].iloc[0]
                ax.plot([r.ci_low, r.ci_high], [y, y], color="#8C8C8D", lw=0.4, zorder=1)
                key = f"{g}|{metric}|{state}"
                vals[key], ivs[key] = round(r.value, 4), [round(r.ci_low, 4), round(r.ci_high, 4)]
                n += 1
                if state == "zero-shot":
                    ax.plot(r.value, y, ls="", marker="o", ms=2.4, mfc=RED, mec=RED, mew=0.4, zorder=6)
                else:
                    ax.plot(r.value, y, ls="", marker="s", ms=3.2, mfc="white", mec=FINE, mew=0.8, zorder=5)
    for g, y in zip(groups, ypos):
        r = f[f.group == g].iloc[0]
        ov.text(127, 163 + (y + 0.55) * 28 / 3.2 - 1.6, g, ha="right", va="center", color=INK)
        ov.text(127, 163 + (y + 0.55) * 28 / 3.2 + 1.6, f"{r.n_duplexes} duplexes, {r.n_sites:,} A's", ha="right", va="center", color=LABEL)
        vals[f"{g}|n_duplexes"], vals[f"{g}|n_sites"] = int(r.n_duplexes), int(r.n_sites)
    ov.text(88.5, 157.0, "f", fontsize=9, fontweight="bold", color=INK, va="center")
    ov.plot([92], [158.6], ls="", marker="o", ms=2.4, mfc=RED, mec=RED, mew=0.4)
    ov.text(94.5, 158.6, "ModFormer, zero-shot", va="center", color=INK)
    ov.plot([130], [158.6], ls="", marker="s", ms=3.2, mfc="white", mec=FINE, mew=0.8)
    ov.text(132.5, 158.6, "ModFormer, fine-tuned", va="center", color=INK)
    out["f"] = {"marks": n, "values": vals, "intervals": ivs}


def ed2(tables=None):
    """Extended Data Figure 2, panels a-f, without the example tile of a and the fine-tuning diagram of b."""
    T = {n: load_table(n) for n in ED2_TABLES} if tables is None else tables
    fig, ov, axmm = _page(200.0)
    out = {}
    ov.text(0.5, 2.4, "a", fontsize=9, fontweight="bold", color=INK, va="center")
    ov.text(0.5, 55.0, "b", fontsize=9, fontweight="bold", color=INK, va="center")
    _a(ov, axmm, T[ED2_TABLES[0]], out)
    _b(ov, axmm, T[ED2_TABLES[1]], out)
    _cd(ov, axmm, T[ED2_TABLES[2]], out)
    _e(ov, axmm, T[ED2_TABLES[3]], out)
    _f(ov, axmm, T[ED2_TABLES[4]], out)
    return fig, {"figure": "ED2", "panels": {k: out[k] for k in "abcdef"}}
