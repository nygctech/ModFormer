"""Terminal visuals for the ``modformer`` command line.

Everything here is built from small functions that *return* strings or *yield*
frames, so the command line decides when and where to print.  The module uses
only the standard library.

Capability handling lives in one place, :func:`detect`.  Color and animation
are switched off when the stream is not a terminal, when ``NO_COLOR`` is set,
when ``TERM=dumb``, or when ``plain=True`` is passed.  In that case every
function returns plain ASCII with no escape codes, and :class:`Progress`
prints at most one line per stage.  Output never exceeds the terminal width,
the screen is never cleared, and the cursor is restored on exit and on
``KeyboardInterrupt``.

Every color comes from one table, :data:`PALETTE`, by role: neutral grays for
text and chrome, green for base pairs, yellow for the site in focus, and one
hue that brightens with the predicted rate.
"""

from __future__ import annotations

import atexit
import itertools
import math
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    NamedTuple,
    Sequence,
    TextIO,
)

__all__ = [
    "PALETTE",
    "Caps",
    "Progress",
    "banner",
    "card_header",
    "detect",
    "duplex",
    "easter_egg",
    "hint_line",
    "play",
    "rate_level",
    "rate_track",
    "top_sites",
    "track_column",
]

RGB = tuple[int, int, int]


# Palette -------------------------------------------------------------------------
#: Every color the module draws with: role -> (24-bit color, 16-color SGR code).
PALETTE: dict[str, tuple[RGB, int]] = {
    "fg": ((233, 238, 240), 97),         # body text, the site's strand
    "secondary": ((154, 163, 166), 37),  # titles, labels, the partner strand
    "dim": ((95, 102, 105), 90),         # chrome: rulers, coordinates, glyphs
    "rule": ((44, 50, 53), 90),          # separator lines, empty bar cells
    "pair": ((85, 255, 85), 92),         # base pairs
    "focus": ((255, 255, 85), 93),       # the site in focus, the prompt
    # The title drawing: graphite, with heavy green rungs.
    "logo_word": ((208, 208, 208), 97),    # the name
    "logo_strand": ((138, 138, 138), 37),  # the M's two hairpin strands
    "logo_rung": ((95, 255, 95), 92),      # the M's base-pair rungs (drawn bold)
    "logo_hi": ((188, 188, 188), 37),      # upper arcs of the helix
    "logo_lo": ((88, 88, 88), 90),         # lower arcs of the helix
    "logo_cross": ((0, 215, 95), 32),      # where the helix strands cross
    # The predicted rate: one hue, dark to bright.
    "rate0": ((22, 74, 74), 90),
    "rate1": ((34, 110, 110), 36),
    "rate2": ((47, 146, 146), 36),
    "rate3": ((60, 183, 183), 36),
    "rate4": ((72, 219, 219), 96),
    "rate5": ((85, 255, 255), 96),
}
RATE_STEPS = 6

_ESC = "\x1b"
_RESET = f"{_ESC}[0m"


def _sgr(style: "Style", caps: "Caps") -> str:
    """24-bit codes when ``caps.color`` is ``"true"``, else the 16-color codes."""
    if not caps.has_color:
        return ""
    codes = ["1"] if style.bold else []
    for role, base in ((style.fg, 38), (style.bg, 48)):
        if role is not None:
            (r, g, b), code = PALETTE[role]
            codes.append(f"{base};2;{r};{g};{b}" if caps.color == "true" else str(code + base - 38))
    return f"{_ESC}[{';'.join(codes)}m" if codes else ""


def rate_level(rate: float, top: float) -> int:
    """The ramp step (0 to 5) of ``rate`` on a scale whose top step starts at ``top``.

    Rates cluster near the top of their range, so a fourth-power scale spreads them.
    """
    if not (_finite(rate) and _finite(top) and top > 0):
        return 0
    if rate >= top:
        return RATE_STEPS - 1
    t = (max(0.0, float(rate)) / top) ** 4
    return min(RATE_STEPS - 2, int(t * (RATE_STEPS - 1)))


# Cursor control ------------------------------------------------------------------
_HIDE_CURSOR = f"{_ESC}[?25l"
_SHOW_CURSOR = f"{_ESC}[?25h"


# Capabilities ------------------------------------------------------------------
@dataclass(frozen=True)
class Caps:
    """What the output stream can do.

    ``color`` is ``"none"``, ``"16"`` or ``"true"``.  ``animate`` allows
    in-place redraws.  ``unicode`` allows block and box-drawing glyphs.
    """

    color: str = "none"
    animate: bool = False
    unicode: bool = False
    width: int = 80

    @property
    def has_color(self) -> bool:
        return self.color != "none"


def _stream_columns(stream: TextIO | None) -> int | None:
    """The columns of the terminal ``stream`` is attached to, or None when it is not one."""
    try:
        columns = os.get_terminal_size(stream.fileno()).columns  # type: ignore[union-attr]
    except (AttributeError, OSError, ValueError, TypeError):
        return None
    return columns if columns > 0 else None


def _terminal_width(stream: TextIO | None = None, default: int = 80) -> int:
    """The width to draw at: the columns of the terminal ``stream`` is attached to, or, when it is
    not a terminal, ``COLUMNS`` and then the other standard streams."""
    own = _stream_columns(stream)
    if own is not None:
        return max(12, own)
    try:
        return max(12, shutil.get_terminal_size((default, 24)).columns)
    except (OSError, ValueError):
        return default


def detect(
    stream: TextIO | None = None,
    plain: bool = False,
    env: Mapping[str, str] | None = None,
) -> Caps:
    """Decide color, animation and glyph support for ``stream``.

    Everything is off for ``plain=True``, a non-terminal stream, ``NO_COLOR``
    (non-empty) and ``TERM=dumb``.  Color is 24-bit when ``COLORTERM`` says
    ``truecolor`` or ``24bit``, and the 16 standard colors otherwise.  The
    width always follows the terminal.
    """
    env = os.environ if env is None else env
    stream = sys.stdout if stream is None else stream
    width = _terminal_width(stream)
    try:
        is_tty = bool(stream.isatty())
    except (AttributeError, ValueError, OSError):
        is_tty = False
    off = (
        plain
        or not is_tty
        or bool(env.get("NO_COLOR"))
        or env.get("TERM", "") == "dumb"
    )
    if off:
        return Caps("none", False, False, width)
    colorterm = env.get("COLORTERM", "").lower()
    depth = "true" if colorterm in ("truecolor", "24bit") else "16"
    encoding = (getattr(stream, "encoding", None) or "").lower()
    return Caps(depth, True, "utf" in encoding, width)


def _resolve(caps: Caps | None, plain: bool, stream: TextIO | None = None) -> Caps:
    if caps is None:
        return detect(stream, plain=plain)
    if plain:
        return Caps("none", False, False, caps.width)
    return caps


# Styling -------------------------------------------------------------------------
class Style(NamedTuple):
    """Palette roles for the foreground and background, and boldness."""

    fg: str | None = None
    bg: str | None = None
    bold: bool = False


NO_STYLE = Style()
_TEXT = Style("fg")
_SECONDARY = Style("secondary")
_DIM = Style("dim")
_RULE = Style("rule")
_PAIR = Style("pair")
_FOCUS = Style("focus", bold=True)


def _rate_style(rate: float, top: float) -> Style:
    return Style(f"rate{rate_level(rate, top)}")


Cell = tuple[str, Style]


def _cells_to_str(cells: Sequence[Cell], caps: Caps) -> str:
    out: list[str] = []
    i = 0
    while i < len(cells):
        style = cells[i][1]
        j = i
        text: list[str] = []
        while j < len(cells) and cells[j][1] == style:
            text.append(cells[j][0])
            j += 1
        chunk = "".join(text)
        prefix = _sgr(style, caps)
        out.append(f"{prefix}{chunk}{_RESET}" if prefix else chunk)
        i = j
    return "".join(out)


class Canvas:
    """A sparse grid of single-width characters with per-cell styles."""

    def __init__(self) -> None:
        self._rows: dict[int, dict[int, Cell]] = {}

    def put(self, x: int, y: int, text: str, style: Style = NO_STYLE) -> None:
        if y < 0:
            return
        row = self._rows.setdefault(y, {})
        for k, ch in enumerate(text):
            if x + k >= 0:
                row[x + k] = (ch, style)

    def height(self) -> int:
        return (max(self._rows) + 1) if self._rows else 0

    def render(self, caps: Caps, width: int | None = None) -> list[str]:
        limit = caps.width if width is None else min(width, caps.width)
        lines: list[str] = []
        for y in range(self.height()):
            row = self._rows.get(y, {})
            if not row:
                lines.append("")
                continue
            end = min(limit, max(row) + 1)
            cells = [row.get(x, (" ", NO_STYLE)) for x in range(end)]
            while cells and cells[-1][0] == " " and cells[-1][1].bg is None:
                cells.pop()
            lines.append(_cells_to_str(cells, caps))
        return lines


def _line(parts: Iterable[Cell | tuple[str, Style]], caps: Caps, width: int | None = None) -> str:
    cv = Canvas()
    x = 0
    for text, style in parts:
        cv.put(x, 0, text, style)
        x += len(text)
    return cv.render(caps, width)[0] if x else ""


# Title and spinner ----------------------------------------------------------------
def _helix_cells(width: int, phase: float, mark: int | None) -> list[Cell]:
    """The helix as half blocks: upper arcs bright, lower arcs darker, crossings green."""
    cells: list[Cell] = []
    for x in range(width):
        s = math.sin(phase + x * 0.75)
        if abs(s) < 0.28:
            cells.append(("█", _PAIR))
            continue
        role = "focus" if x == mark else "rate5" if s >= 0 else "rate3"
        cells.append(("▀" if s >= 0 else "▄", Style(role)))
    return cells


# The title: two RNA hairpins form the M, the name follows, a helix runs underneath. Each cell of
# _LOGO_ROWS has a role letter in _LOGO_ROLES.
_LOGO_ROWS = (
    "▄▀▄   ▄▀▄",
    "█━█   █━█",
    "█━█   █━█        █ █▀▀▀▀",
    "█━█   █━█ █▀█ █▀▀█ █▀▀  █▀█ █▀▀ █▀█▀█ ██▀ █▀▀",
    "█  ▀▄▀  █ ▀▀▀ ▀▀▀▀ ▀    ▀▀▀ ▀   ▀ ▀ ▀ ▀▀▀ ▀",
    "▀       ▀▀▀▀█▄▄▄█▀▀▀█▄▄▄▄▀▀▀▀█▄▄▄█▀▀▀█▄▄▄█▀▀▀",
)
_LOGO_ROLES = (
    "sss   sss",
    "srs   srs",
    "srs   srs        w wwwww",
    "srs   srs www wwww www  www www wwwww www www",
    "s  sss  s www wwww w    www w   w w w www w",
    "s       shhhplllphhhpllllhhhhplllphhhplllphhh",
)
# s hairpin strand, r rung, w name, h upper helix arc, l lower helix arc, p crossing
_LOGO_ROLE_STYLE = {
    "s": Style("logo_strand"),
    "r": Style("logo_rung", bold=True),
    "w": Style("logo_word"),
    "h": Style("logo_hi"),
    "l": Style("logo_lo"),
    "p": Style("logo_cross"),
}
_LOGO_WIDTH = max(len(row) for row in _LOGO_ROWS)


def banner(
    version: str | None = None, caps: Caps | None = None, plain: bool = False
) -> str:
    """The ModFormer title: two RNA hairpins form the M, the name follows, and a helix runs
    underneath, drawn in half blocks.

    Each cell takes the color of its role (name, rungs, strands, helix). Without Unicode, or on
    a terminal narrower than the drawing, it is one line of text that includes ``version`` when
    it fits. The drawing itself shows no version.
    """
    c = _resolve(caps, plain)
    if not c.unicode or c.width < _LOGO_WIDTH:
        tagline = "A-to-I RNA editing sites and rates"
        full = f"{tagline}  v{version}" if version else tagline
        text = f"ModFormer | {full}" if len(full) + 12 <= c.width else "ModFormer"
        return _line([(text, _SECONDARY)], c)
    cv = Canvas()
    for y, (row, roles) in enumerate(zip(_LOGO_ROWS, _LOGO_ROLES)):
        for x, ch in enumerate(row):
            if ch != " ":
                cv.put(x, y, ch, _LOGO_ROLE_STYLE[roles[x]])
    return "\n".join(cv.render(c))


def card_header(
    name: str, model: str, extra: str = "", *, caps: Caps | None = None, plain: bool = False
) -> str:
    """The rule that opens a result: ``── name · model · extra ──``, no wider than the screen.

    A name too long for the line is cut with an ellipsis first, then ``extra`` is dropped;
    ``model`` is always kept.
    """
    c = _resolve(caps, plain)
    dash, sep, cut = ("─", " · ", "…") if c.unicode else ("-", " | ", "...")

    def build(label: str, tail: str) -> list[tuple[str, Style]]:
        parts = [(dash * 2 + " ", _DIM), (label, _TEXT), (sep, _DIM), (model, _SECONDARY)]
        if tail:
            parts += [(sep, _DIM), (tail, _SECONDARY)]
        return parts + [(" " + dash * 2, _DIM)]

    for tail in ([extra, ""] if extra else [""]):
        room = c.width - sum(len(t) for t, _ in build("", tail))
        if len(name) <= room:
            return _line(build(name, tail), c)
        if room > len(cut) + 3:
            return _line(build(name[: room - len(cut)] + cut, tail), c)
    return _line(build(name[: max(1, c.width - sum(len(t) for t, _ in build("", "")) - len(cut))] + cut, ""), c)


def hint_line(text: str, *, caps: Caps | None = None, plain: bool = False) -> str:
    """One dim line, indented, that never runs past the screen."""
    c = _resolve(caps, plain)
    return _line([("  " + text, _DIM)], c, c.width)


# Rate track ----------------------------------------------------------------------
_RAMP_UNICODE = "▁▂▃▄▅▆▇█"
_RAMP_ASCII = ".:-=+*#@"


def track_column(position: int, length: int, ncols: int) -> int:
    """The track column that holds 1-based ``position`` of a ``length``-nt transcript."""
    return min(ncols - 1, max(0, (position - 1) * ncols // max(1, length)))


def _finite(v: Any) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _nice_step(span: int, columns: int, min_gap: int) -> int:
    """Smallest 1/2/5 x 10^k position step that leaves ``min_gap`` columns between ticks."""
    k = 0
    while True:
        for m in (1, 2, 5):
            step = m * 10**k
            if step * columns / max(1, span) >= min_gap:
                return step
        k += 1


def rate_track(
    positions: Sequence[int],
    rates: Sequence[float],
    width: int | None = None,
    *,
    length: int | None = None,
    start: int = 1,
    sequence: str | None = None,
    top: int = 5,
    mark: Sequence[int] | None = None,
    scale: float | None = None,
    caps: Caps | None = None,
    plain: bool = False,
) -> str:
    """The predicted rate along the transcript, or a span of it, as a block track.

    ``positions`` are 1-based; ``rates`` are fractions.  The track covers ``length``
    nucleotides from ``start`` (default: the whole transcript up to the last
    position), and the ruler gives true transcript coordinates.  Each column shows
    the highest rate among the positions it covers: the height is scaled to the
    highest rate drawn, and the brightness to ``scale`` (default: that same rate),
    so a table drawn with the same ``scale`` uses the same colors.

    ``mark`` lists the positions to flag above the track, in order of priority (for
    example the top editing-site scores); by default the ``top`` highest rates are
    flagged.  When ``sequence`` (the transcript) is given and the span fits one
    column per nucleotide, its letters are printed under the track, each adenosine
    in the color of its rate.

    Four lines come back (five with letters): markers, the track, the letters, a
    ruler and ruler labels.
    """
    c = _resolve(caps, plain)
    start = max(1, int(start))
    points = [
        (int(p), float(r))
        for p, r in zip(positions, rates)
        if _finite(p) and _finite(r) and int(p) >= 1
    ]
    end = start + int(length) - 1 if length else max((p for p, _ in points), default=0)
    points = [(p, r) for p, r in points if start <= p <= end]
    if not points:
        return "(no adenosines to show)"
    total = end - start + 1
    ncols = max(1, min(width or c.width, c.width, total))
    letters = sequence is not None and ncols == total

    def column(p: int) -> int:
        return track_column(p - start + 1, total, ncols)

    best: list[float | None] = [None] * ncols
    for p, r in points:
        col = column(p)
        best[col] = r if best[col] is None else max(best[col], r)  # type: ignore[type-var]
    rmax = max(r for _, r in points)
    shade = rmax if scale is None else scale
    ramp = _RAMP_UNICODE if c.unicode else _RAMP_ASCII
    empty = " " if letters or not c.unicode else "·"

    cv = Canvas()
    for col, val in enumerate(best):
        if val is None:
            cv.put(col, 1, empty, _DIM)
            continue
        t = min(1.0, max(0.0, (val / rmax) if rmax > 0 else 0.0))
        level = min(len(ramp) - 1, int(round(t * (len(ramp) - 1))))
        cv.put(col, 1, ramp[level], _rate_style(val, shade))
    row = 2
    if letters:
        rate_at = dict(points)
        for x, ch in enumerate(_clean(sequence or "")[start - 1:end]):
            p = start + x
            cv.put(x, row, ch, _rate_style(rate_at[p], shade) if p in rate_at else _DIM)
        row += 1

    if mark is not None:
        ranked = [(int(p), 0.0) for p in mark if _finite(p) and start <= int(p) <= end]
    else:
        ranked = sorted(points, key=lambda pr: (-pr[1], pr[0]))
    markers: dict[int, int] = {}
    for p, _ in ranked:
        col = column(p)
        if col not in markers and len(markers) < max(0, top):
            markers[col] = p
    # A label may run past the track into the terminal's free columns (a zoomed track is
    # narrower than the screen), so a marker near the right end is labeled on its right like the
    # others, and only a track as wide as the screen puts a label on a marker's left.
    reach = max(ncols, min(width or c.width, c.width))
    occupied = [False] * (reach + 2)
    for col in markers:
        occupied[col] = True
    glyph = "▼" if c.unicode else "v"
    for n, col in enumerate(markers):
        cv.put(col, 0, glyph, _FOCUS if n == 0 else _SECONDARY)

    def free(a: int, b: int) -> bool:
        return a >= 0 and b <= reach and not any(occupied[x] for x in range(a, b))

    # Markers in neighboring columns share one label, their positions in column order, so no
    # glyph is left without a name.
    clusters: list[list[int]] = []
    for col in sorted(markers):
        if clusters and clusters[-1][-1] == col - 1:
            clusters[-1].append(col)
        else:
            clusters.append([col])
    for cols in clusters:
        label = ",".join(str(markers[col]) for col in cols)
        n = len(label)
        first, last = cols[0], cols[-1]
        if free(last + 1, last + 1 + n) and not occupied[min(reach, last + 1 + n)]:
            at = last + 1
        elif free(first - n, first) and (first - n == 0 or not occupied[first - n - 1]):
            at = first - n
        else:
            continue
        for x in range(at, at + n):
            occupied[x] = True
        cv.put(at, 0, label, _DIM)

    tick_mark = "┴" if c.unicode else "+"
    base = "─" if c.unicode else "-"
    for x in range(ncols):
        cv.put(x, row, base, _DIM)
    for _, tick_col, label_col, label in _ruler_ticks(total, ncols, start):
        cv.put(tick_col, row, tick_mark, _DIM)
        cv.put(label_col, row + 1, label, _DIM)
    return "\n".join(cv.render(c, reach))


def _ruler_ticks(total: int, ncols: int, start: int = 1) -> list[tuple[int, int, int, str]]:
    """(position, tick column, label column, label) with labels that never overlap or overflow.

    Positions are true coordinates of a ``total``-nt span from ``start``.  The tick
    always sits in the column that holds its position; only a label that would run
    past the right edge is shifted left to fit.
    """
    end = start + total - 1
    gap = len(str(end)) + 2
    step = _nice_step(total, ncols, gap)
    first = -(-start // step) * step
    wanted = [start] + [p for p in range(first, end + 1, step) if p != start]
    if end not in wanted:
        wanted.append(end)
    ticks: list[tuple[int, int, int, str]] = []
    last_end = -1
    for p in wanted:
        label = str(p)
        tick_col = track_column(p - start + 1, total, ncols)
        col = min(tick_col, ncols - len(label))
        if col < 0 or col <= last_end:
            continue
        ticks.append((p, tick_col, col, label))
        last_end = col + len(label)
    return ticks


# Sites table ---------------------------------------------------------------------
_SITE_KEYS = {
    "position": ("position",),
    "context": ("context",),
    "score": ("edit_score",),
    "rate": ("predicted_rate",),
}


def _site_rows(sites: Any) -> list[tuple[int, str, float, float]]:
    if hasattr(sites, "to_dict"):
        sites = sites.to_dict("records")
    rows: list[tuple[int, str, float, float]] = []
    for s in sites:
        if isinstance(s, Mapping):
            vals = []
            for field in ("position", "context", "score", "rate"):
                key = next((k for k in _SITE_KEYS[field] if k in s), None)
                if key is None:
                    raise KeyError(f"site is missing a {field!r} column")
                vals.append(s[key])
        else:
            vals = list(s)[:4]
        pos, ctx, score, rate = vals
        rows.append((int(pos), str(ctx), float(score), float(rate)))
    return rows


def top_sites(
    sites: Any,
    n: int = 10,
    *,
    scale: float | None = None,
    caps: Caps | None = None,
    plain: bool = False,
) -> str:
    """A table of the top ``n`` sites: position, 5-mer context, edit score, rate.

    ``sites`` is an iterable of ``(position, context, edit_score, rate)``
    tuples or mappings (a pandas data frame of predictions also works).  Sites
    are ranked by editing-site score, then rate, then position; both numbers
    are always shown because they can disagree.  The rate is a percent with an
    inline bar, drawn and colored on ``scale`` (default: the highest rate in the
    table).  The edited adenosine of each context is
    highlighted, also where the context is clipped at an end of the sequence.
    Narrow terminals shorten the bar, then drop it, then close up the columns,
    and finally wrap each site onto two lines.
    """
    c = _resolve(caps, plain)
    rows = _site_rows(sites)
    rows.sort(key=lambda r: (-r[2], -r[3], r[0]))
    rows = rows[: max(0, n)]
    if not rows:
        return "(no sites)"
    rmax = max(r[3] for r in rows) or 1.0
    shade = rmax if scale is None else scale
    w = c.width
    # (x, line) of rank, position, context, score and rate; the score format; the header
    if w >= 34:
        at = [(0, 0), (3, 0), (10, 0), (19, 0), (27, 0)]
        fmt, head = "6.3f", [(0, 0, " #"), (3, 0, "   pos"), (10, 0, "context"),
                             (19, 0, " score"), (27, 0, "  rate")]
    elif w >= 28:
        at = [(0, 0), (3, 0), (10, 0), (16, 0), (22, 0)]
        fmt, head = "5.3f", [(0, 0, " #"), (3, 0, "   pos"), (10, 0, "ctx"),
                             (16, 0, "score"), (22, 0, "  rate")]
    else:
        at = [(0, 0), (3, 0), (10, 0), (3, 1), (9, 1)]
        fmt, head = "5.3f", [(0, 0, " #"), (3, 0, "   pos"), (10, 0, "ctx"),
                             (3, 1, "score"), (9, 1, "  rate")]
    span = 1 + max(dy for _, dy in at)
    bar_w = min(10, w - 36) if w >= 34 else 0
    bar_w = bar_w if bar_w >= 3 else 0

    cv = Canvas()
    for x, dy, text in head:
        cv.put(x, dy, text, _DIM)
    for k, (pos, ctx, score, rate) in enumerate(rows, start=1):
        y = span * k
        (xr, yr), (xp, yp), (xc, yc), (xs, ys), (xt, yt) = at
        cv.put(xr, y + yr, f"{k:>2}", _DIM)
        cv.put(xp, y + yp, f"{pos:>6}", _TEXT)
        hot = _motif_center(pos, ctx)
        for i, ch in enumerate(ctx[:5]):
            cv.put(xc + i, y + yc, ch, _FOCUS if i == hot else _SECONDARY)
        cv.put(xs, y + ys, f"{score:{fmt}}", _TEXT)
        cv.put(xt, y + yt, f"{rate * 100:5.1f}%", _TEXT)
        if bar_w:
            t = min(1.0, max(0.0, rate / shade)) if shade > 0 else 0.0
            cv.put(36, y, _inline_bar(t, bar_w, c), _rate_style(rate, shade))
    return "\n".join(cv.render(c))


def _motif_center(position: int, context: str) -> int:
    """Index of the edited adenosine in a 5-mer context, which is clipped at either end of the
    sequence."""
    hot = min(2, position - 1) if len(context) < 5 else 2
    return hot if 0 <= hot < len(context) else -1


def _inline_bar(t: float, width: int, c: Caps) -> str:
    if c.unicode:
        full, half = divmod(int(round(min(1.0, max(0.0, t)) * width * 2)), 2)
        return "━" * full + ("╸" if half else "")
    eighths = int(round(min(1.0, max(0.0, t)) * width * 8))
    full, part = divmod(eighths, 8)
    cells = full + (1 if part >= 4 else 0)
    return ("#" * cells).ljust(width, ".")


# Dot-bracket helpers ------------------------------------------------------------
def _pairs_of(dot_bracket: str) -> list[int]:
    stack: list[int] = []
    partner = [-1] * len(dot_bracket)
    for i, ch in enumerate(dot_bracket):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            if not stack:
                raise ValueError("unbalanced dot-bracket string")
            j = stack.pop()
            partner[i], partner[j] = j, i
    if stack:
        raise ValueError("unbalanced dot-bracket string")
    return partner


def _clean(sequence: str) -> str:
    return sequence.upper().replace("T", "U")


# Duplex --------------------------------------------------------------------------
_WATSON_CRICK = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")}
_WOBBLE = {("G", "U"), ("U", "G")}
# Grid rows, top to bottom; rows left empty are dropped.
_LABEL_A, _LOOP_A, _STRAND_A, _MARKS, _STRAND_B, _LOOP_B, _LABEL_B = range(7)


def _bounds(span: Any) -> tuple[int, int] | None:
    """``(start, end)`` as integers, or ``None`` for a missing or empty span."""
    try:
        a, b = (int(v) for v in span)
    except (TypeError, ValueError):
        return None
    return (a, b) if a <= b else None


def duplex(
    window: str | None,
    dot_bracket: str | None,
    *,
    origin: int,
    anchor: tuple[int, int],
    partner: tuple[int, int] | None,
    focus: int,
    length: int | None = None,
    width: int | None = None,
    caps: Caps | None = None,
    plain: bool = False,
) -> str:
    """The pairing between a site's anchor and its partner stretch, as two strands.

    ``window`` and ``dot_bracket`` are the fold window and its structure;
    ``origin`` is the transcript position (1-based) of ``window[0]``, which may
    be below 1 where the window is padded with ``N``.  ``anchor`` and ``partner``
    are 1-based inclusive transcript spans and ``focus`` is the site.  Both runs
    are clipped to the real sequence (and to ``length``, the transcript length).

    The anchor is drawn on top, 5' to 3'; the partner below it, 3' to 5', so
    each pair of the dot-bracket with one end in each run is one column with a
    mark (``│`` for a Watson-Crick pair, ``·`` for G-U, ``╎`` otherwise).  Bases
    between pairs sit one row out; the break gives the number of nucleotides
    between the runs.  When the drawing is wider than ``width`` it is cut around
    the site, shifted if needed so that some pairs stay in view, and its cut
    ends are marked.  Without a drawing the result is one status line.
    """
    c = _resolve(caps, plain)
    u = c.unicode
    width = c.width if width is None else min(width, c.width)
    site = int(focus)

    def status(text: str) -> str:
        return _line([(f"  site {site}: {text}", _SECONDARY)], c)

    seq = _clean(window or "")
    try:
        mate = _pairs_of(dot_bracket) if dot_bracket and len(dot_bracket) == len(seq) else None
    except ValueError:
        mate = None
    real = [k for k, ch in enumerate(seq) if ch != "N"]
    if mate is None or not real:
        return status("pairing view unavailable")
    first = max(1, origin + real[0])
    last = min(origin + real[-1], int(length)) if length else origin + real[-1]

    def clip(span: Any) -> tuple[int, int] | None:
        b = _bounds(span) if span is not None else None
        return _bounds((max(b[0], first), min(b[1], last))) if b else None

    top_run, bot_run = clip(anchor), clip(partner)
    if top_run is None or not top_run[0] <= site <= top_run[1]:
        return status("pairing view unavailable")
    if bot_run is None or (bot_run[0] <= top_run[1] and top_run[0] <= bot_run[1]):
        return status("no distinct partner stretch to draw")
    t_lo, t_hi = top_run[0] - origin, top_run[1] - origin
    b_lo, b_hi = bot_run[0] - origin, bot_run[1] - origin
    pairs = [(i, mate[i]) for i in range(t_lo, t_hi + 1) if b_lo <= mate[i] <= b_hi]
    if not pairs:
        return status("no predicted pairs between anchor and partner")

    f = site - origin
    Col = dict  # grid row -> (character, style, window index or None)

    def base(k: int, top: bool) -> tuple[str, Style, int]:
        return (seq[k], _FOCUS if k == f else _TEXT if top else _SECONDARY, k)

    def ends(tops: list[int], bots: list[int], right: bool) -> list[Col]:
        n = max(len(tops), len(bots))
        cols = [Col() for _ in range(n)]
        for run, row, top in ((tops, _STRAND_A, True), (bots, _STRAND_B, False)):
            for k, idx in enumerate(run):
                cols[k if right else n - len(run) + k][row] = base(idx, top)
        return cols

    glyphs = ("│", "·", "╎") if u else ("|", ":", "!")
    (i0, j0), (ik, jk) = pairs[0], pairs[-1]
    cols = ends(list(range(t_lo, i0)), list(range(b_hi, j0, -1)), right=False)
    for n, (i, j) in enumerate(pairs):
        kind = (seq[i], seq[j])
        mark = glyphs[0] if kind in _WATSON_CRICK else glyphs[1] if kind in _WOBBLE else glyphs[2]
        cols.append(Col({_STRAND_A: base(i, True), _MARKS: (mark, _PAIR, None),
                         _STRAND_B: base(j, False)}))
        if n + 1 < len(pairs):
            ups, downs = range(i + 1, pairs[n + 1][0]), range(j - 1, pairs[n + 1][1], -1)
            for k in range(max(len(ups), len(downs))):
                col = Col()
                if k < len(ups):
                    col[_LOOP_A] = base(ups[k], True)
                if k < len(downs):
                    col[_LOOP_B] = base(downs[k], False)
                cols.append(col)
    cols += ends(list(range(ik + 1, t_hi + 1)), list(range(jk - 1, b_lo - 1, -1)), right=True)

    # The break joins the strand ends that are neighbors in the RNA: on the right
    # when the partner lies downstream of the anchor, on the left when upstream.
    downstream = b_lo > t_hi
    gap = (bot_run[0] - top_run[1] - 1) if downstream else (top_run[0] - bot_run[1] - 1)
    gap_text = f"{gap} nt"
    break_w = 2 + len(gap_text)
    indent, gutter = 2, 3
    avail = max(3, width - indent - gutter - break_w)
    fcol = next(x for x, col in enumerate(cols) if any(v[2] == f for v in col.values()))
    lo = min(max(0, fcol - avail // 2), max(0, len(cols) - avail))

    def visible(x: int, at: int) -> bool:  # drawn, not one of the cut marks at the view's ends
        end = min(len(cols), at + avail)
        return at + (at > 0) <= x <= end - 1 - (end < len(cols))

    # Centered on the site, the view can hold no pair at all (a site far from its pairs, in a
    # narrow terminal), and then the drawing would be two loose strands. It slides to the nearest
    # position that keeps the site and shows the most pairs.
    pair_cols = [x for x, col in enumerate(cols) if _MARKS in col]
    if not any(visible(x, lo) for x in pair_cols):
        spots = [t for t in range(max(0, len(cols) - avail) + 1) if visible(fcol, t)]
        lo = max(spots, key=lambda t: (sum(visible(x, t) for x in pair_cols), -abs(t - lo)),
                 default=lo)
    hi = min(len(cols), lo + avail)
    view = cols[lo:hi]
    cut = Col({_STRAND_A: ("…" if u else "~", _DIM, None), _STRAND_B: ("…" if u else "~", _DIM, None)})
    if lo > 0:
        view[0] = cut
    if hi < len(cols):
        view[-1] = cut

    cv = Canvas()
    x0 = gutter if downstream else break_w
    x_end = x0 + len(view)
    where: dict[int, int] = {}
    for x, col in enumerate(view):
        for row, (ch, style, idx) in col.items():
            cv.put(x0 + x, row, ch, style)
            if idx is not None:
                where[idx] = x0 + x
    corner = ("╮", "╯", "╭", "╰", "┆") if u else ("\\", "/", "/", "\\", ")")
    if downstream:
        cv.put(0, _STRAND_A, "5'", _DIM)
        cv.put(0, _STRAND_B, "3'", _DIM)
        cv.put(x_end, _STRAND_A, corner[0], _DIM)
        cv.put(x_end, _MARKS, f"{corner[4]} {gap_text}", _DIM)
        cv.put(x_end, _STRAND_B, corner[1], _DIM)
    else:
        cv.put(x0 - 1, _STRAND_A, corner[2], _DIM)
        cv.put(0, _MARKS, f"{gap_text} {'┆' if u else '('}", _DIM)
        cv.put(x0 - 1, _STRAND_B, corner[3], _DIM)
        cv.put(x_end + 1, _STRAND_A, "3'", _DIM)
        cv.put(x_end + 1, _STRAND_B, "5'", _DIM)

    # Labels: the site first, then the true coordinates of each strand's drawn ends.
    taken: dict[int, set[int]] = {_LABEL_A: set(), _LABEL_B: set()}

    def label(row: int, x: int, text: str, style: Style, right: bool = False) -> bool:
        x = x - len(text) + 1 if right else x
        if x < 0 or x + len(text) > width - indent or taken[row] & set(range(x - 1, x + len(text) + 1)):
            return False
        cv.put(x, row, text, style)
        taken[row] |= set(range(x, x + len(text)))
        return True

    arrow = "▼" if u else "v"
    fx = where.get(f)  # the cut can leave a strand, even the site, outside the view
    if fx is not None:
        label(_LABEL_A, fx, f"{arrow}{site}", _FOCUS) or label(_LABEL_A, fx, f"{site}{arrow}", _FOCUS, right=True)
    for row, run in ((_LABEL_A, range(t_lo, t_hi + 1)), (_LABEL_B, range(b_hi, b_lo - 1, -1))):
        shown = [k for k in run if k in where]
        if not shown:  # none of this strand's bases fell inside the view: no ends to name
            continue
        label(row, where[shown[0]], str(origin + shown[0]), _DIM)
        if len(shown) > 1:
            label(row, where[shown[-1]], str(origin + shown[-1]), _DIM, right=True)

    heading = " " * indent + f"predicted structure at site {site}"
    off_screen = " (pairs off-screen)"
    if not any(visible(x, lo) for x in pair_cols) and len(heading) + len(off_screen) <= width:
        heading += off_screen  # no pair is in view: say so rather than draw two loose strands
    heading = _line([(heading, _SECONDARY)], c)
    rows = [y for y in range(cv.height()) if y in cv._rows]
    lines = cv.render(c, width - indent)
    return "\n".join([heading] + [" " * indent + lines[y] for y in rows])


# Progress ------------------------------------------------------------------------
_HIDDEN_CURSOR_STREAMS: list[TextIO] = []


def _restore_cursors() -> None:
    for stream in list(_HIDDEN_CURSOR_STREAMS):
        _show_cursor(stream)


atexit.register(_restore_cursors)


def _hide_cursor(stream: TextIO) -> None:
    if stream not in _HIDDEN_CURSOR_STREAMS:
        _HIDDEN_CURSOR_STREAMS.append(stream)
        _safe_write(stream, _HIDE_CURSOR)


def _show_cursor(stream: TextIO) -> None:
    if stream in _HIDDEN_CURSOR_STREAMS:
        _HIDDEN_CURSOR_STREAMS.remove(stream)
        _safe_write(stream, _SHOW_CURSOR)


def _safe_write(stream: TextIO, text: str) -> None:
    try:
        stream.write(text)
        stream.flush()
    except (OSError, ValueError):
        pass


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"


_HELIX_WIDTH = 8  # the cells of the helix beside the active stage


class Progress:
    """One updating status line: ``progress(stage, done, total)``.

    On a terminal the active stage redraws in place (turning helix, stage name,
    bar, counts, elapsed time and ETA) and each finished stage stays behind as
    one line.  Without a terminal, with ``NO_COLOR`` or ``plain=True``, exactly
    one plain line is printed per stage when it finishes.  ``total`` may be 0
    or ``None`` when unknown.  Use it as a context manager (or call
    :meth:`close`) so the cursor is always restored.
    """

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        plain: bool = False,
        caps: Caps | None = None,
        clock: Callable[[], float] = time.monotonic,
        min_interval: float = 0.05,
    ) -> None:
        self.stream: TextIO = sys.stderr if stream is None else stream
        self.caps = _resolve(caps, plain, self.stream)
        self._clock = clock
        self._min_interval = min_interval
        self._lock = threading.RLock()
        self._stage: str | None = None
        self._t0 = 0.0
        self._done = 0
        self._frac = 0.0
        self._total = 0
        self._complete = False
        self._printed = False
        self._live = False
        self._last_draw = -1e9
        self._frames = itertools.cycle("|/-\\")  # the spinner of a terminal without Unicode
        self._tick = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- public ------------------------------------------------------------------
    def __call__(self, stage: str, done: float = 0, total: int | None = None) -> None:
        with self._lock:
            now = self._clock()
            total_n = int(total) if _finite(total) and total and total > 0 else 0
            # ``done`` may be fractional (a step part way through): the bar and the percentage
            # follow it, the count shows the whole steps finished.
            frac = float(done) if _finite(done) and done and done > 0 else 0.0
            if total_n and frac > total_n:
                frac = float(total_n)
            done_n = int(frac)
            if stage != self._stage:
                self._finish_stage(now)
                self._stage, self._t0 = stage, now
                self._complete = False
                self._printed = False
            self._done, self._total, self._frac = done_n, total_n, frac
            if total_n and done_n >= total_n:
                self._complete = True
            if self.caps.animate:
                self._draw(now, force=self._complete)
            elif self._complete and not self._printed:
                self._emit_final(now)

    def __enter__(self) -> "Progress":
        if self.caps.animate:
            self._stop.clear()
            self._thread = threading.Thread(target=self._ticker, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Finish the open stage and restore the cursor."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._thread = None
        with self._lock:
            try:
                self._finish_stage(self._clock())
                self._stage = None
            finally:
                _show_cursor(self.stream)

    # -- internals ---------------------------------------------------------------
    def _ticker(self) -> None:
        while not self._stop.wait(0.1):
            with self._lock:
                if self._stage is not None and not self._complete:
                    self._draw(self._clock(), force=False)

    def _finish_stage(self, now: float) -> None:
        if self._stage is None or self._printed:
            return
        self._emit_final(now)

    def _emit_final(self, now: float) -> None:
        self._printed = True
        text = self._build(now, final=True)
        if self.caps.animate and self._live:
            _safe_write(self.stream, f"\r{text}{_ESC}[K\n")
        else:
            _safe_write(self.stream, f"{text}\n")
        self._live = False

    def _draw(self, now: float, force: bool) -> None:
        if self._stage is None or self._printed:
            return
        if not force and now - self._last_draw < self._min_interval:
            return
        self._last_draw = now
        if not self._live:
            _hide_cursor(self.stream)
        text = self._build(now, final=False)
        _safe_write(self.stream, f"\r{text}{_ESC}[K")
        self._live = True

    def _build(self, now: float, final: bool) -> str:
        c = self.caps
        stage = self._stage or ""
        elapsed = now - self._t0
        done, total = self._done, self._total
        if not c.has_color and not c.animate:
            counts = f"{done}/{total}" if total else f"{done}"
            return f"{stage}: {counts} ({_fmt_time(elapsed)})"
        counts = f"{done}/{total}" if total else f"{done}"
        pct = f"{int(100 * self._frac / total):>3d}%" if total else ""
        timing = _fmt_time(elapsed)
        if not final and total and self._frac > 0 and done < total:
            timing += f" ETA {_fmt_time(elapsed * (total - self._frac) / self._frac)}"
        # A terminal that was resized since the caps were made is followed on every redraw.
        live = _stream_columns(self.stream) if c.animate else None
        limit = (max(12, live) if live else c.width) - (1 if c.animate else 0)
        if final:
            mark = "✓" if c.unicode and self._complete else "·" if c.unicode else "*"
            glyph_cells: list[Cell] = [(mark, _SECONDARY if self._complete else _DIM)]
            # Pad a finished row to the helix width so its bar lines up with the live rows; the
            # mark sits at the right of that pad, next to the stage name.
            if c.animate and c.unicode:
                glyph_cells.insert(0, (" " * (_HELIX_WIDTH - 1), NO_STYLE))
        else:
            self._tick += 1
            if c.unicode:
                lit = (self._tick // 2) % _HELIX_WIDTH
                glyph_cells = _helix_cells(_HELIX_WIDTH, self._tick * 0.45, lit)
            else:
                glyph_cells = [(next(self._frames), _SECONDARY)]
        name = stage.ljust(12)
        variants = [
            (24, True, True, True),
            (16, True, True, True),
            (12, False, True, True),
            (10, False, False, True),
            (8, False, False, False),
            (0, False, False, False),
        ]
        parts: list[Cell] = []
        for bar_w, with_eta, with_pct, with_glyph in variants:
            parts = []
            if with_glyph:
                parts.extend(glyph_cells)
                parts.append((" ", NO_STYLE))
            parts.append((name, _TEXT if not final else _SECONDARY))
            if bar_w and not (final and not total):
                parts.append((" ", NO_STYLE))
                parts.extend(self._bar_cells(bar_w, final))
            parts.append((" " + counts, _DIM))
            if with_pct and pct:
                parts.append((" " + pct, _DIM))
            t = timing if with_eta else timing.split(" ETA")[0]
            parts.append(("  " + t, _DIM))
            if sum(len(p[0]) for p in parts) <= limit:
                break
        cv = Canvas()
        x = 0
        for text, style in parts:
            cv.put(x, 0, text, style)
            x += len(text)
        return cv.render(c, limit)[0]

    def _bar_cells(self, width: int, final: bool) -> list[Cell]:
        c = self.caps
        full_ch, empty_ch = ("━", "━") if c.unicode else ("#", "-")
        done_style = _DIM if final else _SECONDARY
        cells: list[Cell] = []
        if self._total:
            filled = int(min(1.0, self._frac / self._total) * width)
            for i in range(width):
                if i < filled:
                    head = i == filled - 1 and not final
                    cells.append((full_ch, Style("rate5") if head else done_style))
                else:
                    cells.append((empty_ch, _RULE))
        else:
            pos = self._tick % (width + 4) - 4
            for i in range(width):
                on = pos <= i < pos + 4 and not final
                cells.append((full_ch, Style("rate3")) if on else (empty_ch, _RULE))
        if not c.unicode:
            return [("[", _DIM), *cells, ("]", _DIM)]
        return cells


# Easter egg ----------------------------------------------------------------------
_EGG_TOP = "GCCAUAGCUACGG"
_EGG_TARGET = 3
_COMPLEMENT = {"A": "U", "U": "A", "G": "C", "C": "G"}


def _egg_frame(c: Caps, nb: int, scan: int | None, state: str) -> str:
    x0 = 4
    top = _EGG_TOP[:nb]
    cv = Canvas()
    glyph = "▼" if c.unicode else "v"
    if scan is not None:
        hot = state in ("glow", "edited") or scan == _EGG_TARGET and state == "scan"
        cv.put(x0 + 3 * scan, 0, glyph, _FOCUS if hot else _SECONDARY)
        if scan == _EGG_TARGET or state != "scan":
            cv.put(x0 + 3 * scan + 2, 0, "ADAR", _SECONDARY)
    cv.put(0, 1, "5'", _DIM)
    cv.put(0, 3, "3'", _DIM)
    bar = "│" if c.unicode else "|"
    for i, base in enumerate(top):
        x = x0 + 3 * i
        edited = i == _EGG_TARGET and state in ("edited", "final")
        glow = i == _EGG_TARGET and state == "glow"
        cv.put(x, 1, "I" if edited else base, _FOCUS if (edited or glow) else _TEXT)
        cv.put(x, 2, ":" if edited else bar, _DIM if edited else _PAIR)
        cv.put(x, 3, _COMPLEMENT[base], _SECONDARY)
    if state == "final":
        arrow = "→" if c.unicode else "->"
        cv.put(x0, 5, f"A {arrow} I   ADAR was here.", _SECONDARY)
    return "\n".join(cv.render(c))


def easter_egg(caps: Caps | None = None, plain: bool = False) -> Iterator[str]:
    """Frames of a short animation: ADAR slides along a duplex and turns an A into an I.

    The last frame is the finished picture, so showing only that frame (what
    :func:`play` does without a terminal) still reads well.
    """
    c = _resolve(caps, plain)
    nb = min(len(_EGG_TOP), max(_EGG_TARGET + 2, (c.width - 6) // 3))
    for scan in range(_EGG_TARGET + 1):
        yield _egg_frame(c, nb, scan, "scan")
    for _ in range(2):
        yield _egg_frame(c, nb, _EGG_TARGET, "glow")
    yield _egg_frame(c, nb, _EGG_TARGET, "edited")
    yield _egg_frame(c, nb, None, "final")


def play(
    frames: Iterable[str],
    stream: TextIO | None = None,
    delay: float = 0.09,
    *,
    caps: Caps | None = None,
    plain: bool = False,
) -> None:
    """Show ``frames`` in place on ``stream`` (the last one stays on screen).

    Without animation only the final frame is written.  The screen is never
    cleared, and the cursor is restored even on ``KeyboardInterrupt``, which is
    swallowed after leaving the current frame in place.
    """
    stream = sys.stdout if stream is None else stream
    c = _resolve(caps, plain, stream)
    if not c.animate:
        last = None
        for last in frames:
            pass
        if last is not None:
            _safe_write(stream, last + "\n")
        return
    height = 0
    _hide_cursor(stream)
    try:
        for frame in frames:
            lines = frame.split("\n")
            if height:
                _safe_write(stream, f"{_ESC}[{height}A\r")
            lines += [""] * (height - len(lines))
            _safe_write(stream, "".join(f"{ln}{_ESC}[K\n" for ln in lines))
            height = max(height, len(lines))
            time.sleep(delay)
    except KeyboardInterrupt:
        _safe_write(stream, "\n")
    finally:
        _show_cursor(stream)
