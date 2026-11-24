"""Input normalization and the fixed tile grid.

A transcript is cut into 128-nt tiles starting at 0, 128, 256, ... Each tile has two
64-nt anchors and every adenosine belongs to exactly one anchor, so each one is scored
once. The structure of a tile is predicted in a 1022-nt window centered on the tile
(447 nt of context on either side), padded with ``N`` where the window runs past the
ends of the transcript.

Selecting a site or a region never moves the grid: it picks the tiles that already
contain the requested positions, so a position gets the same sequence context whether it is
run alone or inside the whole transcript (scores can differ in the last digits with batching).
"""

from __future__ import annotations

import operator
from dataclasses import dataclass

#: Shortest accepted input, in nucleotides.
MIN_LENGTH = 64
#: Width of the folding window.
WINDOW = 1022
#: Width of a tile and the stride of the grid.
TILE = 128
#: Width of an anchor (a tile holds two).
ANCHOR = 64
#: Offset of the tile inside its window: ``tile_start - TILE_OFFSET`` is the window origin.
TILE_OFFSET = (WINDOW - TILE) // 2
#: A tail shorter than this is scored as an extension of the grid.
EXTENSION_BELOW = 64

_ALPHABET = frozenset("ACGUN")


def normalize(seq: str) -> str:
    """Return ``seq`` as upper-case RNA over ``ACGUN``.

    White space is removed, ``T`` becomes ``U``, and anything else outside ``ACGUN`` is
    rejected. The result must be at least :data:`MIN_LENGTH` nucleotides long. Positions
    reported anywhere in the package refer to the string returned here.
    """
    if not isinstance(seq, str):
        raise TypeError(f"sequence must be a str, got {type(seq).__name__}")
    out = "".join(seq.split()).upper().replace("T", "U")
    for i, ch in enumerate(out):
        if ch not in _ALPHABET:
            raise ValueError(
                f"unsupported character {ch!r} at position {i + 1}; "
                "only A, C, G, U (or T) and N are accepted")
    if len(out) < MIN_LENGTH:
        raise ValueError(
            f"sequence is {len(out)} nt; at least {MIN_LENGTH} nt are required")
    return out


@dataclass(frozen=True)
class Tile:
    """One 128-nt grid tile of a transcript (half-open, zero-based coordinates).

    ``ordinal`` is the tile's index on the grid of the whole sequence, whatever subset
    of tiles was selected. ``extension`` marks a final tile shorter than 64 nt; the model was
    not trained on tiles this short.
    """

    ordinal: int
    start: int
    end: int
    extension: bool

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def anchors(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """The two anchor intervals, clipped to the sequence (the second may be empty)."""
        mid = min(self.start + ANCHOR, self.end)
        return (self.start, mid), (mid, self.end)


def _grid(n: int) -> list[Tile]:
    tiles = []
    for ordinal, start in enumerate(range(0, n, TILE)):
        end = min(start + TILE, n)
        tiles.append(Tile(ordinal, start, end, end - start < EXTENSION_BELOW))
    return tiles


def _position(value, what: str) -> int:
    """``value`` as an int position (numpy integers included); anything else is a TypeError."""
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{what} must be an integer position, got {value!r}") from None


def plan_tiles(n: int, site: int | None = None,
               region: tuple[int, int] | None = None) -> list[Tile]:
    """Tiles to run for a sequence of ``n`` nucleotides.

    With no selection every tile of the grid is returned. ``site`` (one-based) selects
    the tile containing it; ``region`` (one-based, inclusive ``(first, last)``) selects
    every tile it touches. At most one of the two may be given. Positions are integers.
    """
    if n < MIN_LENGTH:
        raise ValueError(f"sequence is {n} nt; at least {MIN_LENGTH} nt are required")
    if site is not None and region is not None:
        raise ValueError("give either site or region, not both")
    grid = _grid(n)
    if site is not None:
        site = _position(site, "site")
        if not 1 <= site <= n:
            raise ValueError(f"site {site} is outside 1..{n}")
        return [grid[(site - 1) // TILE]]
    if region is not None:
        first, last = (_position(p, "region") for p in region)
        if first > last:
            raise ValueError(f"region {first}-{last} runs backwards; give the smaller position first")
        if first < 1 or last > n:
            raise ValueError(f"region {first}-{last} is not inside 1..{n}")
        return grid[(first - 1) // TILE:(last - 1) // TILE + 1]
    return grid


def window_for(seq: str, tile_start: int) -> str:
    """The 1022-nt fold window of a tile: ``seq`` from ``tile_start - 447``, ``N``-padded."""
    origin = tile_start - TILE_OFFSET
    lo, hi = max(origin, 0), min(origin + WINDOW, len(seq))
    left = max(-origin, 0)
    body = seq[lo:hi] if hi > lo else ""
    return "N" * left + body + "N" * (WINDOW - left - len(body))
