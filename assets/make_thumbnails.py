"""Render the README thumbnails of the reproduced figures.

    python assets/make_thumbnails.py            # writes assets/fig2.png, fig3.png, fig4.png

The figures are drawn by the same functions the notebooks use (``notebooks/_figures.py``)
from the fetched source tables, then saved small, with no text metadata in the files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))

import _figures  # noqa: E402


def main() -> int:
    out = Path(__file__).resolve().parent
    for name, draw in (("fig2", _figures.figure2), ("fig3", _figures.figure3), ("fig4", _figures.figure4)):
        fig, _ = draw()
        path = out / f"{name}.png"
        fig.savefig(path, dpi=110, metadata={"Software": None})
        print(path.name, path.stat().st_size, "bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
