"""Geometry of the binned editing-rate head."""

from __future__ import annotations

import numpy as np


def rate_bin_centers(n_bins: int = 17, thr: float = 0.10, min_rate: float = 0.002) -> np.ndarray:
    """Centers of the rate bins, shape ``(n_bins,)``, float64, strictly increasing.

    Bin 0 is a point mass at exactly 0.0 (an unedited adenosine). The remaining
    bins form two geometric ladders that meet at ``thr``, so ``thr`` is an exact
    bin edge: the first ladder spans ``[min_rate, thr]`` and the second spans
    ``[thr, 1]``. Each center is the geometric mean of its two edges.
    """
    n_bins = int(n_bins)
    if n_bins < 3:
        raise ValueError(f"n_rate_bins must be at least 3, got {n_bins}")
    if not (0.0 < min_rate < thr < 1.0):
        raise ValueError(f"need 0 < rate_bin_min < edit_rate_thr < 1, got {min_rate} and {thr}")
    n_rest = n_bins - 1
    n_lo = n_rest // 2
    n_hi = n_rest - n_lo
    edges_lo = np.geomspace(min_rate, thr, n_lo + 1)
    edges_hi = np.geomspace(thr, 1.0, n_hi + 1)
    c_lo = np.sqrt(edges_lo[:-1] * edges_lo[1:])
    c_hi = np.sqrt(edges_hi[:-1] * edges_hi[1:])
    return np.concatenate([[0.0], c_lo, c_hi])
