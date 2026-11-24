"""Input pipeline: language-model embedding, folding, and partner-window gathering.

Sub-modules are imported where they are used, so importing this package stays cheap.
"""
from __future__ import annotations

#: The ViennaRNA release this package is validated against: with it, the released
#: predictions are reproduced from raw sequence.
VIENNARNA_VERSION = "2.7.2"


def recipe() -> dict:
    """The input recipe this package implements, in the layout of a release's
    ``config.json`` ``features`` block. A release whose block differs was trained on
    inputs this package would not reproduce."""
    from ..tiling import ANCHOR, TILE, WINDOW
    from . import embed, fold, gather, structure_tags

    return {
        "language_model": embed.MODEL_ID,
        "language_model_revision": embed.REVISION,
        "embedding_window": embed.WINDOW,
        "embedding_stride": embed.STRIDE,
        "fold_window": WINDOW,
        "tile": TILE,
        "anchor_len": ANCHOR,
        "partner_len": gather.PARTNER_LEN,
        "partner_mode": "fold",
        "gather_conf": structure_tags.CONF,
        "gather_gap_tol": structure_tags.GAP_TOL,
        "gather_register_tol": structure_tags.REGISTER_TOL,
        "gather_group_tol": gather.REGISTER_GROUP_TOL,
        "gather_tie_tol": gather.RELATIVE_TIE_TOL,
        "fold": {"engine": "ViennaRNA", "version": VIENNARNA_VERSION,
                 "temperature": fold.TEMPERATURE, "dangles": fold.DANGLES, "salt": fold.SALT},
    }


def check_recipe(release: dict) -> None:
    """Raise if a release's recorded input recipe is not the one this package builds, or
    if the installed ViennaRNA is not the release the package is validated against."""
    import RNA

    want = release.get("features")
    have = recipe()
    if want != have:
        diff = sorted(k for k in set(want or {}) | set(have) if (want or {}).get(k) != have.get(k))
        raise RuntimeError("this model was trained with a different input recipe than this "
                           f"version of modformer builds (differs in: {', '.join(diff)}); "
                           "update modformer")
    installed = getattr(RNA, "__version__", "unknown")
    if installed != VIENNARNA_VERSION:
        raise RuntimeError(f"ViennaRNA {installed} is installed, but modformer is validated with "
                           f"{VIENNARNA_VERSION}; run: pip install ViennaRNA=={VIENNARNA_VERSION}")
