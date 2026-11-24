"""ModFormer: structure-aware prediction of A-to-I RNA editing from sequence.

    >>> import modformer
    >>> table = modformer.predict(seq)             # every adenosine of seq
    >>> table = modformer.predict(seq, site=412)   # only the anchor that holds site 412

The model reads an RNA sequence, embeds it with the RiNALMo language model, folds it with
ViennaRNA, pairs each 64-nt anchor with its most likely duplex partner, and returns two
numbers for every adenosine: an editing-site score and a predicted editing rate.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional, Union

__version__ = "1.0.0"

DEFAULT_MODEL = "ModFormer-24L512-12ep"
FINETUNED_MODEL = "ModFormer-24L512-12ep-FT1ep"

#: Contexts of the fine-tuned model that were trained on transcript tiles, the input layout
#: this package builds. Keys are the names accepted on the command line; values are the context
#: names stored in the released weights ("observed" is the reference-site annotation context).
CONTEXTS = {
    "observed": "catalogue",
    "hek293ft": "hs_hek293ft_drna",
    "neuro2a": "mm_neuro2a_drna",
    "mouse-brain": "mm_brain_drna",
}

COLUMNS = ["position", "context", "edit_score", "predicted_rate",
           "partner_start", "partner_end", "extension"]

#: Column types of the table :func:`predict` returns, also when it is empty.
COLUMN_TYPES = {"position": "int64", "context": "object", "edit_score": "float64",
                "predicted_rate": "float64", "partner_start": "Int64",
                "partner_end": "Int64", "extension": "bool"}

Progress = Optional[Callable[[str, int, int], None]]

_MODELS: dict = {}


def load(model: str = DEFAULT_MODEL, device: Optional[str] = None):
    """Load (once per process) a released model. Returns ``(model, config)``.

    ``device=None``, ``"cuda"`` and the current GPU by index share one loaded copy.
    """
    import torch

    from .features.embed import resolve_device
    from .model import load_model

    dev = resolve_device(device)  # the same rule load_model applies, so the key names the real device
    if dev.type == "cuda" and dev.index is None and torch.cuda.is_available():
        dev = torch.device("cuda", torch.cuda.current_device())
    key = (str(model), str(dev))
    if key not in _MODELS:
        _MODELS[key] = load_model(model, device=device)
    # Also file the copy under the device as the caller spelled it, so a lookup by that name finds it.
    _MODELS[(str(model), str(device))] = _MODELS[key]
    return _MODELS[key]


def _context_id(net, context: Optional[str]) -> Optional[int]:
    names = getattr(net, "context_names", None)
    if names is None:
        if context is not None:
            raise ValueError("this model takes no context; drop the context or use the "
                             f"fine-tuned model ({FINETUNED_MODEL})")
        return None
    if context is None:
        raise ValueError("the fine-tuned model needs a context: one of "
                         + ", ".join(CONTEXTS))
    if context not in CONTEXTS:
        raise ValueError(f"context {context!r} is not available for transcript input; "
                         "choose one of " + ", ".join(CONTEXTS))
    return list(names).index(CONTEXTS[context])


def _local_fold(window: str, dot_bracket: str, center: int, flank: int = 28):
    """``(sequence, dot_bracket, index)`` for the neighborhood of one window position.

    Pairs that leave the neighborhood are shown as unpaired, so the slice is a valid
    structure by itself.
    """
    lo, hi = max(0, center - flank), min(len(window), center + flank + 1)
    while window[lo] == "N":  # the window is N-padded past the ends of the sequence
        lo += 1
    while window[hi - 1] == "N":
        hi -= 1
    stack, keep = [], {}
    for i, ch in enumerate(dot_bracket):
        if ch == "(":
            stack.append(i)
        elif ch == ")" and stack:
            j = stack.pop()
            if lo <= j and i < hi:
                keep[j], keep[i] = "(", ")"
    return window[lo:hi], "".join(keep.get(i, ".") for i in range(lo, hi)), center - lo


#: Share of a tile's predict-stage time assumed to be folding. It only places the progress bar
#: between whole tiles; results do not depend on it.
FOLD_SHARE = 0.15


def _score(net, batch, slice_size: int = 16, progress: Progress = None,
           on_slice: Optional[Callable[[float], None]] = None) -> dict:
    """Run the network over a batch in slices, halving the slice when a GPU runs out of memory."""
    import time

    import numpy as np
    import torch

    from .model import predict_batch

    n = int(batch["embeddings"].shape[0])
    parts, i, waited = [], 0, 0.0
    while i < n:
        j = min(n, i + slice_size)
        piece = {k: v[i:j] for k, v in batch.items()}
        try:
            parts.append(predict_batch(net, piece))
            i = j
            if on_slice:
                on_slice(i / n)
        except torch.OutOfMemoryError:
            # The same patience as the embedding step: a shared GPU may free up.
            from .features.embed import GPU_MIN_BYTES, OOM_RETRY_SECONDS, OOM_WAIT_SECONDS

            torch.cuda.empty_cache()
            if slice_size > 1:
                slice_size = max(1, slice_size // 2)
                continue
            dev = next(net.parameters()).device
            total = torch.cuda.mem_get_info(dev)[1] if dev.type == "cuda" else None
            if total is not None and total < GPU_MIN_BYTES:
                raise RuntimeError(
                    f"this GPU has {total / 2**30:.1f} GiB of memory, and the language model "
                    f"and scoring together need about {GPU_MIN_BYTES // 2**30} GiB. Run on the "
                    "CPU with device='cpu' (on the command line: --device cpu)") from None
            if waited >= float(OOM_WAIT_SECONDS):
                raise RuntimeError(
                    "the GPU ran out of memory on a single tile. Wait for the other work "
                    "on this GPU to finish, or run on the CPU with device='cpu' "
                    "(on the command line: --device cpu)") from None
            if progress:
                progress("waiting for GPU memory", int(waited), int(OOM_WAIT_SECONDS))
            time.sleep(OOM_RETRY_SECONDS)
            waited += OOM_RETRY_SECONDS
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def predict(sequence: str, *, model: str = DEFAULT_MODEL, site: Optional[int] = None,
            region: Optional[tuple] = None, context: Optional[str] = None,
            device: Optional[str] = None, workers: Optional[int] = None,
            batch_tiles: int = 16, progress: Progress = None):
    """Score the adenosines of one RNA sequence.

    ``sequence`` may use T or U. With no selection every adenosine is scored: the sequence
    is cut into 128-nt tiles, each tile is folded inside its 1022-nt window, and the tiles
    are stitched back into one table. ``site`` (1-based) returns only the 64-nt anchor that
    holds that position, and ``region`` (1-based, inclusive ``(first, last)``) only the
    positions inside it. A selection uses the same sequence context as a whole-sequence run and
    only skips the work the other positions would need; small numerical differences can arise
    from how the work is batched, more so on a GPU, where the language model runs in half
    precision.

    ``model`` is a release name (``DEFAULT_MODEL`` or ``FINETUNED_MODEL``) or a directory with
    ``model.safetensors`` and ``config.json``. ``FINETUNED_MODEL`` needs ``context``, a key of
    ``CONTEXTS``; leave it unset for ``DEFAULT_MODEL``. ``device`` defaults to CUDA when it is
    available. ``workers`` sets the number of folding processes (default: all available CPUs) and
    ``batch_tiles`` how many tiles are folded and scored together. ``progress(stage, done, total)``
    is called as work advances.

    Returns a :class:`pandas.DataFrame` with one row per adenosine:

    ``position``        1-based position in the (normalized) sequence
    ``context``         the adenosine with up to two neighbors on each side (fewer at the
                        ends of the sequence)
    ``edit_score``      0-1 score from the editing-site head (how strongly the site resembles
                        observed editing sites; not the fraction of molecules edited)
    ``predicted_rate``  0-1 expected editing rate from the rate head
    ``partner_start``,
    ``partner_end``     1-based span of the partner window the anchor was paired with,
                        clipped to the sequence; empty when the anchor had no partner
                        inside the sequence
    ``extension``       True for positions in a final tile shorter than 64 nt, a length the
                        model was not trained on
    """
    import numpy as np
    import pandas as pd
    import torch

    from .features import check_recipe
    from .features.embed import embed_sequence
    from .features.fold import fold_windows
    from .features.gather import build_examples, collate
    from .features.gather import PARTNER_LEN
    from .tiling import ANCHOR, TILE_OFFSET, WINDOW, normalize, plan_tiles, window_for

    if batch_tiles < 1:
        raise ValueError(f"batch_tiles must be at least 1, got {batch_tiles}")
    seq = normalize(sequence)
    n = len(seq)
    tiles = plan_tiles(n, site=site, region=region)
    net, _ = load(model, device)
    check_recipe(net.release)
    ctx = _context_id(net, context)

    first, last = 1, n  # 1-based inclusive bounds of the positions to report
    if site is not None:
        first = ((site - 1) // ANCHOR) * ANCHOR + 1
        last = min(n, first + ANCHOR - 1)
    elif region is not None:
        first, last = region

    spans = None
    if site is not None or region is not None:
        # Embed only the windows that supply the rows the selected tiles can read.
        spans = [(max(0, t.start - TILE_OFFSET), min(n, t.start - TILE_OFFSET + WINDOW))
                 for t in tiles]
    emb = embed_sequence(seq, device=device, progress=progress, needed=spans)

    rows, folds = [], {}
    done = 0
    if progress:
        progress("predict", 0, len(tiles))  # open the stage before the first, slow, fold
    for lo in range(0, len(tiles), batch_tiles):
        chunk = tiles[lo:lo + batch_tiles]
        windows = [window_for(seq, t.start) for t in chunk]

        def folding(_stage: str, k: int, m: int, base: int = done, size: int = len(chunk)) -> None:
            progress("predict", base + size * FOLD_SHARE * k / m, len(tiles))

        folded = fold_windows(windows, workers=workers, progress=folding if progress else None)
        examples, owners = [], []
        for t, window, (bpp, dot_bracket) in zip(chunk, windows, folded):
            folds[t.start] = dot_bracket
            pair = build_examples(seq, emb, t.start, bpp, dot_bracket, t.ordinal)
            examples.extend(pair)
            owners.extend([t] * len(pair))
        batch = collate(examples)
        if ctx is not None:
            batch["context_id"] = torch.full((len(examples),), ctx, dtype=torch.long)
        if progress:
            progress("predict", done + len(chunk) * FOLD_SHARE, len(tiles))
        def scored(part: float, base: int = done, size: int = len(chunk)) -> None:
            # Stay just below the chunk's end until the chunk is done.
            progress("predict", min(base + size * (FOLD_SHARE + (1 - FOLD_SHARE) * part),
                                    base + size - 1e-6), len(tiles))

        out = _score(net, batch, progress=progress, on_slice=scored if progress else None)
        local = batch["a_position_orig"].numpy()
        for i, t in enumerate(owners):
            p0 = int(batch["partner_starts"][i])
            inside = p0 < n and p0 + PARTNER_LEN > 0
            partner = (max(1, p0 + 1), min(n, p0 + PARTNER_LEN)) if inside else (pd.NA, pd.NA)
            for j in np.flatnonzero(out["valid"][i]):
                p = t.start + int(local[i, j])
                if not first <= p + 1 <= last:
                    continue
                rows.append((p + 1, seq[max(0, p - 2):p + 3],
                             float(out["catalogue_probability"][i, j]),
                             float(out["rate"][i, j]), partner[0], partner[1], t.extension))
        done += len(chunk)
        if progress:
            progress("predict", done, len(tiles))

    table = pd.DataFrame(rows, columns=COLUMNS).sort_values("position", ignore_index=True)
    table = table.astype(COLUMN_TYPES)  # fixed column types, also for an empty table
    table.attrs.update(length=n, tiles=len(tiles), model=str(model), context=context,
                       site=site, region=region, folds=folds)
    _top_fold(table, seq, folds)
    return table


def _top_fold(table, seq: str, folds: dict) -> None:
    """Set ``table.attrs["top_fold"]``: the local fold around the table's top-scoring site.

    ``folds`` maps a tile start to the dot-bracket of its window, as :func:`predict` keeps it.
    """
    from .tiling import TILE, TILE_OFFSET, window_for

    table.attrs.pop("top_fold", None)
    if not len(table):
        return
    top = table.sort_values(["edit_score", "predicted_rate", "position"],
                            ascending=[False, False, True]).iloc[0]
    best = int(top["position"]) - 1
    tile_start = (best // TILE) * TILE
    table.attrs["top_fold"] = _local_fold(window_for(seq, tile_start), folds[tile_start],
                                          best - tile_start + TILE_OFFSET)


def predict_many(records: Union[dict, Iterable[tuple]], **kwargs):
    """Score several sequences. ``records`` maps ids to sequences (or is ``(id, seq)`` pairs).

    Returns one table with a leading ``sequence_id`` column. Each sequence is scored exactly
    as :func:`predict` would score it alone.
    """
    import pandas as pd

    items = list(records.items()) if isinstance(records, dict) else list(records)
    parts = []
    for name, seq in items:
        part = predict(seq, **kwargs)
        part.insert(0, "sequence_id", name)
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["sequence_id"] + COLUMNS).astype(
            {"sequence_id": "object", **COLUMN_TYPES})
    return pd.concat(parts, ignore_index=True)


__all__ = ["predict", "predict_many", "load", "DEFAULT_MODEL", "FINETUNED_MODEL",
           "CONTEXTS", "COLUMNS", "__version__"]
