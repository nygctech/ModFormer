"""RiNALMo language-model embeddings for a whole transcript.

The transcript is embedded once, as it is, in overlapping windows of at most 1022 nt
(stride 128). Inside each window the central positions are kept, the first window keeps
its 5' end and the last its 3' end, so every position comes from exactly one window and
sits at least 447 nt from a window edge wherever the transcript is long enough. The
language model is the public ``multimolecule/rinalmo-giga`` at a pinned revision; its
weights are fetched into the default Hugging Face cache on first use.
"""

from __future__ import annotations

import time
from typing import Callable, Optional, Sequence

import numpy as np

from ..tiling import normalize
# The language model and the snapshot the released weights were trained against.
from ..weights import LM_ID as MODEL_ID
from ..weights import LM_REVISION as REVISION
from ..weights import WeightsError, offline
from .gather import tokenize

WINDOW = 1022
STRIDE = 128
DIM = 1280
#: How long to keep retrying when a shared GPU is out of memory, and how long between tries.
OOM_WAIT_SECONDS = 600
OOM_RETRY_SECONDS = 20
#: The least GPU memory (bytes) on which one window fits: the language model alone peaks at
#: about 4 GiB, and the CUDA context takes more. A smaller card cannot succeed by waiting.
GPU_MIN_BYTES = 5 * 2**30

_MODELS: dict = {}
_NOTED = False


def resolve_device(device=None):
    """The ``torch.device`` to run on.

    An explicit ``device`` is used as given. With none, a GPU is chosen when one exists and
    has at least ``GPU_MIN_BYTES`` of memory; a smaller GPU falls back to the CPU, with one
    note on stderr per process.
    """
    import sys

    import torch

    global _NOTED
    if device is not None:
        return torch.device(device)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    if total >= GPU_MIN_BYTES:
        return torch.device("cuda")
    if not _NOTED:
        _NOTED = True
        print(f"modformer: this GPU has {total / 2**30:.1f} GiB of memory and the language "
              f"model needs about {GPU_MIN_BYTES // 2**30} GiB, so the CPU is used. "
              "Pass device='cuda' (on the command line: --device cuda) to use the GPU anyway.",
              file=sys.stderr)
    return torch.device("cpu")


def load_language_model(device=None):
    """Load (once per process) the language model on ``device``; returns the model."""
    return _load(resolve_device(device))


def plan_windows(n: int, window: int = WINDOW,
                 stride: int = STRIDE) -> list[tuple[int, int, int, int]]:
    """Sliding windows over ``n`` nucleotides.

    Each entry is ``(start, end, keep_from, keep_to)``: the window covers ``[start, end)``
    and contributes its positions ``keep_from:keep_to`` (relative to ``start``). A sequence
    that fits in one window uses a single window and keeps all of it.
    """
    if n <= window:
        return [(0, n, 0, n)]
    margin = (window - stride) // 2
    out = []
    start = 0
    while start < n:
        end = min(start + window, n)
        if start == 0:
            keep = (0, window - margin)
        elif end == n:
            keep = (margin, end - start)
        else:
            keep = (margin, margin + stride)
        out.append((start, end, keep[0], keep[1]))
        if end == n:
            break
        start += stride
    return out


def _load(device):
    import torch
    from multimolecule import RiNALMoModel

    device = torch.device(device)
    asked = str(device)
    if device.type == "cuda" and device.index is None and torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())  # "cuda" and "cuda:0" share a copy
    key = str(device)
    if key not in _MODELS:
        from transformers.utils import logging as hf_logging

        level = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()  # the loader's report is noise at a prompt
        local = {"local_files_only": True} if offline() else {}
        try:
            model = RiNALMoModel.from_pretrained(MODEL_ID, revision=REVISION,
                                                 attn_implementation="sdpa", **local)
        except OSError as exc:
            if not local:
                raise
            raise WeightsError("the RiNALMo language model is not in the Hugging Face cache and "
                               "MODFORMER_OFFLINE is set; run `modformer fetch` once with "
                               "MODFORMER_OFFLINE unset") from exc
        finally:
            hf_logging.set_verbosity(level)
        _MODELS[key] = model.to(torch.device(device)).eval()
    _MODELS[asked] = _MODELS[key]  # a lookup by the caller's spelling ("cuda") finds the same copy
    return _MODELS[key]


def _forward(model, ids, mask, device, on_layer: Optional[Callable[[int, int], None]] = None
             ) -> np.ndarray:
    """One forward pass over a batch of windows.

    ``on_layer(k, layers)``, when given, is called as encoder layer ``k`` (1-based, of
    ``layers``) finishes. The hooks only observe: the batch, and so every number, is the same
    with or without them. On a GPU the call waits for that layer's kernels first, because they
    are queued ahead of the CPU and the count would otherwise run ahead of the work.
    """
    import torch

    handles = []
    if on_layer is not None:
        layers = list(model.encoder.layer)

        def watch(k: int):
            def hook(module, args, output):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                on_layer(k, len(layers))
            return hook

        handles = [layer.register_forward_hook(watch(k)) for k, layer in enumerate(layers, 1)]
    try:
        with torch.no_grad(), torch.autocast("cuda", enabled=(device.type == "cuda")):
            hidden = model(ids.to(device), attention_mask=mask.to(device)).last_hidden_state
    finally:
        for handle in handles:
            handle.remove()
    return hidden.float().cpu().numpy()


def windows_for_spans(n: int, spans: Sequence[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    """The windows of the full-length plan whose kept positions overlap ``spans``.

    ``spans`` are half-open ``(start, end)`` position ranges. The plan is always made for
    the whole sequence, so a position is embedded by the same window, with the same
    context, whether or not the rest of the sequence is embedded too.
    """
    keep = []
    for start, end, keep_from, keep_to in plan_windows(n):
        lo, hi = start + keep_from, start + keep_to
        if any(lo < b and a < hi for a, b in spans):
            keep.append((start, end, keep_from, keep_to))
    return keep


def embed_sequence(seq: str, device=None, batch_size: int = 8,
                   progress: Optional[Callable[[str, int, int], None]] = None,
                   needed: Optional[Sequence[tuple[int, int]]] = None) -> np.ndarray:
    """Embed a normalized sequence: float32 ``[len(seq), 1280]``.

    ``device`` follows :func:`resolve_device`.

    On a GPU the model runs in half precision (autocast). The result is rounded through
    float16, the precision of the embeddings the model was trained on.

    ``needed`` limits the work to the windows that supply the given half-open position
    ranges; every other row of the result is zero and must not be read. A computed row comes
    from the same window of the same full-length plan as it would in a complete run, so it has
    the same sequence context either way. Its last digits can still differ, because fewer
    windows go through the model at once and a matrix product in half precision depends on the
    shape of its batch.
    """
    import torch

    if seq != normalize(seq):
        raise ValueError("embed_sequence expects a sequence already passed through normalize()")
    device = resolve_device(device)
    model = _load(device)

    n = len(seq)
    windows = plan_windows(n) if needed is None else windows_for_spans(n, needed)
    ids, mask = tokenize([seq[s:e] for s, e, _, _ in windows])
    out = np.zeros((n, DIM), dtype=np.float32)

    done, bs, waited = 0, max(1, int(batch_size)), 0.0
    shown = 0.0  # the furthest the count has gone; a retry after an OOM resets it to `done`

    def report(value: float) -> None:
        nonlocal shown
        shown = max(shown, value)
        progress("embed", shown, len(windows))

    if progress:
        report(0)
    while done < len(windows):
        stop = min(done + bs, len(windows))
        def inside(k: int, layers: int, first: int = done, size: int = stop - done) -> None:
            # Windows go through the model in one batch, so the count between two batches is
            # the share of the layers done, kept below the batch's end until it is finished.
            report(min(first + size * k / layers, stop - 1e-6))

        try:
            hidden = _forward(model, ids[done:stop], mask[done:stop], device,
                              on_layer=inside if progress else None)
        except torch.OutOfMemoryError:
            # A shared GPU can be full because of someone else's work, so make the window
            # batch smaller and then wait a while before giving up.
            torch.cuda.empty_cache()
            advanced, shown = shown > done, float(done)  # the failed batch finished no window
            if progress and advanced:
                report(done)
            if bs > 1:
                bs = max(1, bs // 2)
                continue
            dev = torch.device(device)
            total = torch.cuda.mem_get_info(dev)[1] if dev.type == "cuda" else None
            if total is not None and total < GPU_MIN_BYTES:
                raise RuntimeError(
                    f"this GPU has {total / 2**30:.1f} GiB of memory, and embedding one window "
                    f"needs about {GPU_MIN_BYTES // 2**30} GiB. Run on the CPU with "
                    "device='cpu' (on the command line: --device cpu)") from None
            if waited >= float(OOM_WAIT_SECONDS):
                raise RuntimeError(
                    "the GPU ran out of memory on a single window. Wait for the other work "
                    "on this GPU to finish, or run on the CPU with device='cpu' "
                    "(on the command line: --device cpu)") from None
            if progress:
                progress("waiting for GPU memory", int(waited), int(OOM_WAIT_SECONDS))
            time.sleep(OOM_RETRY_SECONDS)
            waited += OOM_RETRY_SECONDS
            continue
        for row, (start, _, keep_from, keep_to) in zip(hidden, windows[done:stop]):
            out[start + keep_from:start + keep_to] = row[1 + keep_from:1 + keep_to]
        done = stop
        if progress:
            report(done)
    return out.astype(np.float16).astype(np.float32)
