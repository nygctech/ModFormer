"""Model definition, weight loading and batch inference for ModFormer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from .modformer import ModFormer, count_float_parameters
from .tags import N_TAG_SCALARS, PAIR_TAG_KEYS, TAG_PREFIX, TAG_SCALAR_KEYS

#: Batch keys the network reads (besides ``context_id`` for a conditioned model).
BATCH_KEYS = (
    "embeddings", "bp_probs", "a_positions", "a_valid", "orig_idx", "tile_valid",
    "w_tag_scalars", "w_struct_annot", "w_valid", "w_seg_id",
) + tuple(TAG_PREFIX + k for k in PAIR_TAG_KEYS)


def _resolve_bundle(name_or_dir: Union[str, Path]) -> Path:
    """A directory holding ``model.safetensors`` and ``config.json``, or a release name."""
    path = Path(name_or_dir).expanduser()
    if path.is_dir():
        return path
    from ..weights import ensure  # imported here so a local directory needs no download code

    return Path(ensure(str(name_or_dir)))


def load_model(name_or_dir: str = "ModFormer-24L512-12ep",
               device: Optional[Union[str, torch.device]] = None) -> Tuple[ModFormer, dict]:
    """Load a released ModFormer model in evaluation mode.

    ``name_or_dir`` is either a directory containing ``model.safetensors`` and
    ``config.json``, or the name of a release, which is fetched on first use. Returns
    ``(model, cfg)`` where ``cfg`` is the architecture dictionary from ``config.json``.
    Loading is strict: every stored tensor must match a parameter of the network.
    The whole ``config.json`` is kept on the model as ``model.release`` (its ``features``
    block records the input recipe the weights were trained with).

    With no ``device``, the network goes on the GPU when there is one with enough memory for
    the language model and on the CPU otherwise.
    """
    from safetensors.torch import load_file

    bundle = _resolve_bundle(name_or_dir)
    with open(bundle / "config.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    cfg = meta["arch"]
    if device is None:
        from ..features.embed import resolve_device

        device = resolve_device()
    model = ModFormer(cfg)
    state = load_file(str(bundle / "model.safetensors"), device="cpu")
    model.load_state_dict(state, strict=True)
    expected = meta.get("n_float_params")
    if expected is not None and count_float_parameters(model) != int(expected):
        raise ValueError(
            f"parameter count {count_float_parameters(model)} does not match the "
            f"{int(expected)} recorded in config.json"
        )
    model.release = meta
    model.to(device)
    model.eval()
    return model, cfg


@torch.no_grad()
def predict_batch(model: ModFormer, batch: Dict[str, torch.Tensor],
                  device: Optional[Union[str, torch.device]] = None) -> Dict[str, np.ndarray]:
    """Score a collated batch without gradients.

    The network always runs in float32, so one batch scores the same, to rounding, on a
    CPU and on a GPU. The embeddings in the batch can differ in their last digits: on a
    GPU the language model runs in half precision (see ``modformer.features.embed``). A
    context-conditioned model also needs ``batch["context_id"]``. Returns numpy arrays of
    shape ``(B, Na)``: ``catalogue_probability`` (sigmoid of the editing-site logit;
    ``edit_score`` in the output table), ``rate`` (expected editing rate under the 17-bin
    distribution) and ``valid`` (false for padded adenosine slots).
    """
    device = torch.device(device) if device is not None else next(model.parameters()).device
    needed = list(BATCH_KEYS)
    if model.context_names is not None:
        needed.append("context_id")
    missing = [k for k in needed if k not in batch]
    if missing:
        raise KeyError(f"batch is missing {missing}")
    inputs = {k: batch[k].to(device) for k in needed}
    model.eval()
    with torch.amp.autocast(device.type, enabled=False):
        logits, logq = model(inputs)
    return {
        "catalogue_probability": torch.sigmoid(logits[..., 0].float()).cpu().numpy(),
        "rate": model.expected_rate(logq.float()).cpu().numpy(),
        "valid": inputs["a_valid"].bool().cpu().numpy(),
    }


__all__ = [
    "ModFormer", "load_model", "predict_batch", "count_float_parameters", "BATCH_KEYS",
    "N_TAG_SCALARS", "PAIR_TAG_KEYS", "TAG_PREFIX", "TAG_SCALAR_KEYS",
]
