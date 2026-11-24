"""The ModFormer network (inference only).

Input per example is 128 positions: a 64-nt anchor whose adenosines are scored,
followed by a 64-nt partner window chosen from the fold. The model reads

* a per-position language-model embedding (``backbone_dim`` wide) for the example,
* the base-pair probability matrix of the example,
* structure tags of the whole 1022-nt fold window (consumed by the 1D tower and then
  gathered to the example) and eight per-position tags gathered to the example.

It returns, for every scored adenosine, an editing-site logit (how strongly the site
resembles observed editing sites) and a distribution over the editing-rate bins.
Training-only code is not part of this module; the parameter and
buffer names match the released weights exactly.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .pair_init import PairInit
from .pairformer import PairFormerBlock, PairRowPooling, mask_pair_by_padding
from .rate_bins import rate_bin_centers
from .tags import N_TAG_SCALARS, PAIR_TAG_KEYS, TAG_PREFIX
from .tower import PreCropTower

CONTEXT_FACTOR_NAMES = ("modality", "organism", "tissue")

#: Architecture options with a single supported value. Anything else in a config is
#: rejected, because the corresponding code path is not part of this package.
_FIXED_OPTIONS = {
    "block_type": "pairformer",
    "trunk_p2s": "pair_weighted",
    "tri_mul": "both",
    "trunk_pair_bias": True,
    "trunk_qk_norm": False,
    "trunk_s2p_period": 1,
    "head_mode": "cls+reg",
    "head_row_pool": "attn",
    "head_row_pool_bp_bias": False,
    "head_pair_diag": True,
    "catalogue_readout": "separate",
    "rate_head": "binned",
    "conv_kernel": "0",
    "struct_attn_bias": False,
    "nt_inject": False,
    "seq_embed_only": False,
    "tower_over_tile": False,
    "tower_str_leg": True,
    "tower_struct_annot": True,
    "pair_ch_bp": True,
    "pair_ch_helix": True,
    "pair_ch_axial": True,
    "pair_ch_relpos": True,
    "pair_lang_branch": True,
    "pair_tower_branch": True,
    "tissue_conditioning": "off",
    "context_cls_bias": False,
    "shuffle_bpp": False,
    "gather": True,
    "context_into_classifier": True,
}


def _int_list(spec) -> List[int]:
    """Parse ``"3,6,9"`` (or an already parsed sequence) into a list of ints."""
    if isinstance(spec, str):
        return [int(t) for t in spec.split(",") if t.strip()]
    return [int(t) for t in spec]


def _check_supported(cfg: dict) -> None:
    """Raise if ``cfg`` asks for an architecture variant this package does not implement."""
    for key, expected in _FIXED_OPTIONS.items():
        if key not in cfg:
            continue
        value = cfg[key]
        if key == "conv_kernel":
            ok = str(value) == expected
        else:
            ok = value == expected
        if not ok:
            raise ValueError(
                f"unsupported architecture option {key}={value!r}; this package implements {expected!r}"
            )


def _gather_positions(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather along the position axis: ``(B, W[, C])`` by ``(B, L)`` gives ``(B, L[, C])``."""
    if x.dim() == 2:
        return x.gather(1, idx)
    return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


class ModFormer(nn.Module):
    """Pair-representation model of adenosine editing.

    ``cfg`` is the ``"arch"`` dictionary stored next to the weights. If it enables
    context conditioning (the fine-tuned model), :meth:`forward` also needs
    ``context_id``, an integer vector indexing ``context_names``.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        _check_supported(cfg)
        d_s, d_p = int(cfg["d_s"]), int(cfg["d_p"])
        n_heads = int(cfg["n_heads"])
        backbone_dim = int(cfg["backbone_dim"])
        tower_dim = int(cfg["tower_dim"])

        self.single_proj = nn.Sequential(nn.LayerNorm(backbone_dim), nn.Linear(backbone_dim, d_s))
        self.tower = PreCropTower(
            n_tag_scalars=N_TAG_SCALARS,
            annot_classes=int(cfg["tower_annot_classes"]),
            motif_dim=int(cfg["tower_motif_dim"]),
            motif_kernels=_int_list(cfg["tower_motif_kernels"]),
            dilations=_int_list(cfg["tower_dilations"]),
            out_dim=tower_dim,
        )
        self.pair_init = PairInit(
            d_single=d_s,
            d_tower=tower_dim,
            d_pair=d_p,
            c_outer_lang=int(cfg["c_outer_lang"]),
            c_outer_tower=int(cfg["c_outer_tower"]),
            d_out_lang=int(cfg["d_out_lang"]),
            d_out_tower=int(cfg["d_out_tower"]),
            cell_hidden=int(cfg["pair_cell_hidden"]),
            within_clamp=int(cfg["relpos_within_clamp"]),
            cross_buckets=int(cfg["relpos_cross_buckets"]),
        )
        s2p_dim = int(cfg["trunk_s2p_dim"])
        self.blocks = nn.ModuleList([
            PairFormerBlock(
                d_single=d_s,
                d_pair=d_p,
                n_head=n_heads,
                d_ff_single=d_s * int(cfg["d_ff_factor_single"]),
                d_ff_pair=d_p * int(cfg["d_ff_factor_pair"]),
                s2p_dim=s2p_dim,
            )
            for _ in range(int(cfg["n_layers"]))
        ])

        # Readout: attention pooling of each scored row of the pair tensor, plus the
        # pair diagonal, concatenated with the single-track vector.
        self.pair_pooling = PairRowPooling(d_s, d_p)
        self.head_norm_single = nn.LayerNorm(d_s)
        self.head_norm_pair = nn.LayerNorm(d_p)
        self.head_norm_diag = nn.LayerNorm(d_p)
        head_dim = d_s + 2 * d_p
        n_bins = int(cfg["n_rate_bins"])
        centers = rate_bin_centers(n_bins, float(cfg["edit_rate_thr"]), float(cfg["rate_bin_min"]))
        self.register_buffer("rate_centers", torch.tensor(centers, dtype=torch.float32),
                             persistent=False)
        # The two heads share an architecture. Index 2 of each is a parameter-free
        # placeholder, so the linear layers keep the names ``*.0`` and ``*.3`` of the
        # released weights.
        self.dist_head = nn.Sequential(
            nn.Linear(head_dim, head_dim // 2), nn.SiLU(), nn.Identity(),
            nn.Linear(head_dim // 2, n_bins),
        )
        self.cat_head = nn.Sequential(
            nn.Linear(head_dim, head_dim // 2), nn.SiLU(), nn.Identity(),
            nn.Linear(head_dim // 2, 1),
        )

        self.context_names: Optional[List[str]] = None
        if cfg.get("context_conditioning", False):
            self._build_context(cfg, head_dim)

    def _build_context(self, cfg: dict, head_dim: int) -> None:
        """Context tables: one embedding per factor, summed into a context vector."""
        names = list(cfg["context_vocab"])
        factors = {k: tuple(v) for k, v in cfg["context_factors"].items()}
        if names[0] != "catalogue":
            raise ValueError("context_vocab[0] must be 'catalogue'")
        vocabs = [["none"] + sorted({f[i] for f in factors.values()} - {"none"}) for i in range(3)]
        ids = [[vocabs[i].index(factors[n][i]) for i in range(3)] for n in names]
        self.context_names = names
        self.register_buffer("context_factor_ids", torch.tensor(ids, dtype=torch.long),
                             persistent=False)
        self.context_embedding = nn.ModuleDict({
            f: nn.Embedding(len(vocabs[i]), head_dim, padding_idx=0,
                            _weight=torch.zeros(len(vocabs[i]), head_dim))
            for i, f in enumerate(CONTEXT_FACTOR_NAMES)
        })

    def context_vector(self, context_id: torch.Tensor) -> torch.Tensor:
        """Sum of the three factor rows of each context id: ``(B,)`` to ``(B, head_dim)``."""
        fid = self.context_factor_ids[context_id.long()]
        vec = None
        for i, f in enumerate(CONTEXT_FACTOR_NAMES):
            row = self.context_embedding[f](fid[:, i])
            vec = row if vec is None else vec + row
        return vec

    def _check_context_id(self, context_id: Optional[torch.Tensor], batch_size: int) -> None:
        n_ctx = len(self.context_names)
        bad = (
            not isinstance(context_id, torch.Tensor)
            or context_id.shape != (batch_size,)
            or context_id.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
            or bool(torch.any((context_id < 0) | (context_id >= n_ctx)))
        )
        if bad:
            raise ValueError(
                f"this model is context-conditioned: pass context_id as an integer tensor "
                f"of shape ({batch_size},) with values in 0..{n_ctx - 1}"
            )

    def expected_rate(self, logq: torch.Tensor) -> torch.Tensor:
        """Expected editing rate: the mean of the bin centers under ``exp(logq)``."""
        return (logq.exp() * self.rate_centers).sum(dim=-1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score the adenosines of a batch of examples.

        Required keys of ``batch`` (``B`` examples, ``L`` = 128 example positions,
        ``W`` = 1022 window positions, ``Na`` scored adenosines padded to a common count):

        ``embeddings`` float ``(B, L, backbone_dim)``; ``bp_probs`` float ``(B, L, L)``;
        ``a_positions`` long ``(B, Na)`` (example index of each adenosine);
        ``orig_idx`` long ``(B, L)`` (window coordinate of each example position);
        ``tile_valid`` bool ``(B, L)``;
        ``w_tag_scalars`` float ``(B, W, 13)``; ``w_struct_annot`` long ``(B, W)``;
        ``w_valid`` bool ``(B, W)``; ``w_seg_id`` long ``(B, W)``; the eight ``tag_*``
        tensors ``(B, L)`` named in ``modformer.model.tags.PAIR_TAG_KEYS``; and
        ``context_id`` long ``(B,)`` for a context-conditioned model.

        Returns ``(logits, logq)``: ``logits`` is ``(B, Na, 2)`` with the editing-site logit
        and the logit of the expected rate, ``logq`` is ``(B, Na, n_bins)``, the
        log-probabilities over the rate bins.
        """
        orig_idx = batch["orig_idx"]
        tile_valid = batch["tile_valid"].bool()
        a_positions = batch["a_positions"]
        tags = {k: batch[TAG_PREFIX + k] for k in PAIR_TAG_KEYS}

        single = self.single_proj(batch["embeddings"])
        ctx = self.tower(batch["w_tag_scalars"], batch["w_struct_annot"],
                         batch["w_valid"], batch["w_seg_id"])
        tower_ctx = _gather_positions(ctx, orig_idx)
        pair = self.pair_init(single, tower_ctx, orig_idx, tags, batch["bp_probs"], tile_valid)
        pair = mask_pair_by_padding(pair, tile_valid)
        for block in self.blocks:
            single, pair = block(single, pair, tile_valid)

        batch_size, length, d_s = single.shape
        d_p = pair.shape[-1]
        a_repr = torch.gather(single, 1, a_positions.unsqueeze(-1).expand(-1, -1, d_s))
        rows = torch.gather(
            pair, 1, a_positions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, length, d_p))
        pooled = self.pair_pooling(rows, a_repr, tile_valid)
        diag = torch.gather(rows, 2, a_positions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, d_p))
        z_diag = diag.squeeze(2)
        head_in = torch.cat([
            self.head_norm_single(a_repr),
            self.head_norm_pair(pooled),
            self.head_norm_diag(z_diag),
        ], dim=-1)

        rate_input = head_in
        cat_input = head_in
        if self.context_names is not None:
            # The context vector is added to the input of both heads.
            context_id = batch.get("context_id")
            self._check_context_id(context_id, batch_size)
            ctx_vec = self.context_vector(context_id).unsqueeze(1).to(head_in.dtype)
            rate_input = head_in + ctx_vec
            cat_input = head_in + ctx_vec
        logq = torch.log_softmax(self.dist_head(rate_input).float(), dim=-1)
        cat_logit = self.cat_head(cat_input)
        rate = self.expected_rate(logq)
        rate_logit = torch.logit(rate.clamp(1e-6, 1.0 - 1e-6)).unsqueeze(-1)
        logits = torch.cat([cat_logit, rate_logit], dim=-1).to(head_in.dtype)
        return logits, logq


def count_float_parameters(model: nn.Module) -> int:
    """Number of elements in all floating-point parameters of ``model``."""
    return int(sum(p.numel() for p in model.parameters() if p.is_floating_point()))


__all__ = ["ModFormer", "count_float_parameters"]
