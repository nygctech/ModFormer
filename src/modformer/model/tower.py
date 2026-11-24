"""One-dimensional tower over the folded window.

The tower turns the per-position structure tags of the full 1022-nt fold window into
a per-position context vector. It runs before the window is gathered down to the
128-position example, so its convolutions see the real, contiguous backbone. It reads
structure only, never the nucleotide identity.

Two stages: a bank of parallel convolutions of different widths (templates for short
stretches of structure), then a residual stack of dilated convolutions that relates
those templates to each other over a long range.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pads(kernel: int, dilation: int) -> tuple:
    """Left and right padding that keeps the length for any kernel width."""
    span = dilation * (kernel - 1)
    left = span // 2
    return left, span - left


def masked_conv1d(x: torch.Tensor, conv: nn.Conv1d, seg_id: torch.Tensor) -> torch.Tensor:
    """Convolve without mixing positions that lie in different backbone segments.

    ``x`` is ``(B, C, W)`` and ``seg_id`` is ``(B, W)``: positions of one unbroken
    stretch of real sequence share an id, and padding is ``-1``. ``conv`` must have
    no padding of its own. A tap that falls in another segment than the output
    position is zeroed, so a convolution never bridges a break in the sequence.
    """
    k, d = conv.kernel_size[0], conv.dilation[0]
    lpad, rpad = _pads(k, d)
    if bool((seg_id == seg_id[:, :1]).all()):
        return conv(F.pad(x, (lpad, rpad)))
    span = d * (k - 1) + 1
    patches = F.pad(x, (lpad, rpad)).unfold(2, span, 1)[..., ::d]
    seg_pad = F.pad(seg_id, (lpad, rpad), value=-1)
    seg_taps = seg_pad.unfold(1, span, 1)[..., ::d]
    same = (seg_taps == seg_id.unsqueeze(-1)).to(x.dtype)
    out = torch.einsum("bcwk,ock->bow", patches * same.unsqueeze(1), conv.weight)
    return out + conv.bias.view(1, -1, 1)


class MotifBank(nn.Module):
    """Parallel convolutions of several widths; each gets an equal share of the channels."""

    def __init__(self, in_dim: int, out_dim: int, kernels: Sequence[int]):
        super().__init__()
        per = out_dim // len(kernels)
        self.convs = nn.ModuleList([nn.Conv1d(in_dim, per, kernel_size=k, padding=0) for k in kernels])
        self.norm = nn.LayerNorm(per * len(kernels))
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, seg_id: torch.Tensor) -> torch.Tensor:
        xc = x.transpose(1, 2)
        out = torch.cat([masked_conv1d(xc, c, seg_id) for c in self.convs], dim=1).transpose(1, 2)
        return self.act(self.norm(out))


class CompositionStack(nn.Module):
    """Residual stack of dilated convolutions, one dilation per layer."""

    def __init__(self, dim: int, dilations: Sequence[int], kernel: int = 3):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(dim, dim, kernel_size=kernel, dilation=d, padding=0) for d in dilations
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in dilations])
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, seg_id: torch.Tensor) -> torch.Tensor:
        for conv, norm in zip(self.convs, self.norms):
            h = masked_conv1d(norm(x).transpose(1, 2), conv, seg_id).transpose(1, 2)
            x = x + self.act(h)
        return x


class PreCropTower(nn.Module):
    """Per-position context from structure tags: ``(B, W, n_tags)`` to ``(B, W, out_dim)``."""

    def __init__(self, n_tag_scalars: int, annot_classes: int, motif_dim: int,
                 motif_kernels: Sequence[int], dilations: Sequence[int], out_dim: int,
                 annot_embed_dim: int = 8):
        super().__init__()
        self.annot_embed = nn.Embedding(annot_classes, annot_embed_dim)
        self.str_motifs = MotifBank(n_tag_scalars + annot_embed_dim, motif_dim, motif_kernels)
        self.proj_in = nn.Linear(motif_dim, out_dim)
        self.composition = CompositionStack(out_dim, dilations)

    def forward(self, tag_scalars: torch.Tensor, struct_annot: torch.Tensor,
                valid: torch.Tensor, seg_id: torch.Tensor) -> torch.Tensor:
        """Positions where ``valid`` is false are zeroed on input and on output."""
        mask = valid.unsqueeze(-1).to(tag_scalars.dtype)
        s = torch.cat([tag_scalars, self.annot_embed(struct_annot)], dim=-1) * mask
        ctx = self.composition(self.proj_in(self.str_motifs(s, seg_id)), seg_id)
        return ctx * mask


__all__ = ["masked_conv1d", "MotifBank", "CompositionStack", "PreCropTower"]
