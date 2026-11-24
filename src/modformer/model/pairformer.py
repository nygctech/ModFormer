"""Building blocks of the pair-representation trunk (inference only).

The trunk keeps two tensors per example: a per-position "single" track ``s`` of shape
``(B, L, d_s)`` and a per-cell "pair" track ``z`` of shape ``(B, L, L, d_p)``.
Each block lets the pair track bias attention on the single track, lets the single
track write back into the pair track, and refines the pair track with triangle
multiplicative updates. The design follows the AlphaFold-style pairformer.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMS normalization with a learned scale, computed in the input dtype."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight.to(x.dtype)


class OuterProductMean(nn.Module):
    """Single track ``(B, L, d_in)`` to a pair tensor ``(B, L, L, d_out)``.

    A low-rank outer product of two projections of the input, followed by a linear
    map. The linear map is applied to the left factor first, so the
    ``(L, L, d_hidden * d_hidden)`` intermediate is never built.
    """

    def __init__(self, d_in: int, d_out: int, d_hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.left = nn.Linear(d_in, d_hidden)
        self.right = nn.Linear(d_in, d_hidden)
        self.out = nn.Linear(d_hidden * d_hidden, d_out)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        s = self.norm(s)
        left = self.left(s)
        right = self.right(s)
        d_h = left.shape[-1]
        weight = self.out.weight.reshape(self.out.out_features, d_h, d_h)
        t = torch.einsum("bif,pfg->bigp", left, weight)
        out = torch.einsum("bigp,bjg->bijp", t, right)
        return out + self.out.bias.to(out.dtype)


class TriangleMultiplication(nn.Module):
    """Triangle multiplicative update of the pair tensor, with a residual.

    ``outgoing`` forms ``u[i, j] = sum_k a[i, k] * b[j, k]`` and ``incoming``
    ``u[i, j] = sum_k a[k, i] * b[k, j]``, where ``a`` and ``b`` are gated projections of
    ``z``. Cells that touch a padded position are zeroed before the sum over ``k``. ``u``
    is normalized, gated and added to ``z``.
    """

    def __init__(self, d_pair: int, mode: str):
        super().__init__()
        if mode not in ("outgoing", "incoming"):
            raise ValueError(f"mode must be 'outgoing' or 'incoming', got {mode!r}")
        self.mode = mode
        self.norm = nn.LayerNorm(d_pair)
        self.norm_out = nn.LayerNorm(d_pair)
        self.left_proj = nn.Linear(d_pair, d_pair)
        self.left_gate = nn.Linear(d_pair, d_pair)
        self.right_proj = nn.Linear(d_pair, d_pair)
        self.right_gate = nn.Linear(d_pair, d_pair)
        self.out_proj = nn.Linear(d_pair, d_pair)
        self.out_gate = nn.Linear(d_pair, d_pair)

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z_norm = self.norm(z)
        left = torch.sigmoid(self.left_gate(z_norm)) * self.left_proj(z_norm)
        right = torch.sigmoid(self.right_gate(z_norm)) * self.right_proj(z_norm)
        cell = (mask.unsqueeze(2) & mask.unsqueeze(1)).unsqueeze(-1).to(left.dtype)
        left = left * cell
        right = right * cell
        if self.mode == "outgoing":
            update = torch.einsum("bikd,bjkd->bijd", left, right)
        else:
            update = torch.einsum("bkid,bkjd->bijd", left, right)
        update = self.norm_out(update)
        gate = torch.sigmoid(self.out_gate(z_norm))
        return z + gate * self.out_proj(update)


class SingleAttentionWithPairBias(nn.Module):
    """Gated multi-head self-attention on the single track, biased by the pair track."""

    def __init__(self, d_single: int, d_pair: int, n_head: int):
        super().__init__()
        if d_single % n_head:
            raise ValueError(f"d_s={d_single} must be divisible by n_heads={n_head}")
        self.n_head = n_head
        self.d_head = d_single // n_head
        self.norm_s = nn.LayerNorm(d_single)
        self.norm_z = nn.LayerNorm(d_pair)
        self.to_qkv = nn.Linear(d_single, 3 * d_single, bias=False)
        self.bias_proj = nn.Linear(d_pair, n_head, bias=False)
        self.gate = nn.Linear(d_single, d_single)
        self.out = nn.Linear(d_single, d_single)

    def forward(self, s: torch.Tensor, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, length, _ = s.shape
        s_norm = self.norm_s(s)
        z_norm = self.norm_z(z)
        q, k, v = self.to_qkv(s_norm).chunk(3, dim=-1)

        def heads(x: torch.Tensor) -> torch.Tensor:
            return x.view(b, length, self.n_head, self.d_head).permute(0, 2, 1, 3)

        q, k, v = heads(q), heads(k), heads(v)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)
        logits = logits + self.bias_proj(z_norm).permute(0, 3, 1, 2)
        pad = (~mask).unsqueeze(1).unsqueeze(2)
        # A large finite negative value rather than -inf keeps a fully masked row finite.
        logits = logits.masked_fill(pad, torch.finfo(logits.dtype).min)
        with torch.amp.autocast("cuda", enabled=False):
            weights = torch.softmax(logits.float(), dim=-1).to(q.dtype)
        out = torch.matmul(weights, v).permute(0, 2, 1, 3).reshape(b, length, -1)
        return s + torch.sigmoid(self.gate(s_norm)) * self.out(out)


class FeedForward(nn.Module):
    """Gated (SwiGLU) transition with a residual connection.

    ``d_ff`` is the width of an equivalent dense MLP; the gated hidden width is two
    thirds of it, rounded up to a multiple of 8.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        hidden = int(2 * d_ff / 3)
        hidden += (-hidden) % 8
        self.norm = nn.LayerNorm(d_model)
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))


class PairRowPooling(nn.Module):
    """Attention pooling of a pair-tensor row into one vector per scored adenosine.

    The adenosine's single-track vector is the query; the keys and values come from
    the cells of its row in the pair tensor. Padded columns are excluded.
    """

    def __init__(self, d_single: int, d_pair: int):
        super().__init__()
        self.norm_single = nn.LayerNorm(d_single)
        self.norm_pair = nn.LayerNorm(d_pair)
        self.query_proj = nn.Linear(d_single, d_pair)
        self.to_k = nn.Linear(d_pair, d_pair)
        self.to_v = nn.Linear(d_pair, d_pair)

    def forward(self, pair_rows: torch.Tensor, single_at_a: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        """``pair_rows`` is ``(B, Na, L, d_p)``, ``single_at_a`` ``(B, Na, d_s)``, ``valid`` ``(B, L)``."""
        query = self.query_proj(self.norm_single(single_at_a)).float()
        keys = self.to_k(self.norm_pair(pair_rows)).float()
        values = self.to_v(self.norm_pair(pair_rows)).float()
        logits = torch.einsum("bnd,bnld->bnl", query, keys) / math.sqrt(query.shape[-1])
        logits = logits.masked_fill(~valid.unsqueeze(1), torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        return torch.einsum("bnl,bnld->bnd", weights, values).to(pair_rows.dtype)


class SingleToPair(nn.Module):
    """Write the single track into the pair track through a low-rank outer product."""

    def __init__(self, d_single: int, d_pair: int, d_hidden: int):
        super().__init__()
        self.opm = OuterProductMean(d_single, d_pair, d_hidden)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.opm(s)


class PairToSingle(nn.Module):
    """Read the pair track into the single track with pair-weighted pooling.

    For each position ``i`` and head ``h`` the pair row gives a softmax over positions
    ``j``; the head's slice of a value projection of the single track is averaged
    with those weights, gated, normalized and projected back to ``d_s``.
    """

    def __init__(self, d_pair: int, d_single: int, n_head: int):
        super().__init__()
        if d_single % n_head:
            raise ValueError(f"d_s={d_single} must be divisible by n_heads={n_head}")
        self.n_head = n_head
        self.d_head = d_single // n_head
        self.norm = nn.LayerNorm(d_pair)
        self.norm_pooled = nn.LayerNorm(d_single)
        self.norm_single = nn.LayerNorm(d_single)
        self.to_w = nn.Linear(d_pair, n_head, bias=False)
        self.to_v = nn.Linear(d_single, d_single, bias=False)
        self.to_g = nn.Linear(d_single, d_single)
        self.out = nn.Linear(d_single, d_single)

    def forward(self, s: torch.Tensor, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        zn = self.norm(z)
        sn = self.norm_single(s)
        b, length, _ = s.shape
        w = self.to_w(zn)
        w = w.masked_fill(~mask.unsqueeze(1).unsqueeze(-1), torch.finfo(w.dtype).min)
        w = torch.softmax(w.float(), dim=2).to(zn.dtype)
        v = self.to_v(sn).view(b, length, self.n_head, self.d_head)
        o = torch.einsum("bijh,bjhd->bihd", w, v).reshape(b, length, -1)
        o = torch.sigmoid(self.to_g(sn)) * o
        return self.out(self.norm_pooled(o))


class PairFormerBlock(nn.Module):
    """One trunk layer.

    Single track: pair-biased attention, then pair-to-single read, then feed-forward.
    Pair track: single-to-pair write, outgoing and incoming triangle updates, then
    feed-forward. The single-to-pair write comes before the triangle updates so that
    the update it makes is propagated within the same layer.
    """

    def __init__(self, d_single: int, d_pair: int, n_head: int, d_ff_single: int,
                 d_ff_pair: int, s2p_dim: int):
        super().__init__()
        self.single_attn = SingleAttentionWithPairBias(d_single, d_pair, n_head)
        self.p2s = PairToSingle(d_pair, d_single, n_head)
        self.single_ff = FeedForward(d_single, d_ff_single)
        self.s2p = SingleToPair(d_single, d_pair, s2p_dim)
        self.tri_mul_out = TriangleMultiplication(d_pair, "outgoing")
        self.tri_mul_in = TriangleMultiplication(d_pair, "incoming")
        self.pair_ff = FeedForward(d_pair, d_ff_pair)

    def forward(self, s: torch.Tensor, z: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.single_attn(s, z, mask)
        s = s + self.p2s(s, z, mask)
        s = self.single_ff(s)
        z = z + self.s2p(s)
        z = z * (mask.unsqueeze(2) & mask.unsqueeze(1)).unsqueeze(-1).to(z.dtype)
        z = self.tri_mul_out(z, mask)
        z = self.tri_mul_in(z, mask)
        z = self.pair_ff(z)
        return s, z


def mask_pair_by_padding(pair: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Zero every pair cell that touches a padded position."""
    cell = valid.unsqueeze(2) & valid.unsqueeze(1)
    return pair * cell.unsqueeze(-1).to(pair.dtype)


__all__ = [
    "RMSNorm", "OuterProductMean", "TriangleMultiplication", "SingleAttentionWithPairBias",
    "FeedForward", "PairRowPooling", "SingleToPair", "PairToSingle", "PairFormerBlock",
    "mask_pair_by_padding",
]
