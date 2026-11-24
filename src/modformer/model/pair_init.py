"""Initial pair tensor of the trunk.

Three sources are concatenated per cell and passed through a small MLP:

* an outer product of the language-model track,
* an outer product of the structure tower's context,
* pairwise channels that cannot be written as a function of one position each: the raw
  base-pair probability, same-helix and same-loop identity, the gap between helices,
  the register along a helix, and the relative position of the two cells.

Relative positions use the coordinates in the fold window (``orig_idx``), never the
index within the gathered example, because its two runs can be far apart in the window.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn

from .pairformer import OuterProductMean, RMSNorm

AXIAL_BUCKETS = (0, 1, 2, 3, 5, 8, 12, 17, 22, 26, 30, 34, 38, 44, 56, 80)
HELICAL_PERIOD = 11.0
CROSS_LOG_SPAN = 10.0
ADJACENT_GAP_EDGES = (1, 3, 6, 12, 24)
PAIRWISE_DIM = 48


def _bucket_signed(x: torch.Tensor, edges=AXIAL_BUCKETS) -> torch.Tensor:
    """Signed magnitude bucketing; zero keeps a class of its own."""
    mag = x.abs()
    b = torch.zeros_like(mag)
    for e in edges[1:]:
        b = b + (mag >= e).to(b.dtype)
    return (torch.sign(x) * b).long()


class PairwiseChannels(nn.Module):
    """Features computed at a cell from the tags of its two positions."""

    def __init__(self, d_out: int, within_clamp: int, cross_buckets: int):
        super().__init__()
        self.within_clamp = int(within_clamp)
        self.cross_buckets = int(cross_buckets)
        e = max(8, d_out // 6)
        n_within = 2 * self.within_clamp + 1
        self.relpos_embed = nn.Embedding(n_within + 2 * self.cross_buckets + 1, e)
        self.axial_embed = nn.Embedding(2 * (len(AXIAL_BUCKETS) - 1) + 2, e)
        self.strand_embed = nn.Embedding(3, e)
        self.helix_embed = nn.Embedding(3, e)
        self.loop_embed = nn.Embedding(3, e)
        self.register_buffer("adj_edges", torch.tensor(ADJACENT_GAP_EDGES, dtype=torch.long))
        self.n_adj = len(ADJACENT_GAP_EDGES) + 1
        self.adj_embed = nn.Embedding(self.n_adj + 1, e)
        # 2 scalars (base-pair probability, confidence) + 2 (helical sin, cos)
        # + relative position (e) + axial and strand (2e) + helix, loop, adjacency (3e).
        self.proj = nn.Linear(4 + e + 2 * e + 3 * e, d_out)

    def forward(self, orig_idx: torch.Tensor, tags: Dict[str, torch.Tensor],
                bp_tile: torch.Tensor) -> torch.Tensor:
        feats = [bp_tile.unsqueeze(-1)]

        mx = tags["max_bp"]
        feats.append(torch.minimum(mx.unsqueeze(2), mx.unsqueeze(1)).unsqueeze(-1))

        delta = orig_idx.unsqueeze(2) - orig_idx.unsqueeze(1)
        near = delta.abs() <= self.within_clamp
        within = delta.clamp(-self.within_clamp, self.within_clamp) + self.within_clamp
        mag = delta.abs().clamp(min=1).to(torch.float32)
        cb = (torch.log2(mag) * (self.cross_buckets / CROSS_LOG_SPAN)).long()
        cb = cb.clamp(0, self.cross_buckets - 1)
        cross = (2 * self.within_clamp + 1) + torch.sign(delta).long() * cb + self.cross_buckets
        feats.append(self.relpos_embed(torch.where(near, within, cross)))

        hid, lid = tags["helix_id"], tags["loop_id"]
        helix_unknown = (hid.unsqueeze(2) < 0) | (hid.unsqueeze(1) < 0)
        same_helix = hid.unsqueeze(2) == hid.unsqueeze(1)
        feats.append(self.helix_embed(torch.where(helix_unknown, 2, same_helix.long())))
        loop_unknown = (lid.unsqueeze(2) < 0) | (lid.unsqueeze(1) < 0)
        same_loop = lid.unsqueeze(2) == lid.unsqueeze(1)
        feats.append(self.loop_embed(torch.where(loop_unknown, 2, same_loop.long())))

        rs, re = tags["helix_run_start"], tags["helix_run_end"]
        gap = torch.maximum(
            rs.unsqueeze(2) - re.unsqueeze(1),
            rs.unsqueeze(1) - re.unsqueeze(2),
        ).clamp(min=0)
        adj = torch.bucketize(gap, self.adj_edges.to(gap.device))
        known = (~helix_unknown) & (~same_helix)
        feats.append(self.adj_embed(torch.where(known, adj, self.n_adj)))

        reg, defined, is5 = tags["register"], tags["register_defined"], tags["is_5p"]
        both = (defined.unsqueeze(2) & defined.unsqueeze(1)
                & (hid.unsqueeze(2) == hid.unsqueeze(1)) & (hid.unsqueeze(2) >= 0))
        same_strand = is5.unsqueeze(2) == is5.unsqueeze(1)
        oi, oj = orig_idx.unsqueeze(2), orig_idx.unsqueeze(1)
        r_i, r_j = reg.unsqueeze(2), reg.unsqueeze(1)
        r_five = torch.where(is5.unsqueeze(2), r_i, r_j)
        axial = torch.where(same_strand, oi - oj, oi + oj - r_five)
        bucket = _bucket_signed(axial) + (len(AXIAL_BUCKETS) - 1)
        undefined = 2 * (len(AXIAL_BUCKETS) - 1) + 1
        feats.append(self.axial_embed(torch.where(both, bucket, undefined)))
        strand = torch.where(both, (~same_strand).long(), 2)
        feats.append(self.strand_embed(strand))
        ang = (2 * math.pi / HELICAL_PERIOD) * axial.to(torch.float32)
        m = both.unsqueeze(-1).to(ang.dtype)
        feats.append(torch.stack([ang.sin(), ang.cos()], dim=-1) * m)

        return self.proj(torch.cat(feats, dim=-1))


class PairInit(nn.Module):
    """Build the initial pair tensor from the single track, tower context and channels."""

    def __init__(self, d_single: int, d_tower: int, d_pair: int, c_outer_lang: int,
                 c_outer_tower: int, d_out_lang: int, d_out_tower: int, cell_hidden: int,
                 within_clamp: int, cross_buckets: int):
        super().__init__()
        self.opm_lang = OuterProductMean(d_single, d_out_lang, c_outer_lang)
        self.opm_tower = OuterProductMean(d_tower, d_out_tower, c_outer_tower)
        self.channels = PairwiseChannels(PAIRWISE_DIM, within_clamp, cross_buckets)
        width = d_out_lang + d_out_tower + PAIRWISE_DIM
        self.cell_mlp = nn.Sequential(
            nn.Linear(width, cell_hidden),
            nn.SiLU(),
            nn.Linear(cell_hidden, d_pair),
        )
        self.out_norm = RMSNorm(d_pair)

    def forward(self, single: torch.Tensor, tower_ctx: torch.Tensor, orig_idx: torch.Tensor,
                tags: Dict[str, torch.Tensor], bp_tile: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        parts = [
            self.opm_lang(single),
            self.opm_tower(tower_ctx),
            self.channels(orig_idx, tags, bp_tile),
        ]
        z = self.out_norm(self.cell_mlp(torch.cat(parts, dim=-1)))
        cell = (valid.unsqueeze(2) & valid.unsqueeze(1)).unsqueeze(-1)
        return z * cell.to(z.dtype)
