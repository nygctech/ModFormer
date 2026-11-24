"""Assembly of model inputs for one tile.

A tile holds two 64-nt anchors. Each anchor becomes one example of 128 positions: the
anchor followed by a 64-nt *partner* window, chosen from the fold as the stretch that the
anchor's confident base pairs point to. Without such pairs, the partner is the stretch that
holds the most base-pair probability from the anchor, and with none at all the nearest legal
stretch.
Both runs keep their natural 5' to 3' order, so a duplex between anchor and partner shows
up as an anti-diagonal of the gathered pair matrix. Only adenosines of the anchor are
scored; the partner is context.

Every per-position tensor of an example is gathered with the same index vector, and one
validity mask (true at real, non-``N`` positions) is shared by all of them.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..tiling import ANCHOR, TILE, TILE_OFFSET, WINDOW, window_for
from .structure_tags import (PAIR_TAG_KEYS, annotation_from_dot_bracket, compute_tags,
                             stack_tag_scalars)

PAD_NT = 4
PARTNER_LEN = 64
RELATIVE_TIE_TOL = 0.95
REGISTER_GROUP_TOL = 8
TAG_PREFIX = "tag_"  # the same prefix as modformer.model.tags; keep the two equal

_NT_CODE = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
_TOKEN = {"A": 6, "C": 7, "G": 8, "U": 9, "T": 9, "N": 10}
_PAD_TOKEN, _CLS_TOKEN, _EOS_TOKEN = 0, 1, 2


def tokenize(seqs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """RiNALMo token ids and attention mask for sequences over ``ACGUN``.

    Returns int64 tensors of shape ``[len(seqs), longest + 2]``: a start token, the
    nucleotides, an end token, then padding (mask zero) up to the longest sequence.
    """
    longest = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), longest + 2), _PAD_TOKEN, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, s in enumerate(seqs):
        row = [_CLS_TOKEN] + [_TOKEN[c] for c in s] + [_EOS_TOKEN]
        ids[i, :len(row)] = torch.tensor(row)
        mask[i, :len(row)] = 1
    return ids, mask


def backbone_segments(is_pad: np.ndarray) -> np.ndarray:
    """Label each maximal run of real sequence with an id; padding gets ``-1``.

    Convolutions in the structure tower use these ids so they never reach across ``N``.
    """
    w = int(is_pad.shape[0])
    out = np.full(w, -1, dtype=np.int64)
    real = ~is_pad
    if not real.any():
        return out
    starts = real & ~np.concatenate([[False], real[:-1]])
    out[real] = (np.cumsum(starts) - 1)[real]
    return out


def _legal_starts(w: int, m: int, a0: int, a1: int, is_pad: np.ndarray) -> np.ndarray:
    """Starts of ``m``-wide windows that are in bounds, off the anchor and hold real sequence."""
    if m <= 0 or m > w:
        return np.empty(0, dtype=np.int64)
    starts = np.arange(0, w - m + 1, dtype=np.int64)
    disjoint = (starts + m <= a0) | (starts >= a1)
    real = np.convolve((~is_pad).astype(np.int32), np.ones(m, dtype=np.int32), mode="valid") > 0
    return starts[disjoint & real[: starts.size]]


def _disjoint_starts(w: int, m: int, a0: int, a1: int) -> np.ndarray:
    """In-bounds starts off the anchor, without requiring real sequence (the all-padding fallback)."""
    if m <= 0 or m > w:
        return np.empty(0, dtype=np.int64)
    starts = np.arange(0, w - m + 1, dtype=np.int64)
    return starts[(starts + m <= a0) | (starts >= a1)]


def _shift_off_anchor(start: int, m: int, w: int, a0: int, a1: int) -> Optional[int]:
    """Smallest shift that puts a window in bounds and off the anchor, or ``None``."""
    start = int(max(0, min(start, w - m)))
    if start + m <= a0 or start >= a1:
        return start
    cands = []
    if a0 - m >= 0:
        cands.append(a0 - m)
    if a1 + m <= w:
        cands.append(a1)
    if not cands:
        return None
    return int(min(cands, key=lambda s: abs(s - start)))


def select_partner_window(bp: np.ndarray, tags: Dict[str, np.ndarray], is_pad: np.ndarray,
                          a0: int, a1: int, m: int, parity: int = 0,
                          relative_tie_tol: float = RELATIVE_TIE_TOL,
                          register_group_tol: int = REGISTER_GROUP_TOL) -> Dict[str, object]:
    """Choose the ``m``-wide partner window for the anchor ``[a0, a1)``.

    1. Take the anchor positions whose most probable partner is confident and lies
       outside the anchor, and group them by pairing register.
    2. In the largest group, center the window on the extent of the partners, then shift
       it the least amount needed to stay in bounds and off the anchor.
    3. With no such group (or an illegal window), take the window with the most
       base-pair mass from the anchor. Every window with at least ``relative_tie_tol`` times
       the best mass counts as tied; among them the nearest wins, then the one with more
       real sequence.
    4. With no mass at all, take the nearest legal window; exact ties are broken by
       ``parity``.

    Returns a dict with ``partner_window_start``, ``mode`` (which rule fired),
    ``duplex_mass`` (pair mass the window captures) and ``off_mass`` (total off-anchor mass).
    """
    w = bp.shape[0]
    anchor = np.arange(a0, a1)

    prof = bp[a0:a1].sum(axis=0).astype(np.float64)
    prof[a0:a1] = 0.0
    prof[is_pad] = 0.0
    off_mass = float(prof.sum())

    conf = tags["register_valid"][anchor]
    part = tags["partner"][anchor]
    off = conf & ((part < a0) | (part >= a1))

    start: Optional[int] = None
    if int(off.sum()) >= 1:
        reg = tags["register"][anchor][off]
        order = np.argsort(reg, kind="stable")
        rs = reg[order]
        brk = np.flatnonzero(np.diff(rs) > register_group_tol)
        groups = np.split(np.arange(rs.size), brk + 1)
        big = max(groups, key=len)
        sel = order[big]
        pin = part[off][sel]
        lo, hi = int(pin.min()), int(pin.max())
        span = hi - lo + 1
        start = _shift_off_anchor(lo - (m - span) // 2, m, w, a0, a1)

    legal = _legal_starts(w, m, a0, a1, is_pad)
    if start is not None and start not in legal:
        start = None

    mode = "dominant_helix"
    if start is None:
        if legal.size == 0:
            mode = "degenerate_allpad"
            disj = _disjoint_starts(w, m, a0, a1)
            if disj.size == 0:
                raise ValueError(
                    f"no in-bounds window of {m} nt off the anchor [{a0}, {a1}) in {w} nt")
            dist = np.where(disj >= a1, disj - a1, a0 - (disj + m))
            near = disj[dist == dist.min()]
            start = int(near[parity % near.size])
        else:
            mode = "mass_argmax"
            cs = np.concatenate([[0.0], np.cumsum(prof)])
            mass = cs[legal + m] - cs[legal]
            best = float(mass.max())
            real_cs = np.concatenate([[0], np.cumsum((~is_pad).astype(np.int64))])
            n_real = real_cs[legal + m] - real_cs[legal]
            if best <= 0.0:
                mode = "degenerate_adjacent"
                dist = np.where(legal >= a1, legal - a1, a0 - (legal + m))
                near = legal[dist == dist.min()]
                near_real = n_real[dist == dist.min()]
                near = near[near_real == near_real.max()]
                start = int(near[parity % near.size])
            else:
                keep = mass >= best * relative_tie_tol
                cand, cand_real = legal[keep], n_real[keep]
                dist = np.where(cand >= a1, cand - a1, a0 - (cand + m))
                order = np.lexsort((-cand_real, dist))
                start = int(cand[order[0]])

    return {
        "partner_window_start": int(start),
        "mode": mode,
        "duplex_mass": float(prof[start:start + m].sum()),
        "off_mass": off_mass,
    }


def build_index_vector(a0: int, a: int, p0: int, m: int,
                       is_pad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Index vector ``[a0, a0+a) ++ [p0, p0+m)`` and its validity mask (true off padding)."""
    w = is_pad.shape[0]
    if a <= 0 or m < 0:
        raise ValueError(f"anchor width {a} must be > 0 and partner width {m} >= 0")
    if a0 < 0 or a0 + a > w or p0 < 0 or p0 + m > w:
        raise ValueError(f"runs [{a0}, {a0 + a}) and [{p0}, {p0 + m}) do not fit in {w}")
    if m > 0 and not (p0 + m <= a0 or p0 >= a0 + a):
        raise ValueError(f"partner run [{p0}, {p0 + m}) overlaps the anchor [{a0}, {a0 + a})")
    idx = np.concatenate([np.arange(a0, a0 + a, dtype=np.int64),
                          np.arange(p0, p0 + m, dtype=np.int64)])
    return idx, ~is_pad[idx]


def _nt_codes(window: str) -> np.ndarray:
    return np.array([_NT_CODE.get(c, PAD_NT) for c in window], dtype=np.int64)


def _gather_embeddings(embeddings: np.ndarray, origin: int,
                       runs: List[Tuple[int, int, int]], length: int) -> np.ndarray:
    """Embedding rows for runs of window positions; rows outside the transcript stay zero."""
    out = np.zeros((length, embeddings.shape[1]), dtype=np.float32)
    n = embeddings.shape[0]
    for dst, w_start, count in runs:
        t_lo, t_hi = w_start + origin, w_start + origin + count
        lo, hi = max(t_lo, 0), min(t_hi, n)
        if hi <= lo:
            continue
        rows = np.asarray(embeddings[lo:hi], dtype=np.float32)
        out[dst + (lo - t_lo):dst + (lo - t_lo) + rows.shape[0]] = rows
    return out


def build_examples(seq: str, embeddings: np.ndarray, tile_start: int, bpp: np.ndarray,
                   dot_bracket: str, tile_id: Optional[int] = None) -> List[dict]:
    """The two anchor examples of the tile starting at ``tile_start``.

    ``seq`` is the whole normalized transcript, ``embeddings`` its ``[len(seq), 1280]``
    embedding, ``bpp`` and ``dot_bracket`` the fold of the tile's window (see
    :func:`modformer.tiling.window_for`). ``tile_id`` breaks ties between equally good
    partner windows through its parity; it defaults to the tile's ordinal on the grid,
    ``tile_start // 128``.

    Each example is a dict. The network reads the tensors ``embeddings``, ``bp_probs``,
    ``orig_idx``, ``tile_valid``, ``a_positions``, the four window tensors
    ``w_tag_scalars``, ``w_struct_annot``, ``w_valid`` and ``w_seg_id``, and the eight
    ``tag_*`` tensors (see ``modformer.model.BATCH_KEYS``). The tensors ``run_id``,
    ``a_mask``, ``input_ids``, ``attention_mask``, ``anchor_half``, ``tile_valid_len``,
    ``a_source_measured`` (always zero) and ``struct_annot`` describe the example and are
    not read by the network. ``a_position_orig`` (tile-relative index of each scored
    adenosine) and the plain ints ``tile_id``, ``tile_start`` and ``partner_start`` (the
    partner window's first position as a zero-based, possibly out-of-range transcript
    coordinate) map results back onto the transcript. A scored adenosine at anchor-local
    index ``a`` of half ``h`` sits at transcript position ``tile_start + h * 64 + a``
    (zero-based).
    """
    if tile_id is None:
        tile_id = tile_start // TILE
    if embeddings.shape[0] != len(seq):
        raise ValueError(f"embeddings cover {embeddings.shape[0]} positions, sequence has {len(seq)}")
    if bpp.shape != (WINDOW, WINDOW):
        raise ValueError(f"bpp must be {WINDOW} x {WINDOW}, got {bpp.shape}")
    if len(dot_bracket) != WINDOW:
        raise ValueError(f"dot_bracket must have {WINDOW} characters, got {len(dot_bracket)}")

    origin = tile_start - TILE_OFFSET
    window = window_for(seq, tile_start)
    nt = _nt_codes(window)
    is_pad = nt == PAD_NT

    bp = np.asarray(bpp, dtype=np.float32)
    if not np.isfinite(bp).all():
        bp = np.where(np.isfinite(bp), bp, 0.0).astype(np.float32)
    tags = compute_tags(bp)
    annot = annotation_from_dot_bracket(
        np.frombuffer(dot_bracket.encode("ascii"), dtype=np.int8)).astype(np.int64)

    shared = {
        "w_tag_scalars": torch.from_numpy(stack_tag_scalars(tags)),
        "w_struct_annot": torch.from_numpy(annot),
        "w_valid": torch.from_numpy(~is_pad),
        "w_seg_id": torch.from_numpy(backbone_segments(is_pad)),
    }
    length = ANCHOR + PARTNER_LEN
    examples = []
    for half in range(TILE // ANCHOR):
        a0 = TILE_OFFSET + half * ANCHOR
        a1 = a0 + ANCHOR
        sel = select_partner_window(bp, tags, is_pad, a0, a1, PARTNER_LEN,
                                    parity=int(tile_id) & 1)
        p0 = int(sel["partner_window_start"])
        gidx, valid = build_index_vector(a0, ANCHOR, p0, PARTNER_LEN, is_pad)
        runs = [(0, a0, ANCHOR), (ANCHOR, p0, PARTNER_LEN)]

        gathered = "".join(window[i] for i in gidx)
        ids, _ = tokenize([gathered])
        attention = torch.ones_like(ids[0])
        attention[1:1 + length] = torch.from_numpy(valid.astype(np.int64))

        a_mask = np.zeros(length, dtype=np.float32)
        for p, ch in enumerate(gathered[:ANCHOR]):
            if ch == "A" and valid[p]:
                a_mask[p] = 1.0
        a_positions = np.flatnonzero(a_mask[:ANCHOR]).astype(np.int64)

        ex = {
            "tile_id": int(tile_id),
            "tile_start": int(tile_start),
            "partner_start": p0 + origin,
            "anchor_half": torch.tensor(half, dtype=torch.long),
            "embeddings": torch.from_numpy(_gather_embeddings(embeddings, origin, runs, length)),
            "input_ids": ids[0],
            "attention_mask": attention,
            "a_mask": torch.from_numpy(a_mask),
            "a_positions": torch.from_numpy(a_positions),
            "a_position_orig": torch.from_numpy(a_positions + half * ANCHOR),
            "a_source_measured": torch.zeros(len(a_positions), dtype=torch.float32),
            "tile_valid_len": torch.tensor(int(valid.sum()), dtype=torch.long),
            "orig_idx": torch.from_numpy(gidx),
            "run_id": torch.from_numpy((np.arange(length) >= ANCHOR).astype(np.int64)),
            "tile_valid": torch.from_numpy(valid),
            "bp_probs": torch.from_numpy(np.ascontiguousarray(bp[np.ix_(gidx, gidx)])),
            "struct_annot": torch.from_numpy(annot[gidx].copy()),
            **shared,
        }
        for key in PAIR_TAG_KEYS:
            v = np.ascontiguousarray(tags[key][gidx])
            if v.dtype not in (np.float32, np.bool_):
                v = v.astype(np.int64)
            ex[TAG_PREFIX + key] = torch.from_numpy(v)
        examples.append(ex)
    return examples


def collate(examples: List[dict]) -> dict:
    """Batch examples from :func:`build_examples` into one dict of tensors.

    The adenosine axis is padded to a multiple of 32 (to 1 when no example has an
    adenosine); ``a_valid`` marks the real entries and padded ``a_positions`` are zero.
    ``tile_starts`` and ``partner_starts`` (int64 ``[B]``) carry the coordinates needed to
    map results back onto the transcript.
    """
    b = len(examples)

    most = max(x["a_positions"].shape[0] for x in examples)
    na = 1 if most == 0 else ((most + 31) // 32) * 32
    a_positions = torch.zeros(b, na, dtype=torch.long)
    a_position_orig = torch.zeros(b, na, dtype=torch.long)
    a_valid = torch.zeros(b, na, dtype=torch.bool)
    a_source_measured = torch.zeros(b, na)
    for i, x in enumerate(examples):
        k = x["a_positions"].shape[0]
        a_positions[i, :k] = x["a_positions"]
        a_position_orig[i, :k] = x["a_position_orig"]
        a_valid[i, :k] = True

    batch = {
        "input_ids": torch.stack([x["input_ids"] for x in examples]),
        "attention_mask": torch.stack([x["attention_mask"] for x in examples]),
        "a_mask": torch.stack([x["a_mask"] for x in examples]),
        "a_positions": a_positions,
        "a_valid": a_valid,
        "a_position_orig": a_position_orig,
        "a_source_measured": a_source_measured,
        "embeddings": torch.stack([x["embeddings"] for x in examples]),
        "bp_probs": torch.stack([x["bp_probs"] for x in examples]),
        "struct_annot": torch.stack([x["struct_annot"] for x in examples]),
        "tile_ids": torch.tensor([int(x["tile_id"]) for x in examples], dtype=torch.long),
        "tile_starts": torch.tensor([int(x["tile_start"]) for x in examples], dtype=torch.long),
        "partner_starts": torch.tensor([int(x["partner_start"]) for x in examples],
                                       dtype=torch.long),
        "anchor_half": torch.stack([x["anchor_half"] for x in examples]),
    }
    for key in ("orig_idx", "run_id", "tile_valid",
                "w_tag_scalars", "w_struct_annot", "w_valid", "w_seg_id"):
        batch[key] = torch.stack([x[key] for x in examples])
    for key in PAIR_TAG_KEYS:
        batch[TAG_PREFIX + key] = torch.stack([x[TAG_PREFIX + key] for x in examples])
    return batch
