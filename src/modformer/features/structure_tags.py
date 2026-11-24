"""Per-position structure descriptors computed from a base-pair probability matrix.

Everything here works on one full fold window (a ``W x W`` matrix), before any
positions are selected, and returns arrays of length ``W``. Two kinds of tag exist:

* scalar tags (``TAG_SCALAR_KEYS``): float32 magnitudes in a bounded range, with -1
  marking "undefined". They are stacked, in a fixed order, as the structure tower's input.
* identity tags (helix, loop and register labels): integers or booleans that are only
  ever compared for equality, never used as magnitudes.

The pairing register of position ``i`` is ``partner(i) + i``. It is constant along a
helix and shared by the two paired positions, so it identifies a duplex from either arm.
It is only meaningful where the pairing is confident (``register_valid``).
"""

from __future__ import annotations

from typing import Dict

import numpy as np

CONF = 0.5
REGISTER_TOL = 4
GAP_TOL = 4
CAP = 30.0
HELIX_LEN_CAP = 512.0
DIST_CAP = 256.0
EPS = 1e-6
UNDEFINED = -1.0

#: Order of the thirteen scalar channels of the tower input, the order the released weights
#: expect. ``modformer.model.tags`` holds the same list; keep the two copies identical.
TAG_SCALAR_KEYS = (
    "max_bp",
    "total_bp",
    "entropy",
    "partner_dist",
    "stem_end_dist",
    "loop_size",
    "loop_span_own",
    "loop_span_opposing",
    "helix_rank_5p",
    "helix_rank_3p",
    "helix_len",
    "helix_frac_5p",
    "dregister",
)

#: The eight tags gathered next to every example's positions (the network's copy is in
#: ``modformer.model.tags``).
PAIR_TAG_KEYS = (
    "max_bp",
    "helix_id",
    "loop_id",
    "register",
    "register_defined",
    "is_5p",
    "helix_run_start",
    "helix_run_end",
)

N_TAG_SCALARS = len(TAG_SCALAR_KEYS)

#: Structure annotation classes derived from the minimum-free-energy dot-bracket.
ANNOT_STEM, ANNOT_HAIRPIN, ANNOT_INTERNAL = 0, 1, 2
ANNOT_BULGE, ANNOT_EXTERIOR, ANNOT_MULTI = 3, 4, 5

_DIST_CACHE: Dict[int, np.ndarray] = {}


def stack_tag_scalars(tags: Dict[str, np.ndarray]) -> np.ndarray:
    """Stack the scalar tags into a ``(W, 13)`` float32 array in ``TAG_SCALAR_KEYS`` order."""
    missing = [k for k in TAG_SCALAR_KEYS if k not in tags]
    if missing:
        raise KeyError(f"tag dict is missing scalar keys {missing}")
    return np.stack([np.asarray(tags[k], dtype=np.float32) for k in TAG_SCALAR_KEYS], axis=-1)


def _dist_matrix(w: int) -> np.ndarray:
    cached = _DIST_CACHE.get(w)
    if cached is None:
        idx = np.arange(w, dtype=np.float32)
        cached = np.abs(idx[None, :] - idx[:, None])
        _DIST_CACHE[w] = cached
    return cached


def pairing_map(bp: np.ndarray, conf: float = CONF) -> Dict[str, np.ndarray]:
    """Most probable partner of each position, its probability and the register.

    ``register_valid`` is true where that probability reaches ``conf`` and the partner
    is not the position itself. ``register`` is filled everywhere but only meaningful
    where ``register_valid``.
    """
    w = bp.shape[0]
    idx = np.arange(w, dtype=np.int32)
    partner = np.asarray(bp.argmax(axis=1), dtype=np.int32)
    max_bp = np.asarray(bp.max(axis=1), dtype=np.float32)
    valid = max_bp >= conf
    valid &= partner != idx
    return {
        "partner": partner,
        "max_bp": max_bp,
        "register": (partner + idx).astype(np.int32),
        "register_valid": valid,
    }


def fill_register(register: np.ndarray, valid: np.ndarray, partner: np.ndarray,
                  gap_tol: int = GAP_TOL,
                  register_tol: int = REGISTER_TOL) -> Dict[str, np.ndarray]:
    """Carry the register across short unpaired runs that sit inside one helix.

    Two consecutive confident positions bound a gap of at most ``gap_tol`` positions;
    when their registers differ by at most ``register_tol`` the gap takes the register of
    the first one, which defines the opposing position of an unpaired base. A gap whose
    bounding positions are partners of each other is a hairpin loop and is left alone.
    """
    out = register.copy()
    was_filled = np.zeros(register.shape[0], dtype=bool)
    v = np.flatnonzero(valid)
    if v.size < 2:
        return {"register": out, "register_filled": was_filled}
    for a, b in zip(v[:-1], v[1:]):
        gap = int(b - a - 1)
        if gap <= 0 or gap > gap_tol:
            continue
        if abs(int(register[a]) - int(register[b])) > register_tol:
            continue
        if int(partner[a]) == int(b) or int(partner[b]) == int(a):
            continue
        out[a + 1:b] = register[a]
        was_filled[a + 1:b] = True
    return {"register": out, "register_filled": was_filled}


def helix_segments(register: np.ndarray, defined: np.ndarray,
                   gap_tol: int = GAP_TOL,
                   register_tol: int = REGISTER_TOL) -> np.ndarray:
    """Label each position with the id of the duplex it belongs to (``-1`` if none).

    Positions are grouped by register, so both arms of a duplex share an id although
    they are far apart in sequence. A register group is split where its 5' arm has a
    gap longer than ``gap_tol``, and the 3' arm inherits the id of its partner.
    """
    w = register.shape[0]
    out = np.full(w, -1, dtype=np.int32)
    v = np.flatnonzero(defined)
    if v.size == 0:
        return out

    order = v[np.argsort(register[v], kind="stable")]
    regs = register[order]
    groups = np.split(order, np.flatnonzero(np.diff(regs) > register_tol) + 1)

    seg = 0
    for g in groups:
        g = np.sort(g)
        reg = int(register[g[0]])
        five = g[2 * g < reg]
        if five.size == 0:
            cuts = np.flatnonzero(np.diff(g) > gap_tol) + 1
            for run in np.split(g, cuts):
                out[run] = seg
                seg += 1
            continue
        cuts = np.flatnonzero(np.diff(five) > gap_tol) + 1
        for run in np.split(five, cuts):
            out[run] = seg
            partners = reg - run
            ok = (partners >= 0) & (partners < w)
            partners = partners[ok]
            out[partners[defined[partners]]] = seg
            seg += 1
    return out


def helix_rank(helix_id: np.ndarray, cap: float = CAP,
               len_cap: float = HELIX_LEN_CAP) -> Dict[str, np.ndarray]:
    """Position of each base inside its duplex.

    ``helix_rank_5p`` and ``helix_rank_3p`` count positions from either end of a
    contiguous run, capped at ``cap`` and scaled to [0, 1]. ``helix_len`` is the
    log-scaled run length and ``helix_frac_5p`` the fractional depth along the run.
    ``helix_run_start`` and ``helix_run_end`` give the run's first and last position.
    Positions outside any helix are ``UNDEFINED`` (or ``-1`` for the integer tags).
    """
    w = helix_id.shape[0]
    r5 = np.full(w, UNDEFINED, dtype=np.float32)
    r3 = np.full(w, UNDEFINED, dtype=np.float32)
    hl = np.full(w, UNDEFINED, dtype=np.float32)
    hf = np.full(w, UNDEFINED, dtype=np.float32)
    rs = np.full(w, -1, dtype=np.int32)
    re = np.full(w, -1, dtype=np.int32)
    if (helix_id >= 0).sum() > 0:
        for hid in np.unique(helix_id[helix_id >= 0]):
            pos = np.flatnonzero(helix_id == hid)
            for run in np.split(pos, np.flatnonzero(np.diff(pos) != 1) + 1):
                n = run.size
                r5[run] = np.minimum(np.arange(n, dtype=np.float32), cap) / cap
                r3[run] = np.minimum(np.arange(n - 1, -1, -1, dtype=np.float32), cap) / cap
                hl[run] = min(np.log1p(n) / np.log1p(len_cap), 1.0)
                hf[run] = np.arange(n, dtype=np.float32) / max(n - 1, 1)
                rs[run] = int(run[0])
                re[run] = int(run[-1])
    return {"helix_rank_5p": r5, "helix_rank_3p": r3,
            "helix_len": hl, "helix_frac_5p": hf,
            "helix_run_start": rs, "helix_run_end": re}


def annotation_from_dot_bracket(db: np.ndarray) -> np.ndarray:
    """Classify every position of a dot-bracket structure into six element classes.

    ``db`` holds the ASCII codes of ``(``, ``)`` and ``.``. Paired positions are stems.
    An unpaired run is classified by the pair that encloses it and the number of helices
    that pair opens directly inside: none is a hairpin loop, one is a bulge (or an
    internal loop when both strands of the enclosing pair are interrupted) and two or
    more is a multiloop. A run enclosed by no pair is exterior.
    """
    w = int(db.shape[0])
    out = np.full(w, ANNOT_EXTERIOR, dtype=np.int8)
    if w == 0:
        return out

    ch = np.asarray(db).astype(np.int16)
    partner = np.full(w, -1, dtype=np.int64)
    enclosing = np.full(w, -1, dtype=np.int64)
    n_children: Dict[int, int] = {}

    stack: list[int] = []
    for i in range(w):
        c = ch[i]
        if c == 40:
            if stack:
                n_children[stack[-1]] = n_children.get(stack[-1], 0) + 1
            else:
                enclosing[i] = -1
            stack.append(i)
            n_children.setdefault(i, 0)
        elif c == 41:
            if not stack:
                continue
            j = stack.pop()
            partner[i] = j
            partner[j] = i
        else:
            if stack:
                enclosing[i] = stack[-1]

    paired = partner >= 0
    out[paired] = ANNOT_STEM

    i = 0
    while i < w:
        if paired[i] or ch[i] in (40, 41):
            i += 1
            continue
        start = i
        while i < w and not paired[i] and ch[i] not in (40, 41):
            i += 1
        run = slice(start, i)
        enc = enclosing[start]
        if enc < 0:
            out[run] = ANNOT_EXTERIOR
            continue
        kids = n_children.get(int(enc), 0)
        if kids == 0:
            out[run] = ANNOT_HAIRPIN
        elif kids == 1:
            out[run] = ANNOT_BULGE
        else:
            out[run] = ANNOT_MULTI
    for enc, kids in n_children.items():
        if kids != 1:
            continue
        close = int(partner[enc]) if partner[enc] >= 0 else -1
        if close < 0:
            continue
        left = np.arange(enc + 1, close)
        un = left[(out[left] == ANNOT_BULGE) & (enclosing[left] == enc)]
        if un.size == 0:
            continue
        child = left[paired[left]]
        if child.size == 0:
            continue
        lo, hi = int(child.min()), int(child.max())
        if (un < lo).any() and (un > hi).any():
            out[un] = ANNOT_INTERNAL
    return out


def delta_register(register: np.ndarray, defined: np.ndarray,
                   cap: float = CAP) -> Dict[str, np.ndarray]:
    """Step of the register between neighboring positions, scaled to [0, 1].

    Zero step (a continuous duplex) maps to 0.5. Undefined at the first position and
    wherever either neighbor has no defined register.
    """
    w = register.shape[0]
    d = np.full(w, UNDEFINED, dtype=np.float32)
    if w < 2:
        return {"dregister": d}
    ok = defined[1:] & defined[:-1]
    step = register[1:].astype(np.int64) - register[:-1].astype(np.int64)
    step = 0.5 + 0.5 * (np.clip(step, -cap, cap) / cap)
    d[1:][ok] = step[ok].astype(np.float32)
    return {"dregister": d}


def loop_geometry(valid: np.ndarray, register: np.ndarray, register_defined: np.ndarray,
                  cap: float = CAP) -> Dict[str, np.ndarray]:
    """Unpaired-run membership and sizes, and distance to the nearest unpaired position.

    ``loop_span_own`` is the length of the unpaired run a position sits in and
    ``loop_span_opposing`` the length of the run found at its opposing position through
    the filled register, which tells a mismatch from a bulge. All lengths are capped at
    ``cap`` and scaled to [0, 1]. ``loop_size`` is the same value as ``loop_span_own``;
    both channels are kept because the released weights read both.
    """
    w = valid.shape[0]
    unpaired = ~valid

    loop_id = np.full(w, -1, dtype=np.int32)
    run_len = np.zeros(w, dtype=np.float32)
    i, seg = 0, 0
    while i < w:
        if not unpaired[i]:
            i += 1
            continue
        j = i
        while j < w and unpaired[j]:
            j += 1
        loop_id[i:j] = seg
        run_len[i:j] = float(j - i)
        seg += 1
        i = j

    span_own = np.minimum(run_len, cap) / cap

    span_opp = np.full(w, UNDEFINED, dtype=np.float32)
    opposing = register - np.arange(w, dtype=register.dtype)
    ok = register_defined & (opposing >= 0) & (opposing < w)
    span_opp[ok] = np.minimum(run_len[opposing[ok]], cap) / cap

    pos = np.arange(w, dtype=np.float32)
    left = np.where(unpaired, pos, -np.inf)
    np.maximum.accumulate(left, out=left)
    right = np.where(unpaired, pos, np.inf)
    right = np.minimum.accumulate(right[::-1])[::-1]
    d = np.minimum(pos - left, right - pos)
    d[~np.isfinite(d)] = cap
    stem_end_dist = (np.minimum(d, cap) / cap).astype(np.float32)

    loop_size = np.minimum(run_len, cap).astype(np.float32) / cap

    return {
        "loop_id": loop_id,
        "loop_span_own": span_own.astype(np.float32),
        "loop_span_opposing": span_opp,
        "stem_end_dist": stem_end_dist,
        "loop_size": loop_size,
    }


def row_reductions(bp: np.ndarray, eps: float = EPS,
                   dist_cap: float = DIST_CAP) -> Dict[str, np.ndarray]:
    """Row sums over the whole partner axis: total pairing mass, entropy, mean distance.

    Entropy is in nats and the mean partner distance is capped at ``dist_cap`` nt and
    divided by it, so neither depends on the window width.
    """
    dist = _dist_matrix(bp.shape[0])
    logbp = np.log(bp + eps)
    return {
        "total_bp": bp.sum(axis=1).astype(np.float32),
        "entropy": (-(bp * logbp).sum(axis=1)).astype(np.float32),
        "partner_dist": ((bp * np.minimum(dist, dist_cap)).sum(axis=1)
                         / dist_cap).astype(np.float32),
    }


def compute_tags(bp: np.ndarray, conf: float = CONF, gap_tol: int = GAP_TOL,
                 register_tol: int = REGISTER_TOL) -> Dict[str, np.ndarray]:
    """All per-position tags of one fold window.

    ``bp`` is a finite square probability matrix. The result maps tag names to arrays of
    length ``W``: the scalar tags (``TAG_SCALAR_KEYS``) as float32 and the identity tags
    (``partner``, ``register``, ``register_valid``, ``register_filled``,
    ``register_defined``, ``is_5p``, ``helix_id``, ``loop_id`` and the helix run bounds).
    """
    w = bp.shape[0]
    if bp.ndim != 2 or bp.shape[1] != w:
        raise ValueError(f"bp must be square (W, W); got {bp.shape}")

    pm = pairing_map(bp, conf=conf)
    filled = fill_register(pm["register"], pm["register_valid"], pm["partner"],
                           gap_tol=gap_tol, register_tol=register_tol)
    reg_defined = pm["register_valid"] | filled["register_filled"]
    hid = helix_segments(filled["register"], reg_defined,
                         gap_tol=gap_tol, register_tol=register_tol)

    out: Dict[str, np.ndarray] = {}
    out.update(pm)
    out["register"] = filled["register"]
    out["register_filled"] = filled["register_filled"]
    out["register_defined"] = reg_defined
    out["helix_id"] = hid
    idx = np.arange(w, dtype=np.int64)
    out["is_5p"] = reg_defined & (2 * idx < out["register"])
    out.update(helix_rank(hid))
    out.update(delta_register(out["register"], reg_defined))
    out.update(loop_geometry(pm["register_valid"], out["register"], reg_defined))
    out.update(row_reductions(bp))
    return out
