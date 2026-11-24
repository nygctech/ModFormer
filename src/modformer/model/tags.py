"""Names of the per-position structure tags the model reads.

The tags are computed from the folded window by ``modformer.features``; this
module only fixes the names and the order that the model's input layout depends on.
"""

#: Scalar tags stacked, in this order, into the ``w_tag_scalars`` tower input.
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
N_TAG_SCALARS = len(TAG_SCALAR_KEYS)

#: Tags gathered to the 128-position example and read by the pairwise channels.
#: In a batch each one is stored under the key ``"tag_" + name``.
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
TAG_PREFIX = "tag_"
