"""Pipeline-wide configuration constants shared across stages.

The resolution thresholds below describe three different decisions:

- `PAIR_MATCH_THRESHOLD`: default probability cutoff used when converting
  scored candidate pairs into binary match / non-match decisions for
  pair-level evaluation and helper utilities.

- `KEEP_SCORE_THRESHOLD`: minimum LightGBM same-entity probability required for
  an edge to enter the retained graph at all.
- `OBJECTIVE_NEUTRAL_THRESHOLD`: the point inside correlation clustering where a
  retained edge stops being weak evidence and starts voting in favor of merging.
- `BASE_CONFIDENCE_REVIEW_THRESHOLD`: post-clustering routing cutoff used for
  reviewer-facing confidence handling. This does not change the graph or the
  clustering objective.
"""

# Default pair-level decision threshold used by matching/evaluation helpers.
# This is intentionally separate from the retained-graph and post-clustering
# thresholds below.
PAIR_MATCH_THRESHOLD = 0.50

# Minimum reranker score required for a pair to survive into the retained graph.
# Lower-scoring pairs are treated as too implausible to spend clustering work on.
# Provisional placeholder until inspection on more realistic data.
KEEP_SCORE_THRESHOLD = 0.60

# Neutral point inside correlation clustering after the pair already survived the
# keep threshold. Retained edges above this vote for merging; retained edges
# below this still exist in the component but vote for splitting.
OBJECTIVE_NEUTRAL_THRESHOLD = 0.80


# Post-clustering routing thresholds based on cluster evidence. These do not
# affect retained-graph construction or the clustering objective itself.
# Provisional placeholder until inspection on more realistic data.
BASE_CONFIDENCE_AUTO_MERGE_THRESHOLD = 0.85
BASE_CONFIDENCE_REVIEW_THRESHOLD = 0.50

# Keep routing policy separate from the evidence score so the same resolver
# output can support low-HITL and more review-heavy operation without changing
# clustering.
ROUTING_PROFILE = "quick_low_hitl"
