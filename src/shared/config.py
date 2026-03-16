"""Pipeline-wide configuration constants shared across stages."""

# Keep only reasonably plausible same-entity edges in the retained graph.
# Provisional placeholder until phase-2 inspection on more realistic data.
KEEP_SCORE_THRESHOLD = 0.60

# Neutral point for the correlation clustering objective in the next step.
# Retained edges above this favor merging; lower retained edges favor splitting.
OBJECTIVE_NEUTRAL_THRESHOLD = 0.80

# Review-routing cutoff applied after clustering. This does not change graph building.
# Provisional placeholder until phase-2 inspection on more realistic data.
REVIEW_CONFIDENCE_THRESHOLD = 0.75
