"""Cluster-first HITL triage interface for resolution outputs.

Provides run selection, routing-bucket exploration, cluster-size distribution,
a sortable cluster table, a merge-candidate overview for checking entities that
were kept apart, and a cluster inspector for edge-level drilldown.

Launch with:  streamlit run src/hitl/streamlit_app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import streamlit as st

from src.hitl.cluster_view import (
    INSPECTED_CLUSTER_KEY,
    INSPECTOR_HISTORY_KEY,
    render_inspector,
    render_unmerged_reason_legend,
)
from src.hitl.queries import (
    BUCKETS_BY_PROFILE,
    PROFILE_ROUTE_COLUMN,
    PROFILES,
    UNMERGED_PRUNED,
    UNMERGED_SPLIT,
    UNMERGED_WEAK,
    bucket_counts,
    bucket_summary,
    discover_run_ids,
    filter_by_bucket,
    load_candidate_table,
    load_cluster_frame_safe,
    resolve_run_thresholds,
    size_distribution,
    top_candidate_scores,
)
from src.hitl.status import (
    diagnostics_sidebar_summary,
    load_diagnostics_safe,
    load_installed_model_version,
    load_run_model_version,
    model_version_warning,
)
from src.hitl.ui_utils import build_option_label, parse_option_id
from src.shared.config import HITL_CANDIDATE_MIN_SCORE, HITL_CONTEXT_WINDOW_CHARS


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
DATA_DIR = Path(os.environ.get("NERINE_DATA_DIR", str(DEFAULT_DATA_DIR))).expanduser()

VIEW_CLUSTERS = "Clusters"
VIEW_CANDIDATES = "Merge candidates"
UNMERGED_REASONS = (UNMERGED_SPLIT, UNMERGED_WEAK, UNMERGED_PRUNED)

st.set_page_config(page_title="Nerine - Cluster Triage", layout="wide")
st.title("Nerine - Cluster Triage")

# Streamlit drops a widget's state while the inspector hides it. Re-assigning
# the candidate filters every run keeps them when the reviewer comes back.
st.session_state.setdefault("candidates_best_only", True)
st.session_state.setdefault("candidates_reasons", list(UNMERGED_REASONS))
for _key in ("candidates_best_only", "candidates_reasons"):
    st.session_state[_key] = st.session_state[_key]


# ── Run discovery ──────────────────────────────────────────────────────────────

@st.cache_data
def _cached_run_ids(data_dir_str: str) -> list[str]:
    """Cache run discovery across Streamlit reruns."""
    return discover_run_ids(Path(data_dir_str))


run_ids = _cached_run_ids(str(DATA_DIR))

if not run_ids:
    st.warning("No resolution runs found. Run the pipeline first.")
    st.stop()


# ── Sidebar controls ──────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Run & Filters")

    selected_run_id = st.selectbox("Run ID", run_ids, key="selected_run_id")

    selected_profile = st.selectbox(
        "Routing profile",
        PROFILES,
        key="selected_profile",
    )

    available_buckets = BUCKETS_BY_PROFILE[selected_profile]

    # Profile-scoped key so each profile remembers its own last bucket choice
    bucket_key = f"bucket_{selected_profile}"
    if st.session_state.get(bucket_key) not in available_buckets:
        st.session_state[bucket_key] = available_buckets[0]

    selected_bucket = st.selectbox(
        "Routing bucket",
        available_buckets,
        key=bucket_key,
    )

    view_mode = st.radio(
        "View",
        (VIEW_CLUSTERS, VIEW_CANDIDATES),
        key="view_mode",
        horizontal=True,
        help=(
            "Merge candidates lists entities in this bucket that the model "
            "scored against an entity in another cluster."
        ),
    )

    st.divider()
    st.subheader("Review settings")
    min_candidate_score = st.slider(
        "Minimum candidate score",
        min_value=0.0,
        max_value=1.0,
        value=HITL_CANDIDATE_MIN_SCORE,
        step=0.05,
        key="min_candidate_score",
        help="Lowest pair score shown as a merge candidate between two clusters.",
    )
    context_window = st.slider(
        "Context window (characters each side)",
        min_value=50,
        max_value=2000,
        value=HITL_CONTEXT_WINDOW_CHARS,
        step=50,
        key="context_window",
    )

    # Scoring model, flagged when it differs from the installed model
    st.divider()
    run_model_version = load_run_model_version(DATA_DIR, selected_run_id)
    st.markdown(f"**Scoring model**  \n`{run_model_version or 'unknown'}`")
    model_warning = model_version_warning(
        run_model_version, load_installed_model_version(DATA_DIR)
    )
    if model_warning:
        st.warning(model_warning)

    # Sidebar diagnostics summary
    st.divider()
    diagnostics = load_diagnostics_safe(DATA_DIR, selected_run_id)
    if diagnostics:
        summary = diagnostics_sidebar_summary(diagnostics)
        for label, value in summary.items():
            st.metric(label, value)
    else:
        st.info("No diagnostics available for this run.")

thresholds = resolve_run_thresholds(diagnostics)


# ── Load and cache cluster data ───────────────────────────────────────────────

@st.cache_data
def _cached_cluster_frame(data_dir_str: str, run_id: str) -> tuple[pl.DataFrame, str | None]:
    """Cache cluster frame loading so the explorer stays responsive on rerun."""
    return load_cluster_frame_safe(Path(data_dir_str), run_id)


cluster_frame, cluster_error = _cached_cluster_frame(str(DATA_DIR), selected_run_id)

if cluster_error is not None:
    st.error(cluster_error)
    st.stop()

if cluster_frame.is_empty():
    st.info("No clusters found for this run.")
    st.stop()


@st.cache_data
def _cached_candidate_table(
    data_dir_str: str,
    run_id: str,
    profile: str,
    min_score: float,
    run_thresholds: tuple[float, float],
) -> pl.DataFrame:
    """Cache cross-cluster merge candidates per run, profile and score floor."""
    frame, _ = _cached_cluster_frame(data_dir_str, run_id)
    return load_candidate_table(
        Path(data_dir_str), run_id, frame, profile, min_score, run_thresholds
    )


candidate_table = _cached_candidate_table(
    str(DATA_DIR), selected_run_id, selected_profile, min_candidate_score, thresholds
)


def _candidates_for_clusters(cluster_ids: list[str]) -> pl.DataFrame:
    """Restrict the candidate table to the given clusters."""
    if candidate_table.is_empty():
        return candidate_table
    return candidate_table.filter(pl.col("cluster_id").is_in(cluster_ids))


# ── Inspector view (shown when a cluster is selected) ─────────────────────────

if st.session_state.get(INSPECTED_CLUSTER_KEY) is not None:
    inspected_id = st.session_state[INSPECTED_CLUSTER_KEY]
    history = st.session_state.setdefault(INSPECTOR_HISTORY_KEY, [])

    back_col, previous_col, _ = st.columns([1, 1, 4])
    if back_col.button("Back to list"):
        st.session_state[INSPECTED_CLUSTER_KEY] = None
        history.clear()
        st.rerun()
    if history and previous_col.button("Back to previous cluster"):
        st.session_state[INSPECTED_CLUSTER_KEY] = history.pop()
        st.rerun()

    # Build the cluster row dict for the inspector header
    row_frame = cluster_frame.filter(pl.col("cluster_id") == inspected_id)
    if row_frame.is_empty():
        st.warning("Cluster not found in current data. Returning to list.")
        st.session_state[INSPECTED_CLUSTER_KEY] = None
        history.clear()
        st.rerun()

    row_dict = row_frame.row(0, named=True)

    # Map the profile-specific route column to a generic "route_action" key
    route_col = PROFILE_ROUTE_COLUMN.get(selected_profile, "")
    row_dict["route_action"] = row_dict.get(route_col, "-")

    render_inspector(
        DATA_DIR,
        selected_run_id,
        row_dict,
        _candidates_for_clusters([inspected_id]),
        context_window=context_window,
        min_candidate_score=min_candidate_score,
        thresholds=thresholds,
    )
    st.stop()


# ── Bucket summary cards ──────────────────────────────────────────────────────

st.subheader(f"Bucket: {selected_bucket}")

counts = bucket_counts(cluster_frame, selected_profile)
summary = bucket_summary(cluster_frame, selected_profile, selected_bucket)

card_col1, card_col2, card_col3, card_col4 = st.columns(4)
card_col1.metric("Clusters", summary["cluster_count"])
card_col2.metric("Entities", summary["entity_count"])
card_col3.metric("Avg size", summary["avg_cluster_size"])
card_col4.metric("Max size", summary["max_cluster_size"])

st.caption(
    "All buckets: "
    + " | ".join(f"{b}: {counts.get(b, 0)}" for b in available_buckets)
)


# ── Filtered frame for current bucket ─────────────────────────────────────────

bucket_frame = filter_by_bucket(cluster_frame, selected_profile, selected_bucket)

if bucket_frame.is_empty():
    st.info("No clusters in this bucket.")
    st.stop()


# ── Merge candidate overview ──────────────────────────────────────────────────

CANDIDATE_OVERVIEW_COLUMNS = [
    "cluster_id",
    "text",
    "type",
    "candidate_text",
    "candidate_type",
    "score",
    "unmerged_reason",
    "candidate_canonical_name",
    "candidate_cluster_size",
    "candidate_route",
    "context",
    "candidate_context",
]

if view_mode == VIEW_CANDIDATES:
    st.subheader("Merge candidates")
    st.caption(
        "Entities in this bucket that the model scored against an entity in "
        f"another cluster at {min_candidate_score:.2f} or higher. Use this to "
        "check whether entities that were kept apart belong together."
    )

    bucket_candidates = _candidates_for_clusters(bucket_frame["cluster_id"].to_list())
    with_candidates = (
        0 if bucket_candidates.is_empty() else bucket_candidates["cluster_id"].n_unique()
    )
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Clusters with a candidate", f"{with_candidates} / {bucket_frame.height}")
    metric_col2.metric("Candidate pairs", bucket_candidates.height)
    metric_col3.metric(
        "Strongest score",
        "-" if bucket_candidates.is_empty() else f"{bucket_candidates['score'].max():.3f}",
    )

    if bucket_candidates.is_empty():
        st.info(
            "No merge candidates in this bucket. Lower the minimum candidate "
            "score in the sidebar to see weaker candidates."
        )
        st.stop()

    render_unmerged_reason_legend(*thresholds)

    filter_col1, filter_col2 = st.columns([1, 2])
    best_only = filter_col1.checkbox(
        "Strongest candidate per cluster only", key="candidates_best_only"
    )
    reasons = filter_col2.multiselect(
        "Why not merged", UNMERGED_REASONS, key="candidates_reasons"
    )

    overview = bucket_candidates.filter(pl.col("unmerged_reason").is_in(reasons))
    if best_only:
        overview = overview.unique(subset="cluster_id", keep="first", maintain_order=True)

    if overview.is_empty():
        st.info("No candidates match the current filters.")
        st.stop()

    st.dataframe(
        overview.select(
            [c for c in CANDIDATE_OVERVIEW_COLUMNS if c in overview.columns]
        ).to_pandas(),
        use_container_width=True,
        hide_index=True,
    )

    canonical_by_cluster = dict(
        zip(bucket_frame["cluster_id"].to_list(), bucket_frame["canonical_name"].to_list())
    )
    candidate_cluster_options = [
        build_option_label(cluster_id, canonical_by_cluster.get(cluster_id, ""))
        for cluster_id in overview["cluster_id"].unique(maintain_order=True).to_list()
    ]
    chosen_label = st.selectbox(
        "Inspect cluster",
        candidate_cluster_options,
        index=None,
        placeholder="Select a cluster to inspect its candidates...",
        key="candidate_cluster_picker",
    )
    chosen = parse_option_id(chosen_label)
    if chosen is not None:
        st.session_state[INSPECTED_CLUSTER_KEY] = chosen
        st.rerun()
    st.stop()


# ── Cluster size distribution chart ───────────────────────────────────────────

st.subheader("Cluster Size Distribution")

dist = size_distribution(bucket_frame)
st.bar_chart(dist.to_pandas().set_index("cluster_size"), y="cluster_count")


# ── Size filter slider ────────────────────────────────────────────────────────

sizes = sorted(bucket_frame["cluster_size"].unique().to_list())
min_size, max_size = int(min(sizes)), int(max(sizes))

if min_size < max_size:
    size_range = st.slider(
        "Filter by cluster size",
        min_value=min_size,
        max_value=max_size,
        value=(min_size, max_size),
        key="size_range",
    )
    display_frame = bucket_frame.filter(
        (pl.col("cluster_size") >= size_range[0])
        & (pl.col("cluster_size") <= size_range[1])
    )
else:
    display_frame = bucket_frame


# ── Cluster table with inspect buttons ───────────────────────────────────────

st.subheader("Clusters")

if display_frame.is_empty():
    st.info("No clusters match the current size filter.")
    st.stop()

DISPLAY_COLUMNS = [
    "cluster_id",
    "canonical_name",
    "canonical_type",
    "cluster_size",
    "base_confidence",
    "top_candidate_score",
    "min_edge_score",
    "density",
    "suspicious_merge",
]

scored_frame = display_frame.join(
    top_candidate_scores(candidate_table), on="cluster_id", how="left"
)
sorted_frame = (
    scored_frame.select([c for c in DISPLAY_COLUMNS if c in scored_frame.columns])
    .sort(
        ["base_confidence", "top_candidate_score"],
        descending=[False, True],
        nulls_last=True,
    )
)

st.dataframe(sorted_frame.to_pandas(), use_container_width=True, hide_index=True)
st.caption(
    "**top_candidate_score** = strongest score between a member and an entity "
    f"in another cluster (empty when below {min_candidate_score:.2f}). Useful "
    "for keep_separate clusters, whose base confidence is always 0."
)

# Cluster selector for inspection (selectbox below the table)
# Show canonical_name next to cluster_id for human readability
cluster_options = [
    build_option_label(row["cluster_id"], row["canonical_name"])
    for row in sorted_frame.select("cluster_id", "canonical_name").iter_rows(named=True)
]
chosen_label = st.selectbox(
    "Inspect cluster",
    cluster_options,
    index=None,
    placeholder="Select a cluster to inspect...",
    key="cluster_picker",
)
# Extract cluster_id from the combined label
chosen = parse_option_id(chosen_label)

if chosen is not None:
    st.session_state[INSPECTED_CLUSTER_KEY] = chosen
    st.rerun()
