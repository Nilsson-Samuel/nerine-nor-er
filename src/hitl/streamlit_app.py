"""Cluster-first HITL triage interface for resolution outputs.

Provides run selection, routing-bucket exploration, cluster-size distribution,
and a sortable cluster table for fast triage of resolution results.

Launch with:  streamlit run src/hitl/streamlit_app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import streamlit as st

from src.hitl.queries import (
    BUCKETS_BY_PROFILE,
    PROFILES,
    bucket_counts,
    bucket_summary,
    discover_run_ids,
    filter_by_bucket,
    load_cluster_frame_safe,
    size_distribution,
)
from src.hitl.status import diagnostics_sidebar_summary, load_diagnostics_safe


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
DATA_DIR = Path(os.environ.get("NERINE_DATA_DIR", str(DEFAULT_DATA_DIR))).expanduser()

st.set_page_config(page_title="Nerine — Cluster Triage", layout="wide")
st.title("Nerine — Cluster Triage")


# ── Run discovery ──────────────────────────────────────────────────────────────

run_ids = discover_run_ids(DATA_DIR)

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

    # Sidebar diagnostics summary
    st.divider()
    diagnostics = load_diagnostics_safe(DATA_DIR, selected_run_id)
    if diagnostics:
        summary = diagnostics_sidebar_summary(diagnostics)
        for label, value in summary.items():
            st.metric(label, value)
    else:
        st.info("No diagnostics available for this run.")


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


# ── Cluster table ─────────────────────────────────────────────────────────────

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
    "min_edge_score",
    "density",
    "suspicious_merge",
]

st.dataframe(
    display_frame.select([c for c in DISPLAY_COLUMNS if c in display_frame.columns])
    .sort("base_confidence")
    .to_pandas(),
    use_container_width=True,
    hide_index=True,
)
