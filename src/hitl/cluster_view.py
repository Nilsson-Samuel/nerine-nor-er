"""Cluster inspector rendering helpers for the Streamlit HITL interface.

Provides Streamlit-specific display functions for the cluster detail view:
header metrics, member table, member drilldown, alias table, edge table
with weakest-link highlight, and SHAP evidence formatting.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import streamlit as st

from src.hitl.queries import (
    build_alias_table,
    build_entity_text_lookup,
    find_weakest_edge,
    format_shap_reasons,
    load_cluster_edges,
    load_cluster_members,
    load_doc_paths,
)


def render_cluster_header(cluster_row: dict[str, Any]) -> None:
    """Display cluster-level metrics as a header row of metric cards."""
    cluster_id = cluster_row.get("cluster_id", "?")
    canonical = cluster_row.get("canonical_name", "")
    label = f"{cluster_id}    canonical_name: {canonical}" if canonical else cluster_id
    st.subheader(f"Cluster - {label}")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Route action", cluster_row.get("route_action", "-"))
    col2.metric("Size", cluster_row.get("cluster_size", "?"))
    col3.metric("Confidence", f"{cluster_row.get('base_confidence', 0):.2f}")
    col4.metric("Min edge score", f"{cluster_row.get('min_edge_score', 0):.2f}")
    col5.metric("Density", f"{cluster_row.get('density', 0):.2f}")

    st.caption(
        "**Confidence** = geometric mean of internal edge scores. "
        "**Density** = actual edges / possible edges. "
        "**Min edge score** = weakest pairwise evidence in the cluster."
    )


def render_member_table(
    members: pl.DataFrame,
    doc_paths: dict[str, str] | None = None,
) -> None:
    """Display the cluster member table with entity-level detail."""
    st.markdown("#### Members")

    if members.is_empty():
        st.info("No member details available for this cluster.")
        return

    display = members.clone()

    # Add human-readable doc path next to doc_id when available
    if doc_paths and "doc_id" in display.columns:
        display = display.with_columns(
            pl.col("doc_id")
            .replace_strict(doc_paths, default=None)
            .alias("doc_path")
        )

    display_cols = [
        "entity_id", "normalized", "text", "type", "doc_id", "doc_path", "count",
    ]
    st.dataframe(
        display.select([c for c in display_cols if c in display.columns]).to_pandas(),
        use_container_width=True,
        hide_index=True,
    )


def render_member_drilldown(members: pl.DataFrame) -> None:
    """Show provenance detail for a selected member entity."""
    if members.is_empty():
        return

    # Show entity text next to entity_id for readability
    options = [
        f"{row['entity_id']}  -  {row['text']}"
        for row in members.select("entity_id", "text").iter_rows(named=True)
    ]
    selected_label = st.selectbox(
        "Select member for provenance detail",
        options,
        index=None,
        placeholder="Choose an entity to inspect...",
        key="inspector_member_select",
    )

    if selected_label is None:
        st.caption("Select a member above to see context and provenance.")
        return

    selected_id = selected_label.split("  -  ")[0]
    row = members.filter(pl.col("entity_id") == selected_id)
    if row.is_empty():
        return

    record = row.row(0, named=True)
    st.markdown("##### Entity provenance")
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"**chunk_id:** `{record.get('chunk_id', '-')}`")
    col2.markdown(f"**char_start:** {record.get('char_start', '-')}")
    col3.markdown(f"**char_end:** {record.get('char_end', '-')}")

    context = record.get("context", "")
    if context:
        st.text_area("Context window", context, height=100, disabled=True)
    else:
        st.caption("No context available for this entity.")


def render_alias_table(members: pl.DataFrame) -> None:
    """Display alias aggregation: normalized form, surface forms, mention count."""
    st.markdown("#### Aliases")

    aliases = build_alias_table(members)
    if aliases.is_empty():
        st.info("No alias information available.")
        return

    # Convert surface_forms list to comma-separated string for display
    display = aliases.with_columns(
        pl.col("surface_forms").list.join(", ").alias("surface_forms")
    )
    st.dataframe(display.to_pandas(), use_container_width=True, hide_index=True)


def render_source_docs(
    members: pl.DataFrame,
    doc_paths: dict[str, str] | None = None,
) -> None:
    """List distinct source document IDs with file paths from cluster members."""
    if members.is_empty() or "doc_id" not in members.columns:
        return

    doc_ids = sorted(members["doc_id"].unique().to_list())
    st.markdown(f"#### Source documents ({len(doc_ids)})")

    # Show as a small table with doc_path when available
    rows = [
        {"doc_id": did, "doc_path": doc_paths.get(did, "-") if doc_paths else "-"}
        for did in doc_ids
    ]
    st.dataframe(
        pl.DataFrame(rows).to_pandas(),
        use_container_width=True,
        hide_index=True,
    )


def render_edge_table(
    edges: pl.DataFrame,
    entity_text: dict[str, str] | None = None,
) -> None:
    """Display internal edge table with weakest-link highlight and SHAP reasons."""
    st.markdown("#### Internal edges")

    if edges.is_empty():
        st.info("No internal edges available (singleton cluster or missing scored pairs).")
        return

    weakest = find_weakest_edge(edges)

    # Prepare display frame: add entity text labels and format SHAP
    display = edges.clone()

    if entity_text:
        display = display.with_columns([
            pl.col("entity_id_a")
            .replace_strict(entity_text, default="")
            .alias("text_a"),
            pl.col("entity_id_b")
            .replace_strict(entity_text, default="")
            .alias("text_b"),
        ])

    if "shap_top5" in display.columns:
        shap_col = display["shap_top5"].to_list()
        shap_strings = [format_shap_reasons(s) for s in shap_col]
        display = display.drop("shap_top5").with_columns(
            pl.Series("shap_reasons", shap_strings)
        )

    # Reorder so text labels appear right after their entity_id columns
    col_order = []
    for c in display.columns:
        col_order.append(c)
        if c == "entity_id_a" and "text_a" in display.columns:
            col_order.append("text_a")
        if c == "entity_id_b" and "text_b" in display.columns:
            col_order.append("text_b")
    # Deduplicate while keeping order (text_a/text_b may already be in the list)
    seen = set()
    ordered = []
    for c in col_order:
        if c not in seen:
            ordered.append(c)
            seen.add(c)

    st.dataframe(
        display.select(ordered).to_pandas(),
        use_container_width=True,
        hide_index=True,
    )

    # Weakest-link highlight with human-readable entity text
    if weakest is not None:
        id_a = weakest["entity_id_a"]
        id_b = weakest["entity_id_b"]
        label_a = f"{id_a} ({entity_text[id_a]})" if entity_text and id_a in entity_text else id_a
        label_b = f"{id_b} ({entity_text[id_b]})" if entity_text and id_b in entity_text else id_b
        st.warning(
            f"**Weakest link:** {label_a} ↔ {label_b} - "
            f"Score = {weakest['score']:.4f}"
        )


def render_inspector(
    data_dir: "Path",
    run_id: str,
    cluster_row: dict[str, Any],
) -> None:
    """Render the full cluster inspector panel.

    Args:
        data_dir: Root data directory for artifact loading.
        run_id: Pipeline run identifier.
        cluster_row: Dict with cluster-level fields from the cluster frame.
    """
    cluster_id = cluster_row["cluster_id"]

    render_cluster_header(cluster_row)

    members = load_cluster_members(data_dir, run_id, cluster_id)
    edges = load_cluster_edges(data_dir, run_id, cluster_id)
    doc_paths = load_doc_paths(data_dir, run_id)
    entity_text = build_entity_text_lookup(members)

    render_member_table(members, doc_paths=doc_paths)
    render_member_drilldown(members)
    render_alias_table(members)
    render_source_docs(members, doc_paths=doc_paths)
    render_edge_table(edges, entity_text=entity_text)
