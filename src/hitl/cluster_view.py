"""Cluster inspector rendering helpers for the Streamlit HITL interface.

Provides Streamlit-specific display functions for the cluster detail view:
header metrics, member table, member contexts, merge candidates outside the
cluster, alias table, edge table with weakest-link highlight, and SHAP
evidence formatting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import streamlit as st

from src.hitl.context import (
    MAX_MENTIONS_PER_ENTITY,
    DocTextIndex,
    build_doc_text_index,
    entity_mention_windows,
    load_run_chunks,
)
from src.hitl.queries import (
    build_alias_table,
    build_entity_text_lookup,
    find_weakest_edge,
    format_shap_reasons,
    load_cluster_edges,
    load_cluster_member_ids,
    load_cluster_members,
    load_doc_paths,
    load_entities_by_id,
)
from src.hitl.ui_utils import (
    escape_dot_label,
    format_mention_html,
    format_metric_value,
)


INSPECTED_CLUSTER_KEY = "inspected_cluster_id"
# Clusters the reviewer came from, so they can step back after following a candidate
INSPECTOR_HISTORY_KEY = "inspector_history"

# Member contexts start expanded for clusters up to this size
EXPAND_CONTEXTS_MAX_MEMBERS = 5

CANDIDATE_DISPLAY_COLUMNS = [
    "text",
    "candidate_text",
    "candidate_type",
    "score",
    "unmerged_reason",
    "candidate_canonical_name",
    "candidate_cluster_size",
    "candidate_route",
    "candidate_doc_path",
]


@st.cache_data
def _cached_doc_paths(data_dir_str: str, run_id: str) -> dict[str, str]:
    """Cache doc path lookup for the selected run."""
    return load_doc_paths(Path(data_dir_str), run_id)


@st.cache_resource
def _cached_doc_index(data_dir_str: str, run_id: str) -> DocTextIndex:
    """Stitch the run's documents once; shared read-only across reruns."""
    return build_doc_text_index(load_run_chunks(Path(data_dir_str), run_id))


def open_cluster(cluster_id: str, from_cluster_id: str | None = None) -> None:
    """Switch the inspector to another cluster, remembering where we came from."""
    if from_cluster_id is not None:
        st.session_state.setdefault(INSPECTOR_HISTORY_KEY, []).append(from_cluster_id)
    st.session_state[INSPECTED_CLUSTER_KEY] = cluster_id


def render_cluster_header(cluster_row: dict[str, Any]) -> None:
    """Display cluster-level metrics as a header row of metric cards."""
    cluster_id = cluster_row.get("cluster_id", "?")
    canonical = cluster_row.get("canonical_name", "")
    label = f"{cluster_id}    canonical_name: {canonical}" if canonical else cluster_id
    st.subheader(f"Cluster - {label}")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Route action", cluster_row.get("route_action", "-"))
    col2.metric("Size", cluster_row.get("cluster_size", "?"))
    col3.metric("Confidence", format_metric_value(cluster_row.get("base_confidence")))
    col4.metric("Min edge score", format_metric_value(cluster_row.get("min_edge_score")))
    col5.metric("Density", format_metric_value(cluster_row.get("density")))

    st.caption(
        "**Confidence** = (0.6 × min edge score + 0.4 × mean edge score) × density. "
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


def _render_entity_contexts(
    record: dict[str, Any],
    index: DocTextIndex,
    window: int,
) -> None:
    """Show highlighted context windows for every mention of one entity."""
    windows = entity_mention_windows(record, index, window)
    if not windows:
        stored = record.get("context")
        if stored:
            st.caption("Source chunks unavailable - showing the stored ±50 character context.")
            st.text(stored)
        else:
            st.caption("No context available for this entity.")
        return

    for mention_window in windows[:MAX_MENTIONS_PER_ENTITY]:
        st.markdown(format_mention_html(mention_window), unsafe_allow_html=True)
    hidden = len(windows) - MAX_MENTIONS_PER_ENTITY
    if hidden > 0:
        st.caption(f"+{hidden} more mentions not shown.")


def render_member_contexts(
    members: pl.DataFrame,
    index: DocTextIndex,
    window: int,
    doc_paths: dict[str, str] | None = None,
) -> None:
    """Show every member's mentions in a wide context window."""
    st.markdown("#### Member contexts")

    if members.is_empty():
        st.info("No member details available for this cluster.")
        return

    st.caption(
        f"±{window} characters around each mention, rebuilt from the source "
        "document. Change the width under Review settings in the sidebar."
    )
    expanded = members.height <= EXPAND_CONTEXTS_MAX_MEMBERS
    for record in members.iter_rows(named=True):
        doc_id = record.get("doc_id", "-")
        doc_label = doc_paths.get(doc_id, doc_id) if doc_paths else doc_id
        title = f"{record.get('text', '?')}  ·  {record.get('type', '-')}  ·  {doc_label}"
        with st.expander(title, expanded=expanded):
            st.caption(
                f"entity_id `{record['entity_id']}` · "
                f"mentions: {record.get('count') or 1}"
            )
            _render_entity_contexts(record, index, window)


def render_unmerged_reason_legend(keep_threshold: float, neutral_threshold: float) -> None:
    """Explain the unmerged_reason values using the run's actual thresholds."""
    st.caption(
        f"**pruned**: score < {keep_threshold:.2f} (keep threshold), so the pair "
        "never reached clustering. "
        f"**weak**: {keep_threshold:.2f} ≤ score ≤ {neutral_threshold:.2f} "
        "(neutral point), so the pair voted against merging. "
        f"**split**: score > {neutral_threshold:.2f}, so the pair voted to merge, "
        "but correlation clustering kept the entities apart because other "
        "pairs in the component disagreed."
    )


def render_merge_candidates(
    cluster_id: str,
    candidates: pl.DataFrame,
    *,
    data_dir: Path,
    run_id: str,
    index: DocTextIndex,
    window: int,
    min_score: float,
    thresholds: tuple[float, float],
    doc_paths: dict[str, str] | None = None,
) -> None:
    """List outside entities that scored against this cluster, with side-by-side compare."""
    st.markdown("#### Merge candidates outside this cluster")

    if candidates.is_empty():
        st.info(
            f"No entity outside this cluster scored {min_score:.2f} or higher "
            "against its members. Lower the minimum candidate score in the "
            "sidebar to see weaker candidates."
        )
        return

    st.caption(
        f"Entities in other clusters that the model scored {min_score:.2f} or "
        "higher against a member of this cluster, strongest first."
    )
    render_unmerged_reason_legend(*thresholds)

    display = candidates
    if doc_paths and "candidate_doc_id" in display.columns:
        display = display.with_columns(
            pl.col("candidate_doc_id")
            .replace_strict(doc_paths, default=None)
            .alias("candidate_doc_path")
        )
    st.dataframe(
        display.select(
            [c for c in CANDIDATE_DISPLAY_COLUMNS if c in display.columns]
        ).to_pandas(),
        use_container_width=True,
        hide_index=True,
    )

    rows = list(candidates.iter_rows(named=True))
    labels = [
        f"{row['text']}  ↔  {row['candidate_text']}  "
        f"({row['score']:.3f}, cluster: {row['candidate_canonical_name']})"
        for row in rows
    ]
    choice = st.selectbox(
        "Compare with candidate",
        range(len(rows)),
        format_func=labels.__getitem__,
        index=None,
        placeholder="Choose a candidate to compare contexts side by side...",
        key=f"candidate_compare_{cluster_id}",
    )
    if choice is None:
        return

    row = rows[choice]
    records = load_entities_by_id(
        data_dir, run_id, [row["entity_id"], row["candidate_entity_id"]]
    )
    by_id = {record["entity_id"]: record for record in records.iter_rows(named=True)}

    left, right = st.columns(2)
    with left:
        st.markdown(f"**In this cluster:** {row['text']} ({row.get('type', '-')})")
        _render_entity_contexts(by_id.get(row["entity_id"], {}), index, window)
    with right:
        st.markdown(
            f"**Candidate:** {row['candidate_text']} ({row.get('candidate_type', '-')})  \n"
            f"Cluster *{row['candidate_canonical_name']}*, size "
            f"{row['candidate_cluster_size']}, routed to `{row['candidate_route']}`"
        )
        _render_entity_contexts(by_id.get(row["candidate_entity_id"], {}), index, window)
        st.button(
            "Open candidate's cluster",
            key=f"open_candidate_cluster_{cluster_id}",
            on_click=open_cluster,
            args=(row["candidate_cluster_id"], cluster_id),
        )
    st.caption(
        f"Pair score {row['score']:.4f} · {row['unmerged_reason']} · "
        f"blocking: {row.get('blocking_source') or '-'}"
    )


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


def render_cluster_graph(members: pl.DataFrame, edges: pl.DataFrame) -> None:
    """Render an undirected Graphviz node-edge diagram of the cluster.

    Nodes are labeled with entity surface text, edges are labeled with their
    score. The weakest edge is highlighted in red/dashed to match the existing
    weakest-link highlight in the edge table.
    """
    if members.is_empty():
        return

    # Build DOT string for an undirected graph
    lines = ["graph cluster {", '  rankdir=LR;', '  node [shape=ellipse];']

    for row in members.select("entity_id", "text").iter_rows(named=True):
        eid = row["entity_id"]
        label = escape_dot_label(row["text"])
        lines.append(f'  "{eid}" [label="{label}"];')

    if edges.is_empty():
        # Singleton cluster — just render the node(s)
        lines.append("}")
        st.markdown("#### Cluster graph")
        st.graphviz_chart("\n".join(lines))
        return

    weakest = find_weakest_edge(edges)
    weakest_key = None
    if weakest is not None:
        weakest_key = (weakest["entity_id_a"], weakest["entity_id_b"])

    for row in edges.iter_rows(named=True):
        id_a = row["entity_id_a"]
        id_b = row["entity_id_b"]
        score_label = f'{row["score"]:.3f}'

        is_weakest = weakest_key is not None and (
            (id_a, id_b) == weakest_key or (id_b, id_a) == weakest_key
        )

        if is_weakest:
            lines.append(
                f'  "{id_a}" -- "{id_b}" '
                f'[label="{score_label}" color=red style=dashed];'
            )
        else:
            lines.append(f'  "{id_a}" -- "{id_b}" [label="{score_label}"];')

    lines.append("}")

    st.markdown("#### Cluster graph")
    st.graphviz_chart("\n".join(lines))


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
    candidates: pl.DataFrame,
    *,
    context_window: int,
    min_candidate_score: float,
    thresholds: tuple[float, float],
) -> None:
    """Render the full cluster inspector panel.

    Args:
        data_dir: Root data directory for artifact loading.
        run_id: Pipeline run identifier.
        cluster_row: Dict with cluster-level fields from the cluster frame.
        candidates: Candidate table rows for this cluster (see build_candidate_table).
        context_window: Characters of context shown on each side of a mention.
        min_candidate_score: Score floor the candidates were filtered with.
        thresholds: The run's (keep, neutral) resolution thresholds.
    """
    cluster_id = cluster_row["cluster_id"]

    render_cluster_header(cluster_row)

    member_ids = load_cluster_member_ids(data_dir, run_id, cluster_id)
    members = load_cluster_members(data_dir, run_id, cluster_id, member_ids=member_ids)
    edges = load_cluster_edges(data_dir, run_id, cluster_id, member_ids=member_ids)
    doc_paths = _cached_doc_paths(str(data_dir), run_id)
    doc_index = _cached_doc_index(str(data_dir), run_id)
    entity_text = build_entity_text_lookup(members)

    render_member_table(members, doc_paths=doc_paths)
    render_member_contexts(members, doc_index, context_window, doc_paths=doc_paths)
    render_merge_candidates(
        cluster_id,
        candidates,
        data_dir=data_dir,
        run_id=run_id,
        index=doc_index,
        window=context_window,
        min_score=min_candidate_score,
        thresholds=thresholds,
        doc_paths=doc_paths,
    )
    render_alias_table(members)
    render_source_docs(members, doc_paths=doc_paths)
    render_cluster_graph(members, edges)
    render_edge_table(edges, entity_text=entity_text)
