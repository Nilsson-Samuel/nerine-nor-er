"""Canonicalization helpers for stable cluster-level resolution outputs."""

from __future__ import annotations

from datetime import date, datetime
from itertools import combinations
from typing import Any, Mapping

import polars as pl


def choose_canonical_name(member_rows: list[dict[str, Any]]) -> str:
    """Choose one deterministic canonical cluster name from member rows."""
    grouped: dict[str, dict[str, Any]] = {}

    for row in member_rows:
        normalized = str(row["normalized"])
        first_key = (str(row["chunk_id"]), int(row["char_start"]))
        candidate = grouped.setdefault(
            normalized,
            {
                "normalized": normalized,
                "total_count": 0,
                "first_chunk_id": first_key[0],
                "first_char_start": first_key[1],
            },
        )
        candidate["total_count"] += int(row["count"])
        if first_key < (candidate["first_chunk_id"], candidate["first_char_start"]):
            candidate["first_chunk_id"] = first_key[0]
            candidate["first_char_start"] = first_key[1]

    ordered = sorted(
        grouped.values(),
        key=lambda row: (
            -len(str(row["normalized"])),
            -int(row["total_count"]),
            str(row["first_chunk_id"]),
            int(row["first_char_start"]),
            str(row["normalized"]),
        ),
    )
    if not ordered:
        raise ValueError("cannot choose canonical_name from an empty cluster")
    return str(ordered[0]["normalized"])


def choose_canonical_type(member_rows: list[dict[str, Any]]) -> str:
    """Choose one deterministic canonical type from member rows."""
    entity_types = sorted({str(row["type"]) for row in member_rows})
    if len(entity_types) != 1:
        raise ValueError(f"cluster has mixed entity types: {entity_types}")
    return entity_types[0]


def aggregate_doc_ids(member_rows: list[dict[str, Any]]) -> list[str]:
    """Collect distinct sorted document IDs referenced by one cluster."""
    return sorted({str(row["doc_id"]) for row in member_rows})


def aggregate_attributes(member_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Derive minimal cluster attributes from currently available entity fields."""
    grouped: dict[str, set[str]] = {
        "chunk_ids": set(),
        "normalized_variants": set(),
        "text_variants": set(),
    }
    for row in member_rows:
        grouped["chunk_ids"].add(str(row["chunk_id"]))
        grouped["normalized_variants"].add(str(row["normalized"]))
        grouped["text_variants"].add(str(row["text"]))
    return {key: sorted(values) for key, values in grouped.items()}


def build_cluster_records(
    cluster_rows: list[dict[str, Any]],
    entities: pl.DataFrame,
    scored_pairs: pl.DataFrame,
    doc_dates: Mapping[str, date | datetime] | None = None,
) -> list[dict[str, Any]]:
    """Build canonicalized cluster records with member and edge lineage."""
    entity_by_id = {
        str(row["entity_id"]): row for row in entities.sort("entity_id").iter_rows(named=True)
    }
    edge_by_key = {
        tuple(sorted((str(row["entity_id_a"]), str(row["entity_id_b"])))): row
        for row in scored_pairs.sort(["entity_id_a", "entity_id_b"]).iter_rows(named=True)
    }

    records: list[dict[str, Any]] = []
    for cluster_row in cluster_rows:
        entity_ids = [str(entity_id) for entity_id in cluster_row["entity_ids"]]
        member_rows = [entity_by_id[entity_id] for entity_id in entity_ids]
        canonical_type = choose_canonical_type(member_rows)
        confidence = float(cluster_row["base_confidence"])
        doc_ids = aggregate_doc_ids(member_rows)
        most_recent_doc_id, most_recent_doc_date = _resolve_most_recent_doc(doc_ids, doc_dates)
        route_action = str(cluster_row["route_action"])

        records.append(
            {
                "cluster_id": str(cluster_row["cluster_id"]),
                "component_id": str(cluster_row["component_id"]),
                "entity_ids": entity_ids,
                "cluster_size": len(entity_ids),
                "canonical_name": choose_canonical_name(member_rows),
                "canonical_type": canonical_type,
                "confidence": confidence,
                "needs_review": route_action == "review",
                "route_action": route_action,
                "routing_actions_by_profile": dict(cluster_row["routing_actions_by_profile"]),
                "clustering_method": str(cluster_row["clustering_method"]),
                "component_objective_score": float(cluster_row["component_objective_score"]),
                "component_solve_elapsed_ms": float(cluster_row["component_solve_elapsed_ms"]),
                "pivot_run_count": int(cluster_row["pivot_run_count"]),
                "actual_edge_count": int(cluster_row["actual_edge_count"]),
                "possible_edge_count": int(cluster_row["possible_edge_count"]),
                "density": float(cluster_row["density"]),
                "min_edge_score": float(cluster_row["min_edge_score"]),
                "avg_edge_score": float(cluster_row["avg_edge_score"]),
                "suspicious_merge": bool(cluster_row["suspicious_merge"]),
                "doc_ids": doc_ids,
                "most_recent_doc_id": most_recent_doc_id,
                "most_recent_doc_date": most_recent_doc_date,
                "attributes": aggregate_attributes(member_rows),
                "members": [_member_lineage_row(row) for row in member_rows],
                "edges": _edge_lineage_rows(entity_ids, edge_by_key),
            }
        )

    return records


def build_resolved_entity_rows(cluster_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand cluster-level records into one resolved row per member entity."""
    rows: list[dict[str, Any]] = []

    for cluster in cluster_records:
        for member in cluster["members"]:
            rows.append(
                {
                    "run_id": member["run_id"],
                    "cluster_id": cluster["cluster_id"],
                    "entity_id": member["entity_id"],
                    "doc_id": member["doc_id"],
                    "entity_type": member["type"],
                    "canonical_name": cluster["canonical_name"],
                    "canonical_type": cluster["canonical_type"],
                    "cluster_size": cluster["cluster_size"],
                    "confidence": cluster["confidence"],
                    "needs_review": cluster["needs_review"],
                    "route_action": cluster["route_action"],
                    "clustering_method": cluster["clustering_method"],
                    "component_id": cluster["component_id"],
                    "doc_ids": cluster["doc_ids"],
                    "most_recent_doc_id": cluster["most_recent_doc_id"],
                    "most_recent_doc_date": cluster["most_recent_doc_date"],
                    "attributes": cluster["attributes"],
                }
            )

    return sorted(rows, key=lambda row: (row["cluster_id"], row["entity_id"]))


def _member_lineage_row(row: dict[str, Any]) -> dict[str, Any]:
    """Keep the upstream entity row fields needed for audit-friendly lineage."""
    return {
        "run_id": str(row["run_id"]),
        "entity_id": str(row["entity_id"]),
        "doc_id": str(row["doc_id"]),
        "chunk_id": str(row["chunk_id"]),
        "text": str(row["text"]),
        "normalized": str(row["normalized"]),
        "type": str(row["type"]),
        "char_start": int(row["char_start"]),
        "char_end": int(row["char_end"]),
        "context": str(row["context"]),
        "count": int(row["count"]),
        "positions": list(row["positions"]),
    }


def _edge_lineage_rows(
    entity_ids: list[str],
    edge_by_key: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach scored edge evidence and SHAP reasons for one cluster."""
    rows: list[dict[str, Any]] = []

    for entity_id_a, entity_id_b in combinations(sorted(entity_ids), 2):
        edge_row = edge_by_key.get((entity_id_a, entity_id_b))
        if edge_row is None:
            continue
        rows.append(
            {
                "entity_id_a": entity_id_a,
                "entity_id_b": entity_id_b,
                "score": float(edge_row["score"]),
                "blocking_methods": sorted(str(method) for method in edge_row["blocking_methods"]),
                "blocking_source": str(edge_row["blocking_source"]),
                "blocking_method_count": int(edge_row["blocking_method_count"]),
                "shap_top5": list(edge_row["shap_top5"]),
            }
        )

    return rows


def _resolve_most_recent_doc(
    doc_ids: list[str],
    doc_dates: Mapping[str, date | datetime] | None,
) -> tuple[str | None, date | None]:
    """Return the most recent cluster doc when usable dates are available."""
    if not doc_dates:
        return None, None

    dated_docs: list[tuple[date, str]] = []
    for doc_id in doc_ids:
        raw_value = doc_dates.get(doc_id)
        if raw_value is None:
            continue
        normalized = _normalize_doc_date(raw_value)
        if normalized is not None:
            dated_docs.append((normalized, doc_id))

    if not dated_docs:
        return None, None

    most_recent_doc_date, most_recent_doc_id = max(dated_docs, key=lambda row: (row[0], row[1]))
    return most_recent_doc_id, most_recent_doc_date


def _normalize_doc_date(value: date | datetime) -> date | None:
    """Normalize a date-like value down to a plain date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None
