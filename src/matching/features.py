"""Matching loaders and feature functions.

Includes:
- DuckDB pair/name loader for candidate pairs.
- DuckDB pair metadata loader for context/type/doc/blocking fields.
- Embedding artifact loader with strict alignment guards.
- Structured identity flags from context/name/type fields.
- Co-occurrence/blocking metadata features.
- String/token features computed in a column-oriented pass: 5 similarity
  scores in [0,1] and 2 binary flags in {0,1}.
"""

from functools import lru_cache
from pathlib import Path
import re
import unicodedata
from typing import NamedTuple

import duckdb
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from metaphone import doublemetaphone
from rapidfuzz.distance import JaroWinkler, Levenshtein

from src.shared.validators import validate_embedding_alignment


# ---------------------------------------------------------------------------
# DuckDB join loader for candidate pairs with entity names
# ---------------------------------------------------------------------------

# Double-join query: resolves names only.
# Ordered so downstream feature builders always see a stable row sequence.
_LOAD_PAIRS_WITH_NAMES_QUERY = """
SELECT
    c.run_id,
    c.entity_id_a,
    c.entity_id_b,
    e1.normalized AS name_a,
    e2.normalized AS name_b
FROM candidate_pairs c
JOIN entities e1 ON c.run_id = e1.run_id AND c.entity_id_a = e1.entity_id
JOIN entities e2 ON c.run_id = e2.run_id AND c.entity_id_b = e2.entity_id
WHERE c.run_id = ?
ORDER BY c.entity_id_a, c.entity_id_b
"""

# Double-join query: resolves pair metadata from entities/candidate_pairs.
# Ordered so downstream feature builders always see a stable row sequence.
_LOAD_PAIRS_WITH_METADATA_QUERY = """
SELECT
    c.run_id,
    c.entity_id_a,
    c.entity_id_b,
    e1.normalized AS name_a,
    e2.normalized AS name_b,
    e1.context AS context_a,
    e2.context AS context_b,
    e1.type AS entity_type_a,
    e2.type AS entity_type_b,
    e1.doc_id AS doc_id_a,
    e2.doc_id AS doc_id_b,
    c.blocking_method_count
FROM candidate_pairs c
JOIN entities e1 ON c.run_id = e1.run_id AND c.entity_id_a = e1.entity_id
JOIN entities e2 ON c.run_id = e2.run_id AND c.entity_id_b = e2.entity_id
WHERE c.run_id = ?
ORDER BY c.entity_id_a, c.entity_id_b
"""

PAIR_NAME_COLUMNS = [
    "run_id",
    "entity_id_a",
    "entity_id_b",
    "name_a",
    "name_b",
]

PAIR_METADATA_COLUMNS = [
    *PAIR_NAME_COLUMNS,
    "context_a",
    "context_b",
    "entity_type_a",
    "entity_type_b",
    "doc_id_a",
    "doc_id_b",
    "blocking_method_count",
]


def _execute_pairs_query(
    db_path_or_con: "duckdb.DuckDBPyConnection | Path | str",
    run_id: str,
    query: str,
) -> pl.DataFrame:
    """Execute a pair loader query using either a connection or data directory."""
    if isinstance(db_path_or_con, duckdb.DuckDBPyConnection):
        table = db_path_or_con.execute(query, [run_id]).fetch_arrow_table()
        return pl.from_arrow(table)

    data_dir = Path(db_path_or_con)
    with duckdb.connect() as con:
        con.register("entities", con.read_parquet(str(data_dir / "entities.parquet")))
        con.register(
            "candidate_pairs",
            con.read_parquet(str(data_dir / "candidate_pairs.parquet")),
        )
        table = con.execute(query, [run_id]).fetch_arrow_table()
        return pl.from_arrow(table)


def load_pairs_with_names(
    db_path_or_con: "duckdb.DuckDBPyConnection | Path | str",
    run_id: str,
) -> pl.DataFrame:
    """Load candidate pairs joined with entity names for a given run_id.

    Accepts either an existing DuckDB connection (with 'entities' and
    'candidate_pairs' already registered) or a data directory path (parquet
    files are registered as temporary views).

    Returns exactly five columns: run_id, entity_id_a, entity_id_b,
    name_a, name_b — ordered by entity_id_a, entity_id_b for stable output.

    Args:
        db_path_or_con: DuckDB connection with tables pre-registered, or a
            directory path containing entities.parquet and candidate_pairs.parquet.
        run_id: Pipeline run identifier to filter on.

    Returns:
        Polars DataFrame with columns
        [run_id, entity_id_a, entity_id_b, name_a, name_b].
    """
    return _execute_pairs_query(
        db_path_or_con=db_path_or_con,
        run_id=run_id,
        query=_LOAD_PAIRS_WITH_NAMES_QUERY,
    ).select(PAIR_NAME_COLUMNS)


def load_pairs_with_metadata(
    db_path_or_con: "duckdb.DuckDBPyConnection | Path | str",
    run_id: str,
) -> pl.DataFrame:
    """Load candidate pairs with name/context/type/doc/blocking metadata.

    Returns exactly the columns listed in PAIR_METADATA_COLUMNS in stable row order.
    """
    return _execute_pairs_query(
        db_path_or_con=db_path_or_con,
        run_id=run_id,
        query=_LOAD_PAIRS_WITH_METADATA_QUERY,
    ).select(PAIR_METADATA_COLUMNS)


# ---------------------------------------------------------------------------
# Embedding artifact loader (foundation for cosine features)
# ---------------------------------------------------------------------------

class EmbeddingArtifacts(NamedTuple):
    """Immutable bundle of loaded embedding artifacts."""

    embeddings: np.ndarray
    context_embeddings: np.ndarray
    embedding_entity_ids: np.ndarray


EMBEDDING_FEATURE_COLUMNS = [
    "cosine_sim_entity",
    "cosine_sim_context",
]


def validate_loaded_embeddings(
    embeddings: np.ndarray,
    context_embeddings: np.ndarray,
    embedding_entity_ids: np.ndarray,
    entity_ids: list[str] | np.ndarray,
) -> None:
    """Validate loaded embedding artifacts by delegating shared invariants."""
    validate_embedding_alignment(
        embeddings=embeddings,
        context_embeddings=context_embeddings,
        embedding_entity_ids=embedding_entity_ids,
        entity_ids=entity_ids,
    )


def build_embedding_id_index(entity_ids: np.ndarray) -> dict[str, int]:
    """Build a deterministic entity_id -> embedding row index map.

    Raises:
        ValueError: If IDs are not unique.
    """
    ids = [str(entity_id) for entity_id in np.asarray(entity_ids).tolist()]
    index = {entity_id: i for i, entity_id in enumerate(ids)}
    if len(index) != len(ids):
        raise ValueError("embedding_entity_ids must contain unique IDs")
    return index


def load_embedding_artifacts(data_dir: Path | str, run_id: str) -> EmbeddingArtifacts:
    """Load embedding artifacts for one run and enforce alignment invariants.

    Required files:
    - embeddings.npy
    - context_embeddings.npy
    - embedding_entity_ids.npy
    - entities.parquet (source of the expected per-run entity_id order)
    """
    data_dir = Path(data_dir)
    embeddings = np.load(data_dir / "embeddings.npy", allow_pickle=False)
    context_embeddings = np.load(data_dir / "context_embeddings.npy", allow_pickle=False)
    embedding_entity_ids = np.load(data_dir / "embedding_entity_ids.npy", allow_pickle=False)
    entities = (
        pl.from_arrow(
            pq.read_table(
                data_dir / "entities.parquet",
                columns=["entity_id", "run_id"],
                filters=[("run_id", "=", run_id)],
            )
        )
        .sort("entity_id")
    )
    entity_ids = entities["entity_id"].to_list()
    if not entity_ids:
        raise ValueError(f"run_id not found in entities.parquet: {run_id}")

    validate_loaded_embeddings(
        embeddings=embeddings,
        context_embeddings=context_embeddings,
        embedding_entity_ids=embedding_entity_ids,
        entity_ids=entity_ids,
    )
    return EmbeddingArtifacts(
        embeddings=embeddings,
        context_embeddings=context_embeddings,
        embedding_entity_ids=embedding_entity_ids,
    )


def cosine_sim_from_lookup(
    entity_id_a: str,
    entity_id_b: str,
    matrix: np.ndarray,
    id_index: dict[str, int],
) -> float:
    """Compute cosine similarity from entity IDs and an aligned embedding matrix.

    Embeddings are expected to be L2-normalized upstream, so cosine is a dot
    product. Numeric noise is clipped into [-1.0, 1.0].
    """
    score = float(np.dot(matrix[id_index[entity_id_a]], matrix[id_index[entity_id_b]]))
    return float(np.clip(score, -1.0, 1.0))


def _lookup_embedding_row_indices(
    entity_ids: pl.Series,
    id_index: dict[str, int],
    column_name: str,
) -> np.ndarray:
    """Map entity IDs to embedding row indices with clear missing-ID errors."""
    values = entity_ids.to_list()
    try:
        return np.fromiter((id_index[entity_id] for entity_id in values), dtype=np.int64)
    except KeyError as exc:
        raise ValueError(
            f"{column_name} contains ID {exc.args[0]!r} missing from embedding_entity_ids"
        ) from exc


def build_embedding_features(
    pairs_df: pl.DataFrame,
    artifacts: EmbeddingArtifacts,
) -> pl.DataFrame:
    """Compute cosine_sim_entity and cosine_sim_context for each pair row.

    Returns raw cosine similarities in [-1.0, 1.0], preserving input row order.
    """
    if pairs_df.is_empty():
        return pl.DataFrame(
            schema={
                "cosine_sim_entity": pl.Float64,
                "cosine_sim_context": pl.Float64,
            }
        )

    id_index = build_embedding_id_index(artifacts.embedding_entity_ids)
    idx_a = _lookup_embedding_row_indices(pairs_df["entity_id_a"], id_index, "entity_id_a")
    idx_b = _lookup_embedding_row_indices(pairs_df["entity_id_b"], id_index, "entity_id_b")

    entity_cosine = np.einsum(
        "ij,ij->i",
        artifacts.embeddings[idx_a],
        artifacts.embeddings[idx_b],
    )
    context_cosine = np.einsum(
        "ij,ij->i",
        artifacts.context_embeddings[idx_a],
        artifacts.context_embeddings[idx_b],
    )

    return pl.DataFrame(
        {
            "cosine_sim_entity": np.clip(entity_cosine, -1.0, 1.0).astype(np.float64),
            "cosine_sim_context": np.clip(context_cosine, -1.0, 1.0).astype(np.float64),
        }
    )


# ---------------------------------------------------------------------------
# Norwegian phonetic pre-normalization (shared with blocking stage)
# ---------------------------------------------------------------------------

def _phonetic_normalize(name: str) -> str:
    """Map Norwegian characters to ASCII for Double Metaphone.

    Must match the blocking-stage rule exactly: æ->ae, ø->o, å->a,
    skj->sk, kj->k, hj->j, gj->j. Multi-char rules applied first so
    'skj' is not partially consumed by the 'kj' rule.
    """
    name = name.replace("skj", "sk").replace("kj", "k")
    name = name.replace("hj", "j").replace("gj", "j")
    name = name.replace("æ", "ae").replace("ø", "o").replace("å", "a")
    return name


# ---------------------------------------------------------------------------
# Structured identity feature helpers
# ---------------------------------------------------------------------------

NORWEGIAN_ID_RE = re.compile(r"\b\d{11}\b")

STRUCTURED_IDENTITY_FEATURE_COLUMNS = [
    "norwegian_id_match",
    "first_name_match",
    "last_name_match",
]


def norwegian_id_match(context_a: str, context_b: str) -> int:
    """Return 1 if both contexts contain an overlapping 11-digit identifier."""
    ids_in_context_a = set(NORWEGIAN_ID_RE.findall(context_a or ""))
    ids_in_context_b = set(NORWEGIAN_ID_RE.findall(context_b or ""))
    return int(bool(ids_in_context_a) and bool(ids_in_context_b & ids_in_context_a))


def _name_tokens(name: str) -> list[str]:
    """Whitespace-tokenize a name after trimming outer whitespace."""
    return (name or "").strip().split()


def _normalize_name_token(token: str) -> str:
    """Normalize a single name token for Unicode-safe equality checks."""
    return unicodedata.normalize("NFC", token).casefold()


def _is_per_pair(type_a: str, type_b: str) -> bool:
    """True when both entity types are PER (case-insensitive)."""
    return (type_a or "").upper() == "PER" and (type_b or "").upper() == "PER"


def first_name_match(name_a: str, name_b: str, type_a: str, type_b: str) -> int:
    """Return 1 if PER-only first tokens match, else 0."""
    if not _is_per_pair(type_a, type_b):
        return 0
    tokens_a = _name_tokens(name_a)
    tokens_b = _name_tokens(name_b)
    if not tokens_a or not tokens_b:
        return 0
    return int(_normalize_name_token(tokens_a[0]) == _normalize_name_token(tokens_b[0]))


def last_name_match(name_a: str, name_b: str, type_a: str, type_b: str) -> int:
    """Return 1 if PER-only last tokens match, else 0."""
    if not _is_per_pair(type_a, type_b):
        return 0
    tokens_a = _name_tokens(name_a)
    tokens_b = _name_tokens(name_b)
    if not tokens_a or not tokens_b:
        return 0
    return int(_normalize_name_token(tokens_a[-1]) == _normalize_name_token(tokens_b[-1]))


def build_structured_identity_features(pairs_df: pl.DataFrame) -> pl.DataFrame:
    """Build structured identity flags in stable row order.

    Expects input columns:
    - context_a, context_b
    - name_a, name_b
    - entity_type_a, entity_type_b
    """
    if pairs_df.is_empty():
        return pl.DataFrame(
            schema={
                "norwegian_id_match": pl.Int64,
                "first_name_match": pl.Int64,
                "last_name_match": pl.Int64,
            }
        )

    ids_a = pl.col("context_a").fill_null("").str.extract_all(NORWEGIAN_ID_RE.pattern)
    ids_b = pl.col("context_b").fill_null("").str.extract_all(NORWEGIAN_ID_RE.pattern)

    per_pair = (
        pl.col("entity_type_a").fill_null("").str.to_uppercase().eq("PER")
        & pl.col("entity_type_b").fill_null("").str.to_uppercase().eq("PER")
    )
    first_a = pl.col("name_a").fill_null("").str.extract(r"^\s*(\S+)", group_index=1)
    first_b = pl.col("name_b").fill_null("").str.extract(r"^\s*(\S+)", group_index=1)
    last_a = pl.col("name_a").fill_null("").str.extract(r"(\S+)\s*$", group_index=1)
    last_b = pl.col("name_b").fill_null("").str.extract(r"(\S+)\s*$", group_index=1)
    normalized_first_a = first_a.map_elements(_normalize_name_token, return_dtype=pl.String)
    normalized_first_b = first_b.map_elements(_normalize_name_token, return_dtype=pl.String)
    normalized_last_a = last_a.map_elements(_normalize_name_token, return_dtype=pl.String)
    normalized_last_b = last_b.map_elements(_normalize_name_token, return_dtype=pl.String)

    return pairs_df.select(
        (
            (ids_a.list.len() > 0) & (ids_a.list.set_intersection(ids_b).list.len() > 0)
        ).cast(pl.Int64).alias("norwegian_id_match"),
        pl.when(per_pair & first_a.is_not_null() & first_b.is_not_null())
        .then(normalized_first_a.eq(normalized_first_b))
        .otherwise(False)
        .cast(pl.Int64)
        .alias("first_name_match"),
        pl.when(per_pair & last_a.is_not_null() & last_b.is_not_null())
        .then(normalized_last_a.eq(normalized_last_b))
        .otherwise(False)
        .cast(pl.Int64)
        .alias("last_name_match"),
    )


# ---------------------------------------------------------------------------
# Co-occurrence + blocking metadata feature helpers
# ---------------------------------------------------------------------------

COOCCURRENCE_META_FEATURE_COLUMNS = [
    "shared_doc_count",
    "blocking_method_count",
]


def shared_doc_count(
    doc_id_a: str | None,
    doc_id_b: str | None,
) -> int:
    """Return 1 when both entities share the same non-empty doc_id, else 0.

    With the current handoff schema, each entity has a single doc_id scalar.
    This makes shared_doc_count a binary co-occurrence signal (0/1).
    """
    if not doc_id_a or not doc_id_b:
        return 0
    return int(doc_id_a == doc_id_b)


def build_cooccurrence_meta_features(pairs_df: pl.DataFrame) -> pl.DataFrame:
    """Build shared_doc_count and blocking_method_count in stable row order."""
    if pairs_df.is_empty():
        return pl.DataFrame(
            schema={
                "shared_doc_count": pl.Int64,
                "blocking_method_count": pairs_df.schema.get("blocking_method_count", pl.Int64),
            }
        )

    doc_id_a = pl.col("doc_id_a").fill_null("")
    doc_id_b = pl.col("doc_id_b").fill_null("")
    return pairs_df.select(
        ((doc_id_a != "") & (doc_id_b != "") & doc_id_a.eq(doc_id_b))
        .cast(pl.Int64)
        .alias("shared_doc_count"),
        pl.col("blocking_method_count"),
    ).select(COOCCURRENCE_META_FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# String/token feature helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=65_536)
def _token_set(name: str) -> frozenset[str]:
    """Cache whitespace-tokenized names for repeated pair comparisons."""
    if not name:
        return frozenset()
    return frozenset(name.split())


@lru_cache(maxsize=65_536)
def _char_trigrams(name: str) -> frozenset[str]:
    """Cache character 3-grams for repeated compound-name comparisons."""
    if len(name) < 3:
        return frozenset()
    return frozenset(name[i:i + 3] for i in range(len(name) - 2))


@lru_cache(maxsize=65_536)
def _abbreviation_tokens(name: str) -> tuple[str, ...]:
    """Cache abbreviation tokenization after dot normalization."""
    return tuple(name.replace(".", " ").lower().split())


@lru_cache(maxsize=65_536)
def _double_metaphone_codes(name: str) -> frozenset[str]:
    """Cache phonetic codes because the same names recur across many pairs."""
    if not name:
        return frozenset()

    normed = _phonetic_normalize(name)
    codes: set[str] = set()
    for tok in normed.split():
        if len(tok) < 2:
            continue
        primary, secondary = doublemetaphone(tok)
        if primary:
            codes.add(primary)
        if secondary and secondary != primary:
            codes.add(secondary)
    return frozenset(codes)


def jaro_winkler_similarity(a: str, b: str) -> float:
    """Jaro-Winkler similarity in [0, 1]. Returns 0.0 for empty inputs."""
    if not a or not b:
        return 0.0
    return JaroWinkler.similarity(a, b)


def levenshtein_ratio_similarity(a: str, b: str) -> float:
    """Normalized Levenshtein similarity in [0, 1]. Returns 0.0 for empty inputs."""
    if not a or not b:
        return 0.0
    return Levenshtein.normalized_similarity(a, b)


def token_jaccard_similarity(a: str, b: str) -> float:
    """Jaccard similarity of whitespace-tokenized name sets, in [0, 1]."""
    if not a or not b:
        return 0.0
    toks_a = _token_set(a)
    toks_b = _token_set(b)
    if not toks_a or not toks_b:
        return 0.0
    return len(toks_a & toks_b) / len(toks_a | toks_b)


def token_containment_ratio(a: str, b: str) -> float:
    """Max directional token containment between two names, in [0, 1].

    Captures subset-name patterns like "DNB" contained in "DNB ASA".
    Equivalent to intersection / min(|A|, |B|).
    """
    if not a or not b:
        return 0.0
    toks_a = _token_set(a)
    toks_b = _token_set(b)
    if not toks_a or not toks_b:
        return 0.0
    inter = len(toks_a & toks_b)
    return max(inter / len(toks_a), inter / len(toks_b))


def char_trigram_jaccard_similarity(a: str, b: str) -> float:
    """Character 3-gram Jaccard similarity in [0, 1].

    Effective for compound words and substring matching in Norwegian.
    """
    if not a or not b:
        return 0.0
    tgrams_a = _char_trigrams(a)
    tgrams_b = _char_trigrams(b)
    if not tgrams_a or not tgrams_b:
        return 0.0
    return len(tgrams_a & tgrams_b) / len(tgrams_a | tgrams_b)


def _is_abbreviation(short: str, long: str) -> bool:
    """True if short is plausibly a prefix-abbreviation of long."""
    # Normalize dots to spaces so "D.N.B." tokenizes as ["D", "N", "B"].
    s_toks = _abbreviation_tokens(short)
    l_toks = _abbreviation_tokens(long)
    if not s_toks or not l_toks or len(l_toks) < 2:
        return False

    initials = "".join(t[0] for t in l_toks)

    # Full acronym: "dnb" matches initials of "den norske bank"
    if "".join(s_toks) == initials:
        return True

    # First-token acronym: "dnb" in "dnb asa" matches initials of "den norske bank"
    if s_toks[0] == initials:
        return True

    # Token-prefix alignment: "p hansen" ↔ "per hansen" (at least one strict prefix)
    if len(s_toks) == len(l_toks):
        if all(lt.startswith(st) for st, lt in zip(s_toks, l_toks)):
            if any(len(st) < len(lt) for st, lt in zip(s_toks, l_toks)):
                return True

    return False


def abbreviation_match_flag(a: str, b: str) -> int:
    """1 if one name is a plausible prefix-abbreviation of the other, else 0."""
    if not a or not b:
        return 0
    if a.casefold() == b.casefold():
        return 1
    return int(_is_abbreviation(a, b) or _is_abbreviation(b, a))


def double_metaphone_overlap_flag(a: str, b: str) -> int:
    """1 if any Double Metaphone code overlaps after Norwegian normalization, else 0.

    Applies the same phonetic pre-normalization as the blocking stage
    (æ->ae, ø->o, å->a, etc.) before encoding each name token.
    """
    if not a or not b:
        return 0
    return int(bool(_double_metaphone_codes(a) & _double_metaphone_codes(b)))


# ---------------------------------------------------------------------------
# Aggregate builder: applies all 7 string/token features in one column-oriented pass
# ---------------------------------------------------------------------------

# Column names for the 7 string/token features.
STRING_FEATURE_COLUMNS = [
    "jaro_winkler_similarity",
    "levenshtein_ratio_similarity",
    "token_jaccard_similarity",
    "token_containment_ratio",
    "char_trigram_jaccard_similarity",
    "abbreviation_match_flag",
    "double_metaphone_overlap_flag",
]


def build_string_features(pairs_df: pl.DataFrame) -> pl.DataFrame:
    """Compute 7 string/token features for each candidate pair.

    Accepts the output of load_pairs_with_names (columns: run_id, entity_id_a,
    entity_id_b, name_a, name_b). Returns pair key columns + 7 feature columns.
    No nulls in the output — empty/None names are treated as empty strings.
    Uses cached helper preprocessing to reduce repeated Python work on common
    names across large pair tables.

    Args:
        pairs_df: Polars DataFrame from load_pairs_with_names.

    Returns:
        Polars DataFrame with columns [run_id, entity_id_a, entity_id_b]
        plus 7 feature columns listed in STRING_FEATURE_COLUMNS.
    """
    if pairs_df.is_empty():
        return pl.DataFrame(
            schema={
                "run_id": pairs_df.schema.get("run_id", pl.Utf8),
                "entity_id_a": pairs_df.schema.get("entity_id_a", pl.Utf8),
                "entity_id_b": pairs_df.schema.get("entity_id_b", pl.Utf8),
                "jaro_winkler_similarity": pl.Float64,
                "levenshtein_ratio_similarity": pl.Float64,
                "token_jaccard_similarity": pl.Float64,
                "token_containment_ratio": pl.Float64,
                "char_trigram_jaccard_similarity": pl.Float64,
                "abbreviation_match_flag": pl.Int64,
                "double_metaphone_overlap_flag": pl.Int64,
            }
        )

    names_a = pairs_df["name_a"].fill_null("").to_list()
    names_b = pairs_df["name_b"].fill_null("").to_list()
    pair_keys = pairs_df.select("run_id", "entity_id_a", "entity_id_b")

    return pair_keys.with_columns(
        pl.Series(
            "jaro_winkler_similarity",
            [jaro_winkler_similarity(a, b) for a, b in zip(names_a, names_b)],
            dtype=pl.Float64,
        ),
        pl.Series(
            "levenshtein_ratio_similarity",
            [levenshtein_ratio_similarity(a, b) for a, b in zip(names_a, names_b)],
            dtype=pl.Float64,
        ),
        pl.Series(
            "token_jaccard_similarity",
            [token_jaccard_similarity(a, b) for a, b in zip(names_a, names_b)],
            dtype=pl.Float64,
        ),
        pl.Series(
            "token_containment_ratio",
            [token_containment_ratio(a, b) for a, b in zip(names_a, names_b)],
            dtype=pl.Float64,
        ),
        pl.Series(
            "char_trigram_jaccard_similarity",
            [char_trigram_jaccard_similarity(a, b) for a, b in zip(names_a, names_b)],
            dtype=pl.Float64,
        ),
        pl.Series(
            "abbreviation_match_flag",
            [abbreviation_match_flag(a, b) for a, b in zip(names_a, names_b)],
            dtype=pl.Int64,
        ),
        pl.Series(
            "double_metaphone_overlap_flag",
            [double_metaphone_overlap_flag(a, b) for a, b in zip(names_a, names_b)],
            dtype=pl.Int64,
        ),
    )
