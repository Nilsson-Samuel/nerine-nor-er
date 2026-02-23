"""DuckDB join loader and string/token feature functions for matching.

The loader resolves entity names from candidate pairs via a double JOIN.
String/token features are computed row-wise: 5 similarity scores in [0,1]
and 2 binary flags in {0,1}. No embedding or DuckDB dependency in the
feature functions — pure Python + rapidfuzz + metaphone.
"""

from pathlib import Path

import duckdb
import polars as pl
from metaphone import doublemetaphone
from rapidfuzz.distance import JaroWinkler, Levenshtein


# ---------------------------------------------------------------------------
# DuckDB join loader for candidate pairs with entity names
# ---------------------------------------------------------------------------

# Double-join query: resolves name_a/name_b from entities for each candidate pair.
# Ordered so downstream feature builders always see a stable row sequence.
_LOAD_PAIRS_QUERY = """
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
    if isinstance(db_path_or_con, duckdb.DuckDBPyConnection):
        table = db_path_or_con.execute(_LOAD_PAIRS_QUERY, [run_id]).fetch_arrow_table()
        return pl.from_arrow(table)

    # Path-based: use a context-managed connection so it is closed on exit.
    # read_parquet() takes the path as a Python argument - no SQL string interpolation.
    data_dir = Path(db_path_or_con)
    with duckdb.connect() as con:
        con.register("entities", con.read_parquet(str(data_dir / "entities.parquet")))
        con.register(
            "candidate_pairs",
            con.read_parquet(str(data_dir / "candidate_pairs.parquet")),
        )
        table = con.execute(_LOAD_PAIRS_QUERY, [run_id]).fetch_arrow_table()
        return pl.from_arrow(table)


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
# String/token feature helpers (pure Python, no DuckDB)
# ---------------------------------------------------------------------------

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
    toks_a = set(a.split())
    toks_b = set(b.split())
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
    toks_a = set(a.split())
    toks_b = set(b.split())
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
    tgrams_a = {a[i:i + 3] for i in range(len(a) - 2)}
    tgrams_b = {b[i:i + 3] for i in range(len(b) - 2)}
    if not tgrams_a or not tgrams_b:
        return 0.0
    return len(tgrams_a & tgrams_b) / len(tgrams_a | tgrams_b)


def _is_abbreviation(short: str, long: str) -> bool:
    """True if short is plausibly a prefix-abbreviation of long."""
    # Normalize dots to spaces so "D.N.B." tokenizes as ["D", "N", "B"].
    s_toks = short.replace(".", " ").lower().split()
    l_toks = long.replace(".", " ").lower().split()
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

    def _codes(name: str) -> set[str]:
        """Collect all non-empty Double Metaphone codes for a name's tokens."""
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
        return codes

    return int(bool(_codes(a) & _codes(b)))


# ---------------------------------------------------------------------------
# Aggregate builder: applies all 7 string/token features row-wise
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

    rows = []
    for row in pairs_df.iter_rows(named=True):
        a = row["name_a"] or ""
        b = row["name_b"] or ""
        rows.append({
            "run_id": row["run_id"],
            "entity_id_a": row["entity_id_a"],
            "entity_id_b": row["entity_id_b"],
            "jaro_winkler_similarity": jaro_winkler_similarity(a, b),
            "levenshtein_ratio_similarity": levenshtein_ratio_similarity(a, b),
            "token_jaccard_similarity": token_jaccard_similarity(a, b),
            "token_containment_ratio": token_containment_ratio(a, b),
            "char_trigram_jaccard_similarity": char_trigram_jaccard_similarity(a, b),
            "abbreviation_match_flag": abbreviation_match_flag(a, b),
            "double_metaphone_overlap_flag": double_metaphone_overlap_flag(a, b),
        })
    return pl.DataFrame(rows)
