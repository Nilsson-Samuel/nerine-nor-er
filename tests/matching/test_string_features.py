"""Unit tests for string/token feature helpers and build_string_features.

Covers per-helper validation:
- Null/empty guard: returns 0.0 (similarities) or 0 (flags).
- Range: similarities in [0, 1]; flags in {0, 1}.
- Perfect match: identical names score 1.0 (similarities) or 1 (flags).
- Known pair: hand-crafted input with expected approximate values.

Integration test for build_string_features:
- All 7 feature columns present.
- No null values.
- Key columns preserved.
"""

from pathlib import Path

import polars as pl
import pytest

from src.shared.fixtures import DEFAULT_RUN_ID, write_mock_handoff
from src.matching.features import (
    STRING_FEATURE_COLUMNS,
    abbreviation_match_flag,
    build_string_features,
    char_trigram_jaccard_similarity,
    double_metaphone_overlap_flag,
    jaro_winkler_similarity,
    levenshtein_ratio_similarity,
    load_pairs_with_names,
    token_containment_ratio,
    token_jaccard_similarity,
)

# Similarity helpers (return float in [0, 1])
_SIMILARITY_FUNCS = [
    jaro_winkler_similarity,
    levenshtein_ratio_similarity,
    token_jaccard_similarity,
    token_containment_ratio,
    char_trigram_jaccard_similarity,
]

# Flag helpers (return int in {0, 1})
_FLAG_FUNCS = [
    abbreviation_match_flag,
    double_metaphone_overlap_flag,
]

# Mock fixture produces 2 candidate pairs (see src/shared/fixtures.py).
_EXPECTED_PAIR_COUNT = 2


@pytest.fixture()
def handoff_dir(tmp_path: Path) -> Path:
    """Write mock handoff fixture files and return the directory."""
    write_mock_handoff(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Empty / None guard — all helpers return 0.0 or 0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("func", _SIMILARITY_FUNCS, ids=lambda f: f.__name__)
def test_similarity_empty_a_returns_zero(func):
    assert func("", "test") == 0.0


@pytest.mark.parametrize("func", _SIMILARITY_FUNCS, ids=lambda f: f.__name__)
def test_similarity_empty_b_returns_zero(func):
    assert func("test", "") == 0.0


@pytest.mark.parametrize("func", _SIMILARITY_FUNCS, ids=lambda f: f.__name__)
def test_similarity_both_empty_returns_zero(func):
    assert func("", "") == 0.0


@pytest.mark.parametrize("func", _SIMILARITY_FUNCS, ids=lambda f: f.__name__)
def test_similarity_none_a_returns_zero(func):
    assert func(None, "test") == 0.0  # type: ignore[arg-type]


@pytest.mark.parametrize("func", _SIMILARITY_FUNCS, ids=lambda f: f.__name__)
def test_similarity_none_b_returns_zero(func):
    assert func("test", None) == 0.0  # type: ignore[arg-type]


@pytest.mark.parametrize("func", _FLAG_FUNCS, ids=lambda f: f.__name__)
def test_flag_empty_a_returns_zero(func):
    assert func("", "test") == 0


@pytest.mark.parametrize("func", _FLAG_FUNCS, ids=lambda f: f.__name__)
def test_flag_empty_b_returns_zero(func):
    assert func("test", "") == 0


@pytest.mark.parametrize("func", _FLAG_FUNCS, ids=lambda f: f.__name__)
def test_flag_none_a_returns_zero(func):
    assert func(None, "test") == 0  # type: ignore[arg-type]


@pytest.mark.parametrize("func", _FLAG_FUNCS, ids=lambda f: f.__name__)
def test_flag_none_b_returns_zero(func):
    assert func("test", None) == 0  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Identical names — perfect match
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("func", _SIMILARITY_FUNCS, ids=lambda f: f.__name__)
def test_similarity_identical_returns_one(func):
    assert func("per hansen", "per hansen") == 1.0


@pytest.mark.parametrize("func", _FLAG_FUNCS, ids=lambda f: f.__name__)
def test_flag_identical_returns_one(func):
    assert func("per hansen", "per hansen") == 1


# ---------------------------------------------------------------------------
# Range bounds — similarity in [0, 1], flags in {0, 1}
# ---------------------------------------------------------------------------

_DIVERSE_PAIRS = [
    ("per hansen", "per johansen"),
    ("anders", "andreas"),
    ("dnb asa", "den norske bank"),
    ("x", "y"),
    ("oslo politidistrikt", "oslo pd"),
]


@pytest.mark.parametrize("a, b", _DIVERSE_PAIRS)
@pytest.mark.parametrize("func", _SIMILARITY_FUNCS, ids=lambda f: f.__name__)
def test_similarity_range(func, a, b):
    val = func(a, b)
    assert 0.0 <= val <= 1.0, f"{func.__name__}({a!r}, {b!r}) = {val}"


@pytest.mark.parametrize("a, b", _DIVERSE_PAIRS)
@pytest.mark.parametrize("func", _FLAG_FUNCS, ids=lambda f: f.__name__)
def test_flag_values(func, a, b):
    val = func(a, b)
    assert val in (0, 1), f"{func.__name__}({a!r}, {b!r}) = {val}"


# ---------------------------------------------------------------------------
# Known-pair sanity — hand-crafted with expected approximate values
# ---------------------------------------------------------------------------

def test_jaro_winkler_close_names():
    # "per hansen" vs "per johansen": share prefix "per " → high JW
    val = jaro_winkler_similarity("per hansen", "per johansen")
    assert 0.7 < val < 1.0


def test_levenshtein_close_names():
    val = levenshtein_ratio_similarity("per hansen", "per johansen")
    assert 0.6 < val < 1.0


def test_token_jaccard_partial_overlap():
    # {"per", "hansen"} ∩ {"per", "johansen"} = {"per"}, union size 3 → 1/3
    val = token_jaccard_similarity("per hansen", "per johansen")
    assert abs(val - 1.0 / 3.0) < 1e-9


def test_token_containment_subset():
    # "dnb" is a single token, "dnb asa" has {"dnb", "asa"} — containment = 1/1 = 1.0
    val = token_containment_ratio("dnb", "dnb asa")
    assert val == 1.0


def test_token_containment_disjoint():
    val = token_containment_ratio("abc", "xyz")
    assert val == 0.0


def test_char_trigram_similar():
    # "hansen" and "johansen" share many 3-grams ("han", "ans", "nse", "sen")
    val = char_trigram_jaccard_similarity("hansen", "johansen")
    assert val > 0.3


def test_char_trigram_short_strings():
    # Strings shorter than 3 chars produce no trigrams → 0.0
    assert char_trigram_jaccard_similarity("ab", "ab") == 0.0


def test_abbreviation_acronym():
    # "dnb" is the initials of "den norske bank"
    assert abbreviation_match_flag("dnb", "den norske bank") == 1


def test_abbreviation_acronym_with_suffix():
    # "dnb asa" — first token "dnb" matches initials of "den norske bank"
    assert abbreviation_match_flag("dnb asa", "den norske bank") == 1


def test_abbreviation_token_prefix():
    # "p hansen" → "p" is prefix of "per"
    assert abbreviation_match_flag("p hansen", "per hansen") == 1


def test_abbreviation_dotted_initials():
    # "d.n.b." should match "den norske bank" after dot normalization
    assert abbreviation_match_flag("d.n.b.", "den norske bank") == 1


def test_abbreviation_all_caps_acronym():
    # Should be case-insensitive for user-entered acronym variants.
    assert abbreviation_match_flag("DNB", "den norske bank") == 1


def test_abbreviation_no_match():
    # "anders johansen" is not an abbreviation of "per hansen"
    assert abbreviation_match_flag("anders johansen", "per hansen") == 0


def test_abbreviation_single_token_no_match():
    # Single-token vs single-token: not an abbreviation scenario
    assert abbreviation_match_flag("per", "anders") == 0


def test_metaphone_same_sound():
    # "hansen" and "hansen" share metaphone codes
    assert double_metaphone_overlap_flag("hansen", "hansen") == 1


def test_metaphone_similar_sound():
    # "hansen" and "hanssen" should share phonetic codes
    assert double_metaphone_overlap_flag("hansen", "hanssen") == 1


def test_metaphone_different_sound():
    # Completely different names — no phonetic overlap expected
    assert double_metaphone_overlap_flag("per", "bank") == 0


def test_metaphone_norwegian_chars():
    # "bjørn" → phonetic_normalize → "bjorn", "bjorn" → same codes
    assert double_metaphone_overlap_flag("bjørn", "bjorn") == 1


def test_metaphone_single_char_tokens_skipped():
    # Single-char tokens are skipped; "a b" has no codes → 0
    assert double_metaphone_overlap_flag("a b", "a b") == 0


# ---------------------------------------------------------------------------
# build_string_features — integration against mock fixture
# ---------------------------------------------------------------------------

def test_build_has_all_feature_columns(handoff_dir: Path):
    pairs = load_pairs_with_names(handoff_dir, DEFAULT_RUN_ID)
    result = build_string_features(pairs)
    for col in STRING_FEATURE_COLUMNS:
        assert col in result.columns, f"Missing feature column: {col}"


def test_build_has_key_columns(handoff_dir: Path):
    pairs = load_pairs_with_names(handoff_dir, DEFAULT_RUN_ID)
    result = build_string_features(pairs)
    for col in ("run_id", "entity_id_a", "entity_id_b"):
        assert col in result.columns, f"Missing key column: {col}"


def test_build_row_count_matches(handoff_dir: Path):
    pairs = load_pairs_with_names(handoff_dir, DEFAULT_RUN_ID)
    result = build_string_features(pairs)
    assert len(result) == _EXPECTED_PAIR_COUNT


def test_build_no_null_values(handoff_dir: Path):
    pairs = load_pairs_with_names(handoff_dir, DEFAULT_RUN_ID)
    result = build_string_features(pairs)
    for col in STRING_FEATURE_COLUMNS:
        null_count = result[col].null_count()
        assert null_count == 0, f"Column {col} has {null_count} nulls"


def test_build_similarity_range(handoff_dir: Path):
    pairs = load_pairs_with_names(handoff_dir, DEFAULT_RUN_ID)
    result = build_string_features(pairs)
    sim_cols = STRING_FEATURE_COLUMNS[:5]  # first 5 are similarities
    for col in sim_cols:
        vals = result[col]
        assert vals.min() >= 0.0, f"{col} has value below 0"
        assert vals.max() <= 1.0, f"{col} has value above 1"


def test_build_flag_values(handoff_dir: Path):
    pairs = load_pairs_with_names(handoff_dir, DEFAULT_RUN_ID)
    result = build_string_features(pairs)
    flag_cols = STRING_FEATURE_COLUMNS[5:]  # last 2 are flags
    for col in flag_cols:
        unique = set(result[col].to_list())
        assert unique <= {0, 1}, f"{col} has values outside {{0, 1}}: {unique}"


def test_build_identical_pair_scores_high(handoff_dir: Path):
    """The PER pair has identical normalized names ('per hansen' ↔ 'per hansen').

    All similarity features should be 1.0 and flags should be 1.
    """
    pairs = load_pairs_with_names(handoff_dir, DEFAULT_RUN_ID)
    result = build_string_features(pairs)
    # The fixture PER pair has identical normalized names.
    # Find the row where all similarities are 1.0 (the identical pair).
    sim_cols = STRING_FEATURE_COLUMNS[:5]
    for i in range(len(result)):
        row = result.row(i, named=True)
        sims_all_one = all(row[c] == 1.0 for c in sim_cols)
        if sims_all_one:
            # Verify flags are also 1
            assert row["abbreviation_match_flag"] == 1
            assert row["double_metaphone_overlap_flag"] == 1
            return
    pytest.fail("Expected at least one row with all similarities == 1.0 (identical pair)")


def test_build_column_count(handoff_dir: Path):
    pairs = load_pairs_with_names(handoff_dir, DEFAULT_RUN_ID)
    result = build_string_features(pairs)
    # 3 key columns + 7 feature columns = 10 total
    assert len(result.columns) == 10


def test_build_excludes_name_columns(handoff_dir: Path):
    """Output should not include name_a / name_b — only key + features."""
    pairs = load_pairs_with_names(handoff_dir, DEFAULT_RUN_ID)
    result = build_string_features(pairs)
    assert "name_a" not in result.columns
    assert "name_b" not in result.columns


def test_build_empty_input_preserves_output_schema():
    pairs = pl.DataFrame(
        schema={
            "run_id": pl.Utf8,
            "entity_id_a": pl.Utf8,
            "entity_id_b": pl.Utf8,
            "name_a": pl.Utf8,
            "name_b": pl.Utf8,
        }
    )
    result = build_string_features(pairs)
    assert len(result) == 0
    assert result.columns == ["run_id", "entity_id_a", "entity_id_b", *STRING_FEATURE_COLUMNS]
