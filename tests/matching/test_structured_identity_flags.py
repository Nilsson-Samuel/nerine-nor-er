"""Unit tests for structured identity feature helpers and builder."""

import polars as pl

from src.matching.features import (
    STRUCTURED_IDENTITY_FEATURE_COLUMNS,
    build_structured_identity_features,
    first_name_match,
    last_name_match,
    norwegian_id_match,
)


def test_norwegian_id_match_positive_overlap() -> None:
    context_a = "Person has id 12345678901 and lives in Oslo."
    context_b = "Seen with id 12345678901 in passport metadata."
    assert norwegian_id_match(context_a, context_b) == 1


def test_norwegian_id_match_no_overlap() -> None:
    context_a = "Context with 12345678901 only."
    context_b = "Context with 10987654321 only."
    assert norwegian_id_match(context_a, context_b) == 0


def test_norwegian_id_match_missing_values() -> None:
    assert norwegian_id_match(None, "12345678901") == 0  # type: ignore[arg-type]
    assert norwegian_id_match("", "") == 0


def test_first_name_match_per_positive() -> None:
    assert first_name_match("Per Hansen", "per johansen", "PER", "PER") == 1


def test_first_name_match_per_negative() -> None:
    assert first_name_match("Per Hansen", "Anders Hansen", "PER", "PER") == 0


def test_first_name_match_non_per_forced_zero() -> None:
    assert first_name_match("DNB ASA", "DNB Bank", "ORG", "ORG") == 0


def test_last_name_match_per_positive() -> None:
    assert last_name_match("Per Hansen", "Anders hansen", "PER", "PER") == 1


def test_last_name_match_per_negative() -> None:
    assert last_name_match("Per Hansen", "Per Johansen", "PER", "PER") == 0


def test_last_name_match_non_per_forced_zero() -> None:
    assert last_name_match("DNB ASA", "Nordea ASA", "ORG", "ORG") == 0


def test_build_structured_identity_features_values_and_order() -> None:
    pairs = pl.DataFrame(
        {
            "context_a": [
                "id 12345678901 in report",
                "id 12345678901 in report",
                "no ids here",
            ],
            "context_b": [
                "duplicate id 12345678901 in witness note",
                "different id 10987654321 in witness note",
                "still no ids",
            ],
            "name_a": ["Per Hansen", "Per Hansen", "DNB ASA"],
            "name_b": ["Per Hansen", "Anders Johansen", "Nordea ASA"],
            "entity_type_a": ["PER", "PER", "ORG"],
            "entity_type_b": ["PER", "PER", "ORG"],
        }
    )

    result = build_structured_identity_features(pairs)

    assert result.columns == STRUCTURED_IDENTITY_FEATURE_COLUMNS
    assert result["norwegian_id_match"].to_list() == [1, 0, 0]
    assert result["first_name_match"].to_list() == [1, 0, 0]
    assert result["last_name_match"].to_list() == [1, 0, 0]


def test_build_structured_identity_features_binary_int_outputs() -> None:
    pairs = pl.DataFrame(
        {
            "context_a": ["id 12345678901"],
            "context_b": ["id 12345678901"],
            "name_a": ["Per Hansen"],
            "name_b": ["Per Hansen"],
            "entity_type_a": ["PER"],
            "entity_type_b": ["PER"],
        }
    )
    result = build_structured_identity_features(pairs)
    for col in STRUCTURED_IDENTITY_FEATURE_COLUMNS:
        values = set(result[col].to_list())
        assert values <= {0, 1}


def test_build_structured_identity_features_empty_input_schema() -> None:
    pairs = pl.DataFrame(
        schema={
            "context_a": pl.Utf8,
            "context_b": pl.Utf8,
            "name_a": pl.Utf8,
            "name_b": pl.Utf8,
            "entity_type_a": pl.Utf8,
            "entity_type_b": pl.Utf8,
        }
    )
    result = build_structured_identity_features(pairs)
    assert len(result) == 0
    assert result.columns == STRUCTURED_IDENTITY_FEATURE_COLUMNS
