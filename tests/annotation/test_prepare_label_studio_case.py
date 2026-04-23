from __future__ import annotations

from pathlib import Path

from scripts.prepare_label_studio_case import parse_brief_yaml, parse_inline_list


def test_parse_inline_list_preserves_commas_inside_quotes() -> None:
    assert parse_inline_list('["ACME, Inc.", Other]') == ["ACME, Inc.", "Other"]


def test_parse_brief_yaml_preserves_aliases_with_commas(tmp_path: Path) -> None:
    brief_path = tmp_path / "brief.yaml"
    brief_path.write_text(
        "\n".join(
            [
                "case_id: case_demo_01",
                "core_entities:",
                "  - id: org_001",
                "    type: ORG",
                '    canonical: "ACME, Inc."',
                '    aliases: ["ACME, Inc.", ACME]',
            ]
        ),
        encoding="utf-8",
    )

    case_id, entities = parse_brief_yaml(brief_path)

    assert case_id == "case_demo_01"
    assert len(entities) == 1
    assert entities[0].aliases == ["ACME, Inc.", "ACME"]
