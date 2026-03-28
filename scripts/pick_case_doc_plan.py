#!/usr/bin/env python3
"""Pick a small 4-document scaffold variation for a new ER case."""

from __future__ import annotations

import argparse
import json
import random


DOC_1 = {
    "doc_id": "doc_01",
    "filename": "01-anmeldelse.docx",
    "genre": "anmeldelse / rapport",
    "focus": "innledende hendelse, første respons og navngitte aktører",
}

DOC_2_OPTIONS = [
    {
        "doc_id": "doc_02",
        "filename": "02-vitneavhoer.docx",
        "genre": "vitneavhør",
        "focus": "etternavn alene og overlappende familiekontekst",
    },
    {
        "doc_id": "doc_02",
        "filename": "02-naboforklaring.docx",
        "genre": "naboforklaring",
        "focus": "delvise navn, indirekte observasjoner og lokasjonsvariasjon",
    },
    {
        "doc_id": "doc_02",
        "filename": "02-kollegaavhoer.docx",
        "genre": "kollegaavhør",
        "focus": "organisasjonsaliaser, rollebaserte omtaler og tidslinjebrudd",
    },
    {
        "doc_id": "doc_02",
        "filename": "02-patruljesammendrag.docx",
        "genre": "patruljesammendrag",
        "focus": "åstedsobservasjoner, kortformer og ordlyd fra første respons",
    },
]

DOC_3_OPTIONS = [
    {
        "doc_id": "doc_03",
        "filename": "03-mistenktavhoer.docx",
        "genre": "mistenktavhør",
        "focus": "selvforklarende tidslinje, tvetydige referanser og benektelser",
    },
    {
        "doc_id": "doc_03",
        "filename": "03-avhoer-person-av-interesse.docx",
        "genre": "avhør av person av interesse",
        "focus": "uklar involvering, delvise mentions og motstridende detaljer",
    },
]

DOC_4_OPTIONS = [
    {
        "doc_id": "doc_04",
        "filename": "04-etterforskningsnotat.docx",
        "genre": "etterforskningsnotat",
        "focus": "motstrid på tvers av dokumenter, aliaskjeder og sentrale item-omtaler",
    },
    {
        "doc_id": "doc_04",
        "filename": "04-tidslinjesammendrag.docx",
        "genre": "tidslinjesammendrag",
        "focus": "hendelsesrekkefølge, tidsreferanser og gjentatte lokasjonsvarianter",
    },
    {
        "doc_id": "doc_04",
        "filename": "04-beslagsoppsummering.docx",
        "genre": "beslagsoppsummering",
        "focus": "item-omtaler, enheter, kjøretøy og aliaser i støttedokumenter",
    },
    {
        "doc_id": "doc_04",
        "filename": "04-kommunikasjonslogg.docx",
        "genre": "kommunikasjonslogg",
        "focus": "anrop, meldinger, brukernavn og tvetydighet mellom person og organisasjon",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    plan = [
        DOC_1,
        rng.choice(DOC_2_OPTIONS),
        rng.choice(DOC_3_OPTIONS),
        rng.choice(DOC_4_OPTIONS),
    ]
    print(json.dumps(plan, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
