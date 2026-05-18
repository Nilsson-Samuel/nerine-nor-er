"""Generate a tiny Norwegian-style demo case for end-to-end pipeline testing.

Writes a handful of short PDF and DOCX files into <out_dir>, each containing
plausible-looking investigative text with several entities that recur across
documents under different surface forms. The point is to give the pipeline
something realistic to ingest, extract, block, match, and resolve so the
HITL UI shows non-empty clusters.

Run inside the Docker image:
    docker compose run --rm pipeline \
        python scripts/make_demo_case.py --out-dir /app/data/raw/demo_case
"""

from __future__ import annotations

import argparse
from pathlib import Path

import fitz  # PyMuPDF
from docx import Document


# Each tuple: (filename, body_text). The same entities recur across documents
# under different surface forms so blocking + matching has something to do.
DOCUMENTS: list[tuple[str, str]] = [
    (
        "rapport_2024_001.pdf",
        """POLITIRAPPORT - Sak 2024/001

Anmelder: Ola Nordmann, født 12.03.1981, bosatt i Storgata 14, 0182 Oslo.
Anmeldelsen gjelder mistanke om økonomisk utroskap ved DNB ASA.

Den fornærmede oppgir at det ble overført NOK 250 000 fra konto
1234.56.78901 til en konto tilhørende Kari Hansen den 4. mars 2024.
Hansen er ansatt i Den Norske Bank (DNB) ved kontoret på Aker Brygge.

Saksbehandler: Politibetjent Per Olsen, Oslo politidistrikt.
""",
    ),
    (
        "vitneavhor_kari_hansen.pdf",
        """VITNEAVHØR

Vitnet Kari Hansen, født 22.07.1989, ble avhørt 8. mars 2024 ved
Politiet i Oslo. Hansen arbeider som rådgiver i DNB og oppgir at
overføringen til hennes konto var en feilført transaksjon utført av
en kollega ved navn Lars Berg.

Hansen kjenner ikke Ola Nordmann personlig, men har sett navnet i
DNBs interne systemer i forbindelse med en lånesøknad fra 2023.

Avhørsleder: Politioverbetjent Anne Lie, Oslo politidistrikt.
""",
    ),
    (
        "epost_dnb_compliance.pdf",
        """E-POST FRA DNB COMPLIANCE

Fra: compliance@dnb.no
Til: politiet@oslo-politidistrikt.no
Emne: Sak 2024/001 - utlevering av kontoopplysninger

Vedlagt finner dere kontoutskrift for konto 1234.56.78901 tilhørende
O. Nordmann. Vi bekrefter også at Hansen, K. er ansatt ved Aker Brygge-
kontoret. Lars Berg sluttet i banken 1. februar 2024.

Med vennlig hilsen,
DNB Compliance
""",
    ),
    (
        "tilleggsrapport.docx",
        """TILLEGGSRAPPORT - Sak 2024/001

I forbindelse med etterforskningen ble en BMW X5 med registreringsnummer
EK 12345 observert utenfor adressen til Nordmann, Ola den 5. mars 2024.
Kjøretøyet er registrert på Berg, Lars.

Politibetjent Olsen, P. har gjennomført ytterligere undersøkelser ved
Oslo politidistrikt og bekrefter at både Den Norske Bank og DNB ASA
viser til samme juridiske enhet.
""",
    ),
]


def write_pdf(path: Path, body: str) -> None:
    """Write `body` into a one-page A4 PDF using PyMuPDF."""
    doc = fitz.open()
    page = doc.new_page()
    rect = fitz.Rect(50, 50, page.rect.width - 50, page.rect.height - 50)
    page.insert_textbox(rect, body, fontsize=11, fontname="helv")
    doc.save(str(path))
    doc.close()


def write_docx(path: Path, body: str) -> None:
    """Write `body` into a minimal DOCX, one paragraph per line."""
    doc = Document()
    for line in body.splitlines():
        doc.add_paragraph(line)
    doc.save(str(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, help="Where to write the demo case files.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for filename, body in DOCUMENTS:
        target = out_dir / filename
        if filename.lower().endswith(".pdf"):
            write_pdf(target, body)
        elif filename.lower().endswith(".docx"):
            write_docx(target, body)
        else:
            raise ValueError(f"unsupported extension for {filename}")
        print(f"wrote {target}")

    print(f"\nDone. {len(DOCUMENTS)} files written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
