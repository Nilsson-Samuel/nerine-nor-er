# Annotation Tooling — Quick Start

This README summarizes the end-to-end annotation workflow for the Nerine project. It covers creating synthetic case folders, preparing them for Label Studio, reviewing annotations, and exporting gold-standard data.


## Overview of the Pipeline

```
1. Create case folder  →  2. Prepare for Label Studio  →  3. Annotate in Label Studio  →  4. Flatten export to gold CSV  →  5. (Optional) LLM draft for group_id
```

## PIVCOLF ontology
The entity types follow the PIVCOLF ontology:
- PER — Individual persons (names, initials, aliases referring to one person)
- ORG — Organizations (companies, institutions, agencies)
- LOC — Locations (places, addresses, buildings, rooms, properties)
- ITEM — Physical objects (identifiable items relevant in context)
- VEH — Vehicles (vehicle types or registration identifiers)
- COMM — Communication identifiers (phone numbers, emails, usernames)
- FIN — Financial identifiers (account numbers, financial IDs)

---

## Step 0 — Create a New Case Folder

Every annotation run starts with a **new** case folder under `data/raw/` with a new `case_id`. Never modify an existing case.
**The rest of step 0 is intended for creating of new synthetic LLM generated cases. Not needed for existing real cases**

 
**Key rules:**

- Use 4 documents per case
- Use Norwegian filenames and Norwegian police-style text
- Keep documents compact (roughly 1 page each)
- Repeat `PER`, `ORG`, and `LOC` mentions across documents; include `ITEM` where natural
- `VEH`, `COMM`, and `FIN` can be added when they fit naturally, but should not dominate the case.
- Each case folder must contain: `brief.yaml`, `source-notes.md`, and 4 `.docx` files

**Generate a document scaffold:**

```bash
python3 scripts/pick_case_doc_plan.py           # random
python3 scripts/pick_case_doc_plan.py --seed 42  # reproducible
```

This prints a 4-document JSON plan (always starting with `01-anmeldelse.docx`) that you paste into `brief.yaml` as `doc_plan`. The script does **not** create the folder or write `.docx` files — you do that yourself.

**Recommended order:**

1. Create a new folder under `data/raw/` with a new `case_id`. 
2. Run `pick_case_doc_plan.py` and paste the plan into `brief.yaml`. 
3. Write `source-notes.md` with entity notes and a compact source summary.
4. Write the 4 `.docx` documents using the generated filenames and genres.
5. Proceed to Step 1 only after the case folder is complete.

---

## Step 1 — Prepare a Case for Label Studio

```bash
python3 scripts/prepare_label_studio_case.py <case_root>
```

Optional flags:

```bash
--seed-patterns <patterns.json>   # extra seed patterns beyond brief.yaml
--output-dir <custom_output_dir>  # override default output location
```

**What it expects** inside `<case_root>`:

- `brief.yaml`
- `docs/*.docx`

**What it produces** under `<case_root>/annotation/`:

| File | Description |
|---|---|
| `label_config.xml` | Label Studio project config |
| `label_studio_import.json` | Plain task payload (no pre-annotations) |
| `label_studio_predictions.json` | Task payload with seeded pre-annotations |
| `doc_manifest.json` | Document metadata |
| `extracted_text/*.txt` | Plain text extracted from each DOCX |

The script parses `brief.yaml` aliases (and optional JSON patterns) to find seed matches in the extracted text, producing pre-annotations for: **PER, ORG, LOC, ITEM, VEH, COMM, FIN**.

For real cases, skip the synthetic/LLM case-generation parts. The script only needs:

- `brief.yaml` with at least `case_id`
- `docs/*.docx` with the real case documents

Optional, but useful:

- `core_entities` in `brief.yaml` if known aliases should become seeded pre-annotations
- `annotation_seed_patterns.json` or `--seed-patterns <patterns.json>` for extra seed phrases

Not needed for real cases:

- LLM-generated synthetic documents
- `pick_case_doc_plan.py`
- The 4-document synthetic case rule
- `source-notes.md`, unless the team wants private notes about the case

Minimal real-case brief.yaml:


```text case_id: crime_case_001``` 

If core_entities is missing, the script still works; it just creates blank Label Studio tasks without seeded pre-annotations. Label Studio is not required for evaluation. It is just the current tool for creating/reviewing span labels and then flattening them into the project’s gold CSV format.

For evaluation, what matters is the reviewed gold CSV. (more on that below)

---

## Step 2 — Annotate in Label Studio

1. Create a new Label Studio project for span labeling.
2. Upload `label_config.xml` as the labeling config.
3. Import `label_studio_predictions.json` (includes pre-annotations) — or `label_studio_import.json` if you want a blank start.
4. Review and correct annotations in the Label Studio UI.

---

## Step 3 — Flatten Export to Gold CSV

After reviewing annotations in Label Studio, export and flatten:

```bash
python3 scripts/flatten_label_studio_export.py <export_path>
```

Useful flags:

```bash
--validate-only                          # check without writing output
--output <gold_csv_path>                 # custom output path
--drop-empty-zero-length-labels          # discard zero-length spans
--normalized-export-output <norm.json>   # write a normalized copy of the export
```

**Default output:** `<export dir>/gold_annotations.csv`

**Validation checks performed:**

- Each task has exactly one active annotation
- Offsets match the document text exactly
- Each span has exactly one entity type
- No duplicate or overlapping spans
- The export contains exactly one case

**Gold CSV columns:**

```
case_id, doc_id, doc_name, mention_id, char_start, char_end, text, entity_type, group_id, canonical_text, notes
```

---

## Step 4 (Optional) — LLM Draft for `group_id`

Use an LLM to produce a **review draft** that fills `group_id`, `canonical_text`, and `notes` while keeping all other columns unchanged.

**Required inputs:**

- `annotation/gold_annotations.csv` — the mention list
- `annotation/label_studio_export*.flattenable.json` — full document text
- `brief.yaml` — known entities, aliases, preferred IDs

**Output:** a new file, e.g. `annotation/gold_annotations.group_id_draft.csv` (never overwrite the original).

**`group_id` convention:**

- Known entity from `brief.yaml` → use that ID (e.g. `per_001`, `org_003`)
- Uncertain → use a draft ID (e.g. `per_draft_001`, `org_draft_001`)

**Confidence notes** — each `notes` entry should start with a confidence tag:

- `high:` — clear match (e.g. "exact brief alias for Hampton Parish Sheriff's Office")
- `medium:` — likely match needing context (e.g. "surname-only, resolved from nearby sentence")
- `low:` — ambiguous (e.g. "short form ambiguous between Dana Parker and the firm")

**Key ambiguity checks:**

- Surname ambiguity (e.g. `Murdoch` could match multiple people)
- Person vs. organization (e.g. `Parker`)
- Similar location names (e.g. `River Road` vs. `River Kennels`)
- Org alias chains (e.g. `SID` / `State Investigation Division`)

**Review priority:** focus on `low` and `medium` confidence rows, ambiguous surnames, and ambiguous org short-forms. When in doubt, the LLM should create a separate draft group rather than force an aggressive merge.
