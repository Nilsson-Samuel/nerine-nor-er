You are generating synthetic identity groups for matching-stage training data. 

Return exactly one JSON object and nothing else.
Do not use markdown.
Do not use code fences.
Do not add explanation text before or after the JSON.

Output file target: identity_groups_llm_draft.json

Strict output contract
1) Root object must contain exactly these keys:
   - run_id
   - groups
   - hard_negatives
2) No extra root keys allowed.
3) run_id must be lowercase, trimmed, non-empty, format: synthetic_run_### (example: synthetic_run_001).
4) groups must be a non-empty array.
5) hard_negatives must be an array.

Allowed entity types
- PER
- ITEM
- COMM
- VEH
- LOC
- ORG
- FIN

Quota targets (group counts)
- PER: 24-30
- ITEM: 13-17
- COMM: 5-7
- VEH: 5-7
- LOC: 3-4
- ORG: 1-3
- FIN: 1-2
- Total groups target: 52-70

Each group object rules
1) Required keys exactly:
   - group_id
   - entity_type
   - variants
   - doc_ids
   - metadata
2) group_id:
   - unique across all groups
   - lowercase snake_case
   - should start with type prefix (per_, item_, comm_, veh_, loc_, org_, fin_)
3) entity_type:
   - one of the allowed types above
4) variants:
   - array length 2-4
   - each variant must contain exactly:
     - text
     - normalized
     - context
   - text must be non-empty
   - normalized must be lowercase and trimmed
   - context must be non-empty and realistic Norwegian police-report style
5) doc_ids:
   - array length >= 2
   - each value non-empty
6) metadata:
   - object with at least 1 key
   - PER should include fnr when realistic
   - VEH should include regnr when relevant
   - ORG should include orgnr when relevant
   - FIN should include account-like reference when relevant

Hard negatives rules
1) Count must be 8-12.
2) Each object must contain exactly:
   - group_id_a
   - group_id_b
   - reason
3) group_id_a and group_id_b must reference existing groups and must differ.
4) Use difficult negatives: similar surface forms but different real identities.
5) Prefer same-type hard negatives.
6) No duplicate unordered pairs.

Quality rules
1) Use realistic Norwegian names, places, items, communication identifiers, vehicles, organizations, and financial references.
2) Include natural formatting variation per type:
   - PER: initials, reversed order, punctuation variants
   - COMM: phone/account/handle formatting variants
   - VEH: plate/model formatting variants
   - ORG: acronym/full-name variants
   - FIN: account formatting variants
3) Keep all references synthetic and non-sensitive.

Expected size
- Approximate output size: 40-100 KB JSON.
- Typical variant count: 120-260 total variant rows.

Failure mode
- If you cannot satisfy all constraints, return only:
  {"error":{"code":"CONSTRAINT_UNSATISFIED","reason":"<short reason>"}}

Example shape (abbreviated)
{
  "run_id": "synthetic_run_001",
  "groups": [
    {
      "group_id": "per_hansen_01",
      "entity_type": "PER",
      "variants": [
        {
          "text": "Per Hansen",
          "normalized": "per hansen",
          "context": "Per Hansen ble avhørt vedrørende ..."
        },
        {
          "text": "P. Hansen",
          "normalized": "p hansen",
          "context": "Vitnet P. Hansen forklarte at ..."
        }
      ],
      "doc_ids": ["doc_per_01", "doc_per_21"],
      "metadata": {"fnr": "12025545226"}
    }
  ],
  "hard_negatives": [
    {
      "group_id_a": "per_hansen_01",
      "group_id_b": "per_hansen_02",
      "reason": "Same name pattern, different fødselsnummer"
    }
  ]
}
