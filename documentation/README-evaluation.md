# Evaluation Guide

This guide is for running and interpreting evaluation of the entity-resolution
pipeline on reviewed cases.

## Summary

This guide explains how to evaluate the Nerine entity-resolution pipeline against
reviewed case annotations. It covers the three main evaluation paths: a completed
single-run evaluation, held-out case-fold evaluation, and case-fold Optuna tuning
for LightGBM parameters and resolution thresholds. The safest default for
reviewed case reporting is held-out case-fold evaluation: train on reviewed
cases, hold one case completely out, then evaluate the full pipeline result on
that held-out case. This gives a more realistic signal than evaluating on the
same case used to create training labels.

The guide also defines the expected reviewed gold CSV format. The key reviewed
field is `group_id`, because final cluster metrics use it as the answer for
which mentions refer to the same real-world entity. Other fields such as
`doc_name`, `canonical_text`, and `notes` are kept for remapping, review, and
auditability even when the evaluator does not need them directly.

For normal reporting, use `scripts/run_case_fold_eval.py`. It prepares each
fold, trains a model only on train cases, scores and resolves only the held-out
case, runs evaluation, and writes fold-level plus aggregate reports. Use
`scripts/run_case_fold_tuning.py` before the final reporting run only when model
parameters or resolution thresholds are still being selected. Tuning uses the
same case/fold config, but it answers a different question: which LightGBM
parameters and resolution thresholds perform best when judged by held-out final
cluster quality.

The primary metrics to inspect are precision-weighted final cluster metrics,
especially pairwise F0.5, pairwise precision, and pairwise recall. Pairwise
precision is the clearest false-merge risk signal, while recall shows whether
the model still finds enough true links to be useful. Blocking recall should be
checked early because missed candidate pairs cannot be recovered by matching or
resolution. False merges should be reviewed before false splits because this
domain has a higher cost for confidently linking unrelated people,
organizations, accounts, or locations.

For classified or larger evaluation setups, keep one case folder per
investigation case, produce one reviewed gold CSV per case, define explicit
held-out folds, smoke-test one fold, then run all folds into a fresh output
root. Keep output roots separate per evaluation round so reports remain
reproducible and earlier runs are not overwritten.

## Table of Contents

| Section | Tiny summary |
|---|---|
| [Summary](#summary) | Quick orientation on evaluation choices, gold data, metrics, and reporting flow. |
| [Evaluation Alternatives](#evaluation-alternatives) | Compares single-run evaluation, held-out case-fold evaluation, case-fold tuning, and internal matching tuning. |
| [Relevant Code Files](#relevant-code-files) | Maps the scripts, modules, and tests that support evaluation and tuning. |
| [Gold CSV Requirements](#gold-csv-requirements) | Defines required and recommended columns for reviewed annotation files. |
| [Case-Fold Evaluation](#case-fold-evaluation) | Describes the preferred held-out evaluation workflow and fold processing steps. |
| [Config Format](#config-format) | Shows the JSON shape for cases, folds, default paths, and explicit path overrides. |
| [Environment Setup](#environment-setup) | Lists the Python environment setup needed before running evaluation commands. |
| [Run Command](#run-command) | Gives the basic case-fold evaluation command. |
| [Case-Fold Evaluation Flags](#case-fold-evaluation-flags) | Explains the main evaluator flags in priority order. |
| [Case-Fold Outputs](#case-fold-outputs) | Explains the aggregate, fold-level, and held-out case reports that get written. |
| [Relation To Optuna Tuning](#relation-to-optuna-tuning) | Separates final evaluation from parameter search and explains the tuning objective. |
| [Why This Optuna Target](#why-this-optuna-target) | Explains why Optuna optimizes held-out final-cluster pairwise F-beta. |
| [Tuning Workflow And Search Space](#tuning-workflow-and-search-space) | Explains when to tune, trial counts, resolution-threshold search, and study cost. |
| [Case-Fold Tuning Command](#case-fold-tuning-command) | Gives the recommended command shape for a case-fold Optuna study. |
| [Case-Fold Tuning Flags](#case-fold-tuning-flags) | Explains tuning flags, persistent storage, recall guardrails, and thresholds. |
| [Single-Run Evaluation](#single-run-evaluation) | Shows how to evaluate one already completed pipeline run against one gold CSV. |
| [Single-Run Evaluation Flags](#single-run-evaluation-flags) | Lists the required and optional flags for `src.evaluation.run`. |
| [Pipeline Flags](#pipeline-flags) | Lists the pipeline flags usually needed before a single-run evaluation. |
| [Metrics To Read First](#metrics-to-read-first) | Prioritizes pairwise, B-cubed, blocking, matching, extraction, and alignment metrics. |
| [Large Classified Dataset Workflow](#large-classified-dataset-workflow) | Gives the recommended larger-case workflow from case folders to clean full evaluation. |
| [Interpretation Rules](#interpretation-rules) | States what must be included when reporting results and reviewing errors. |

## Evaluation Alternatives

| Alternative | Main entrypoint | Use when | Limitation |
|---|---|---|---|
| Single completed-run evaluation | `src/evaluation/run.py` | One pipeline run already exists and needs metrics against one reviewed gold CSV. | Not a train/test split by itself. If the model was trained on the same case, results can be optimistic. |
| Held-out case-fold evaluation | `scripts/run_case_fold_eval.py` | Several reviewed cases exist and one case should be evaluated without being used for training. | Needs reviewed `group_id` values and enough cases to train every fold. |
| Case-fold Optuna tuning | `scripts/run_case_fold_tuning.py` + `src/matching/fold_tuning.py` | Tune LightGBM parameters and resolution thresholds against held-out case quality. | More computationally expensive than evaluation because every trial trains, scores, resolves, and evaluates folds. |
| Internal matching Optuna | `src/matching/tuning.py` through `src/matching/run.py` | Quick tuning on labeled pair rows inside one run. | Optimizes pairwise validation labels, not final held-out cluster quality. Less relevant for case-held-out generalization. |

For reviewed case reporting, prefer `scripts/run_case_fold_eval.py`, and use
`scripts/run_case_fold_tuning.py` for selecting model parameters and resolution
thresholds before a final evaluation run.

## Relevant Code Files

| File | When it is needed |
|---|---|
| `scripts/run_case_fold_eval.py` | Main case-fold evaluator. Use this when several reviewed case folders exist and held-out metrics are needed. |
| `scripts/run_case_fold_tuning.py` | CLI wrapper for case-fold Optuna tuning. Use before final evaluation if tuning LightGBM parameters or resolution thresholds. |
| `src/matching/fold_tuning.py` | Implements the Optuna objective: train fold model, score held-out case, run resolution with trial thresholds, evaluate, then combine held-out pairwise F-beta across folds with a geometric mean. |
| `src/matching/fold_preparation.py` | Prepares reusable per-case artifacts for tuning: ingestion, extraction, blocking, matching features, and gold-derived labels. |
| `src/matching/fold_training.py` | Loads labeled rows from train cases, trains a fold-specific LightGBM model, and writes fold summaries. |
| `src/evaluation/run.py` | Converts reviewed gold CSV rows into labels and computes extraction, blocking, matching, and final clustering metrics. |
| `src/evaluation/metrics.py` | Metric definitions for mention, pairwise, ARI/NMI, and B-cubed scores. |
| `src/evaluation/error_analysis.py` | False-merge, false-split, and extraction-error summaries in evaluation reports. |
| `src/pipeline.py` | Single-case end-to-end pipeline runner when evaluation is not using the fold runner. |
| `src/shared/paths.py` | Explains the per-run output layout under `data_dir/runs/rid_<encoded_run_id>/...`. |
| `tests/matching/test_case_fold_tuning.py` | Safety checks for case-fold tuning, persistent studies, stale best params, and input fingerprinting. |
| `tests/evaluation/test_run.py` | Evaluation bridge and report behavior, including required gold columns and label generation. |

## Gold CSV Requirements

The evaluation code reads gold annotations from a CSV. The reviewed file should
normally be named like this inside each case folder:

```text
data/raw/<case_name>/annotation/gold_annotations.group_id_reviewed.csv
```

The key point is that `group_id` must be human-reviewed, because the cluster metrics treat `group_id` as the
reference answer for which mentions refer to the same real-world entity.

The full gold CSV shape is:

```text
case_id, doc_id, doc_name, mention_id, char_start, char_end, text, entity_type, group_id, canonical_text, notes
```

Not every column is used directly by `src/evaluation/run.py`, but the reviewed
CSV should keep all of them when possible.

| Column | Needed by code? | Keep it? | Why |
|---|---:|---:|---|
| `doc_id` | Yes | Yes | Primary document key. The evaluator can use it directly when it matches the pipeline run. |
| `mention_id` | Yes | Yes | Stable mention identity for deterministic matching and sorting. |
| `char_start` | Yes | Yes | Start offset for span matching and text remapping. |
| `char_end` | Yes | Yes | End offset for span matching and text remapping. |
| `text` | Yes | Yes | Exact mention text. Also used as fallback when offsets need remapping to the pipeline's normalized text. |
| `entity_type` | Yes | Yes | Keeps PER, ORG, LOC, ITEM, VEH, COMM, and FIN evaluations type-safe. |
| `group_id` | Yes | Yes | The gold entity identity. This is the most important reviewed field for entity-resolution evaluation. |
| `doc_name` | Optional in code, highly useful | Yes | Allows the evaluator to remap gold `doc_id` values to run-local document IDs when filenames match. This is important if document IDs are regenerated in a separate environment. |
| `case_id` | No | Yes | Operational provenance. Useful when combining exports, auditing cases, or checking that one CSV belongs to one case. |
| `canonical_text` | No | Yes | Human-readable group name for review and debugging. Helps reviewers inspect false merges and false splits. |
| `notes` | No | Yes | Audit trail for ambiguous decisions. Useful when a classified dataset cannot be shared back to the project team. |

Minimum technical CSV for `src/evaluation/run.py`:

```text
doc_id, mention_id, char_start, char_end, text, entity_type, group_id
```

Recommended reviewed CSV:

```text
case_id, doc_id, doc_name, mention_id, char_start, char_end, text, entity_type, group_id, canonical_text, notes
```

## Case-Fold Evaluation

`scripts/run_case_fold_eval.py` is the main held-out case evaluator. For each
fold it does this:

1. Runs ingestion, extraction, blocking, and matching feature generation for
   every case in that fold.
2. Converts train-case reviewed gold CSV files into `labels.parquet`.
3. Trains one fold-specific LightGBM model on the train cases only.
4. Scores only the held-out case with that frozen model.
5. Runs resolution on the held-out case.
6. Runs `src/evaluation/run.py` against the held-out case gold CSV.
7. Writes one fold report and one aggregate report across all selected folds.

The held-out case must not appear in `train_cases`. The runner rejects duplicate
train cases and folds that reference unknown cases.

### Config Format

The case-fold runner expects one JSON config:

```json
{
  "source_root": "data/raw",
  "cases": {
    "case_a": {},
    "case_b": {},
    "case_c": {}
  },
  "folds": [
    {
      "name": "fold_case_a_held_out",
      "held_out_case": "case_a",
      "train_cases": ["case_b", "case_c"]
    }
  ]
}
```

With empty case objects, the runner uses:

```text
data/raw/<case_name>/annotation/gold_annotations.group_id_reviewed.csv
```

Use explicit paths only when evaluation material is stored outside that layout:

```json
{
  "source_root": "/secure/eval/raw",
  "cases": {
    "classified_case_01": {
      "case_root": "/secure/eval/raw/classified_case_01",
      "gold_path": "/secure/eval/raw/classified_case_01/annotation/gold_annotations.group_id_reviewed.csv"
    }
  },
  "folds": []
}
```

### Environment Setup

Run the evaluator from a Python environment with the project dependencies
installed. One conda-based setup is:

```bash
conda create -n nerine-eval python=3.12 -y
conda activate nerine-eval
python -m pip install --upgrade pip
python -m pip install -r requirements-ci.txt
```

Run these commands from the repository root so the relative paths below resolve
correctly.

### Run Command

```bash
python3 scripts/run_case_fold_eval.py case_folds.json \
  --output-root data/cv-reviewed
```

### Case-Fold Evaluation Flags

| Flag | Importance | Description |
|---|---:|---|
| `config` | 1 | Required JSON file defining cases and folds. This controls train/held-out separation. |
| `--output-root` | 2 | Root directory for fold outputs. Use a fresh path per evaluation round, for example `data/cv-reviewed-2026-04`. |
| `--fold-name` | 3 | Run only selected folds from a larger config. Repeat it to run several named folds. |
| `--match-threshold` | 4 | Score threshold for held-out matching evaluation. Default comes from `src.shared.config.PAIR_MATCH_THRESHOLD`. Use only when deliberately testing threshold sensitivity. |
| `--keep-score-threshold` | 5 | Resolution threshold for keeping scored pairs as retained graph edges. Default comes from `src.shared.config.KEEP_SCORE_THRESHOLD`. Use tuned values for the final clean reporting run. |
| `--objective-neutral-threshold` | 6 | Resolution threshold where retained edges become merge-positive in the correlation-clustering objective. Must be at least `0.10` above `--keep-score-threshold`. Use tuned values for the final clean reporting run. |
| `--enable-shap` | 7 | Generate SHAP explanations for held-out scoring. Useful for audit examples, but slower and not required for aggregate metrics. |

### Case-Fold Outputs

| Output | Scope | What it contains |
|---|---|---|
| `<output-root>/fold_reports.md` | All folds | Human-readable overview across folds, including final clustering metrics, resolution thresholds, and macro averages. Good for reading or sharing. |
| `<output-root>/fold_reports.csv` | All folds | Main machine-readable summary table for comparing held-out results across folds, including resolution thresholds and retained-graph diagnostics. Read this first for exact values. |
| `<output-root>/fold_reports.json` | All folds | Machine-readable aggregate metadata, resolution thresholds, and fold report payload. Useful for scripts or archiving exact values. |
| `<output-root>/<fold_name>/fold_summary.md` | One fold | Human-readable fold report with train cases, held-out case, final clustering metrics, resolution threshold context, stage context, and links to detailed reports. |
| `<output-root>/<fold_name>/fold_summary.json` | One fold | Machine-readable fold setup and high-level run metadata, including train cases, held-out case, and output locations. |
| `<output-root>/<fold_name>/fold_metrics.csv` | One fold | Machine-readable per-fold metrics in table form. Useful when inspecting one held-out case in detail. |
| `<output-root>/<fold_name>/reranker_model.txt` | One fold | The LightGBM model trained on that fold's train cases. |
| `<output-root>/<fold_name>/reranker_model_metadata.json` | One fold | Model metadata and training context for the fold-specific reranker. |
| `<output-root>/<fold_name>/<held_out_case>/runs/rid_<encoded_run_id>/evaluation/evaluation_report.md` | Held-out case | Human-readable detailed evaluation report with inputs, scope, final metrics, stage metrics, retained-graph diagnostics, regression checks, and error examples. |
| `<output-root>/<fold_name>/<held_out_case>/runs/rid_<encoded_run_id>/evaluation/evaluation_report.json` | Held-out case | Detailed evaluation report with metrics, alignment details, resolution diagnostics, false merges, false splits, and extraction errors. |
| `<output-root>/<fold_name>/<held_out_case>/runs/rid_<encoded_run_id>/evaluation/labels.parquet` | Held-out case | Gold-derived labels aligned to the evaluated run. Useful for debugging and later controlled experiments. |

Read `fold_reports.md` for a quick overview and `fold_reports.csv` for exact
cross-fold values. Treat the JSON, CSV, and Parquet files as the
machine-readable source of truth, then inspect the held-out detailed evaluation
report for false merges, false splits, and extraction errors.

To inspect the resolved clusters of a held-out case interactively, point the
HITL Streamlit app at that case's data directory. See
`documents/README-hitl.md` for the launch command, sidebar
controls, and cluster inspector fields.

## Relation To Optuna Tuning

Case-fold evaluation and case-fold Optuna tuning use the same case/fold config,
but they answer different questions.

`scripts/run_case_fold_eval.py` answers:

```text
How well does the current pipeline perform when each case is held out?
```

`scripts/run_case_fold_tuning.py` answers:

```text
Which LightGBM parameters and resolution thresholds work best when judged by held-out final cluster quality?
```

The tuning objective lives in `src/matching/fold_tuning.py`. For each Optuna
trial it:

1. Suggests LightGBM parameters through `src/matching/tuning.py` and a valid
   resolution threshold pair.
2. Trains one fold model on the train cases.
3. Scores the held-out case.
4. Runs resolution.
5. Runs `src/evaluation/run.py`.
6. Computes held-out `pairwise_f_beta` from final cluster pairwise precision
   and recall.
7. Returns the macro-average `pairwise_f_beta` across folds.

This is important: case-fold tuning optimizes final resolved clusters, not only
pair-level classifier validation. That makes it more relevant for deployment than
the internal matching Optuna path when the goal is real case generalization.

By default, case-fold tuning uses pairwise F0.5. Pairwise precision directly
measures false-merge risk: among predicted same-cluster pairs, how many are
truly same-entity pairs. F0.5 keeps that precision focus while still penalizing
low recall. Use `--pairwise-beta` to adjust the precision/recall tradeoff.

### Why This Optuna Target

We use held-out final-cluster `pairwise_f_beta` as the Optuna target because it
matches the main risk in this project: false merges are more serious than false
splits. The metric is calculated from final resolved clusters, not just from the
LightGBM pair classifier. For each held-out fold, evaluation first compares all
predicted same-cluster entity pairs against the reviewed `group_id` pairs, then
computes pairwise precision and recall. This pair view is useful for tuning because
false merges show up directly as wrong predicted pairs.

The score is then calculated as:

```text
F_beta = (1 + beta^2) * precision * recall / ((beta^2 * precision) + recall)
```

With the default beta `0.5`, precision is weighted higher than recall. This
means Optuna prefers settings that avoid linking unrelated entities, while still
penalizing settings that become too conservative and miss many true links. The
final Optuna value is the macro-average of this score across held-out folds, so
one strong case cannot hide weak generalization on another case.

Pairwise precision and recall are computed by converting both reviewed gold
groups and predicted clusters into same-entity pairs. A cluster with three
mentions creates three pairs; a cluster with two mentions creates one pair.
Precision measures how many predicted same-cluster pairs are correct. Recall
measures how many reviewed same-entity pairs were recovered.

### Future Work: Weighted Optuna Target

A possible refinement is to extend the per-fold objective from pairwise F0.5
alone to a weighted composite:

    T = GeoMean_over_folds[ 0.5 * pairwise_F0.5 + 0.5 * ELM_F0.5 ]

ELM is a B-cubed-style element-centric score that does not count a mention's
match with itself. Singletons and tiny-cluster mentions, which are trivially
easy for entity resolution, no longer inflate the score. This forces the tuner
to do well on both large hub clusters (captured by pairwise, which is
quadratic in cluster size) and the long tail of small clusters (captured by
ELM, which is linear in cluster size).

Equal 0.5 / 0.5 weighting is the intended default. Pairwise already encodes
"hubs matter more" through its quadratic scaling; weighting pairwise higher
externally would double-count that preference. ELM is meant to act as a
binding counter-signal, not a supplementary hub signal.

This refinement is out of scope for the current implementation but is a
reasonable next step for datasets with many tiny clusters, or when tuning
produces a model that looks pairwise-strong but B-cubed-weak.


### Tuning Workflow And Search Space

Run tuning before the final held-out evaluation if model parameters or
resolution thresholds are still being selected. After choosing parameters and
thresholds, run `scripts/run_case_fold_eval.py` again on a clean output root for
the result you want to report.

The current search space tunes seven LightGBM parameters plus two resolution
thresholds. Use `2` trials only for smoke testing. For a real study, start
around `50` trials if compute is limited; use roughly `180-270` trials when
possible, which is about `20-30` trials per tuned parameter. Remember that each
trial runs every selected fold, so total cost is approximately
`n_trials * fold_count`.

Resolution threshold tuning is part of case-fold tuning by default. The search
space is:

```text
keep_score_threshold: 0.45 to 0.75
objective_neutral_threshold: 0.65 to 0.90
```

Only valid pairs are evaluated:

```text
objective_neutral_threshold >= keep_score_threshold + 0.10
```

The internal matching Optuna path does not tune these thresholds because it does
not run resolution inside its objective.

### Case-Fold Tuning Command

```bash
python3 scripts/run_case_fold_tuning.py case_folds.json \
  --output-root data/cv-tuning-reviewed \
  --mode study \
  --n-trials 120 \
  --pairwise-beta 0.5 \
  --min-pairwise-recall 0.70 \
  --storage sqlite:///data/cv-tuning-reviewed/optuna.db
```

### Case-Fold Tuning Flags

| Flag | Importance | Description |
|---|---:|---|
| `config` | 1 | Required JSON file with the same case/fold shape as `scripts/run_case_fold_eval.py`. |
| `--output-root` | 2 | Directory for prepared artifacts, trial outputs, and reports. If omitted, a timestamped child is created under `data/cv_tuning`. |
| `--n-trials` | 3 | Number of Optuna trials. Use `2` for smoke testing, around `50` for a constrained initial study, and roughly `180-270` for a fuller study of the current nine-parameter search space. |
| `--mode` | 4 | `smoke` or `study`. This is recorded in reports. Use `study` for real tuning. |
| `--fold-name` | 5 | Restrict tuning to selected folds. Repeat for several folds. Useful for smoke tests or rerunning one failed fold. |
| `--pairwise-beta` | 6 | F-beta parameter for the pairwise tuning objective. Default `0.5` weights precision higher than recall. Lower values are more precision-heavy; `1.0` is normal F1. |
| `--min-pairwise-recall` | 7 | Optional usefulness guardrail. Best params are trusted only if every held-out fold reaches this pairwise recall minimum. |
| `--storage` | 8 | Optional Optuna storage URL, for example SQLite. Use this for persistent tuning and dashboard inspection. |
| `--study-name` | 9 | Stable Optuna study name for persistent storage. Default is `case_fold_lightgbm`. |
| `--match-threshold` | 10 | Score threshold used during held-out evaluation inside each trial. Keep default unless testing threshold policy. |
| `--keep-trial-artifacts` | 11 | Keep full per-trial fold run directories for debugging. By default, tuning deletes those heavy fold artifacts after metrics or failure details are recorded. |

There are no tuning flags for the resolution threshold search space. The two
resolution thresholds are included automatically in every case-fold Optuna
trial, and the selected values are written as `best_resolution_thresholds`.

By default, case-fold tuning uses summary-only artifact retention. Each trial
still trains, scores, resolves, and evaluates every selected fold, but after a
fold metric row has been captured the trial-local model, scored pairs,
resolution outputs, and detailed held-out reports are removed. This keeps long
Optuna runs from growing roughly with `n_trials * fold_count * full_run_outputs`.
The reusable prepared case artifacts under `<output-root>/prepared/cases/` are
kept once per case and reused across trials. Add `--keep-trial-artifacts` only
when debugging a specific trial needs the full run tree.

Case-fold tuning writes:

```text
<output-root>/fold_tuning_summary.json
<output-root>/fold_tuning_trials.csv
<output-root>/fold_tuning_report.md
<output-root>/case_fold_optuna_best_params.json
<output-root>/prepared/cases/
<output-root>/trials/trial_0000/trial_summary.json
<output-root>/trials/trial_0000/fold_metrics.csv
```

When `--keep-trial-artifacts` is set, each trial also keeps full fold
subdirectories such as:

```text
<output-root>/trials/trial_0000/<fold_name>/reranker_model.txt
<output-root>/trials/trial_0000/<fold_name>/case_run/runs/rid_<encoded_run_id>/
```

Persistent studies are guarded by an input fingerprint in `src/matching/fold_tuning.py`.
If case inputs, folds, objective settings, resolution-threshold search space, or
search-space version differ, the same persistent study is rejected instead of
silently mixing incompatible runs.

### Optuna Dashboard

Install once: `python -m pip install optuna-dashboard`.

Open any saved tuning study with:
`optuna-dashboard sqlite:///data/<tuning-run-dir>/optuna.db`

Then visit `http://127.0.0.1:8080/`. Use `fold_tuning_report.md` for trusted-trial interpretation.


## Single-Run Evaluation

Use `src/evaluation/run.py` when the pipeline has already produced artifacts for
one run and the goal is to evaluate that run against one gold CSV.

Typical flow:

```bash
python3 -m src.pipeline \
  --case-root data/raw/case_a \
  --data-dir data/processed-case-a \
  --run-id case_a_eval

python3 -m src.evaluation.run \
  --data-dir data/processed-case-a \
  --run-id case_a_eval \
  --gold-path data/raw/case_a/annotation/gold_annotations.group_id_reviewed.csv
```

### Single-Run Evaluation Flags

| Flag | Importance | Description |
|---|---:|---|
| `--data-dir` | 1 | Required directory containing the completed run artifacts. |
| `--run-id` | 2 | Required run identifier to evaluate. |
| `--gold-path` | 3 | Reviewed gold CSV. Defaults to `data/gold_annotations.csv`, but deployment runs should pass the case-local reviewed file explicitly. |
| `--match-threshold` | 4 | Probability threshold for score-based pair evaluation. Keep default unless testing threshold sensitivity. |
| `--baseline-report` | 5 | Earlier `evaluation_report.json` for regression drift checks. |
| `--shared-labels-path` | 6 | Optional shared `labels.parquet` store for later training. Mostly useful for controlled experiments. |
| `--shared-labels-allowed-doc-ids` | 7 | Optional document allowlist when writing shared labels. Use only if intentionally restricting label export. |

### Pipeline Flags

| Flag | Importance | Description |
|---|---:|---|
| `--case-root` | 1 | Required case folder with PDF/DOCX input documents. |
| `--data-dir` | 2 | Required pipeline output directory. |
| `--run-id` | 3 | Optional stable run ID. Use one for reproducibility. |
| `--enable-shap` | 4 | Generate SHAP top-5 explanations during matching scoring. Useful for audit examples, slower on large data. |

## Metrics To Read First

| Metric | Where | Interpretation |
|---|---|---|
| `pairwise_f_beta` / `pairwise_f0_5` | Fold tuning reports and final cluster metrics | Primary Optuna tuning objective. Default F0.5 weights precision higher than recall because false merges are high-cost. |
| `pairwise_precision` | Final cluster metrics | Most direct false-merge risk signal: among predicted same-cluster pairs, how many are truly same-entity pairs. |
| `pairwise_recall` | Final cluster metrics | Measures how many true same-entity pairs were recovered. Use as a usefulness guardrail so tuning does not become too conservative. |
| `keep_score_threshold` / `objective_neutral_threshold` | Fold reports, tuning reports, best-params artifact, and `stage_metrics.resolution.thresholds` | Resolution thresholds selected or used for the run. These control retained graph edges and merge-positive objective evidence. |
| `retained_edge_count` / `component_count` | Fold reports and `stage_metrics.resolution` | Retained-graph diagnostics. Large shifts can explain final metric changes even when matching scores look similar. |
| `bcubed_f0_5` | Final cluster metrics | Supporting precision-weighted cluster metric. Useful for interpretation, but no longer the main tuning target. |
| `bcubed_precision` | Final cluster metrics | Element-centric precision view. Low value means final clusters often contain entities from different gold groups. |
| `bcubed_recall` | Final cluster metrics | Element-centric recall view. Low value means one gold group is split across predicted clusters. |
| `ari` | Final cluster metrics | Adjusted Rand Index. Measures overall agreement between predicted and gold clusters, adjusted for chance. Useful as a compact clustering sanity check. |
| `nmi` | Final cluster metrics | Normalized Mutual Information. Measures how much information the predicted clusters share with the gold groups. Useful as a second whole-clustering view. |
| `blocking_positive_pair_recall` | Fold reports and `stage_metrics.blocking` | If low, true links were missed before matching and cannot be recovered later. |
| `matching_pairwise_precision` / `matching_pairwise_recall` | Fold reports and `stage_metrics.matching` | Shows whether the scored candidate-pair model is too aggressive or too conservative before resolution. |
| Extraction `precision` / `recall` / `f1` | `stage_metrics.extraction` | Measures exact mention span/type recovery before entity resolution. |
| `matched_mention_rate` | `alignment` | Shows how much of the reviewed gold could be bridged to predicted mentions for cluster evaluation. |

False merges should be reviewed first. In this domain, a false merge can imply a
connection between unrelated people, organizations, accounts, or locations.
False splits are still important, but they are usually easier for a human review
workflow to repair than a confident wrong merge.

## Large Classified Dataset Workflow

For a classified evaluation:

1. Create one case folder per investigation case, each with PDF/DOCX documents.
2. Produce a reviewed gold CSV per case with reviewed `group_id` values.
3. Keep the recommended full gold CSV columns, even though the evaluator only
   requires a subset.
4. Define explicit held-out folds in a JSON config.
5. Run a one-fold smoke test with `--fold-name` and a fresh `--output-root`.
6. Inspect the held-out detailed evaluation report and verify that the
   evaluation entity count is nonzero.
7. Run all folds.
8. Read `fold_reports.csv` and inspect the worst folds by `pairwise_f0_5`,
   `pairwise_precision`, `pairwise_recall`, and `bcubed_f0_5`.
9. If tuning is needed, run case-fold Optuna tuning, select parameters, then run
   a clean case-fold evaluation again for reporting.
10. Keep output roots per evaluation round so reports remain reproducible and
    earlier outputs are not overwritten.

Recommended first command on a new classified setup:

```bash
python3 scripts/run_case_fold_eval.py classified_case_folds.json \
  --output-root data/cv-classified-smoke \
  --fold-name <one_fold_name>
```

After the smoke fold is valid:

```bash
python3 scripts/run_case_fold_eval.py classified_case_folds.json \
  --output-root data/cv-classified-full
```

## Interpretation Rules

A strong evaluation should use varied cases: different document styles,
different entity densities, different ambiguity levels, and different annotation
reviewers if possible. Results should state the number of cases, the train/test
fold setup, the reviewed gold source, the output root, and the most important
false-merge examples.

When reporting results review these:

1. Number of reviewed cases and fold design.
2. Final pairwise cluster metrics, especially pairwise F0.5, precision, and
   recall.
3. Blocking recall, because missed candidate pairs limit every later stage.
4. Matching precision/recall, to explain over-merging or under-linking.
5. Supporting cluster metrics such as B-cubed F0.5, ARI, and NMI.
6. Concrete false-merge and false-split examples from the detailed reports.
