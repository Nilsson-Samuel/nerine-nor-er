# HITL Streamlit Guide

This guide is for launching and navigating the human-in-the-loop (HITL) cluster
triage interface over a completed resolution run.

## Run Command

Run from the repository root in the same Python environment used for evaluation
(see *Environment Setup* in `README-evaluation.md`). Currently there is no
Docker image; launch Streamlit directly.

```bash
NERINE_DATA_DIR=<path-to-run-data-dir> \
  streamlit run src/hitl/streamlit_app.py --browser.gatherUsageStats false
```

If `NERINE_DATA_DIR` is not set, the app falls back to `data/processed` under
the repo root. Set the variable explicitly whenever the run lives elsewhere.
`--browser.gatherUsageStats false` disables Streamlit's usage telemetry and is
appropriate for offline Kripos-style use.

### Data Dir Examples

| Scenario | `NERINE_DATA_DIR` |
|---|---|
| Default pipeline run from repo root | unset (falls back to `data/processed`) |
| Named single-case pipeline run | `data/processed-case-a` |
| Case-fold held-out case output | `data/<output-root>/<fold_name>/<held_out_case>` |

For case-fold outputs, point at the held-out case directory, not the fold root
or the output root. That directory contains the per-run resolution artifacts
under `runs/rid_<encoded_run_id>/`, which is what the app discovers.

---

## Summary

The HITL app is a Streamlit interface for reviewing resolved clusters produced
by the resolution stage. It loads cluster artifacts for one pipeline run and
lets a reviewer browse clusters by routing profile and routing bucket, inspect
individual clusters, and drill down into members, aliases, edge-level evidence,
and SHAP explanations. A merge-candidate view lists entities the model scored
against an entity in a different cluster, so a reviewer can check whether
entities that were kept apart (for example `keep_separate` singletons) should
have been merged.

The app does not require a gold CSV and is independent from the evaluation
scripts. It can be pointed at any directory that contains a per-run resolution
output: a normal `src/pipeline.py` run, a case-fold held-out case output, or
another case-specific run.

### The sections below are reference details.
The GUI itself is intuitive: run
selector, profile, bucket, cluster table, inspector. Read on only when a field
or routing choice needs clarification.



## Relevant Code Files

| File | Role |
|---|---|
| `src/hitl/streamlit_app.py` | Streamlit entrypoint. Sidebar run/profile/bucket/view selection and review settings, size chart, cluster table, merge-candidate overview, inspector routing. |
| `src/hitl/queries.py` | Run discovery, cluster frame loading, bucket filtering, inspector queries (members, edges, aliases), cross-cluster merge candidates. |
| `src/hitl/context.py` | Rebuilds document text from the run's chunks and cuts wide context windows around mentions. |
| `src/hitl/cluster_view.py` | Inspector rendering: header, member table, member contexts, merge candidates, aliases, edge table with weakest-link. |
| `src/hitl/status.py` | Safe diagnostics loading and the compact sidebar summary. |
| `src/hitl/ui_utils.py` | Option-label formatting and small UI helpers. |

## Sidebar Controls

| Control | Purpose |
|---|---|
| Run ID | Selects which run under `NERINE_DATA_DIR` to load. Populated automatically from resolution outputs. |
| Routing profile | Switches between `balanced_hitl` and `quick_low_hitl`. Controls which route column and bucket set is active. |
| Routing bucket | Filters the cluster table to one bucket, for example `auto_merge`, `review`, `keep_separate`, or `defer`. |
| View | `Clusters` shows the cluster table. `Merge candidates` lists cross-cluster candidates for the clusters in the bucket. |
| Minimum candidate score | Lowest pair score shown as a merge candidate between two clusters. Default `HITL_CANDIDATE_MIN_SCORE` in `src/shared/config.py`. |
| Context window | Characters shown on each side of a mention in the inspector. Default `HITL_CONTEXT_WINDOW_CHARS` in `src/shared/config.py`. |
| Scoring model | `model_version` of the reranker that scored the run. A warning appears when it differs from the model installed in `NERINE_DATA_DIR` (`reranker_model_metadata.json`), which is the model a rerun would use. Check this before evaluating a run. |
| Diagnostics summary | Compact stage summary drawn from resolution diagnostics. Missing or partial diagnostics never crash the app. |

## Routing Profiles And Buckets

| Profile | Buckets | When to use |
|---|---|---|
| `balanced_hitl` | `auto_merge`, `review`, `keep_separate` | Default reviewer workflow. `review` surfaces the cluster set a human should look at. |
| `quick_low_hitl` | `auto_merge`, `defer`, `keep_separate` | Lighter-touch workflow. `defer` is the HITL-lite version of `review` for less strict triage. |

Each profile remembers its own last-selected bucket, so switching profiles does
not overwrite the other profile's bucket choice.

## Cluster Table Columns

| Column | Meaning |
|---|---|
| `cluster_id` | Stable cluster identifier from resolution. |
| `canonical_name` | Canonical surface form selected during canonicalization. |
| `canonical_type` | PIVCOLF entity type of the cluster. |
| `cluster_size` | Number of entities in the cluster. |
| `base_confidence` | `(0.6 × min edge score + 0.4 × mean edge score) × density`. Always 0 for singletons. |
| `top_candidate_score` | Strongest score between a member and an entity in another cluster. Empty when no pair reaches the minimum candidate score. Gives `keep_separate` clusters a score to sort and evaluate by. |
| `min_edge_score` | Weakest pairwise evidence in the cluster. |
| `density` | Actual edges divided by possible edges. |
| `suspicious_merge` | Flag for clusters where the weakest link is unusually low. |

The table is sorted by ascending `base_confidence` so the riskiest clusters are
at the top. A size filter above the table lets the reviewer narrow to a size
range without changing the bucket.

## Cluster Inspector

Selecting a cluster under the table opens the inspector. It shows:

| View | What it contains |
|---|---|
| Header metrics | Route action, cluster size, confidence, min edge score, density. |
| Members | Entity rows for the cluster with normalized form, surface text, type, doc, and offsets. |
| Member contexts | Every mention of every member with the mention highlighted and a configurable window of surrounding text. The text is rebuilt from the run's chunks, so it can cross chunk boundaries. |
| Merge candidates | Entities in other clusters that scored against a member, with the score, why they were not merged, and the candidate cluster's name, size and bucket. Choosing a candidate shows both entities' contexts side by side, with a button that opens the candidate's cluster. |
| Aliases | Aggregated surface forms inside the cluster, grouped by normalized form. |
| Edges | Pair-level edges between members with score, blocking source, and SHAP top-5 reasons. The weakest link is highlighted. |

Use the inspector to judge whether a cluster should be kept as a merge, split,
or flagged. Edge-level SHAP reasons and the weakest link are the main evidence
for false-merge review. Merge candidates are the main evidence for missed-merge
review. After opening a candidate's cluster, `Back to previous cluster` returns
to the cluster you came from.

## Merge Candidates

`keep_separate` clusters have no internal edges, so their base confidence is 0
and the cluster inspector alone cannot say whether they should have joined
another cluster. Merge candidates fill that gap. A candidate is any scored pair
at or above the minimum candidate score whose two entities ended up in
different clusters. Each candidate gets one of three reasons, using the
thresholds the run was resolved with:

| Reason | Meaning |
|---|---|
| `split: clustering kept apart` | Score above the neutral point (`OBJECTIVE_NEUTRAL_THRESHOLD`). The pair voted to merge, but correlation clustering kept the entities apart because other pairs disagreed. |
| `weak: below neutral point` | Score between the keep threshold and the neutral point. The pair was in the graph but voted against merging. |
| `pruned: below keep threshold` | Score below `KEEP_SCORE_THRESHOLD`. The pair never reached clustering. |

The `Merge candidates` view lists these for every cluster in the selected
bucket, strongest first. By default it shows one row per cluster, the
strongest candidate. Untick that option to see every candidate, or filter by
reason. Both entities' stored short contexts are in the table. Pick a cluster
under the table to open it in the inspector and compare wide contexts. The
table's download button exports the current rows to CSV for evaluation notes.

## When Nothing Appears

| Symptom | First check |
|---|---|
| "No resolution runs found." | `NERINE_DATA_DIR` points at a parent folder, not a run data dir. Point at the directory that contains `runs/rid_*` or the default `data/processed`. |
| Cluster table empty in a bucket | Switch profile or bucket. Some buckets are legitimately empty for small cases. |
| Sidebar diagnostics missing | Resolution diagnostics file is absent for that run. The app continues; this is informational only. |
| Stale data after a new run | Streamlit caches per run. Use the top-right menu to rerun or clear cache after writing new artifacts. |

For held-out evaluation context (fold layout, gold CSV, metrics, tuning), see
`documents/README-evaluation.md`.
