# NERINE — Kripos Handoff Runbook

This bundle delivers the NERINE entity-resolution pipeline as a self-contained
Docker image. Everything required to run on an air-gapped machine — Python
runtime, all dependencies, the Norwegian NER and SBERT models, and the tuned
LightGBM reranker — is baked into the image. No internet access is required
on the target machine after `docker load`.

## Bundle contents

| File                      | Purpose                                        |
|---------------------------|------------------------------------------------|
| `nerine-image.tar.gz`     | The Docker image (load with `docker load`).   |
| `docker-compose.yml`      | Service definitions for HITL UI and pipeline. |
| `RUNBOOK.md`              | This file.                                    |
| `SHA256SUMS`              | Integrity checksums for every file above.     |

Verify the bundle is intact before loading:

```bash
sha256sum --check SHA256SUMS
```

## Prerequisites on the target machine

- Linux, macOS, or Windows with Docker Engine ≥ 24 (or Docker Desktop).
- About **8 GB free disk** for the loaded image (the gzipped tarball is ~3 GB).
- A modern CPU with AVX2 (any laptop/desktop from the last decade qualifies).
- No GPU is required — CPU inference is the supported path.

## One-time setup

From the bundle directory:

```bash
# 1. Load the image into Docker (takes 1–2 min; verifies layer integrity).
gunzip -c nerine-image.tar.gz | docker load

# 2. Confirm the image is registered.
docker images nerine

# 3. Prepare host folders Docker will bind-mount.
mkdir -p data/raw data/processed/runs
```

That's it for installation. Nothing else gets downloaded.

## Running the HITL review UI

```bash
docker compose up hitl
```

Open `http://localhost:8501` in a browser on the same machine. While the
service is running, the terminal stays attached and shows Streamlit logs.
Stop it with `Ctrl+C`.

The first time you start it, the UI will say "No resolution runs found" —
this is correct. You produce runs by invoking the pipeline (next section).

**Important:** if you start the HITL service *before* producing any runs,
Streamlit caches the empty result. A pipeline run completed afterward will
not appear on a browser refresh — you have to restart the HITL service for
it to show up. See *Troubleshooting → New runs do not appear in HITL* for
the fix. Avoiding the trap entirely: run the pipeline first, then start
HITL.

## Running the pipeline on a case

Drop your case files (PDF and/or DOCX) into a new folder under `data/raw/`:

```
data/raw/<your_case_id>/
  document1.pdf
  document2.pdf
  notater.docx
  ...
```

Then run the pipeline against that folder:

```bash
docker compose run --rm pipeline \
  python -m src.pipeline \
    --case-root /app/data/raw/<your_case_id> \
    --data-dir  /app/data/processed
```

Expected runtime: **roughly 1–3 minutes per ~10 short documents** on a recent
laptop CPU. The first stage (text extraction) is fast; NER and SBERT
embedding are the heaviest steps.

When the run finishes, refresh the HITL tab — your new run will appear in the
Run ID dropdown in the sidebar.

## Output locations

All artifacts land under `data/processed/runs/<encoded_run_id>/` on the host:

| Subfolder            | Contents                                                      |
|----------------------|---------------------------------------------------------------|
| `ingestion/`         | One row per ingested document, chunked text.                  |
| `extraction/`        | Detected entity mentions with type and surface form.          |
| `blocking/`          | Candidate pairs proposed for matching.                        |
| `matching/`          | Per-pair feature vectors and LightGBM scores.                 |
| `resolution/`        | Final clusters, routing buckets, and diagnostics.             |
| `pipeline/`          | Per-run summary JSON with stage timings and counts.           |

The structured outputs are Parquet (open with DuckDB, Polars, pandas, or any
modern data tool). The pipeline summary is plain JSON.

## What the tuned model does and does not know

The image ships with model
`lightgbm_kripos_conservative_trial128_20260511`. Relevant facts:

- Trained on **3,880 labeled pairs** from four reviewed cases
  (`case_epstein_real_01`, `case_palme_real_01`, `case_tall_pines_01`,
  `styles_case_manual`). Held-out pairwise F0.5 ≈ 0.894.
- Selected as a **conservative variant** (Optuna trial 128 rather than the
  absolute best trusted trial) to reduce the rate of false merges under
  small-data uncertainty — the right tradeoff when false merges are
  operationally costly.
- Recommended thresholds are recorded in the metadata at
  `/app/data/processed/reranker_model_metadata.json` inside the image.

If your cases differ substantially from the training corpus (different
investigation type, very different document genres, non-Norwegian text),
the model may under-perform. The HITL routing buckets are designed to
surface uncertain clusters specifically so a reviewer can catch this.

## Troubleshooting

### `permission denied … docker.sock`

Your user is not in the `docker` group. On Linux:
```bash
sudo usermod -aG docker $USER
# log out and back in (or open a fresh shell)
```

### `Cannot connect to the Docker daemon`

The daemon is not running. On Linux: `sudo service docker start`. On
Docker Desktop: open the app and wait for the whale icon to be solid.

### `docker compose` reports "unknown command"

You have Docker Engine but the Compose v2 plugin is not installed.
Do *not* install `docker-compose` (with a hyphen) — that is the legacy
Python-based v1 tool and is incompatible with modern Docker daemons.
You will see `_retrieve_server_version` crashes or `unexpected keyword
argument 'chunked'` errors.

Install Compose v2 instead. The package name depends on which apt
repo provides Docker on your distribution:

```bash
# Ubuntu 24.04+ (universe repo):
sudo apt install -y docker-compose-v2

# Docker's official apt repo:
sudo apt install -y docker-compose-plugin
```

If neither package is found, install the binary directly:

```bash
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -fsSL \
  https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
```

Then verify (note the space — `docker compose`, not `docker-compose`):

```bash
docker compose version
```

If you previously installed the legacy v1, remove it so it does not
shadow the plugin:

```bash
sudo apt remove -y docker-compose
```

### Pipeline error: "Could not open … reranker_model.txt"

The bind mount may be shadowing the baked-in model. Verify with:
```bash
docker compose run --rm pipeline ls -lh /app/data/processed/reranker_model.txt
```
The file should be ~340 KB. If it is missing, the compose file's volume
section was edited — restore the granular mounts shown in the shipped
`docker-compose.yml` (only `./data/raw` and `./data/processed/runs`).

### Streamlit shows runs but the cluster view is empty

The run probably failed before resolution finished. Open
`data/processed/runs/<id>/pipeline/pipeline_summary.json` and check the
`failed_stage` and `error` fields.

### New runs do not appear in HITL

Symptom: you ran the pipeline successfully (six stage folders are present
on disk under `data/processed/runs/<id>/`, including a populated
`resolution/` folder) but HITL still reports "No resolution runs found"
after a browser refresh.

Cause: Streamlit's `@st.cache_data` decorator caches the empty
run-discovery result from when HITL was first opened. A browser refresh
does not invalidate this cache; only a process restart or a manual cache
clear does.

Fix — restart the HITL service:
```bash
# In the terminal where HITL is running:
#   Press Ctrl+C to stop it.
docker compose up hitl
```
Wait for `You can now view your Streamlit app in your browser`, then
refresh `http://localhost:8501`. The new run will appear in the Run ID
dropdown.

Alternative without restart: in the HITL browser tab, open the menu
(≡ icon, top right) → **Clear cache** → then **Rerun**.

To verify the run is actually on disk before assuming this is the cache
issue, the following two commands should both list `rid_<...>`:
```bash
ls data/processed/runs/
docker compose exec hitl ls /app/data/processed/runs/
```

### Host filesystem shows files owned by root

By default the container runs as root and writes files owned by root to
bind-mounted volumes. For day-to-day operation this is harmless. If it
gets in the way (e.g. you cannot delete a run folder), pass your UID:GID
when running the container:
```bash
docker compose run --rm --user "$(id -u):$(id -g)" pipeline ...
```

## Support

For pipeline behaviour questions, contact the NERINE bachelor team at NTNU.
For Docker-level issues unrelated to NERINE, see the official Docker docs.
