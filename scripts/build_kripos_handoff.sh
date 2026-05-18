#!/usr/bin/env bash
# Produce a self-contained NERINE handoff bundle for Kripos.
#
# Output:
#   dist/kripos-handoff-YYYY-MM-DD/
#     nerine-image.tar.gz       (the Docker image, ~3 GB compressed)
#     docker-compose.yml        (granular bind mounts; no source needed)
#     RUNBOOK.md                (one-page operator guide)
#     SHA256SUMS                (integrity checksums)
#
# Usage (from the repository root):
#   ./scripts/build_kripos_handoff.sh
#
# Environment overrides:
#   IMAGE_TAG     default: nerine:latest
#   OUTPUT_ROOT   default: dist
#   GZIP_LEVEL    default: 6   (1 = fastest, 9 = smallest)

set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-nerine:latest}"
OUTPUT_ROOT="${OUTPUT_ROOT:-dist}"
GZIP_LEVEL="${GZIP_LEVEL:-6}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

stamp="$(date -u +%Y-%m-%d)"
bundle_dir="$OUTPUT_ROOT/kripos-handoff-$stamp"
image_archive="$bundle_dir/nerine-image.tar.gz"
runbook_src="$REPO_ROOT/documentation/kripos-handoff-runbook.md"
compose_src="$REPO_ROOT/docker-compose.yml"

echo "==> Preparing $bundle_dir"
mkdir -p "$bundle_dir"

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker is not on PATH. Install Docker or enable WSL integration." >&2
    exit 1
fi

if ! docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
    echo "==> Image $IMAGE_TAG not found locally; building it."
    docker compose --progress=plain build
fi

# Sanity-check that the tuned model is baked into the image at the right path.
echo "==> Verifying tuned model is present in image"
if ! docker run --rm --entrypoint /bin/sh "$IMAGE_TAG" \
        -c '[ -s /app/data/processed/reranker_model.txt ] && [ -s /app/data/processed/reranker_model_metadata.json ]'; then
    echo "ERROR: image is missing the tuned LightGBM reranker." >&2
    echo "       Check the COPY data/lightgbm_kripos_conservative_trial128_*/..." >&2
    echo "       lines in the Dockerfile and rebuild." >&2
    exit 1
fi

baked_model_sha="$(docker run --rm --entrypoint /bin/sh "$IMAGE_TAG" \
    -c 'sha256sum /app/data/processed/reranker_model.txt' | awk '{print $1}')"
echo "    baked model sha256: $baked_model_sha"

echo "==> Saving image to $image_archive (this is the slow step, ~30-90s)"
docker save "$IMAGE_TAG" | gzip "-$GZIP_LEVEL" > "$image_archive"

echo "==> Copying compose file and runbook"
cp "$compose_src" "$bundle_dir/docker-compose.yml"
cp "$runbook_src" "$bundle_dir/RUNBOOK.md"

echo "==> Computing checksums"
( cd "$bundle_dir" && sha256sum nerine-image.tar.gz docker-compose.yml RUNBOOK.md > SHA256SUMS )

echo
echo "==> Bundle ready:"
ls -lh "$bundle_dir" | awk 'NR>1 {printf "    %-10s %s\n", $5, $NF}'
echo
echo "    Bundle path: $bundle_dir"
echo "    Image SHA  : $baked_model_sha (tuned model inside the image)"
echo
echo "    Transfer the directory to Kripos via your approved channel."
echo "    They run the steps in RUNBOOK.md to load and start the system."
