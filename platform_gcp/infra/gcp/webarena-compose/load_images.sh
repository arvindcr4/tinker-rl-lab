#!/usr/bin/env bash
# Download WebArena docker images from archive.org (CMU mirror fallback),
# docker-load them, and verify tags match docker-compose.yaml expectations.
# Run ONCE during GCP image build (~30 min + ~25GB disk).
set -euo pipefail
: "${WEBARENA_SHA256_MANIFEST:?Provide a reviewed archive SHA-256 manifest}"
: "${WEBARENA_SHA256_ANCHOR:?Provide its independently reviewed SHA-256 digest}"
: "${KIWIX_IMAGE:?Provide ghcr.io/kiwix/kiwix-serve@sha256:DIGEST}"
[[ "$WEBARENA_SHA256_ANCHOR" =~ ^[a-f0-9]{64}$ ]] || exit 2
[[ "$KIWIX_IMAGE" =~ ^ghcr.io/kiwix/kiwix-serve@sha256:[a-f0-9]{64}$ ]] || exit 2
MANIFEST=$(realpath "$WEBARENA_SHA256_MANIFEST")
printf '%s  %s\n' "$WEBARENA_SHA256_ANCHOR" "$MANIFEST" | sha256sum -c -

WORK=${WORK:-/opt/webarena-images}
GCS_BUCKET=${WEBARENA_TAR_GCS_BUCKET:-}
sudo mkdir -p "$WORK"
cd "$WORK"

declare -A IMAGES=(
  [shopping_final_0712.tar]="https://archive.org/download/webarena-env-shopping-image/shopping_final_0712.tar"
  [shopping_admin_final_0719.tar]="https://archive.org/download/webarena-env-shopping-admin-image/shopping_admin_final_0719.tar"
  [postmill-populated-exposed-withimg.tar]="https://archive.org/download/webarena-env-forum-image/postmill-populated-exposed-withimg.tar"
  [gitlab-populated-final-port8023.tar]="https://archive.org/download/webarena-env-gitlab-image/gitlab-populated-final-port8023.tar"
)

download_one() {
  local name="$1"
  local url="$2"
  if [ -f "$name" ]; then
    echo "[skip] $name already present"
    return 0
  fi
  if [ -n "$GCS_BUCKET" ]; then
    echo "[gcs] $name <- $GCS_BUCKET/$name"
    if sudo gsutil -m -o "GSUtil:parallel_thread_count=8" cp "$GCS_BUCKET/$name" "$name.part"; then
      mv "$name.part" "$name"
      return 0
    fi
    echo "[gcs-fail] trying authenticated HTTPS source"
  fi
  echo "[download] $name <- $url"
  curl --proto '=https' --proto-redir '=https' -fL --retry 3 -o "$name.part" "$url"
  mv "$name.part" "$name"
}

for name in "${!IMAGES[@]}"; do
  download_one "$name" "${IMAGES[$name]}"
  expected=$(awk -v target="$name" '$2 == target || $2 == "*" target { print $1 }' "$MANIFEST")
  [[ "$expected" =~ ^[a-f0-9]{64}$ ]] || { echo "Missing/ambiguous trusted archive digest" >&2; exit 2; }
  printf '%s  %s\n' "$expected" "$name" | sha256sum -c -
  echo "[load] $name"
  sudo docker load --input "$name"
  echo "[cleanup] $name (free disk)"
  sudo rm -f "$name"
done

echo ""
echo "=== Loaded images ==="
sudo docker images | grep -E 'shopping_final|shopping_admin|postmill|gitlab-populated' || true

# Also pull kiwix-serve from Docker Hub (small, ~50MB)
sudo docker pull "$KIWIX_IMAGE"
# Compose resolves this local alias to the digest-pinned bytes loaded above.
sudo docker tag "$KIWIX_IMAGE" ghcr.io/kiwix/kiwix-serve:latest

# Build a tiny homepage image (assumes /opt/webarena-homepage/ is provisioned from the repo)
if [ -d /opt/webarena-homepage ]; then
  cd /opt/webarena-homepage
  sudo docker build -t webarena-homepage:local .
  cd "$WORK"
fi

echo "[done] WebArena images loaded."
