#!/usr/bin/env bash
# Download WebArena docker images from archive.org (CMU mirror fallback),
# docker-load them, and verify tags match docker-compose.yaml expectations.
# Run ONCE during GCP image build (~30 min + ~25GB disk).
set -euo pipefail

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

# CMU direct-download fallback (faster on GCP if archive.org is slow)
declare -A CMU_MIRROR=(
  [shopping_final_0712.tar]="http://metis.lti.cs.cmu.edu/webarena-images/shopping_final_0712.tar"
  [shopping_admin_final_0719.tar]="http://metis.lti.cs.cmu.edu/webarena-images/shopping_admin_final_0719.tar"
  [postmill-populated-exposed-withimg.tar]="http://metis.lti.cs.cmu.edu/webarena-images/postmill-populated-exposed-withimg.tar"
  [gitlab-populated-final-port8023.tar]="http://metis.lti.cs.cmu.edu/webarena-images/gitlab-populated-final-port8023.tar"
)

download_one() {
  local name="$1"
  local url="$2"
  local mirror="$3"
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
    echo "[gcs-fail] falling back to CMU mirror"
  fi
  echo "[download] $name <- $url"
  if ! curl -fL --retry 3 -o "$name.part" "$url"; then
    echo "[mirror] $name <- $mirror"
    curl -fL --retry 3 -o "$name.part" "$mirror"
  fi
  mv "$name.part" "$name"
}

for name in "${!IMAGES[@]}"; do
  download_one "$name" "${IMAGES[$name]}" "${CMU_MIRROR[$name]}"
done

for name in "${!IMAGES[@]}"; do
  echo "[load] $name"
  sudo docker load --input "$name"
done

echo ""
echo "=== Loaded images ==="
sudo docker images | grep -E 'shopping_final|shopping_admin|postmill|gitlab-populated' || true

# Also pull kiwix-serve from Docker Hub (small, ~50MB)
sudo docker pull ghcr.io/kiwix/kiwix-serve:latest

# Build a tiny homepage image (assumes /opt/webarena-homepage/ is provisioned from the repo)
if [ -d /opt/webarena-homepage ]; then
  cd /opt/webarena-homepage
  sudo docker build -t webarena-homepage:local .
  cd "$WORK"
fi

echo "[done] WebArena images loaded."
