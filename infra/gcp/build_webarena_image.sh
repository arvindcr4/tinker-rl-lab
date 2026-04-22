#!/usr/bin/env bash
# Build a reusable GCP image with WebArena dockers preloaded + Python + Playwright.
# ~60 min first run (dominated by archive.org pulls). Cuts subsequent VM boots to ~3 min.
set -euo pipefail

: "${GCP_PROJECT:?set GCP_PROJECT (e.g. electric-armor-388216)}"
: "${GCP_REGION:=us-central1}"
: "${GCP_ZONE:=us-central1-a}"
: "${IMAGE_NAME:=webarena-v1}"
: "${IMAGE_FAMILY:=webarena}"
: "${BUILDER_DISK_GB:=300}"
: "${SERVICE_ACCOUNT:=webarena-runner@${GCP_PROJECT}.iam.gserviceaccount.com}"
BUILDER_NAME="webarena-builder-$$"
REPO_DIR="$(cd "$(dirname "$0")"/../.. && pwd)"

echo "==> Project: $GCP_PROJECT  Zone: $GCP_ZONE  Disk: ${BUILDER_DISK_GB}GB"
gcloud config set project "$GCP_PROJECT"

echo "==> Creating builder VM..."
gcloud compute instances create "$BUILDER_NAME" \
  --zone="$GCP_ZONE" \
  --machine-type=e2-standard-4 \
  --image-family=ubuntu-2404-lts-amd64 \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size="${BUILDER_DISK_GB}GB" \
  --boot-disk-type=pd-balanced \
  --service-account="$SERVICE_ACCOUNT" \
  --scopes=cloud-platform

trap 'echo "==> Cleaning builder..."; gcloud compute instances delete -q "$BUILDER_NAME" --zone="$GCP_ZONE" || true' EXIT

echo "==> Waiting for SSH..."
for i in {1..30}; do
  if gcloud compute ssh "$BUILDER_NAME" --zone="$GCP_ZONE" -- "echo ready" 2>/dev/null; then
    break
  fi
  sleep 10
done

echo "==> Uploading webarena-compose + load_images.sh..."
gcloud compute scp --recurse \
  "$REPO_DIR/infra/gcp/webarena-compose" \
  "$BUILDER_NAME:/tmp/webarena-compose" \
  --zone="$GCP_ZONE"

cat > /tmp/webarena_setup.sh <<'SETUP'
#!/usr/bin/env bash
set -euo pipefail
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg git python3-pip python3-venv \
  build-essential libnss3 libatk-bridge2.0-0 libcups2 libxkbcommon0 \
  libgbm1 libasound2t64 libgtk-3-0

# Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER" || true

# Python env
sudo python3 -m venv /opt/tinker
sudo /opt/tinker/bin/pip install -U pip setuptools wheel

# Pin numpy + pyarrow to versions with cp312 wheels BEFORE pulling datasets
sudo /opt/tinker/bin/pip install --prefer-binary "numpy>=1.26,<3" "pyarrow>=16,<22"

sudo /opt/tinker/bin/pip install --prefer-binary \
  "tinker>=0.1" "transformers>=4.44" "gymnasium>=0.29" \
  "playwright>=1.44" "browsergym-core" "browsergym-miniwob" \
  "browsergym-webarena" "browsergym-experiments" \
  "datasets" "tqdm" "pyyaml" "google-cloud-storage" "google-cloud-secret-manager" \
  "wandb" "huggingface_hub"

sudo /opt/tinker/bin/playwright install chromium
sudo /opt/tinker/bin/playwright install-deps chromium

# Clone tinker-rl-lab for startup script to use
sudo rm -rf /opt/tinker-rl-lab
sudo git clone --depth 1 https://github.com/arvindcr4/tinker-rl-lab /opt/tinker-rl-lab || true

# Stage WebArena compose dir
sudo mkdir -p /opt/webarena-compose
sudo cp -r /tmp/webarena-compose/* /opt/webarena-compose/

# Clone homepage Flask app from webarena repo
sudo rm -rf /opt/webarena-homepage
sudo git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/web-arena-x/webarena /tmp/wa-src
(cd /tmp/wa-src && sudo git sparse-checkout set environment_docker/webarena-homepage)
sudo mv /tmp/wa-src/environment_docker/webarena-homepage /opt/webarena-homepage
sudo rm -rf /tmp/wa-src

# Download + load WebArena docker images (~30 min)
sudo bash /opt/webarena-compose/load_images.sh

echo "==> setup complete"
SETUP

gcloud compute scp /tmp/webarena_setup.sh "$BUILDER_NAME:/tmp/webarena_setup.sh" --zone="$GCP_ZONE"
gcloud compute ssh "$BUILDER_NAME" --zone="$GCP_ZONE" -- "bash /tmp/webarena_setup.sh"

echo "==> Stopping builder for image snapshot..."
gcloud compute instances stop "$BUILDER_NAME" --zone="$GCP_ZONE"

echo "==> Creating image $IMAGE_NAME (family=$IMAGE_FAMILY)..."
gcloud compute images create "$IMAGE_NAME" \
  --source-disk="$BUILDER_NAME" \
  --source-disk-zone="$GCP_ZONE" \
  --family="$IMAGE_FAMILY" \
  --description="WebArena + Tinker + BrowserGym preloaded ($(date -u +%Y-%m-%d))"

echo "==> DONE. Image: $IMAGE_NAME  Family: $IMAGE_FAMILY"
