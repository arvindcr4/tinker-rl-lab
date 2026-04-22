#!/usr/bin/env bash
# Deploy an N-worker Managed Instance Group that runs the parallel WebArena eval
# and uploads per-shard results to a GCS bucket.
#
# Prereqs:
#   - build_webarena_image.sh has already built image family 'webarena'
#   - A GCS bucket exists: gs://$BUCKET
#   - TINKER_API_KEY saved in Secret Manager as 'tinker-api-key' (or injected via --metadata-from-file)
#
# Usage:
#   GCP_PROJECT=electric-armor-388216 BUCKET=my-webarena-results \
#   NUM_WORKERS=10 ./infra/gcp/deploy_mig.sh
set -euo pipefail

: "${GCP_PROJECT:?set GCP_PROJECT}"
: "${GCP_ZONE:=us-central1-a}"
: "${BUCKET:?set BUCKET (gs://... without prefix)}"
: "${NUM_WORKERS:=10}"
: "${BENCHMARK:=webarena_verified}"
: "${MODEL:=Qwen/Qwen3-8B}"
: "${MAX_STEPS:=30}"
: "${IMAGE_FAMILY:=webarena}"
: "${MACHINE_TYPE:=e2-standard-4}"
: "${DISK_GB:=1000}"
: "${PREEMPTIBLE:=false}"
: "${WANDB_PROJECT:=}"
: "${HF_REPO:=}"
: "${HF_PRIVATE:=false}"
: "${SERVICE_ACCOUNT:=webarena-runner@${GCP_PROJECT}.iam.gserviceaccount.com}"
RUN_ID="$(date -u +%Y%m%d-%H%M%S)"
MIG_NAME="webarena-mig-$RUN_ID"
TEMPLATE_NAME="webarena-tpl-$RUN_ID"

gcloud config set project "$GCP_PROJECT"

echo "==> Writing startup script..."
cat > /tmp/webarena_startup.sh <<STARTUP
#!/usr/bin/env bash
set -euxo pipefail
WORKER_ID=\$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/worker-id)
NUM_WORKERS=\$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/num-workers)
BENCHMARK=\$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/benchmark)
MODEL=\$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/model)
MAX_STEPS=\$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/max-steps)
BUCKET=\$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket)
RUN_ID=\$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/run-id)
WANDB_PROJECT=\$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/wandb-project || echo "")
HF_REPO=\$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/hf-repo || echo "")
HF_PRIVATE=\$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/hf-private || echo "false")

PROJECT_ID=\$(gcloud config get-value project)

# Required: Tinker key
export TINKER_API_KEY=\$(gcloud secrets versions access latest --secret=tinker-api-key --project="\$PROJECT_ID")

# Optional: W&B + HF Hub tokens (silently skip if secret not present)
export WANDB_API_KEY=\$(gcloud secrets versions access latest --secret=wandb-api-key --project="\$PROJECT_ID" 2>/dev/null || echo "")
export HF_TOKEN=\$(gcloud secrets versions access latest --secret=hf-token --project="\$PROJECT_ID" 2>/dev/null || echo "")
if [ -z "\$WANDB_API_KEY" ]; then WANDB_PROJECT=""; fi
if [ -z "\$HF_TOKEN" ]; then HF_REPO=""; fi

# Launch WebArena docker stack
cd /opt/webarena-compose
sudo docker compose up -d
sleep 90

# Site URLs (all on localhost)
export WEBARENA_SHOPPING_URL=http://localhost:7770
export WEBARENA_SHOPPING_ADMIN_URL=http://localhost:7780
export WEBARENA_REDDIT_URL=http://localhost:9999
export WEBARENA_GITLAB_URL=http://localhost:8023
export WEBARENA_WIKIPEDIA_URL=http://localhost:8888
export WEBARENA_MAP_URL=http://public-map.server.invalid
export WEBARENA_HOMEPAGE_URL=http://localhost:4399

sudo git -C /opt/tinker-rl-lab pull --ff-only || true

cd /opt/tinker-rl-lab
HF_FLAGS=""
if [ -n "\$HF_REPO" ]; then
  HF_FLAGS="--hf-repo \$HF_REPO"
  if [ "\$HF_PRIVATE" = "true" ]; then HF_FLAGS="\$HF_FLAGS --hf-private"; fi
fi
WANDB_FLAGS=""
if [ -n "\$WANDB_PROJECT" ]; then
  WANDB_FLAGS="--wandb-project \$WANDB_PROJECT --run-name webarena-\$RUN_ID-w\$WORKER_ID"
fi

/opt/tinker/bin/python -m experiments.webarena.react_eval \\
  --benchmark "\$BENCHMARK" \\
  --tasks all \\
  --model "\$MODEL" \\
  --max-steps "\$MAX_STEPS" \\
  --concurrency 5 \\
  --out "/tmp/results_shard_\${WORKER_ID}.jsonl" \\
  --shard "\${WORKER_ID}/\${NUM_WORKERS}" \\
  \$WANDB_FLAGS \$HF_FLAGS

gsutil cp "/tmp/results_shard_\${WORKER_ID}.jsonl" \\
  "gs://\${BUCKET}/\${RUN_ID}/results_shard_\${WORKER_ID}.jsonl"

sudo shutdown -h now
STARTUP

PREEMPT_FLAG=""
if [ "$PREEMPTIBLE" = "true" ]; then
  PREEMPT_FLAG="--preemptible"
fi

echo "==> Creating instance template $TEMPLATE_NAME..."
gcloud compute instance-templates create "$TEMPLATE_NAME" \
  --machine-type="$MACHINE_TYPE" \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$GCP_PROJECT" \
  --boot-disk-size="${DISK_GB}GB" \
  --boot-disk-type=pd-balanced \
  --service-account="$SERVICE_ACCOUNT" \
  --scopes=cloud-platform \
  --metadata-from-file=startup-script=/tmp/webarena_startup.sh \
  --metadata=num-workers="$NUM_WORKERS",benchmark="$BENCHMARK",model="$MODEL",max-steps="$MAX_STEPS",bucket="$BUCKET",run-id="$RUN_ID",wandb-project="$WANDB_PROJECT",hf-repo="$HF_REPO",hf-private="$HF_PRIVATE" \
  $PREEMPT_FLAG

echo "==> Creating MIG $MIG_NAME with $NUM_WORKERS workers..."
gcloud compute instance-groups managed create "$MIG_NAME" \
  --zone="$GCP_ZONE" \
  --template="$TEMPLATE_NAME" \
  --size="$NUM_WORKERS"

# MIG doesn't inject per-instance metadata automatically; patch each instance's worker-id
echo "==> Patching per-instance worker-id..."
i=0
for instance in $(gcloud compute instance-groups managed list-instances "$MIG_NAME" --zone="$GCP_ZONE" --format="value(name)"); do
  gcloud compute instances add-metadata "$instance" \
    --zone="$GCP_ZONE" \
    --metadata="worker-id=$i"
  i=$((i+1))
done

echo ""
echo "==> Deployed run_id=$RUN_ID with $NUM_WORKERS workers"
echo "==> Results will appear at gs://$BUCKET/$RUN_ID/"
echo "==> Monitor:    gcloud compute instance-groups managed list-instances $MIG_NAME --zone=$GCP_ZONE"
echo "==> Collect:    gsutil ls gs://$BUCKET/$RUN_ID/"
echo "==> Aggregate:  gsutil cp gs://$BUCKET/$RUN_ID/*.jsonl ./results/ && \\"
echo "                python -m experiments.webarena.aggregate --inputs 'results/*.jsonl' --out final.json"
echo "==> Cleanup:    gcloud compute instance-groups managed delete $MIG_NAME --zone=$GCP_ZONE -q && \\"
echo "                gcloud compute instance-templates delete $TEMPLATE_NAME -q"
