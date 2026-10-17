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
#   NUM_WORKERS=10 ./platform_gcp/infra/gcp/deploy_mig.sh
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
# Required local contract validation happens before any gcloud invocation.
: "${WEBARENA_TASK_MANIFEST:?Provide an explicit reviewed task/site manifest}"
: "${WEBARENA_TASK_MANIFEST_SHA256:?Provide its independently reviewed digest}"
CONTRACT_TOOL="$(dirname "$0")/webarena_contract.py"
TASKS=$(python3 "$CONTRACT_TOOL" "$WEBARENA_TASK_MANIFEST" "$WEBARENA_TASK_MANIFEST_SHA256" "$BENCHMARK" --field tasks)
REQUIRED_SITES=$(python3 "$CONTRACT_TOOL" "$WEBARENA_TASK_MANIFEST" "$WEBARENA_TASK_MANIFEST_SHA256" "$BENCHMARK" --field required_sites)
MAX_RUN_SECONDS=${MAX_RUN_SECONDS:-21600}
: "${TINKER_REPO_REVISION:?Provide the reviewed commit baked into the worker image}"
[[ "$TINKER_REPO_REVISION" =~ ^[a-f0-9]{40}$ ]] || exit 2
[[ "$MAX_RUN_SECONDS" =~ ^[1-9][0-9]*$ && "$NUM_WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid worker count/deadline" >&2; exit 2; }
# These are resource ceilings, not a currency budget or permission to launch.
[[ "$NUM_WORKERS" =~ ^[1-9][0-9]?$ ]] && (( NUM_WORKERS <= 10 )) || { echo "Worker count exceeds safety ceiling" >&2; exit 2; }
[[ "$DISK_GB" =~ ^[1-9][0-9]{0,3}$ ]] && (( DISK_GB <= 1000 )) || { echo "Disk exceeds safety ceiling" >&2; exit 2; }
[[ "$MAX_RUN_SECONDS" =~ ^[1-9][0-9]{0,4}$ ]] && (( MAX_RUN_SECONDS <= 86400 )) || { echo "Runtime exceeds safety ceiling" >&2; exit 2; }
[[ "$MACHINE_TYPE" = "e2-standard-4" ]] || { echo "Unreviewed worker machine type" >&2; exit 2; }
command -v timeout >/dev/null || { echo "GNU timeout is required" >&2; exit 2; }
gcloud() { timeout --kill-after=10s 120s gcloud "$@"; }
gsutil() { timeout --kill-after=10s 120s gsutil "$@"; }
RUN_ID="$(date -u +%Y%m%d-%H%M%S)-$(head -c 4 /dev/urandom | od -An -tx1 | tr -d ' \n')"
MIG_NAME="webarena-mig-$RUN_ID"
TEMPLATE_NAME="webarena-tpl-$RUN_ID"

# Local ownership records prevent cleanup of resources from another invocation.
TEMPLATE_CREATED=false
MIG_CREATED=false
TEMPLATE_ID=""
MIG_ID=""
STARTUP_FILE=$(mktemp)
cleanup() {
  status=$?
  trap - EXIT INT TERM HUP
  rm -f "$STARTUP_FILE"
  if $MIG_CREATED && [ -n "$MIG_ID" ]; then
    current=$(gcloud compute instance-groups managed describe "$MIG_NAME" --project="$GCP_PROJECT" --zone="$GCP_ZONE" --format='value(id)' 2>/dev/null || true)
    if [ "$current" = "$MIG_ID" ]; then
      gcloud compute instance-groups managed delete "$MIG_NAME" --project="$GCP_PROJECT" --zone="$GCP_ZONE" -q || status=1
    fi
  fi
  if $TEMPLATE_CREATED && [ -n "$TEMPLATE_ID" ]; then
    current=$(gcloud compute instance-templates describe "$TEMPLATE_NAME" --project="$GCP_PROJECT" --format='value(id)' 2>/dev/null || true)
    if [ "$current" = "$TEMPLATE_ID" ]; then
      gcloud compute instance-templates delete "$TEMPLATE_NAME" --project="$GCP_PROJECT" -q || status=1
    fi
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP
gcloud config set project "$GCP_PROJECT"

echo "==> Writing startup script..."
cat > "$STARTUP_FILE" <<STARTUP
#!/usr/bin/env bash
set -euo pipefail

# Poll metadata until worker-id is a valid integer. MIG sets worker-id via
# per-instance metadata AFTER VM creation (gcloud compute instances
# add-metadata), so the startup-script can fire before the metadata lands
# and read back a 404 HTML page. Block here until worker-id is numeric.
MD="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
_mdget() { curl -sf -H "Metadata-Flavor: Google" "\$MD/\$1" 2>/dev/null || echo ""; }
for i in \$(seq 1 60); do
  WORKER_ID=\$(_mdget worker-id)
  if [[ "\$WORKER_ID" =~ ^[0-9]+\$ ]]; then break; fi
  echo "Waiting for worker-id metadata to land (attempt \$i/60, got: '\${WORKER_ID:0:40}')..."
  sleep 5
done
if ! [[ "\$WORKER_ID" =~ ^[0-9]+\$ ]]; then
  echo "FATAL: worker-id never landed after 5 minutes; aborting startup" >&2
  exit 1
fi
NUM_WORKERS=\$(_mdget num-workers)
BENCHMARK=\$(_mdget benchmark)
MODEL=\$(_mdget model)
MAX_STEPS=\$(_mdget max-steps)
BUCKET=\$(_mdget bucket)
RUN_ID=\$(_mdget run-id)
WANDB_PROJECT=\$(_mdget wandb-project)
HF_REPO=\$(_mdget hf-repo)
HF_PRIVATE=\$(_mdget hf-private)
[ -z "\$HF_PRIVATE" ] && HF_PRIVATE=false

PHASE=infrastructure
# Restore previous immutable-run shard progress before sampling.
RESULT="/tmp/results_shard_\${WORKER_ID}.jsonl"
REMOTE="gs://\${BUCKET}/\${RUN_ID}/results_shard_\${WORKER_ID}.jsonl"
gsutil cp "\$REMOTE" "\$RESULT" 2>/dev/null || true
# Persist progress at bounded intervals; at most one interval may be repeated.
(
  while sleep 60; do
    if [ -f "\$RESULT" ]; then gsutil cp "\$RESULT" "\$REMOTE" || true; fi
  done
) &
SYNC_PID=\$!
finish_shard() {
  rc=\$?
  trap - EXIT
  kill "\$SYNC_PID" 2>/dev/null || true
  if [ -f "\$RESULT" ]; then gsutil cp "\$RESULT" "\$REMOTE" || exit 1; fi
  printf '{"protocol":"webarena-shard-v2","experiment_protocol":"webarena-restricted-services-v2","task_manifest_sha256":"$WEBARENA_TASK_MANIFEST_SHA256","repo_revision":"$TINKER_REPO_REVISION","run_id":"%s","shard":%s,"exit_code":%s,"phase":"%s"}\\n' "\$RUN_ID" "\$WORKER_ID" "\$rc" "\$PHASE" > /tmp/shard_receipt.json
  gsutil cp /tmp/shard_receipt.json "gs://\${BUCKET}/\${RUN_ID}/shard_\${WORKER_ID}.receipt.json" || exit 1
  exit "\$rc"
}
trap finish_shard EXIT


PROJECT_ID=\$(gcloud config get-value project)

# Required: Tinker key
TINKER_API_KEY=\$(gcloud secrets versions access latest --secret=tinker-api-key --project="\$PROJECT_ID")
: "\${TINKER_API_KEY:?Tinker credential is empty}"
export TINKER_API_KEY

# Optional: W&B + HF Hub tokens (silently skip if secret not present)
export WANDB_API_KEY=\$(gcloud secrets versions access latest --secret=wandb-api-key --project="\$PROJECT_ID" 2>/dev/null || echo "")
export HF_TOKEN=\$(gcloud secrets versions access latest --secret=hf-token --project="\$PROJECT_ID" 2>/dev/null || echo "")
# Optional: OpenAI key for BrowserGym llm_fuzzy_match judge (~15% of WebArena tasks)
export OPENAI_API_KEY=\$(gcloud secrets versions access latest --secret=openai-api-key --project="\$PROJECT_ID" 2>/dev/null || echo "")
if [ -z "\$WANDB_API_KEY" ]; then WANDB_PROJECT=""; fi
if [ -z "\$HF_TOKEN" ]; then HF_REPO=""; fi

# Launch WebArena docker stack
cd /opt/webarena-compose
sudo docker compose up -d
# Wait for containers to stabilize (gitlab/magento are slow).
# Required-service readiness is checked by reset_urls.sh; no fixed sleep is proof.
sleep 5

# Manual setup and startup use one fail-closed reset implementation.
export WA_REQUIRED_SERVICES="$REQUIRED_SITES"
sudo --preserve-env=WA_REQUIRED_SERVICES bash /opt/webarena-compose/reset_urls.sh

# Site URLs — BrowserGym WebArena expects WA_* naming
export WA_SHOPPING=http://localhost:7770
export WA_SHOPPING_ADMIN=http://localhost:7780
export WA_REDDIT=http://localhost:9999
export WA_GITLAB=http://localhost:8023
# Unsupported Wikipedia/map tasks were rejected by the local manifest preflight.
unset WA_WIKIPEDIA WA_MAP
export WA_HOMEPAGE=http://localhost:4399

# Patch missing deps at runtime (image was frozen before we discovered orjson requirement)
/opt/tinker/bin/python -c "import orjson"

test "\$(git -C /opt/tinker-rl-lab rev-parse HEAD)" = "$TINKER_REPO_REVISION" || { echo "Worker image code revision mismatch" >&2; exit 1; }

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


# NOTE: --concurrency 1 is mandatory below. Playwright's sync API maintains
# a single client per process pinned to the thread that created it. With
# --concurrency >1, multiple episodes share the process but run on different
# worker threads (even with single-worker executors per episode), and
# Playwright raises "cannot switch to a different thread" at env.reset.
# To parallelize, scale horizontally (more VMs), not --concurrency.
PHASE=evaluation
timeout "$MAX_RUN_SECONDS" /opt/tinker/bin/python -m platform_hybrid.experiments.webarena.react_eval \\
  --benchmark "\$BENCHMARK" \\
  --tasks "$TASKS" \\
  --resume \\
  --model "\$MODEL" \\
  --max-steps "\$MAX_STEPS" \\
  --concurrency 1 \\
  --out "/tmp/results_shard_\${WORKER_ID}.jsonl" \\
  --shard "\${WORKER_ID}/\${NUM_WORKERS}" \\
  \$WANDB_FLAGS \$HF_FLAGS

gsutil cp "/tmp/results_shard_\${WORKER_ID}.jsonl" \\
  "gs://\${BUCKET}/\${RUN_ID}/results_shard_\${WORKER_ID}.jsonl"

# The EXIT trap publishes a terminal receipt. The foreground controller owns
# group deletion, which also reduces desired capacity rather than recreating VMs.
exit 0
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
  --metadata-from-file=startup-script="$STARTUP_FILE" \
  --metadata=num-workers="$NUM_WORKERS",benchmark="$BENCHMARK",model="$MODEL",max-steps="$MAX_STEPS",bucket="$BUCKET",run-id="$RUN_ID",wandb-project="$WANDB_PROJECT",hf-repo="$HF_REPO",hf-private="$HF_PRIVATE" \
  $PREEMPT_FLAG

TEMPLATE_CREATED=true
TEMPLATE_ID=$(gcloud compute instance-templates describe "$TEMPLATE_NAME" --project="$GCP_PROJECT" --format='value(id)')
echo "==> Creating MIG $MIG_NAME with $NUM_WORKERS workers..."
gcloud compute instance-groups managed create "$MIG_NAME" \
  --zone="$GCP_ZONE" \
  --template="$TEMPLATE_NAME" \
  --size="$NUM_WORKERS"

MIG_CREATED=true
MIG_ID=$(gcloud compute instance-groups managed describe "$MIG_NAME" --project="$GCP_PROJECT" --zone="$GCP_ZONE" --format='value(id)')
# Store stateful shard metadata so a replacement inherits its assignment.
echo "==> Patching per-instance worker-id..."
i=0
for instance in $(gcloud compute instance-groups managed list-instances "$MIG_NAME" --zone="$GCP_ZONE" --format="value(name)"); do
  gcloud compute instance-groups managed instance-configs create "$MIG_NAME" \
    --project="$GCP_PROJECT" --zone="$GCP_ZONE" --instance="$instance" \
    --stateful-metadata="worker-id=$i"
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
echo "                python -m platform_hybrid.experiments.webarena.aggregate --inputs 'results/*.jsonl' --out final.json"
echo "==> Cleanup:    gcloud compute instance-groups managed delete $MIG_NAME --zone=$GCP_ZONE -q && \\"
echo "                gcloud compute instance-templates delete $TEMPLATE_NAME -q"

# Foreground bounded controller. SIGKILL/host loss still needs an independent
# deployment expiry controller; never describe this local trap as that guarantee.
deadline=$((SECONDS + MAX_RUN_SECONDS + 900))
while [ "$SECONDS" -lt "$deadline" ]; do
  complete=0
  failed=0
  for ((shard=0; shard<NUM_WORKERS; shard++)); do
    receipt=$(gsutil cat "gs://$BUCKET/$RUN_ID/shard_${shard}.receipt.json" 2>/dev/null || true)
    if [ -n "$receipt" ]; then
      if printf '%s' "$receipt" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d["protocol"]=="webarena-shard-v2" and d["run_id"]==sys.argv[1] and d["shard"]==int(sys.argv[2]) and d["task_manifest_sha256"]==sys.argv[3] and d["repo_revision"]==sys.argv[4]; sys.exit(0 if d["exit_code"]==0 else 1)' "$RUN_ID" "$shard" "$WEBARENA_TASK_MANIFEST_SHA256" "$TINKER_REPO_REVISION"; then
        complete=$((complete+1))
      else
        failed=1
      fi
    fi
  done
  if [ "$failed" -ne 0 ]; then echo "A shard failed; retaining durable partial artifacts" >&2; exit 1; fi
  if [ "$complete" -eq "$NUM_WORKERS" ]; then echo "All terminal shard receipts collected; releasing owned resources"; exit 0; fi
  sleep 30
done
echo "Deadline exceeded before complete shard coverage" >&2
exit 1
