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

PROJECT_ID=\$(gcloud config get-value project)

# Required: Tinker key
export TINKER_API_KEY=\$(gcloud secrets versions access latest --secret=tinker-api-key --project="\$PROJECT_ID")

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
sleep 120

# Reset baked-in CMU reference URLs to localhost. WebArena images hard-code
# http://metis.lti.cs.cmu.edu:<port> in their DB/config so a Location: 302
# from the reference deployment follows back to CMU and dies (CMU is
# unreachable from most GCP VMs). Do NOT skip this step — without it
# Playwright's Page.goto gets ERR_CONNECTION_REFUSED on every shopping/
# shopping_admin/gitlab task.
echo "==> Resetting WebArena base URLs to localhost..."
# Shopping (Magento OneStopShop, port 7770)
sudo docker exec webarena-shopping /var/www/magento2/bin/magento \
  setup:store-config:set --base-url="http://localhost:7770" 2>/dev/null || true
sudo docker exec webarena-shopping /var/www/magento2/bin/magento \
  setup:store-config:set --base-url-secure="http://localhost:7770" 2>/dev/null || true
sudo docker exec webarena-shopping mysql -u magentouser -pMyPassword magentodb \
  -e "UPDATE core_config_data SET value='http://localhost:7770/' WHERE path LIKE 'web/%/base_url';" 2>/dev/null || true
sudo docker exec webarena-shopping /var/www/magento2/bin/magento cache:flush 2>/dev/null || true
# Shopping Admin (Magento CMS Admin, port 7780)
sudo docker exec webarena-shopping-admin /var/www/magento2/bin/magento \
  setup:store-config:set --base-url="http://localhost:7780" 2>/dev/null || true
sudo docker exec webarena-shopping-admin /var/www/magento2/bin/magento \
  setup:store-config:set --base-url-secure="http://localhost:7780" 2>/dev/null || true
sudo docker exec webarena-shopping-admin mysql -u magentouser -pMyPassword magentodb \
  -e "UPDATE core_config_data SET value='http://localhost:7780/' WHERE path LIKE 'web/%/base_url';" 2>/dev/null || true
sudo docker exec webarena-shopping-admin /var/www/magento2/bin/magento cache:flush 2>/dev/null || true
# GitLab (port 8023)
sudo docker exec webarena-gitlab sed -i \
  "s|^external_url.*|external_url 'http://localhost:8023'|" /etc/gitlab/gitlab.rb 2>/dev/null || true
sudo docker exec webarena-gitlab gitlab-ctl reconfigure >/dev/null 2>&1 || true
echo "==> URL reset complete."


# Site URLs — BrowserGym WebArena expects WA_* naming
export WA_SHOPPING=http://localhost:7770
export WA_SHOPPING_ADMIN=http://localhost:7780
export WA_REDDIT=http://localhost:9999
export WA_GITLAB=http://localhost:8023
export WA_WIKIPEDIA=http://localhost:8888
export WA_MAP=http://public-map.server.invalid
export WA_HOMEPAGE=http://localhost:4399

# Patch missing deps at runtime (image was frozen before we discovered orjson requirement)
sudo /opt/tinker/bin/pip install --quiet orjson || true

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

# NOTE: --concurrency 1 is mandatory below. Playwright's sync API maintains
# a single client per process pinned to the thread that created it. With
# --concurrency >1, multiple episodes share the process but run on different
# worker threads (even with single-worker executors per episode), and
# Playwright raises "cannot switch to a different thread" at env.reset.
# To parallelize, scale horizontally (more VMs), not --concurrency.
/opt/tinker/bin/python -m experiments.webarena.react_eval \\
  --benchmark "\$BENCHMARK" \\
  --tasks all \\
  --model "\$MODEL" \\
  --max-steps "\$MAX_STEPS" \\
  --concurrency 1 \\
  --out "/tmp/results_shard_\${WORKER_ID}.jsonl" \\
  --shard "\${WORKER_ID}/\${NUM_WORKERS}" \\
  \$WANDB_FLAGS \$HF_FLAGS

gsutil cp "/tmp/results_shard_\${WORKER_ID}.jsonl" \\
  "gs://\${BUCKET}/\${RUN_ID}/results_shard_\${WORKER_ID}.jsonl"

echo "==> SHARD \$WORKER_ID COMPLETE. VM idle until manual cleanup." | sudo tee /tmp/EVAL_DONE
# Do NOT shutdown — MIG would recreate without worker-id metadata and loop forever.
# Sleep indefinitely so MIG keeps this VM; user deletes MIG when all shards are in GCS.
sleep infinity
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
