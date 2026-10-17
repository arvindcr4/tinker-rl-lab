> Review-10 safety gate: the historical commands below are not deployment-ready
> without the required immutable-input variables and reviewed task/site manifest.
> Full WebArena is blocked for missing forum/map/Wikipedia/judge contracts.
> No live deployment was validated. See `docs/reviews/pro_review_10_resolution.md`.

# WebArena-verified parallel eval on GCP

Fastest path to a full WebArena-verified score: 50-way parallel eval across 10 VMs,
each with its own WebArena stack + one Chromium worker. Wall-clock ~60 min after
image is built.

## One-time prerequisites

```sh
# 1. Log into gcloud (token is currently expired)
gcloud auth login
gcloud auth application-default login

# 2. Confirm billing + project
gcloud config set project electric-armor-388216
gcloud services enable compute.googleapis.com secretmanager.googleapis.com storage.googleapis.com

# 3. Store Tinker API key in Secret Manager (so VMs can pull it without checking in)
echo -n "$TINKER_API_KEY" | gcloud secrets create tinker-api-key --data-file=-
# or update:
# echo -n "$TINKER_API_KEY" | gcloud secrets versions add tinker-api-key --data-file=-

# 4. Create a GCS bucket for per-shard results
gsutil mb -l us-central1 gs://YOUR-WEBARENA-BUCKET
```

## Step 1 — Build the custom image (~45 min, one-time)

```sh
GCP_PROJECT=electric-armor-388216 GCP_ZONE=us-central1-a \
  ./platform_gcp/infra/gcp/build_webarena_image.sh
```

Produces image family `webarena`. Skip this step for subsequent eval runs.

## Step 2 — Deploy the managed instance group (~5 min to spin up)

```sh
GCP_PROJECT=electric-armor-388216 \
BUCKET=YOUR-WEBARENA-BUCKET \
NUM_WORKERS=10 \
BENCHMARK=webarena_verified \
MODEL=Qwen/Qwen3-8B \
MAX_STEPS=30 \
PREEMPTIBLE=false \
  ./platform_gcp/infra/gcp/deploy_mig.sh
```

Each VM:
1. Boots from `webarena` image (~2 min)
2. Starts the WebArena Docker stack locally (~3 min)
3. Fetches Tinker key from Secret Manager
4. Runs its shard of tasks (`k/N`) at concurrency=1
5. Uploads `results_shard_k.jsonl` to `gs://$BUCKET/$RUN_ID/`
6. Publishes a terminal shard receipt; the foreground launcher deletes its owned MIG and template after all receipts arrive or the deadline expires

## Step 3 — Aggregate

```sh
RUN_ID=20260422-120000
mkdir -p results/
gsutil cp gs://YOUR-WEBARENA-BUCKET/$RUN_ID/*.jsonl results/
python -m platform_hybrid.experiments.webarena.aggregate \
  --inputs 'results/*.jsonl' --out results/final_$RUN_ID.json
```

Outputs a JSON with `success_rate`, `mean_reward`, per-task breakdown.

## Cost estimate

| Phase | Time | Cost |
|---|---|---|
| Image build (one-time) | 45 min × 1 e2-standard-4 | ~$0.10 |
| Eval run (10 VMs, 1 hr wall) | 1 hr × 10 × $0.134/hr | ~$1.34 |
| Disk: 10 × 1000 GB × 1 hr | | ~$1.10 |
| Egress: negligible | | ~$0 |
| **GCP total per run** | | **~$2.50** |
| Tinker sampling | 300 tasks × 30 steps × ~8k tokens | ~$30–80 |

With `PREEMPTIBLE=true`: cut GCP cost ~70% (VM may be evicted; resume via
`react_eval.py --resume`).

## Gotchas

- WebArena-verified has ~300 tasks. 10 workers × ~30 tasks × ~2 min/task = ~60 min.
- If a VM is preempted mid-shard, re-run that shard with `--resume`.
- `/dev/shm` must be ≥2GB for Chromium. `e2-standard-4` defaults to 64MB — the
  Docker compose in `webarena-compose/` must bind-mount `/dev/shm` with `size=2g`.
- WebArena's `full_reset()` happens once per epoch inside each VM; Docker stack
  itself boots from a known-good snapshot every time.

## Troubleshooting

```sh
# Watch an instance's startup logs
gcloud compute instances get-serial-port-output <instance-name> --zone=us-central1-a

# SSH into a worker
gcloud compute ssh <instance-name> --zone=us-central1-a
# Then:
sudo docker ps
sudo tail -f /var/log/syslog
cat /tmp/results_shard_*.jsonl | wc -l
```

## Quick local smoke (before GCP)

Validates the ReAct env works against MiniWoB locally — free, 10 min:

```sh
export TINKER_API_KEY=...
export MINIWOB_URL=http://localhost:8000/miniwob/
# In another terminal: cd miniwob-plusplus/miniwob/html && python -m http.server 8000

python -m platform_hybrid.experiments.webarena.react_eval \
  --benchmark miniwob \
  --tasks miniwob.click-button,miniwob.choose-list,miniwob.enter-text \
  --model Qwen/Qwen3-8B \
  --max-steps 10 --concurrency 2 \
  --out /tmp/miniwob_smoke.jsonl
```

Expected: non-zero success rate on `miniwob.click-button` (easy task).
If score is 0, debug **before** spending GCP $.

## Lifecycle limits after review 10

The v2 launcher runs a foreground controller (`MAX_RUN_SECONDS`, default six
hours, plus a 15-minute startup allowance), preserves worker IDs in per-instance
stateful metadata, restores `--resume` JSONL, and uploads progress every minute.
Normal completion, failures, and caught signals clean up only resources whose
recorded immutable IDs still match. Leave this controller running.

This is not an independent expiry service: launcher host loss or SIGKILL, or an
ambiguous provider create response, still requires operator reconciliation and
cleanup. Do not deploy unattended until an independently authorized expiry
controller exists. GCP API/CLI compatibility and replacement behavior require
staging validation; no cloud test was performed. Up to one minute of progress
may be repeated after preemption; aggregation must separately validate task IDs,
artifact hashes and infrastructure-error treatment. Do not pool these runs with
historical results without their protocol identity.

### Required immutable inputs

Image build now requires `TINKER_REPO_REVISION`, `WEBARENA_REPO_REVISION`,
`WEBARENA_SHA256_MANIFEST`, its separately trusted `WEBARENA_SHA256_ANCHOR`, and a
`KIWIX_IMAGE` reference ending in `@sha256:<digest>`. Missing or mismatched inputs
fail before builder creation. The manifest lists SHA-256 values and exact archive
basenames. Obtain the anchor from a reviewed release, not from an untrusted
bundle alongside the archives.

MIG launch requires the image's exact `TINKER_REPO_REVISION`,
`WEBARENA_TASK_MANIFEST` and its separately reviewed
`WEBARENA_TASK_MANIFEST_SHA256`. The manifest uses schema
`webarena-service-contract-v1`, names the benchmark, and enumerates `tasks` as
objects with `id`, `required_sites`, and optional `requires_judge`. Only
shopping, shopping_admin, gitlab and homepage task dependencies are currently
supported. Actual task IDs and dependencies must come from the reviewed benchmark
manifest; no example or automatically inferred subset is authorized here.
