# Reproducing tinker-rl-lab results

**Replay boundary:** the current v2 primary records require access to the named Modal volume; a public clone alone cannot recompute them. Stored-score recomputation, completion regrading, inference and training are separate verification levels. Verify run configuration and evaluation identity before pairing; the compact replay below validates numeric records but cannot authenticate missing provenance. Trainer-output conversion to the §D statistics JSONL layout remains an incomplete manual step. Run shell recipes with `set -euo pipefail` (and use `bash -o pipefail` inside Docker pipelines) so training failures cannot be hidden by `tee`. No command here authorizes paid execution.

Copy-pasteable commands for reviewers. Every command marked **[offline]** was run
on a laptop (CPU, no API keys) with the timing shown; commands marked **[costed]**
need a GPU or a paid API and state the hardware and hours.

## Which paper, which entry point

| Document | LaTeX entry point | What this file reproduces |
|---|---|---|
| Stack-provenance paper (claim C; A and B are case studies) | `platform_hybrid/paper/paper_P6_registry.tex` (artifact: `platform_hybrid/registry/`) | Parts A, B, C below |
| Thesis | `reports/public_revision_2026-10-04/thesis/thesis_master.tex` | same-stack §6.5.5 (Part A), ZVF (Part B) |
| April 2026 NeurIPS compendium (historical) | `platform_hybrid/paper/archive/absorbed/U01_main_compendium/main.tex` (moved there from `platform_hybrid/paper/main.tex`, which no longer exists) | Part D (Semester 3 headline) |

The anonymized compendium source is `platform_local/blind_review/main_anon.tex`.
There is no anonymized variant of `paper_P6_registry.tex`; build one with
`tools/anonymize_export.py` (writes `dist/anon_export/`).

## Pin the revision you reproduce

There is no frozen release tag for the current work (`v1.0.0-neurips-2026` was
never pushed). Record the exact commit you ran instead of a placeholder:

```bash
set -euo pipefail
git rev-parse HEAD                   # paste this hash into your report
git status --porcelain | head        # must be empty for a clean reproduction
git describe --tags --always --dirty # human-readable, e.g. v3.0-neurips-submission-NNN-gabcdef
```

To reproduce the April 19, 2026 Semester 3 submission state (Part D), check out
the commit recorded in `ARTIFACT.md` (`9019b5c`) and confirm it with
`git rev-parse 9019b5c` (expect `9019b5c946f17ffe4761b5f4eac13b86d1e51cc0`).
The Docker build stamps the commit into the image:
`--build-arg GIT_COMMIT=$(git rev-parse HEAD)`.

## Environment for the offline parts

Parts A–C offline need only Python ≥ 3.11 and the standard library. The test suite
needs the project venv:

```bash
set -euo pipefail
uv venv .venv --python 3.12 && source .venv/bin/activate && uv pip install -r requirements.txt
source .venv/bin/activate     # smoke_test.sh calls bare `pytest`; without the venv step 3 fails on `import torch`
bash platform_modal/scripts/smoke_test.sh      # [offline] 6 checks
python -m pytest -q tests/test_samestack_statistics.py tests/test_pcd_vs_zvf.py   # [offline] 50 tests, ~1 s
```

---

## Part A — Same-stack GRPO vs PPO (Qwen2.5-1.5B-Instruct, GSM8K-CoT)

Both versions use one trainer for every arm (same model, LoRA r=16 on q/k/v/o,
64 generations/step, 30 steps, K=2 inner epochs, clip 0.2, lr 1e-5, A10G, bf16);
only the advantage estimator and prompts×group split differ.

| | v1 | v2 |
|---|---|---|
| Script | `platform_hybrid/experiments/modal/modal_samestack_gsm8k_cot.py` | `platform_hybrid/experiments/modal/modal_samestack_gsm8k_cot_v2.py` |
| Colab port (no Modal) | — | `platform_colab/publication_fixes/samestack_v2_colab.py` |
| Token cap / critic | 200 (binding) / cold value head | 512 / zero-init probe, 4 warm-up batches |
| Eval | 200 test items, 5 seeds | full 1,319 test items, 10 seeds |
| Stored results | `platform_hybrid/experiments/results/samestack_gsm8k_cot{,_full}.json` | Modal volume `tinkerrl-results:/samestack_gsm8k_v2s/` (per run: `<arm>_s<seed>.json`, `<arm>_s<seed>_texts.json.gz`, `adapters/<arm>_s<seed>/`); repo holds only `samestack_gsm8k_cot_v2_smoke*.json` |
| Adapters | not kept | HF `arvindcr4/samestack-v2-qwen2.5-1.5b-gsm8k-lora` (27 LoRA adapters, one subfolder per arm×seed) |

v2 status: 27/40 runs finished (grpo_g8 ×10, grpo_g2 ×10, ppo ×7); `ppo` seeds
4096/5555/7777 and all 10 `ppo_8x8` runs are missing.

### A.1 Analysis-only replay from stored per-run JSONs  [offline]

Re-derives every contrast with the v2 script's own `paired`/`signflip_p`/`holm`
functions, loaded via `ast` so neither Modal nor torch is imported. Save as
`/tmp/samestack_replay.py` and run from the repo root:

```bash
set -euo pipefail
cat > /tmp/samestack_replay.py <<'PY'
import ast, glob, itertools, json, math, os, sys
SRC = "platform_hybrid/experiments/modal/modal_samestack_gsm8k_cot_v2.py"
tree = ast.parse(open(SRC).read())
keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in
        {"_t_cdf", "_t_ppf", "signflip_p", "paired", "holm"}) or
        (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "MARGIN" for t in n.targets))]
ns = {"math": math, "itertools": itertools}
exec(compile(ast.Module(body=keep, type_ignores=[]), SRC, "exec"), ns)
paired, holm = ns["paired"], ns["holm"]
if sys.argv[1] == "v1":
    runs = json.load(open("platform_hybrid/experiments/results/samestack_gsm8k_cot.json"))["runs"]
    primary = [("grpo_g8", "ppo"), ("grpo_g8", "grpo_g2")]
else:
    runs = [json.load(open(f)) for f in glob.glob(os.path.join(sys.argv[2], "*_s*.json"))]
    primary = [("grpo_g8", "ppo"), ("grpo_g8", "ppo_8x8"), ("grpo_g8", "grpo_g2")]
if not runs:
    raise ValueError("No result records found")
post = {}
seen = set()
for r in runs:
    key = (r["arm"], r["seed"])
    if key in seen:
        raise ValueError(f"Duplicate arm/seed: {key}")
    seen.add(key)
    for field in ("heldout_pre_acc", "heldout_post_acc"):
        value = r[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid {field}: {key}")
    post.setdefault(r["arm"], {})[r["seed"]] = r["heldout_post_acc"]
for a, v in sorted(post.items()):
    pre = [r["heldout_pre_acc"] for r in runs if r["arm"] == a]
    print(f"{a:8s} n={len(v):2d} pre={sum(pre)/len(pre):.4f} post={sum(v.values())/len(v):.4f}")
cs = {f"{x}-{y}": paired(post[x], post[y]) for x, y in primary
      if x in post and y in post and len(set(post[x]) & set(post[y])) >= 2}
family = [f"{x}-{y}" for x, y in primary]
hp = holm({k: cs[k]["p_two_sided"] if k in cs else 1.0 for k in family})
for k in family:
    if k not in cs:
        print(f"{k}: NOT_RUN; declared-family analysis incomplete")
for k, c in cs.items():
    print(f"{k:16s} n={c['n_seeds']:2d} diff={c['mean_diff']:+.4f} "
          f"CI95=[{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}] p_t={c['p_two_sided']:.4f} "
          f"signflip={c['signflip_p']:.4f} holm_t={hp[k]:.4f} TOST(+/-{c['tost_margin']}) p={c['tost_p']:.4f}")
PY

# v1 (stored in the repo)  — 0.03 s
python3 /tmp/samestack_replay.py v1

# v2: copy the per-run files off the Modal volume (reads are free; needs `modal token set`)
mkdir -p /tmp/v2s && (cd /tmp/v2s && modal volume get tinkerrl-results samestack_gsm8k_v2s)   # ~1 min, ~0.5 GB incl. adapters
python3 /tmp/samestack_replay.py v2 /tmp/v2s/samestack_gsm8k_v2s      # 0.03 s
```

Historical output (preserved v2 snapshot; the repaired replay retains the full registered family and therefore changes Holm-adjusted output, not the underlying results):

```
# v1
grpo_g8-ppo      n= 5 diff=+0.0500 CI95=[+0.0154,+0.0846] p_t=0.0159 signflip=0.0625 holm_t=0.0318 TOST(+/-0.02) p=0.9632
grpo_g8-grpo_g2  n= 5 diff=-0.0050 CI95=[-0.0376,+0.0276] p_t=0.6918 signflip=0.7500 holm_t=0.6918 TOST(+/-0.02) p=0.1350
# v2   (arms: grpo_g2 0.6547->0.6908, grpo_g8 0.6546->0.6987, ppo 0.6540->0.6905)
grpo_g8-ppo      n= 7 diff=+0.0076 CI95=[-0.0034,+0.0185] p_t=0.1415 signflip=0.1562 holm_t=0.1415 TOST(+/-0.02) p=0.0162
grpo_g8-grpo_g2  n=10 diff=+0.0079 CI95=[+0.0002,+0.0156] p_t=0.0452 signflip=0.0508 holm_t=0.0904 TOST(+/-0.02) p=0.0030
```

Notes. (1) v1 PPO post-training accuracy (0.195) is below its base (0.200), so
the v1 +0.050 is "GRPO learned, PPO did not", not an estimator comparison; the
exact sign-flip p (0.0625) is the honest small-n test. (2) The historical output above used available contrasts only; the repaired replay keeps missing contrasts as internal p=1 placeholders, displays NOT_RUN, and retains the pre-registered v2 family of three contrasts;
with m=3 the Holm-adjusted p values are 0.28 (g8−ppo) and 0.14 (g8−g2).
(3) Item-level regrading: each `<arm>_s<seed>_texts.json.gz` holds the greedy
completions; re-score them with the script's `\boxed{}` extractor to audit the
stored per-item correctness (`pre_correct`/`post_correct` in `<arm>_s<seed>.json`).

### A.2 Full reruns  [costed]

```bash
set -euo pipefail
# v1 — 15 runs (3 arms × 5 seeds), Modal A10G; ~3.5 A10G-h total, ~14 min/run
(cd platform_hybrid && modal run experiments/modal/modal_samestack_gsm8k_cot.py)

# v2 — 40 runs (4 arms × 10 seeds), Modal A10G; ~42–48 min/run, ~29 A10G-h total
((cd platform_hybrid && modal run experiments/modal/modal_samestack_gsm8k_cot_v2.py) --smoke)   # 4 tiny runs first
(cd platform_hybrid && modal run experiments/modal/modal_samestack_gsm8k_cot_v2.py)

# v2 on any single GPU without Modal (Colab A100/L4/T4-high-RAM or a local CUDA box):
pip install "torch>=2.3" "transformers>=4.46" "peft>=0.13" "datasets>=3.0" "numpy<2" accelerate safetensors
python platform_colab/publication_fixes/samestack_v2_colab.py --arm ppo --seed 4096 --out_dir /content/v2s
#   one arm×seed per call; then: python3 /tmp/samestack_replay.py v2 /content/v2s
```

Finishing the missing 13 v2 runs (`ppo` × {4096, 5555, 7777}, `ppo_8x8` × 10)
is ~10 A10G-h. The Modal entrypoint writes the aggregate to
`platform_hybrid/experiments/results/samestack_gsm8k_cot_v2{,_full}.json`; the
Colab port writes only per-run files (it adds an `env` block with hardware and
package versions).

---

## Part B — Zero-Variance Fraction (ZVF)

### B.1 ZVF recompute from stored reward tensors  [offline, 0.03 s]

`platform_hybrid/experiments/results/tinker_gsm8k_zvf_s{42,123,456}.json` store
the 200×8 binary reward matrix per seed (Qwen3-8B via Tinker sampling, G=8,
T=1.0). ZVF is the fraction of prompts whose 8 rewards are all equal.

```bash
set -euo pipefail
python3 - <<'PY'
import json
R = "platform_hybrid/experiments/results/"
h = lambda p, G=8: p**G + (1 - p)**G          # homogeneous-binomial ZVF prediction
for s in (42, 123, 456):
    d = json.load(open(f"{R}tinker_gsm8k_zvf_s{s}.json"))
    ks = [sum(int(x) for x in pp["rewards"]) for pp in d["per_problem"]]
    zvf = sum(k in (0, 8) for k in ks) / len(ks); acc = sum(ks) / (8 * len(ks))
    print(f"seed {s}: ZVF={zvf:.3f} (stored {d['overall_zvf']:.3f})  acc={acc:.4f}  "
          f"h8(acc)={h(acc):.4f}  all-correct={sum(k==8 for k in ks)} all-wrong={sum(k==0 for k in ks)}")
PY
```

Expected: `ZVF=0.130 / 0.190 / 0.155` (matches `tinker_gsm8k_zvf_summary.json`,
mean 0.158). Interpretation for the paper: ZVF is, to first order, the binomial
identity mean_i h_G(p_i) over per-prompt solve rates; the pooled single-p
prediction (≈0.05) is lower than the observed values. Heterogeneous latent prompt probabilities raise expected degeneracy under conditional independence; these counts do not uniquely identify heterogeneity rather than dependence or batch effects. Plug-in estimates also have finite-sample error.

Regenerating the tensors is **[costed]** (Tinker API, Qwen3-8B, 200×8 samples
per seed, a few minutes and < $1 per seed):

```bash
set -euo pipefail
export TINKER_API_KEY=...
python3 platform_hybrid/experiments/tinker_direct_eval.py --model Qwen/Qwen3-8B \
    --n-problems 200 --group-size 8 --temperature 1.0 --seed 42 \
    --out platform_hybrid/experiments/results/tinker_gsm8k_zvf_s42.json
```

### B.2 W&B `zvf-audit` audit

Offline snapshot  **[offline, 0.04 s]** — the repo stores a W&B export of 940 runs
(368 in project `zvf-audit`) at
`platform_hybrid/experiments/results/tinker_wandb_registry/wandb_runs.jsonl`:

```bash
set -euo pipefail
python3 - <<'PY'
import json, statistics as st
from collections import defaultdict
rs = [json.loads(l) for l in open("platform_hybrid/experiments/results/tinker_wandb_registry/wandb_runs.jsonl")]
z = [r for r in rs if r["project"] == "zvf-audit"]
by = defaultdict(list)
for r in z:
    if "mean_zvf" in r["summary"]:
        by[r["model"].split("/")[-1]].append(r["summary"]["mean_zvf"])
print(f"zvf-audit runs: {len(z)}; with summary mean_zvf: {sum(map(len, by.values()))}")
for m, v in sorted(by.items(), key=lambda kv: st.mean(kv[1])):
    print(f"  {m:36s} n={len(v):3d} mean_zvf={st.mean(v):.3f}")
PY
```

Expected: `zvf-audit runs: 368; with summary mean_zvf: 350`, from Qwen3-32B
(0.253) to Llama-3.2-1B (0.973).

Live re-aggregation (needs read access to the W&B project; network, no GPU):

```bash
set -euo pipefail
WANDB_API_KEY=... python3 platform_hybrid/experiments/tinker-runs/wandb_zvf_audit_agg.py
# writes platform_hybrid/experiments/tinker-runs/results/zvf_audit_summary.{json,md}
```

Not recomputable from stored data: corr(‖∇‖, p(1−p)) = +0.71 (only the scalar
was kept).

ZVF adapter (one GRPO run, Qwen3.5-4B, seed 123, 15 steps, Tinker):
HF `arvindcr4/zvf-grpo-qwen3.5-4b-gsm8k-s123`; run record
`platform_hybrid/experiments/tinker-runs/results/h2h2_grpo_Qwen3.5-4B_s123_G8_Dmedium_R16_S15.json`.

---

## Part C — Stack-provenance registry and `stackdiff`

Artifact: `platform_hybrid/registry/` (`schema.json`, `entries/*.json`,
`query.py`, `README.md`). `query.py` uses only the standard library and works from
the repo root or from `platform_hybrid/`. Its CLI is `query.py <subcommand> [args]`;
`python3 platform_hybrid/registry/query.py --help` lists the current subcommands
and `query.py <subcommand> --help` the arguments. The subcommands the paper uses:

```bash
set -euo pipefail
R=platform_hybrid/registry/query.py
python3 $R list                                  # [offline] one line per stack entry
python3 $R validate                              # [offline] schema check over every entry
python3 $R badge tinker_grpo_qwen3-8b_gsm8k      # [offline] MIN-REPORT coverage badge for one entry
python3 $R query --item reference_kl             # [offline] one MIN-REPORT item across entries
python3 $R stackdiff tinker_grpo_qwen3-8b_gsm8k trl_grpo_qwen3-8b_gsm8k   # [offline] the paper's main exhibit
python3 $R stackdiff colab-open_dapo_e3 tinker_dapo_qwen3.5-4b_gsm8k      # [offline] the "DAPO" pair
```

Each runs in < 0.1 s (`stackdiff --all` in < 1 s). `stackdiff A B` prints the shared
claimed label, every differing field tagged with a risk level from `[R1]` (nuisance)
to `[R5]` (outcome-critical: checkpoint, reward, prompt/chat template, dataset, eval
protocol), a verdict line, field coverage, and an `UNDERSPECIFIED` flag when fewer than
80% of the compared fields are known on both sides. Useful flags (see
`query.py stackdiff --help`):

```bash
set -euo pipefail
python3 $R stackdiff --sources tinker_grpo_qwen3-8b_gsm8k trl_grpo_qwen3-8b_gsm8k  # provenance path for each field
python3 $R stackdiff -v tinker_grpo_qwen3-8b_gsm8k trl_grpo_qwen3-8b_gsm8k         # also list unknown fields
python3 $R stackdiff --all                                                          # R-level distribution over all stack pairs
python3 -m pytest -q tests/test_registry_stackdiff.py                               # [offline] stackdiff unit tests, < 1 s
```

Entry ids are the file stems in `platform_hybrid/registry/entries/`. The `stackdiff`
implementation and the entries were being revised on 2026-10-16, so the exact field
lists and counts printed may change. Re-run the commands above at your pinned commit
rather than comparing against numbers quoted in a PDF.

Recomputing the numbers the paper attaches to registry pairs (Tinker vs. TRL
gap, the 0.034 band) uses the stored results under
`platform_hybrid/experiments/results/` named in each entry's `provenance`
block; see `platform_hybrid/registry/README.md`.

---

## Part D — Semester 3 headline (April 2026 compendium)

This reproduces the Semester 3 headline result on shared infrastructure, as
reported in the archived compendium
(`platform_hybrid/paper/archive/absorbed/U01_main_compendium/main.tex`) and the
capstone final report (`platform_tinker/reports/final/capstone_final_report.md`).
The Semester 4 (P1–P8) paper-to-evidence map is
[`platform_hybrid/sem 4 work/EXPERIMENTS.md`](platform_hybrid/sem%204%20work/EXPERIMENTS.md).

Headline: **GRPO on Qwen3-8B, GSM8K, 30 steps, LoRA rank 32, group size 8** —
peak accuracy **62.5 %**, last-10 average **34.4 %** (compendium Table 2,
`main.tex:582`; W&B run
[`scale_gsm8k_qwen3-8b`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-world-class/runs/vv6uu72m)).
Tolerance for reruns: **±5 pp** on last-10 and **±10 pp** on peak (§D.8).

### D.0 One-command reviewer flow

```bash
set -euo pipefail
docker build \
  --build-arg GIT_COMMIT=$(git rev-parse HEAD) \
  --build-arg GIT_REF=$(git rev-parse --abbrev-ref HEAD) \
  --build-arg BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
  -t tinker-rl-lab:repro .                                             # ~6 min cold
docker run --rm tinker-rl-lab:repro bash platform_modal/scripts/smoke_test.sh            # [offline] < 10 min
docker run --rm -e TINKER_API_KEY tinker-rl-lab:repro bash platform_modal/scripts/smoke_test.sh   # [costed] ~8 min, < $0.25
docker run --gpus all --rm -e TINKER_API_KEY -e WANDB_API_KEY -e HF_TOKEN \
    -v $(pwd)/results:/workspace/tinker-rl-lab/results tinker-rl-lab:repro \
    bash -o pipefail -c 'python platform_tinker/grpo_gsm8k_base.py --model Qwen/Qwen3-8B --seed 42 --rank 32 \
        --steps 30 --group 8 --batch 2 --lr 1e-5 --tag gsm8k_qwen3_8b_s42 2>&1 | tee results/gsm8k_qwen3_8b_s42.log'
                                                                       # [costed] Tinker API, ~5 h wall
python utils/verify_results.py --results-dir results/ \
    --expected-results platform_hybrid/paper/expected_results.json    # exit 0 = pass
```

Expected per-seed block at the end of the training log (printed by
`platform_tinker/tinkerrl/grpo_cli.py`; fractions):

```
[grpo_cli] Seed 42 done.
  avg_last10    : 0.344
  peak_reward   : 0.625
```

### D.1 Prerequisites

#### D.1.1 Option A — Docker (recommended)

```bash
set -euo pipefail
git clone https://github.com/arvindcr4/tinker-rl-lab.git
cd tinker-rl-lab
git rev-parse HEAD        # record this; see "Pin the revision you reproduce"
# Semester 3 submission state only (ARTIFACT.md): git checkout 9019b5c

docker build -t tinker-rl-lab:repro .
docker run --gpus all -it \
    -e TINKER_API_KEY -e WANDB_API_KEY -e HF_TOKEN \
    -v $(pwd)/results:/workspace/tinker-rl-lab/results \
    tinker-rl-lab:repro bash
```

Build tested on: Docker 24.x, nvidia-container-toolkit ≥ 1.14, NVIDIA driver ≥ 550.

#### D.1.2 Option B — Local virtualenv

```bash
set -euo pipefail
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Optional: install the project so "tinkerrl" CLI shortcuts work
pip install -e .
```

Requires Python 3.11+; CUDA 12.4+ and NVIDIA driver ≥ 550 only for local GPU training.
Activate the venv before `smoke_test.sh` (it calls bare `pytest`).

#### D.1.3 Required environment variables

| Variable          | Required for                                 | Where to get it                                  |
|-------------------|----------------------------------------------|--------------------------------------------------|
| `TINKER_API_KEY`  | All Tinker experiments (GRPO, scaling)       | https://tinker-console.thinkingmachines.ai       |
| `WANDB_API_KEY`   | W&B logging (optional; set `WANDB_MODE=offline` to skip) | https://wandb.ai/authorize             |
| `HF_TOKEN`        | HuggingFace model / dataset download         | https://huggingface.co/settings/tokens           |
| `MODAL_TOKEN_ID`  | Re-running the Modal H100 PPO baselines only | https://modal.com/settings/tokens                |
| `MODAL_TOKEN_SECRET` | "                                         | "                                                |

A template lives in [`.env.example`](./.env.example).

---

### D.2 The 10-minute smoke test  (reviewer entry-point)

```bash
set -euo pipefail
bash platform_modal/scripts/smoke_test.sh                   # [offline] 6/6 checks, 67 s on an M-series laptop (2026-10-16)
TINKER_API_KEY=... bash platform_modal/scripts/smoke_test.sh # ~8 min — live Tinker wire-protocol
```

What it verifies — see the header of
[`platform_modal/scripts/smoke_test.sh`](./platform_modal/scripts/smoke_test.sh) for the full list:

1. Core library imports and versions
2. `utils.seed.set_global_seed(42)` is deterministic across Python/NumPy/PyTorch
3. `pytest tests/` green (unit tests)
4. GSM8K dataset loads and the answer-extraction regex works
5. GRPO reward + advantage normalization round-trip
6. File-surface check (all required artifact files present + parseable)
7. [optional] 3-step GRPO on `Qwen/Qwen3.5-4B` against live Tinker (wire-protocol only)

Target wall-clock: **< 10 min**. If it exceeds that, the script prints a warning
but still exits 0 so reviewers see the failure mode.

---

### D.3 Headline result — GRPO Qwen3-8B GSM8K (compendium Table 2)

#### D.3.1 Seeds

The canonical multi-seed set is **`{42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768}`** (all experiments
table; see `ARTIFACT.md §4.3`). The headline single-seed run uses `seed=42`.

#### D.3.2 Command

```bash
set -euo pipefail
for SEED in 42 123 456 789 1024; do
    python platform_tinker/grpo_gsm8k_base.py \
        --model Qwen/Qwen3-8B \
        --seed  $SEED \
        --rank  32 \
        --steps 30 \
        --group 8 \
        --batch 2 \
        --lr    1e-5 \
        --tag   "gsm8k_qwen3_8b_s${SEED}" \
        2>&1 | tee results/gsm8k_qwen3_8b_s${SEED}.log
done
```

#### D.3.3 Expected per-seed output

Each log ends with a per-seed block of the form (fractions, not percent):

```
[grpo_cli] Seed SEED done.
  avg_last10    : 0.XXX
  peak_reward   : 0.YYY
```

`utils/verify_results.py` (§D.8) preserves the seed from this block and applies only
an explicitly applicable reference. The shipped headline reference is scoped to
seed 42. Other seeds and legacy unscoped references report `UNVERIFIED` and exit
nonzero, rather than being compared with seed 42. Provide a `seeds` mapping in an
expectations JSON file when independently supported per-seed expectations exist:
`{"experiment_name": {"seeds": {"42": {"last10": 0.344, "peak": 0.625}}}}`.
Use the actual experiment key and verified source values; this structural example
is not a reference for an arbitrary experiment. A flat reference can instead
carry `seed: 42`. Only references deliberately defined across seeds should carry
`seed_independent: true`. Missing or conflicting seed identities cannot satisfy
a seed-specific reference. `--strict` requires every declared seed reference to
be represented. The historical table below is retained as a source transcription;
this change does not promote its rows into newly authenticated measurements.

| seed | last-10 (paper)* | peak (paper)* | W&B run (reference) |
|------|------------------|---------------|---------------------|
| 42   | 34.4 %           | 62.5 %        | scale_gsm8k_qwen3-8b (`vv6uu72m`) |
| 123  | 39.4 %           | 68.8 %        | grpo_qwen3-8b_gsm8k_seed123 (`c1zca2qb`) |
| 456  | 29.4 %           | 100.0 %       | grpo_qwen3-8b_gsm8k_seed456 (`igejxwwl`) |
| 789  | 28.8 %           | 68.8 %        | grpo_qwen3-8b_gsm8k_seed789 (`28bzhlsa`) |
| 1024 | —                | —             | (reviewer-only rerun)              |

*Values copied from the W&B project
`arvindcr4-pes-university/tinker-rl-lab-world-class`.  The headline value
reported in the paper is **last-10 = 34.4 %, peak = 62.5 %** — the `seed=42`
row above.

#### D.3.4 Aggregating the 5-seed run

```bash
set -euo pipefail
python utils/stats.py \
    --results-dir results/ \
    --experiment  gsm8k_qwen3_8b \
    --bootstrap-samples 10000 \
    --output-dir analysis/
```

`utils/stats.py` reads `results/<experiment>/seed_<N>/*.jsonl` (one JSON object
per line; the last line's `reward/mean`, `accuracy` or `eval/percent_correct` is
the seed's score) and prints the mean, SE and a percentile-bootstrap 95% CI.
`--format csv`, `latex` or `both` (default) writes `results_table.csv`,
`results_table.tex` or both under `--output-dir`, using that same summary.
Missing/nonfinite metrics and empty inputs fail instead of silently producing
zero scores. `--rliable` exits with a clear usage error: these files do not
define the normalized task matrix required for an aggregate comparison.
The trainer does **not** write that layout: it prints the per-seed block above to
the log. Write one `metrics.jsonl` per seed (e.g. `{"accuracy": <avg_last10>}`)
before running this step.

---

### D.4 Cross-framework / ablation experiments

#### D.4.1 Group-size ablation (compendium Table 2 Group-Size block, archived `main.tex:605-610`)

```bash
set -euo pipefail
for G in 2 4 8 16; do
    python platform_tinker/grpo_gsm8k_base.py \
        --model Qwen/Qwen3-8B --seed 42 --rank 32 \
        --steps 30 --group $G --batch 2 --lr 1e-5 \
        --tag "gsm8k_qwen3_8b_g${G}" \
        2>&1 | tee results/gsm8k_qwen3_8b_g${G}.log
done
```

Expected (paper Table 2 Group-Size block):

| G   | peak  | last-10 |
|-----|-------|---------|
| 2   | 50.0% | 37.5%   |
| 4   | 75.0% | 52.1%   |
| 8   | 62.5% | 34.4%   |
| 16  | 71.9% | 38.0%   |

#### D.4.2 Atropos launcher (5-seed, fully parameterized YAML)

Run from `platform_tinker/atropos/` (the `tinker_atropos` package lives there)
with the Atropos dependencies from `requirements.txt` installed (`atroposlib`,
`tenacity`, `latex2sympy2-extended`, `math-verify`; the local dev venv lacks them).
`launch_training.py` has **no `--seed` flag**; seeds are selected through the
per-seed YAML configs (`gsm8k_qwen_8b_seed{1,2,3}.yaml`, which set
`data_seed` 137 / 256 / 512 rather than the §D.3.1 seed list).

```bash
set -euo pipefail
cd platform_tinker/atropos
pip install -e .   # installs the tinker_atropos package the env server imports
# Terminal 1 — Atropos rollout server (installed by atroposlib)
run-api
# Terminal 2 — environment
python tinker_atropos/environments/gsm8k_tinker.py serve \
       --config configs/gsm8k_qwen_8b.yaml
# Terminal 3 — trainer (one run per seed config)
for CFG in gsm8k_qwen_8b gsm8k_qwen_8b_seed1 gsm8k_qwen_8b_seed2 gsm8k_qwen_8b_seed3; do
    python launch_training.py --config configs/${CFG}.yaml
done
```

#### D.4.3 Size ladder (Qwen 0.6B → 30B MoE)

Same environment as §D.4.2 (run from `platform_tinker/atropos/`). One seed per
config; there is no seed override flag.

```bash
set -euo pipefail
cd platform_tinker/atropos
for CFG in gsm8k_qwen_0_6b gsm8k_qwen_1_7b gsm8k_qwen_4b \
           gsm8k_qwen_8b   gsm8k_qwen_14b \
           gsm8k_qwen_30b_moe; do
    python launch_training.py --config configs/${CFG}.yaml
done
```

#### D.4.4 Modal H100 PPO baselines

```bash
set -euo pipefail
modal run platform_modal/scripts/modal_run_experiments.py
```

Requires `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`. The entrypoint takes no
arguments: it runs the four arithmetic-env baselines (TRL GRPO on Qwen2.5-0.5B,
SB3 / CleanRL / Tianshou PPO) over the seeds hard-coded in `SEEDS` and writes
`modal_results_all.json`. There is no single-experiment `ppo_gsm8k_qwen3_8b`
selector.

#### D.4.5 Cross-library RL baselines (arithmetic env)

```bash
set -euo pipefail
for exp in platform_hybrid/experiments/implementations/{trl_grpo_math,sb3_ppo_math,\
cleanrl_ppo_math,tianshou_ppo_math}.py; do
    bash platform_modal/scripts/run_seeds.sh "python $exp" 42 123 456 789 1024
done
```

---

### D.5 Figures and tables

```bash
set -euo pipefail
# 1. Per-experiment mean / SE / bootstrap CI over results/<exp>/seed_<N>/*.jsonl
python utils/stats.py --results-dir results/ \
       --bootstrap-samples 10000 --output-dir analysis/

# 2. Paper figures (matplotlib, 300 dpi PDF + PNG). No CLI flags: reads
#    platform_hybrid/experiments/results/ and writes paper/figures/ relative to
#    the current directory, so run it from the repository root.
python platform_modal/scripts/make_paper_figures.py
```

`utils/stats.py` accepts `--format {latex,csv,both}` and `--rliable`, but its
`--format` writes the requested CSV/LaTeX outputs; `--rliable` fails explicitly when unsupported. See §D.3.4 for the repaired contract.

---

### D.6 Wall-clock budget (reviewer planning)

| Step                                   | Hardware        | Wall-clock       |
|----------------------------------------|-----------------|------------------|
| `docker build` (cold)                  | any             | ~6 min           |
| `platform_modal/scripts/smoke_test.sh` (offline)      | CPU             | ~2 min           |
| `platform_modal/scripts/smoke_test.sh` (with Tinker)  | any + net       | ~8 min           |
| Headline GRPO Qwen3-8B GSM8K (1 seed)  | 1x A100-80GB    | ~5 h             |
| Full 5-seed headline                   | 1x A100-80GB    | ~25 h            |
| Group-size ablation (4 runs × 1 seed)  | 1x A100-80GB    | ~20 h            |
| Size ladder 0.6B–14B (5-seed each)     | 1–2x A100-80GB  | ~130 h           |
| Size ladder 30B MoE (3-seed)           | 4x A100-80GB    | ~96 h            |
| **Total paper reproduction**           |                 | **~446 GPU-h**   |

A reviewer wishing to only verify the headline should plan for
**~5 GPU-hours on a single A100-80GB**, or use the smoke test
(< 10 min, CPU-only) as a sanity check.

---

### D.7 Data provenance

| Dataset            | Source                                | Split / rows             | Pinned revision |
|--------------------|---------------------------------------|--------------------------|-----------------|
| GSM8K              | `openai/gsm8k` @ HF                   | `main/train` (7 473)     | `740312a` (2026-03-23) |
| GSM8K (eval)       | `openai/gsm8k` @ HF                   | `main/test`  (1 319)     | `740312a`        |
| HumanEval          | `openai/openai_humaneval` @ HF        | `test` (164)             | `2025-11-27 snapshot` |
| ArenaHard Auto     | `lm-sys/arena-hard-auto-v0.1` @ HF    | `test` (500)             | `2024-03-01 snapshot` |
| MATH-500           | `HuggingFaceH4/MATH-500` @ HF         | `test` (500)             | `2025-09-10 snapshot` |
| NoRobots (SFT)     | `HuggingFaceH4/no_robots` @ HF        | `train_sft` (9 500)      | `2024-11-18 snapshot` |
| OpenThoughts3 (KD) | `open-thoughts/OpenThoughts3` @ HF    | `train` (subsample 20k)  | `2025-12-02 snapshot` |

Exact shas are recorded in [`ARTIFACT.md §3.4`](./ARTIFACT.md).

---

### D.8 Verifying results

```bash
set -euo pipefail
python utils/verify_results.py \
    --results-dir results/ \
    --expected-results platform_hybrid/paper/expected_results.json \
    --last10-tolerance 0.05 \
    --peak-tolerance   0.10
```

Exit codes: `0` every matched run is within tolerance; `1` a run is outside
tolerance **or no result file matched an expected experiment**; `2` the results
directory or the expectations file is missing. Add `--strict` to also fail when
any expected experiment has no matching result. Result files are `*.json`
(`last10_avg`, `peak`) or `*.log` captures of the trainer output; the log's file
stem is the experiment tag, so name logs as in §D.3.2 / §D.4.1.

**Why the tolerances are not tighter.**
Even with fixed seeds and CuDNN determinism (`utils.seed.set_global_seed`),
three sources of residual non-determinism remain:

1. **Tinker server scheduling** — remote LoRA training is not bit-deterministic
   across scheduler decisions; runs replayed months later drift by 2–4 pts on
   GSM8K last-10 (confirmed via `grpo_qwen3-8b_gsm8k_s123` re-runs:
   `c1zca2qb`, `snm9mxni`, `9lleqd1g`).
2. **CUDA / cuBLAS non-determinism** for some matmul ops on H100/A100.
3. **Sampling temperature** `0.8` with top-p 0.95 — even deterministic seeds
   still drift because the sampler is in a remote VM.

The compendium's Welch t-test on PPO-vs-GRPO (archived `main.tex:512`, `p=0.7605`) shows the
run-to-run variance on this setup is of the order of the reported effect.

---

### D.9 Known gotchas

- **`transformers` version drift** — `requirements.txt` pins
  `transformers>=4.56.2,<4.60.0`; versions outside that range may rename or
  remove Qwen3 config aliases. If you see `AttributeError: Qwen3Config`,
  re-check the installed version against the pin.
- **GSM8K load_dataset script mode** — HF deprecated `trust_remote_code` for
  the GSM8K loader in datasets 3.0; we pass `trust_remote_code=True` only
  where needed. Upgrading past `datasets<3.2` may require a `load_dataset`
  signature change.
- **W&B offline mode** — the Docker image defaults to `WANDB_MODE=offline`
  so reviewers without a W&B account can run without errors. Remove that
  env var to log online.
- **Tinker rate limits** — new API keys are rate-limited to ~2 concurrent
  training clients. The for-loop in §D.3.2 runs seeds serially.

---

### D.10 Contact

Open an issue on https://github.com/arvindcr4/tinker-rl-lab or email the
corresponding author listed in the paper front-matter. When reporting a
discrepancy, include the output of `git rev-parse HEAD`.
