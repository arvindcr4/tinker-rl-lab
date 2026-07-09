# tinker-rl-lab — Repository INDEX

> **Agent navigation map.** Every folder in this repo has its own `INDEX.md`; this is the top-level
> router. When you need a file, read the relevant folder's `INDEX.md` first instead of grepping blind.
>
> **What this repo is:** *"A Unified Benchmark for RL Post-Training of Language Models"* — PES University
> MTech capstone (Group 6), prepped for NeurIPS 2026 blind review. GRPO/PPO/DPO post-training of LLMs
> across many RL frameworks (Tinker, SkyRL, verl, OpenRLHF, TRL, Atropos), plus a heavy
> paper / reproducibility / audit apparatus.

## Read first (entry points)
- `README.md` — master overview: repo layout, the 8 components, quick-start commands, provenance.
- `AGENTS.md` + `CONTEXT-MAP.md` — agent workflow + domain-doc routing (→ `contexts/research-engineering/CONTEXT.md`).
- `REPRODUCE.md` — copy-pasteable reviewer commands to reproduce the headline Qwen3-8B GSM8K GRPO result.
- `ARTIFACT.md` — how results map to the paper; provenance table (repo/commit/W&B/DOI).
- `CHANGELOG.md` — repo release history; latest tag `v3.0-neurips-submission`.

## ⭐ The "4 pillars" — the paper's de-confound experiments → `experiments/modal/`
| Pillar | Script | What it isolates |
|--------|--------|------------------|
| 1 · PPO vs GRPO (same-stack) | `experiments/modal/modal_samestack_ppo_grpo.py` | only the advantage estimator differs |
| 2 · Zero-Variance Fraction (ZVF) | `experiments/modal/modal_groupsize_zvf_sweep.py` | measured ZVF + confounders |
| 3 · Trainability / group size | `experiments/modal/modal_groupsize_zvf_sweep.py` | G∈{2,4,8,16} × seeds + held-out |
| 4 · Held-out generalization | `experiments/modal/modal_drgrpo_gsm8k_cot.py` | Dr.GRPO vs GRPO GSM8K-CoT, pre→post McNemar |

## Directory map (each links to its own INDEX.md)

### `experiments/`
RL/SFT/distillation experiments (GRPO/PPO/DPO on GSM8K, tool-use, code) + the aggregation → statistics →
paper-rendering pipeline. Start at `experiments/experiment_summary.md` (all 44 runs) and
`experiments/master_results.{json,csv}`. The 4 pillars live in `experiments/modal/`; Tinker-API GRPO
campaigns in `experiments/tinker-runs/`; the "structural ceiling" study in `experiments/10x_structural_ceiling/`;
raw traces/eval in `experiments/results/`.

### `atropos/`
Tinker ↔ Atropos (NousResearch RL environments) integration. GRPO training on Atropos envs via the hosted
Tinker API + an Unsloth/TRL baseline. `tinker_atropos/` = core package (`trainer.py`, envs: GSM8K/MATH/tool-use/
HumanEval/logp-steering); `configs/` (per model×env YAML; `sweep_results/` = 108-config lr×lora×bs×gs sweep);
`notebooks/`. Entry: `launch_training.py`, `serve.py`, `run_experiment.sh`.

### Paper & writing — `paper/` · `reports/` · `blind_review/` · `submission/` · `capstone-literature-survey/`
- `paper/` — canonical LaTeX (`main.tex` + `main_anon.tex`, `sections/*.tex`, `figures/`+`figures/v2/`, `tikz/`, `reviewer_points.yaml`).
- `reports/final/` — final capstone report, held-out GSM8K eval code + seed JSONs, reviewer-response material (`addendum/`, `chatgpt_responses/`).
- `blind_review/` — anonymized NeurIPS package (`main_anon.pdf/.tex`, `anonymize_*.py`, `AUDIT.md`, `SUBMISSION_MANIFEST.md`).
- `submission/contents/` — packaged reviewer bundle (MANIFEST, checksums, REVIEWER_README).
- `capstone-literature-survey/` — Chapter-2 background survey (RLHF→PPO→DPO→GRPO→R1, PEFT, scaling laws).

### RL framework integrations (each wraps a different backend behind the Tinker API)
- `skyrl/` — SkyRL tx: a **local** Tinker API server on your own / vast.ai / Colab GPUs.
- `tinkerrl/` — consolidated GRPO loop vs the hosted Tinker API (source of truth for `grpo_*` scripts).
- `verl/` — verl (Volcano Engine RL / HybridFlow, Ray + vLLM).
- `openrlhf/` — OpenRLHF (Ray + vLLM; PPO/DAPO/REINFORCE++).
- `trl_integrations/` — HuggingFace TRL (reference same-stack runner).
- `unified/` — one launcher: `python -m unified.launcher` dispatches across all of the above.
- `huggingface/` — Hub checkpoint upload · `tests/` — GRPO + util smoke tests · `scripts/` — ~25 experiment/figure/eval/stats utils · `reproducibility/` — cheap claim-verification checks.

### Integrations, tooling & misc
- `ai-scientist-template/` + `ai-scientist-v2-integration/` — Sakana AI-Scientist (v1 + v2/BFTS) that autonomously run GRPO experiments and write/self-review papers.
- `agentic-rl-finetuning/` — Axolotl SFT-QLoRA → DPO Colab pipeline for large Qwen3 models Tinker RL can't host.
- `contexts/` + `docs/` + `.codex/` — agent/skill machinery: domain vocab & ADRs, issue-tracker/triage conventions, Codex subagent roles.
- `demo_recording/` + `grpo_ablation_results/` — talk deliverables (HF-Space demo media; ablation + reviewer-objections report).
- `graphify-out/` (repo knowledge graph + ~171-file cache), `.firecrawl/` (raw web-search dumps), `.github/workflows/` (ruff + pytest + reproducibility CI).

## Root-level files (by theme)
- **Entry docs:** `README.md`, `AGENTS.md`, `CONTEXT-MAP.md`, `CONTRIBUTING.md`, `REPRODUCE.md`, `CHANGELOG.md`.
- **Submission checklists & meta:** `ELEVATION_ROADMAP.md`, `ARTIFACT.md`, `LIMITATIONS_AND_IMPACT.md`, `FINAL_HANDOFF.md`, `INTEGRATION_LOG.md`, `NEURIPS_CHECKLIST.md`, `NEURIPS_CHECKLIST_FINAL.md`, `ACM_CHECKLIST.md`, `BASELINES.md`, `BENCHMARKS_COMPARISON.md`, `COMPUTE.md`.
- **Top-level GRPO drivers** (need `TINKER_API_KEY`): `grpo_gsm8k_base.py` (parameterized multi-seed), `grpo_tooluse_tinker.py` (Qwen3-8B tool-use), `grpo_100_{math,synthetic,xlam}.py`, `grpo_exp_{a,b,c,d}_*.py` (LR/temp/dataset ablations).
- **Audit suite:** `run_all_audits.py` orchestrates 16 `*_audit.py` scripts (paper/claim/abstract/heldout/anonymization/submission/blind-review integrity). Each prints `METRIC …=N`.
- **Notebooks:** `advanced_rl_colab.ipynb` (Dr.GRPO/DAPO/DPO), `ppo_reinforce_baselines_colab.ipynb`, `submission_colab.ipynb`, `skyrl-tinker-colab.ipynb`.
- **Build/deps:** `pyproject.toml` (pkg `tinkerrl`), `requirements.txt`, `uv.lock`, `Dockerfile`, `docker-compose.yml`, `sweep.yaml`, `CITATION.cff`, `.env.example`.
- **Runners:** `run_one.sh`, `run_coding.sh`, `run_heldout_all_seeds.sh`, `vast_run.sh`, `run_ai_scientist.sh`, `autoresearch*.sh`.
- **Utilities / codemods:** `patch*.py`, `inject_patch.py`, `fix_eval.py`, `refactor_seeds.py`, `upload_tinker_to_wandb.py`.
- **Result data:** `modal_results_all.json`, `integration_audit.json`, `GRPO_Ablation_results.zip`, `group6.pdf`.
- **Unrelated side task:** `train_xgboost.py` + `*_data.csv` + `xgboost_results.json` (synthetic fraud detection — not RL/LLM).

## Find it fast
- the **pillar experiments** → `experiments/modal/`
- **consolidated results** → `experiments/master_results.{json,csv}`, `experiments/experiment_summary.md`
- the **paper source** → `paper/main.tex` (sections in `paper/sections/`, figures in `paper/figures/`)
- the **anonymized submission** → `blind_review/`
- **reproduce the headline result** → `REPRODUCE.md`
- **run all integrity audits** → `python run_all_audits.py`
- **add a new RL framework/task** → `CONTRIBUTING.md`, then `unified/`
- **a one-shot GRPO run** → top-level `grpo_*.py` (export `TINKER_API_KEY` first)

## Known issues / caveats (surfaced during indexing)
- `grpo_exp_b/c.py` docstrings are **stale copies** of `exp_a` (code differs; docstrings don't).
- `team-*.pplx.md` and `verify_links_entities.txt` contain **real names/handles → NOT blind-safe**; exclude from anon package.
- `blind_review/tinker-rl-lab-anon.tar.gz` is **stale vs `SUBMISSION_MANIFEST.md`** (~97 MB on disk vs 27 MB / different SHA-256) — regenerate before any integrity check.
- **No `local_*.py` native ports** exist in this checkout (those were generated only on the Lightning studio copy); the `modal_*.py` pillar scripts are canonical here.
- Hardcoded machine-specific paths in `ai-scientist-v2-integration/patch.sh`, `inject_patch.py`, `demo_recording/concat*.txt` — break off their origin machine.

---
*83 per-folder `INDEX.md` files + this root map (84 total), generated 2026-07-02.*
