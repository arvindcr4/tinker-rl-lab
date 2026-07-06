# Phase‑1 ESA — Code Walkthrough & Live Demo Runbook

Audience: two Great Learning examiners. Goal: show a **working** unified GRPO/RLVR post‑training benchmark + the diagnostic pipeline behind the 8‑study portfolio. Honest framing (per our own rigor review): this is a **measurement/diagnostic benchmark + reproducibility tooling**, not a "GRPO wins" leaderboard.

Total demo budget: ~8–10 min live + Q&A. Everything below has been run and verified on 2026‑07‑06.

---

## 0. One‑line setup (before the panel joins)
```bash
cd ~/tinker-rl-lab
set -a; source .env.minimax; set +a          # loads TINKER_API_KEY + WANDB_API_KEY
```
Interpreter is `.venv/bin/python` (has torch + tinker + transformers + datasets). `python`→`python3` shim at `.localbin/python`.

## 1. The 30‑second architecture (say this while showing the repo tree)
```
tinker-rl-lab/
├── tinkerrl/grpo.py        # core GRPO loop: sample → advantage → loss → optim (Tinker backend)
├── skyrl/ atropos/ verl/ openrlhf/ trl_integrations/   # per-framework adapters
├── unified/launcher.py     # one config → any framework
├── experiments/            # runs + results (telemetry JSONL, group tensors)
│   └── openings/           # the 8-study experiments (campaign.py, curriculum_grpo.py, ...)
├── paper/                  # 8 papers P1–P8 + unified main.tex
└── reports/esa_phase1/     # this deck, adversarial reviews, findings
```
**Pitch:** "Same task/reward/decoding across frameworks (TRL, veRL, OpenRLHF, Tinker) so differences are attributable to the *stack*, not the experiment. Per‑step telemetry (reward, ZVF, gradient‑utilisation, entropy, length, KL) feeds eight focused studies P1–P8."

## 2. LIVE DEMO A — a real GRPO training run on Tinker (~3–5 min)
```bash
.venv/bin/python grpo_gsm8k_base.py --model Qwen/Qwen3.5-4B --steps 6 --group 4 --rank 8 --seed 0 --tag esa_demo
```
Point at the streaming output:
```
1/6 | loss=0.000 | reward=1.000 | acc=100%     <- group collapsed (all-correct) → ZERO gradient
2/6 | loss=-17.4 | reward=0.250 | acc=25%       <- mixed group → real gradient
...
Zero-loss steps: 5/6 (83%)
```
**The teaching moment (this is a real finding, not a bug):** "≈83% of steps produced *zero gradient* — every sample in the group got the same reward, so the GRPO advantage is zero. This is the **Zero‑Variance Fraction (ZVF)** problem — the core diagnostic of paper P2, reproduced live."

## 3. LIVE DEMO B — telemetry → diagnostic finding (~2 min)
```bash
.venv/bin/python experiments/openings/p2_collapse_analysis.py     # recomputes P2 from real tensors
```
Shows across 4 methods: **ZVF 0.72–0.77** (≈¾ of every batch wasted), **~93% of collapse is already‑solved easy prompts**, and that a naive cross‑prompt fix just injects a difficulty confounder. → motivates the curriculum/token‑budget direction.

## 4. LIVE DEMO C — the experiment + verification loop on W&B (~2 min)
Open **wandb.ai/arvindcr4-pes-university/rlvr-openings** (project `campaign` group). Show:
- the multi‑seed baseline‑vs‑curriculum runs (matched, ≥3 seeds — real statistical power),
- the P3 group‑size sweep (G=2 collapses/no‑learn → G=4 best → G≥8 diminishing).
**Pitch:** "Every claim is logged, reproducible, and independently verified (we run findings through frontier models — kimi/codex — to catch overclaims before they reach a paper)."

## 5. The 8‑study portfolio (one slide of code→paper mapping, for Q&A)
| Paper | Code / result | Status |
|---|---|---|
| P1 scaling (layer‑freeze) | `experiments/openings/p1_layer_profile.py` (Colab GPU) | white‑box, running |
| P2 ZVF | `p2_collapse_analysis.py`, `curriculum_grpo.py` | measured; redirected to curriculum |
| P3 group size | `parallel_sweep.py`, `campaign.py` | measured (G=4 best, multi‑seed running) |
| P4 length bias | KL‑surprise mask (Colab) | designed |
| P5 MIN‑REPORT / provenance | `registry/`, provenance protocol | systems (flagship) |
| P6 GRPO‑Registry | `registry/schema.json`, `entries/` | systems (fold into P5) |
| P7 controller | PID / control‑theory | designed |
| P8 integrity | `p8_openings/`, telemetry auditor | measured (AUROC 0.63 vs reward‑only 0.35) |

## 6. Answers to the questions the panel WILL ask (rehearsed)
- **"Is this just GRPO wins?"** No — it's a diagnostic benchmark. Headline honest results: stack choice dominates; ZVF wastes ~¾ of gradient steps; held‑out RL gains are within noise → we report that, not hide it.
- **"Single‑seed?"** Early runs were; the current `campaign.py` runs matched baselines across ≥3 seeds — that's the fix, live on W&B.
- **"Novelty vs AERO/NGRPO/Dr.GRPO?"** We differentiate explicitly; the strongest original bet is P5 (a GRPO provenance/reporting standard — no one has done it rigorously). Prior art verified on arXiv.
- **"Reproducibility?"** Docker + pinned deps + W&B + every analysis script in‑repo (e.g. `p2_collapse_analysis.py` recomputes the JSON).

## 7. Fallback if live Tinker is slow/unavailable
Best offline fallback (zero network, always works, uses `.venv`):
```bash
.venv/bin/python experiments/openings/p2_collapse_analysis.py   # recomputes the ZVF/collapse finding from saved real tensors
```
Also: open the already-logged W&B runs (Demo C) — they persist. (`scripts/smoke_test.sh` calls bare `python`; run it as `PATH="$PWD/.venv/bin:$PATH" bash scripts/smoke_test.sh` if you use it, or just use the command above.)
