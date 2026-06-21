# Colab-only ZVF experiments

Follow-up experiments that **cannot run on Tinker** and therefore justify a raw
GPU box (Colab). The unifying observation: the paper's own *Limitations* section
is a map of Tinker's structural constraints —
*"closed-source Tinker (cannot audit the GRPO loss/normalization)"*, *LoRA-only*,
*single-seed*, *no gradient visibility*. Each constraint below is turned into an
experiment.

All scripts are **standalone by design**: `colab run` ships a single file to a
fresh VM, so each script duplicates the small shared harness (problem gen,
generation, sequence log-prob, GRPO advantage) rather than importing a module.

| Exp | Question | Why Tinker can't | Maps to |
|-----|----------|------------------|---------|
| **E1** | Does GRPO gradient magnitude track `S = p(1-p)` (inverted-U), as Theory T3 assumes? | needs per-step **gradient norm** from the open backward pass | Pillar 2 (theory), `THEORY_NOTES.md` T3 gap |
| **E2** | Does the ZVF trajectory / held-out gain differ between **LoRA and full FT**? | Tinker is **LoRA-only** | Pillar 4 (the LoRA stack item), paper future-work |
| **E3** | Re-implementing GRPO/Dr.GRPO/DAPO in **one open, auditable trainer** with the stack held fixed — which gains survive? Does adaptive-G cut ZVF live? | Tinker's loss kernel is **closed & unswappable** | Pillar 4 audit + Pillar 3 (`zvf-triage`) |

## E1 — gradient ↔ ZVF (`e1_grad_signal.py`)

**Pilot finding (T4, Qwen2.5-0.5B-Instruct, n=9):** the harness works (full
gradient access confirmed). The naive `corr(grad_norm, GU)` came out **−0.375**,
which is the *right* answer pointing at the correct test:

- `GU = 1 − ZVF` is **monotone** in difficulty, but gradient signal is
  **inverted-U** (peaks at medium: easy 556.7 < **medium 636.5** > hard 610.7) —
  exactly T3's `S = p(1−p)·(1−h_G(p))`.
- So grad_norm *anti*-correlates with GU **because** the theory's signal term is
  non-monotone while GU isn't. The correct theory test is
  **`grad_norm vs p(1−p)`**, not vs GU.

**Corrections applied in the committed script:** summed (not mean-normalized)
gradient norm + signal-per-rollout; difficulty spans `p∈[~1,~0]` so ZVF→1 at
*both* ends; **ERF** (format-compliance) logged to split "wrong" from
"format-gated"; multi-seed.

## E2 — LoRA vs full FT (`e2_lora_vs_fullft.py`)

Holds task/data/seed/compute fixed and flips only the LoRA↔full axis. Logs ZVF/GU
trajectory, training-reward trajectory, and held-out accuracy delta per arm.
0.5B full-FT fits a T4, so no A100 needed for the pilot.

## E3 — open reproducibility audit (`e3_open_audit.py`)

Re-implements `grpo` / `drgrpo` (no `/std`) / `dapo` (asymmetric clip + dynamic
sampling) **in one loop** with cached old-policy log-probs, importance ratio, and
2 inner epochs so clipping actually engages. Plus a `grpo_adaptiveG` arm that
escalates G under sustained ZVF (the `zvf-triage` controller). Matched compute;
reports per-arm held-out delta, mean ZVF, and rollout count.

## Running

```bash
colab run --gpu T4  --timeout 900  e1_grad_signal.py
colab run --gpu T4  --timeout 1200 e2_lora_vs_fullft.py
colab run --gpu T4  --timeout 1200 e3_open_audit.py
# scale: bump SEEDS / STEPS / model in-file; use --gpu A100 for >1B full-FT (E2)
```

Each prints a `*_RESULT {json}` line (and per-step logs). Sessions auto-tear-down,
so nothing keeps burning compute units.

## Caveats (honest scope)

- Pilots use a weak 0.5B model on synthetic arithmetic — they validate the
  *harness and the measurement design*, not publishable effect sizes.
- E3 uses sequence-level ratios (GSPO-ish); token-level is a refinement.
- Synthetic-arithmetic difficulty is a proxy for the GSM8K step-count terciles
  used in the main sweep; swap in the real dataset for the production run.
