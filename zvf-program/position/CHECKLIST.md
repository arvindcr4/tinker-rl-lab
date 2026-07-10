# MIN-REPORT-RL — Author Checklist

A copy-pasteable minimum-reportable-stack for **any GRPO-family RL post-training paper**
(GRPO, DAPO, GSPO, Dr.GRPO, MAD-GRPO, AERO, CPPO, NGRPO, Scaf-GRPO, GRESO, EDGE-GRPO,
DARS, TreePo, …).

**Why this exists.** Algorithm labels are under-specified treatments. In a controlled audit,
a *nominally identical* GRPO config (same model, group size, learning rate, dataset, seed,
step budget) produced **84.4%** [TODO:trace to v1 audit citation] last-10 training reward on one backend and **5.0%** [TODO:trace to v1 audit citation] on
another — a ~17× gap [TODO:trace to v1 audit citation] with **no visible hyperparameter difference**. The label was constant;
the stack was not. Each item below is on the list because it is a **documented lever that can
flip a head-to-head comparison**. If two papers both report these eight fields, a reader can
tell whether their comparison is confounded.

Report all eight. They are a *minimum*, not a maximum.

**Provenance / scope of the worked numbers in this checklist.** The concrete
numbers used as motivating examples below (61.6–89.6% prompt-token
loss magnitude; the 17× matched-config gap; the 82.0→83.3% held-out
control at p=0.26; the 95.0–98.1% / 95.0% Llama-3.3-70B run)
are inherited from the v1 audit paper that motivated this position
piece. **[TODO:trace to v1 audit citation]** A reader who wants to
verify the numbers should consult that paper; this checklist presents
them as illustrative of the *kind* of stack-driven flip, not as
re-derivable from a fresh run. Once the v1 audit paper is in
citation scope, the references go in the `\bibliography{...}` and
the [TODO:trace] markers are replaced with real `\cite{}` keys.

---

## The 8 items

### 1. Loss form
- **Report:** PPO importance ratio used? (yes/no). Clipped? bounds (incl. asymmetric
  "clip-higher")? Token mask: completion-only or whole-sequence? Advantage normalization:
  per-group / per-batch / running estimate?
- **Why it can flip:** the token mask reassigns gradient. In one diagnostic [TODO:trace to v1 audit citation], **61.6–89.6%**
  of full-sequence loss magnitude came from *prompt* tokens, not completion tokens — a
  whole-sequence and a completion-only mask are different objectives sharing a name. Dr.GRPO
  *is* a normalization change; GSPO *is* an IS-granularity change. If the baseline loss form
  is unreported, the variant's gain is unattributable.
- **Good:** "Token-masked completion-only; PPO ratio with symmetric clip ε=0.2; per-group
  advantage normalization (subtract group mean, divide by group std)."
- **Bad:** "We use the standard GRPO loss."

### 2. Reference policy + KL handling
- **Report:** frozen reference policy retained? (yes/no). KL term in the **loss** or folded
  into the **reward**? KL coefficient + schedule. Forward/reverse, per-token/per-sequence.
- **Why it can flip:** KL placement changes the objective and steady-state exploration. A
  runner with *no* frozen reference (one optimizer step per rollout) is not doing KL-regularized
  GRPO even at matched LR. Dual-anchor methods shift steady-state ZVF and must be recalibrated.
- **Good:** "Frozen reference = SFT checkpoint; KL in the loss, β=0.04 constant; reverse KL,
  per-token."
- **Bad:** "KL regularization as usual." / (silent on whether a reference exists)

### 3. Sampler / backend / precision
- **Report:** rollout engine (vLLM / SGLang / managed API / trainer `.generate`). Decoding
  params (temperature, top-p, max_tokens). Logit precision in sampler vs. trainer
  (bf16/fp16/fp32). Same tokenizer + chat template in sampler and trainer? (yes/no).
- **Why it can flip:** the sampler *defines* the rollout distribution the group-relative update
  consumes. A managed runner vs. an open vLLM path gave the **17×** [TODO:trace to v1 audit citation] matched-config gap. Sampler
  precision shifts the probability of mixed-reward groups, hence available gradient.
- **Good:** "vLLM 0.x rollouts, bf16; trainer logits bf16; temp 0.8, top-p 1.0, max_tokens 512;
  identical tokenizer/chat template across both."
- **Bad:** "Generated with default settings."

### 4. Per-step ZVF and GU trajectory
- **Report:** per-step **ZVF** (fraction of prompts whose G completions all get identical
  reward → zero gradient) and **GU = 1 − ZVF**, logged every step (release the trajectory).
- **Why it can flip:** ZVF/GU reveals whether the run had *any* usable learning signal. A
  method can "win" purely because its stack produced lower ZVF (more usable groups), not because
  its algorithm is better. Without the trajectory, a collapsed run and a saturated run look
  identical from final reward. (Mixed-group probability:
  `P(usable) ≈ (1/N) Σ_x [1 − (1−p_x)^G − p_x^G]`.) Cheap collapse triage: first-5-step rule
  ZVF ≥ 80% with reward ≤ 5%.
- **Good:** "ZVF/GU logged per step (released CSV); mean ZVF@25 = 0.43; no run trips the
  collapse rule."
- **Bad:** Only final reward reported; no per-step signal telemetry.
- **Substitutes allowed:** under dense process rewards or scaffolded exploration ZVF
  degenerates → report per-step reward variance or gradient-norm variance instead. Item 4
  mandates *a usable-signal trajectory*, of which ZVF/GU is the default.

### 5. Group-size schedule (fixed or adaptive)
- **Report:** group size G at every step; the rule that changes it if adaptive (e.g. AERO-style
  double/halve on rolling ZVF).
- **Why it can flip:** G sets the mixed-group probability and was one of the strongest single
  knobs measured (G = 2→4→8→16 moved last-10 reward across a wide band). Adaptive-G vs. fixed-G
  partly credits the variant for spending more compute on hard prompts — separate that from the
  algorithmic claim.
- **Good:** "Fixed G=8 throughout." / "Adaptive G∈{4,…,16}, baseline 8, double when rolling
  ZVF>0.8, halve when <0.3, window 10."
- **Bad:** "Group size 8." (when it was actually adaptive)

### 6. Held-out split distinct from the reward environment
- **Report:** a held-out slice **disjoint** from training prompts, scored by a harness, with N
  and a CI, reported **separately** from online training reward.
- **Why it can flip:** training reward is dynamics, not capability. A clean paired control
  improved only **82.0% → 83.3% (p=0.26)** [TODO:trace to v1 audit citation] despite near-saturated training reward; selecting
  checkpoints by training reward produced a spurious 87–95% "capability" band. Four 70B seeds
  ranged 95.0–98.1% on training last-10 [TODO:trace to v1 audit citation] yet all landed on 95.0% held-out (sampling noise, not
  generalization).
- **Good:** "Held-out = 500 GSM8K test problems, seed 0, disjoint from every training batch;
  base 82.0% → post 83.3% [TODO:trace to v1 audit citation], Wilson 95% CI, paired per-prompt p=0.54 [TODO:trace to v1 audit citation]."</update>
- **Bad:** Reporting training-set reward as the capability number; "accuracy 94%" with no split
  stated.

### 7. Decontamination probe results
- **Report:** train/test contamination check (n-gram or embedding overlap between training
  prompts and the held-out/benchmark slice) AND parser behavior on adversarial format-only
  inputs.
- **Why it can flip:** verifiable rewards still admit reward hacking — parser artifacts, format
  shortcuts, length effects, train-prompt overfitting. Overlap → "gain" may be memorization;
  parser rewards a format token → "gain" may be a shortcut.
- **Good:** "Max 8-gram overlap between train and held-out = 0.0%; parser rejects 100% of
  format-only (no-answer) adversarial inputs."
- **Bad:** No contamination check; parser behavior unstated.

### 8. Pass@k curves alongside pass@1
- **Report:** held-out pass@k at k ∈ {1, 8, 32} (or a stated subset with justification),
  plus the sampling temperature and completion budget used for the estimate.
- **Why it can flip:** pass@1 conflates *sharpening* the output distribution (concentrating
  probability on already-reachable solutions) with *expanding* the set of reachable solutions.
  The TMLR agentic-RL survey (arXiv:2509.02547, §6.4) finds ~2/3 of RL-for-reasoning papers
  report only pass@1, while studies reporting pass@k frontiers repeatedly find base models
  matching or overtaking RL-tuned ones at large k. Group-relative training acts directly on
  the sampling distribution, so a pass@1 ranking between two GRPO variants can invert at k=32.
- **Good:** "Held-out pass@{1,8,32} at T=1.0, 32 completions/problem: base 61/78/89%,
  post-RL 72/80/89% — gain is concentrated at k=1, consistent with distribution sharpening."
- **Bad:** Single pass@1 number, greedy decoding, no k>1 evidence.

---

## Fillable appendix template (drop into your paper)

```
================ MIN-REPORT-RL BLOCK ================
Method label:            <e.g. GRPO / DAPO / GSPO / ...>
Base checkpoint:         <model id + revision/hash>
Tokenizer + chat tmpl:   <id; same in sampler and trainer? yes/no>

[1] LOSS FORM
    PPO ratio used:      <yes/no>
    Clip:                <none / symmetric eps=__ / asymmetric lo=__ hi=__>
    Token mask:          <completion-only / whole-sequence>
    Advantage norm.:     <per-group / per-batch / running; formula>

[2] REFERENCE + KL
    Frozen reference:    <yes (=__ checkpoint) / no>
    KL location:         <in loss / in reward / absent>
    KL coeff + schedule: <beta=__, schedule=__>
    KL estimator:        <forward/reverse, per-token/per-seq>

[3] SAMPLER / BACKEND / PRECISION
    Rollout engine:      <vLLM/SGLang/managed API/trainer.generate; version>
    Decoding:            <temp=__, top_p=__, max_tokens=__>
    Precision:           <sampler=__, trainer=__>

[4] ZVF / GU TRAJECTORY
    Per-step logged:     <yes/no; artifact link>
    Mean ZVF@25:         <__>      Mean GU@25: <__>
    Collapse rule trip:  <__ / S seeds>   (rule: ZVF>=80% & reward<=5% in first 5 steps)
    (or substitute metric under dense/scaffolded rewards: __________)

[5] GROUP-SIZE SCHEDULE
    Type:                <fixed G=__ / adaptive (rule=__, bounds=__, window=__)>

[6] HELD-OUT SPLIT
    Slice:               <dataset/split, N=__, seed=__, disjoint from train? yes/no>
    Base vs post:        <__% -> __%, CI=__, test=__ (p=__)>
    Reported separately from training reward: <yes/no>

[7] DECONTAMINATION
    Train/test overlap:  <metric=__, value=__>
    Parser adversarial:  <format-only reject rate=__%>

[8] PASS@K
    Held-out pass@k:     <k=1: __%, k=8: __%, k=32: __%>
    Estimate config:     <temp=__, completions/problem=__>
    Base-model pass@k:   <k=1: __%, k=8: __%, k=32: __%>  (same config)
====================================================
```

---

## One-line self-test
If a reader cannot reproduce your ranking of method A vs. method B **from the block above
alone**, the block is incomplete — find the unreported lever and add it.
