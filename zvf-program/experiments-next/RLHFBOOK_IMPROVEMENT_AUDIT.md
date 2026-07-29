# RLHF Book and CS2824 improvement audit

Date: 2026-07-29  
Book snapshot: Nathan Lambert, *Reinforcement Learning from Human Feedback*, source commit `3624df9ef62177c2c3d6d824f5c2bb740f31041f` (2026-07-28)  
Scope: ZVF/GRPO experiment design, the r4-2 campaign, the spectral/entropic follow-up, and the flagship manuscript

## Outcome

The book supports the repo's central diagnosis: group-relative methods need within-prompt reward contrast, loss reduction and importance-ratio details matter, and evaluation must be treated as an experimental system rather than a single score. The repo is already unusually strong on immutable replay, exact source binding, matched token/FLOP ledgers, seed-level analysis, held-out row hashes, and explicit non-claims.

The main improvement is therefore not another controller. It is a stricter identification ladder for the new spectral and entropic ideas:

1. isolate them from the frozen r4-2 evidence;
2. prove that their auxiliary score predicts independently verified correctness inside homogeneous reward strata;
3. construct a feasible non-cap-saturated variable-length positive control;
4. log clipping, policy drift, length, parser, and contrast telemetry;
5. require matched-cost multi-seed held-out gains before describing any signal as learning or reward recovery.

The Harvard CS2824 extension adds a foundations gate before the empirical ladder: every theorem-shaped statement must expose its formal domain, coverage and distribution assumptions, approximation class, LLM mapping, and falsifier. Its detailed mapping is in [`HARVARD_CS2824_IMPROVEMENT_AUDIT.md`](HARVARD_CS2824_IMPROVEMENT_AUDIT.md).

The combined executable contract is [`rlhfbook_followup_preregistration.json`](rlhfbook_followup_preregistration.json), validated by [`verify_rlhfbook_followup.py`](verify_rlhfbook_followup.py).

## Evidence boundary discovered during the audit

The accepted r4-2 units bind `zvf-program/flagship/pilot/objective.py` to SHA-256 `980a56a1651299a5adbe7a0927c13b12d42d9d7e1a36205500a24d5eeba9b61b`. The live file currently hashes to `20f420f3901c75bdd3d91f2796d7afe903cd6cf950172249cefe3c18ef995a0b` because later experimental work changed it.

Consequences:

- the live checkout must not be used as the verified source for accepted r4-2 receipts;
- the frozen `review_bundle.zip` remains the review surface and passes when extracted and verified independently;
- spectral/entropic work needs a new source namespace, protocol fingerprint, artifact root, and run identity;
- r4-2 remains a registered fixed-corpus off-policy replay conformance/feasibility study, not canonical on-policy GRPO training.

This is a provenance separation, not a claim that the accepted artifacts are invalid.

## Book-to-repo matrix

| Book lesson | Current evidence | Improvement now required |
|---|---|---|
| [Ch. 6: GRPO requires useful group contrast](https://rlhfbook.com/c/06-policy-gradients) | ZVF, exact zero-gradient relations, and group-size theory are first-class in the repo. | Report all-wrong, all-correct, and mixed fractions separately; stratify by prompt/task; compare observed degeneracy with `p(x)^G + (1-p(x))^G`; token-match temperature and group-size sweeps. |
| [Ch. 6: loss aggregation changes weighting](https://rlhfbook.com/c/06-policy-gradients) | S1 freezes per-completion versus global-token reductions; the failed filtered regime was intended as a variable-length positive control. | Freshly preregister a feasible control with selected-row length CV at least `0.35`, cap-hit rate at most `0.05`, identical prompt schedules and starting checkpoints, and arm-specific completion hashes for on-policy arms. Never weaken the old failed gate. |
| [Ch. 6: off-policy reuse needs explicit correction](https://rlhfbook.com/c/06-policy-gradients) | r4-2 intentionally consumes immutable stored groups and stored old log-probabilities. | Keep its claims at replay/conformance scope. A learning follow-up must regenerate on-policy completions or bound learner-sampler lag and preregister the correction. |
| [Ch. 6 and Ch. 15: clipping and KL reveal optimization health](https://rlhfbook.com/c/15-regularization) | Current accepted receipts emphasize losses, gradient relations, norms, and cost ledgers; pilot `beta` is `0`. | Log policy-ratio quantiles, clip fractions split by advantage sign, approximate KL to the old policy, KL to the initial reference, and checkpoint quality versus drift. |
| [Ch. 14: a proxy can improve while true quality degrades](https://rlhfbook.com/c/14-over-optimization) | The synthetic spectral harness creates nonzero auxiliary advantages and gradients on equal terminal rewards. | Describe this only as auxiliary-signal injection. Before training, require independent correctness labels, length/entropy controls, and a variance-matched placebo. Synthetic gradient retention is not reward recovery. |
| [Ch. 16: prompt, parser, sampling, and token budgets can dominate evaluation](https://rlhfbook.com/c/16-evaluation) | r4-2 uses hashed rows and deterministic greedy decoding, but a single strict parser and one 128-row surface. | Separate correctness, extraction, and format validity; audit parser FP/FN with an independent checker; separate development from untouched test; add pass@8 and format perturbation at fixed inference tokens. |
| [Ch. 16: contamination is a confound](https://rlhfbook.com/c/16-evaluation) | Train and held-out rows are disjoint inside the registered datasets; base-model contamination is not ruled out. | Run exact-normalized and 8-gram prompt-overlap checks, report their scope, and treat perturbation sensitivity as a warning rather than proof of contamination. |
| [Appendix C: evaluation and training variance are distinct](https://rlhfbook.com/c/appendix-c-practical) | The pilot uses seed-level pairing and avoids checkpoint pseudo-replication. | Retain at least five paired training seeds for confirmatory claims and repeated sampled evaluation where decoding is stochastic; do not select a positive training outlier. |

## Highest-priority gates

### 0. Foundations and assumption gate

Before an optimization claim enters the experiment, bind it to an exact result and record the formal state/action/trajectory mapping, policy and comparator, data-generating distribution, support or coverage condition, function class, and approximation/estimation assumptions. Results for tabular MDPs or specific parameterizations remain analogies until those conditions are established for the language-model setting. In particular, a zero or small gradient can indicate missing rewarding-trajectory coverage rather than optimality, and PPO-style clipping does not itself prove a policy-space trust region.

### 1. Offline alignment gate

This is the cheapest decisive test for spectral/entropic work. Build independently labeled all-correct and all-wrong terminal-reward groups, then test whether the auxiliary score adds out-of-sample information after controlling for length, token entropy, task, model scale, and reward stratum. The primary estimand is prompt-clustered, cross-fitted untouched-test log-loss reduction, with a `0.01`-nat minimum effect and a variance-matched placebo. AUC is secondary and is computed only where the independent target has both classes; a one-class required stratum is `NOT_IDENTIFIABLE`, never a pass. The exact split, bootstrap, multiplicity, power, and receipt rules are frozen in [`offline_falsification_packet.json`](offline_falsification_packet.json). Failure is a useful negative result and blocks training.

The current synthetic harness cannot pass this gate because it has no independent correctness label, no held-out model behavior, and no placebo.

### 2. Feasible variable-length positive control

The r4-2 filtered pool was cap-saturated at 512 generated tokens and produced selected-row CV `0`, below the frozen `0.35` gate. A fresh design must change the task or generation cap under a new amendment, not retrospectively weaken the old contract. Before any arm is launched, require:

- selected-row completion-length CV `>= 0.35`;
- cap-hit rate `<= 0.05`;
- EOS, parse-success, and answer-format rates;
- identical prompt-schedule and initial-checkpoint fingerprints across arms, with arm-specific realized-completion fingerprints for on-policy arms;
- equal charged tokens and measured FLOPs within `1%`.

### 3. Optimization-health telemetry

For every update, record:

- all-wrong, all-correct, mixed, reward mean, and reward standard deviation;
- policy-ratio `q05/q50/q95`;
- positive- and negative-advantage clipping fractions;
- approximate KL to the data-collection policy and KL to the initial reference;
- completion length mean/p95, EOS rate, cap-hit rate, format validity, and parser disagreement;
- charged generated tokens, measured FLOPs, pass@1, and sampled pass@8.

This separates "the auxiliary loss has a gradient" from "the policy changes safely and usefully."

### 4. Manuscript promotion rule

Keep `zvf-program/flagship/paper/main.tex` at its present methods/reproducibility and registered-feasibility scope. Promotion to a mitigation, controller-benefit, spectral-learning, or causal-training paper requires all prospective gates plus multi-seed matched-cost held-out evidence. Until then, use "auxiliary signal," "synthetic objective diagnostic," and "proposed" rather than "reward recovery," "eliminates starvation," or "improves learning."

`zvf-program/flagship/paper/spectral_entropy_paper.tex` is explicitly marked as a superseded, non-submittable draft because it presents `64.2%`, `1.1%`, `+7.9%`, and `+7.8%` as achieved results without run-level evidence. `spectral_entropy_paper_kdense_revision.tex` is the safer research draft because it limits the current result to synthetic diagnostics and makes the learning study prospective.

The adaptive-group-size evidence also remains exploratory: `platform_hybrid/experiments/group_size_token_normalized.py` can substitute hard-coded fallback rows when raw runs are missing. No adaptive-G claim should use those rows. A promotion gate must require measured seed-level records only and retain the measured-versus-fallback source label in every exported row.

## Verification

From the repository root:

```bash
python3 zvf-program/experiments-next/verify_rlhfbook_followup.py
python3 zvf-program/experiments-next/verify_rlhfbook_followup.py --deep-review-bundle
python3 -m pytest -q tests/test_rlhfbook_followup.py
```

The first command must pass while reporting that the mutable live objective does not match the accepted r4-2 source. The deep command must extract the frozen bundle to a temporary directory and pass its own claim verifier. Neither command authorizes training.
