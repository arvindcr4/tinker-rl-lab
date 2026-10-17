# Registry demonstration for an examiner

Slide 6 shows an actual comparison of two retained DAPO-labelled records. The ZIP includes the standard-library CLI, schema and these two records. It does not include the full registry corpus.

From the repository or extracted ZIP root:

```sh
python3 platform_hybrid/registry/query.py stackdiff \
  colab-open_dapo_e3 tinker_dapo_qwen3.5-4b_gsm8k
```

Expected output (revised CLI, 16 October 2026) lists the outcome-critical differences — `checkpoint.id: 'Qwen/Qwen2.5-0.5B-Instruct' -> 'Qwen/Qwen3.5-4B'`, the reward definition, prompt and chat template, `data.dataset` (synthetic addition against GSM8K), `eval.metric` and `eval.heldout` — followed by algorithm, hyperparameter and runtime differences, and ends with `verdict: R5 outcome-critical differences` and `coverage: 0.87 of compared fields known on both sides; 7 unknown`. The command exits 0; add `--exit-level` to exit with the level (5). The full corpus distribution comes from `stackdiff --all`: over 378 pairs, R0 39, R2 1, R3 10, R4 26, R5 302, with 342 underspecified (median coverage 0.19); among the 78 same-checkpoint pairs, R0 39, R2 1, R3 10, R4 26, R5 2. The 11 October version's output on this pair (KL coefficient `0.0 -> None`, `R5 unauditable`, exit code 5) is reproducible from git `HEAD`.

Explain the point: both records carry the label "DAPO", yet they differ in checkpoint, reward, data and evaluation protocol, so an outcome difference between them cannot be attributed to the label. A field unknown on either side is reported in the coverage line rather than raising the verdict.

The records describe different tasks and incomplete configurations. They can contain conflicting fields; this CLI does not authenticate the upstream execution or reconcile such conflicts. The demonstration establishes that the comparison tool works on retained records. It is not a controlled outcome comparison, proof of a score change, or a new model evaluation.

Show this after the ninety-second reward demo only if the examiner asks how the registry works. Keep the primary talk within its twelve-minute route.
