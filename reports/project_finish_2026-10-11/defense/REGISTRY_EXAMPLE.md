# Registry demonstration for an examiner

Slide 6 shows an actual comparison of two retained DAPO-labelled records. The ZIP includes the standard-library CLI, schema and these two records. It does not include the full registry corpus.

From the repository or extracted ZIP root:

```sh
python3 platform_hybrid/registry/query.py stackdiff \
  colab-open_dapo_e3 tinker_dapo_qwen3.5-4b_gsm8k
```

Expected output includes the KL coefficient difference `0.0 -> None` and ends with `R5 unauditable: a differing item is unreported (null) on at least one side`. The command deliberately returns **exit code 5**, matching the R5 severity. That is the expected result of this demonstration.

Explain the distinction: `0.0` records an explicit claim of no KL penalty, whereas `null` records no value. The tool exposes this reporting gap even though both records use the same algorithm label. Other recorded differences also contribute to the verdict.

The records describe different tasks and incomplete configurations. They can contain conflicting fields; this CLI does not authenticate the upstream execution or reconcile such conflicts. The demonstration establishes that the comparison tool works on retained records. It is not a controlled outcome comparison, proof of a score change, or a new model evaluation.

Show this after the ninety-second reward demo only if the examiner asks how the registry works. Keep the primary talk within its twelve-minute route.
