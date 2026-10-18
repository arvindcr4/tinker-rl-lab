**Segment scope**: Review of `platform_hybrid/paper/paper_P2_zvf_falsification_note.tex` (short retraction note on ZVF, sign-wrong Bernoulli, and beta-binomial difficulty dispersion).

# Potential Mistakes and Improvements

- **[Major] Inability to Separate Latent Prompt Heterogeneity from Within-Prompt Rollout Correlation**
  Section~3 (lines 108--114) demonstrates that the observed ZVF on GSM8K ($0.158$) is roughly $2.9\times$ the homogeneous Bernoulli value ($0.054$) and is accurately predicted by a two-parameter beta-binomial fit ($0.170$, intra-class correlation $\mathrm{ICC} = 0.160$). From this, the note concludes that ZVF excess is a "difficulty-dispersion functional." However, as conceded in lines 111--114, because seeds 42, 123, and 456 evaluated 200 distinct problem questions each rather than repeatedly evaluating the same 200 questions across seeds, the artifact data cannot formally separate between-prompt difficulty dispersion ($\sigma_p^2$) from within-prompt rollout correlation (e.g. autoregressive token herding or generation mode collapse). Attributing count overdispersion ($\chi^2_7 = 8119$) entirely to prompt difficulty spread is a model-dependent assumption rather than an empirically verified mechanism.

- **[Major] Unfilled Prospective Empirical Test Placeholder in Document Body**
  Section~3 (lines 122--127) contains an unfulfilled experimental placeholder in the core body text:
  *"`\textbf{[Placeholder --- to be filled from \nolinkurl{experiments/results/publication_fixes/}].} A held-out prediction experiment estimates each prompt's $p_i$ from an independent rollout sample and predicts batch ZVF as $\frac1n\sum_i h_G(p_i)$ at several $G$. Results are not yet available and no claim in this note depends on them.`"*
  Leaving a prominent placeholder in a published correction note leaves the intended empirical validation incomplete. The note should either present the completed Colab prediction results or remove the placeholder block.

- **[Major] Non-Identifiability of Finite-Sample Null Distribution for Trajectory Residuals**
  In Section~2 (lines 78--84), the note addresses the negative synthetic arithmetic divergence ($\delta_t = h_G(\widehat{\bar p}_t) - \widehat{\mathrm{ZVF}}_t = -0.067$), explaining that $\mathbb{E}[h_G(p_i)] \ge h_G(\bar p)$ by Jensen's inequality. However, while the population inequality is mathematically indisputable, transferring it to the finite-sample empirical statistic is confounded by non-linear plug-in bias in $\mathbb{E}[h_G(\widehat{\bar p}_t)]$. As documented in `p2_delta_artifact.py` (line 114), because prompt probabilities and within-step raw group structures were not logged for the 480 steps, the finite-sample null distribution cannot be identified from aggregate rows. The reported negative sign should be strictly qualified as a descriptive sample mean rather than a proven negative expectation under finite-sample sampling.

# Minor Corrections and Typos

- **Citation Key and Author Mismatch (`zhang2026greso` vs.\ Zheng)**: In Section~4 (lines 134--135) and the bibliography (lines 172--175), the citation key is written as `zhang2026greso`, but the authors are listed as *"H.~Zheng, Y.~Zhou, B.~R.~Bartoldson, B.~Kailkhura, F.~Lai, J.~Zhao, and B.~Chen"*, and the year is 2025 (arXiv:2506.02177). The cite key should be corrected to reflect the author Zheng.
- **Withheld Inferential Uncertainty for Arithmetic Trajectory**: In line 82, the text notes that *"the historical step-iid interval is withheld."* While discarding an invalid step-i.i.d. assumption is statistically appropriate, reporting a block-bootstrap or run-clustered confidence interval would provide necessary uncertainty quantification for the $-0.067$ statistic.
- **Underfull Hbox in Bibliography**: Line 172 produces an underfull `\hbox` warning (badness 1502) in the natbib bibliography environment for the DAPO entry (`yu2025dapo`).
