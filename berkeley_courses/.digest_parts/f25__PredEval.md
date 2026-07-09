### Predictable Noise and patterns from millions of questions
*Speaker:* Sida Wang

#### Key claims / techniques
- Modern LLM generative and agentic benchmarks are far smaller than historical ML test sets (e.g., HumanEval 164, MBPP+ 378, SWE-bench-Verified 500, T-Bench 80), making statistical power a central concern.
- Although each generated answer is rich and test-evaluated, model behavior is highly inconsistent: weaker models sometimes solve hard problems and stronger models fail easy ones, suggesting memorization and high per-sample variance.
- Many reported 2–10% improvements on small benchmarks are not statistically significant; standard error and paired/unpaired comparisons should be used to interpret results.
- Introduced Eval-Arena (crux-eval.github.io/eval-arena), which performs pairwise model comparisons and statistical testing across benchmarks so users can read off noise levels directly.
- Predictable noise: SE(A) and SE(A-B) are roughly similar (correlation ~ 0.5), with dependence on overall accuracy; empirical per-problem success probabilities follow Beta-like distributions.
- Signal-to-noise analysis shows most code-generation benchmarks have sig/noise < 2 (e.g., HumanEval 1.1, MBPP 1.9, SWE-bench-Verified low), meaning gains from doubling model size are often unmeasurable.
- Attempts to reduce noise via filtering, reweighting, or item-response modeling largely failed because model inconsistency and memorization dominate benchmark noise.
- Simple multiple-choice benchmarks (MMLU, TriviaQA, GSM8K) have much better signal-to-noise than complex generation benchmarks.
- Practical recommendations: run multiple seeds, report results on more benchmarks, collect larger datasets or richer per-example signals, and share full question-level results.
- Shared question-level outputs and leaderboard-level statistical tables are preferred over trusting each paper to do its own testing.

#### Relevance hooks
- Evals-with-error-bars: directly addresses statistical testing, standard-error estimation, and confidence intervals for LLM benchmarks.
- Agent evaluation methodology: analyzes SWE-bench and other agentic evaluations where small sample sizes and high token costs complicate reliable measurement.
- RL reproducibility standards / GRPO/RL post-training benchmarking: emphasis on multi-seed evaluation, predictable noise floors, and shared per-sample results is highly relevant to measuring small post-training improvements.

#### Cited paper titles (verbatim only)
- None explicitly listed in extracted text.

Index row: f25 | PredEval.pdf | Sida Wang | Small generative/agentic benchmarks exhibit high predictable statistical noise that often swamps reported model improvements; Eval-Arena provides pairwise statistical comparisons and recommends larger datasets, multi-seed runs, and shared question-level results. | ok
