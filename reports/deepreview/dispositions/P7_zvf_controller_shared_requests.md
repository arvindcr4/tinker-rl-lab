# P7_zvf_controller: shared-file requests

P7 no longer inputs `sections/_shared_methods.tex` (replaced by `sections/p7_methods.tex`), so none of the items below blocks P7. They are filed for the other papers that still include the shared files.

## references.bib: add three entries (verified on the arXiv API, 2026-10-18)

P7 currently cites these inline with their arXiv IDs. Once the keys exist, `p7_intro.tex` can switch to `\citet{li2025knapsackrl}`, `\citet{fang2026dynamo}`, `\citet{xu2025pods}`.

```bibtex
@article{li2025knapsackrl,
  title   = {Knapsack {RL}: Unlocking Exploration of {LLMs} via Optimizing Budget Allocation},
  author  = {Li, Ziniu and Chen, Congliang and Yang, Tianyun and Ding, Tian and Sun, Ruoyu and Zhang, Ge and Huang, Wenhao and Luo, Zhi-Quan},
  journal = {arXiv preprint arXiv:2509.25849},
  year    = {2025},
  doi     = {10.48550/arXiv.2509.25849}
}
@article{fang2026dynamo,
  title   = {How to Allocate, How to Learn? Dynamic Rollout Allocation and Advantage Modulation for Policy Optimization},
  author  = {Fang, Yangyi and Lin, Jiaye and Fu, Xiaoliang and Qin, Cong and Shi, Haolin and Hu, Chaowen and Pan, Lu and Zeng, Ke and Cai, Xunliang},
  journal = {arXiv preprint arXiv:2602.19208},
  year    = {2026},
  doi     = {10.48550/arXiv.2602.19208}
}
@article{xu2025pods,
  title   = {Not All Rollouts are Useful: Down-Sampling Rollouts in {LLM} Reinforcement Learning},
  author  = {Xu, Yixuan Even and Savani, Yash and Fang, Fei and Kolter, J. Zico},
  journal = {arXiv preprint arXiv:2504.13818},
  year    = {2025},
  doi     = {10.48550/arXiv.2504.13818}
}
```

## sections/_shared_methods.tex (for papers that still include it)

1. Figure 1 (`tikz/taxonomy.pdf`) renders as `[missing figure artifact: tikz/taxonomy.pdf]` in the worktree build. Same for `tikz/reward_flow.pdf` and `tikz/pipeline.pdf`. The figure also says "8 models tested" for GSM8K where some including papers audit nine, and its HumanEval leaf uses the circle symbol labelled "Partial" while the legend defines the circle as "Planned" (Deep Review P7 #418).
2. Casing: `verl` and `veRL` in the same paragraph (cross-launcher roster sentence). Pick `veRL` throughout (P7 #451).
3. Power paragraph, unequal groups (n1=5, n2=15). Current text gives only the pooled MDE (d=1.530, df=18; recomputed, correct). Suggested addition, recomputed with scipy noncentral t: "under equal variances the Welch–Satterthwaite df is 6.89 and the 80%-power MDE is d=1.69". The reviewer's figures (df 6.26, d 1.709) are slightly off; use 6.89 / 1.69 (P7 #423).
