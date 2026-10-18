# Shared-file change requests from P10_zvf_theory / P11_reproducibility_audit

All target `platform_hybrid/paper/references.bib` (shared; not edited by this cluster).

## 1. Remove informal notes from `note` fields (P10 #509, P11 #663)
`plainnat` prints `note` after the DOI, so every entry shows the arXiv id twice plus a reading note.

| key | old `note` | new `note` |
|---|---|---|
| tong2025drgrpo | `arXiv:2503.20783; introduces Dr.\ GRPO` | delete the field (DOI already carries the id) or `arXiv:2503.20783` |
| dars2025 | `arXiv:2508.13755; introduces DARS` | `arXiv:2508.13755` |
| greso2025 | `arXiv:2506.02177; introduces GRESO` | `arXiv:2506.02177` |
| zhang2026aero | `arXiv:2602.14338; introduces AERO` | `arXiv:2602.14338` |
| qwen2025gspo | `arXiv:2507.18071; GSPO` | `arXiv:2507.18071` |
| agenticrlsurvey2026 | `arXiv:2509.02547; accepted by TMLR` | `arXiv:2509.02547` (venue already in `journal`) |

Reason: informal annotations in bibliography fields; duplicate arXiv strings. Portfolio-wide decision whether to drop `note` when `doi` is an arXiv DOI is the coordinator's.

Venue upgrades suggested by the reviewer (Dr.GRPO at COLM 2025, DARS at ICML 2026, GRESO at NeurIPS 2025) were **not verified** by this cluster; verify before applying.

## 2. Full author lists (P11 #657) — verified against the arXiv API on 2026-10-18
- `nan2025ngrpo`: `author = {Nan, Gongrui and Chen, Siye and Huang, Jing and Lu, Mengyu and Wang, Dexun and Xie, Chunmei and Xiong, Weiqi and Zeng, Xianzhou and Zhou, Qixuan and Li, Yadong and Xu, Xingzhong}` (old: `Nan, Gongrui and others`).
- `zhang2025scaffgrpo`: `author = {Zhang, Xichen and Wu, Sitong and Zhu, Yinghao and Tan, Haoru and Yu, Shaozuo and He, Ziyi and Jia, Jiaya}` (old: `Zhang, Xichen and others`). arXiv comment: "Accepted by ICLR 2026" — optionally `booktitle = {International Conference on Learning Representations (ICLR)}, year = {2026}`.

## 3. New entries (P10 #511, #498) — existence verified
```bibtex
@article{brown2001interval,
  author  = {Brown, Lawrence D. and Cai, T. Tony and DasGupta, Anirban},
  title   = {Interval Estimation for a Binomial Proportion},
  journal = {Statistical Science},
  volume  = {16}, number = {2}, pages = {101--133}, year = {2001},
  doi     = {10.1214/ss/1009213286}
}
@article{nguyen2026adaptiverollout,
  author  = {Nguyen, Hieu Trung and Nguyen, Bao and Ma, Wenao and Zhao, Yuzhi and She, Ruifeng and Nguyen, Viet Anh},
  title   = {Adaptive Rollout Allocation for Online Reinforcement Learning with Verifiable Rewards},
  journal = {arXiv preprint}, year = {2026},
  doi     = {10.48550/arXiv.2602.01601}
}
@unpublished{zvfcontroller2026,
  author = {C R, Arvind and the ZVF Program},
  title  = {From Diagnostic to Controller: A Signal-Starvation Theory of {GRPO} and an Adaptive Group-Size Intervention},
  year   = {2026},
  note   = {Companion working paper in the Tinker RL Lab manuscript bundle}
}
```
Once added, P10 can cite `brown2001interval` next to the Wilson/Clopper–Pearson statements (Sec. 3, Cor. 1), `nguyen2026adaptiverollout` in the related-work paragraph, and replace "(in preparation)" in Sec. 6 with `\cite{zvfcontroller2026}`. P10 does not cite them yet, to avoid undefined citations.
