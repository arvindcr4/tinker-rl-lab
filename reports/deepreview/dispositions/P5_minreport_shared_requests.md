# P5_minreport — shared-file change requests

## 1. `platform_hybrid/paper/references.bib`: add three verified entries

All three were checked to exist on 2026-10-18 (arXiv / USENIX / IBM pages). P5 currently names them in prose (no `\cite`) so the build has no undefined citations. After you add the entries, swap the prose for citations as shown in section 2.

```bibtex
@article{yue2025does,
  title   = {Does Reinforcement Learning Really Incentivize Reasoning Capacity in {LLMs} Beyond the Base Model?},
  author  = {Yue, Yang and Chen, Zhiqi and Lu, Rui and Zhao, Andrew and Wang, Zhaokai and Yue, Yang and Song, Shiji and Huang, Gao},
  journal = {arXiv preprint arXiv:2504.13837},
  year    = {2025}
}
@article{arnold2019factsheets,
  title   = {{FactSheets}: Increasing Trust in {AI} Services through Supplier's Declarations of Conformity},
  author  = {Arnold, Matthew and Bellamy, Rachel K. E. and Hind, Michael and Houde, Stephanie and Mehta, Sameep and Mojsilovi{\'c}, Aleksandra and Nair, Ravi and Ramamurthy, Karthikeyan Natesan and Olteanu, Alexandra and Piorkowski, David and Reimer, Darrell and Richards, John and Tsay, Jason and Varshney, Kush R.},
  journal = {IBM Journal of Research and Development},
  volume  = {63},
  number  = {4/5},
  pages   = {6:1--6:13},
  year    = {2019},
  doi     = {10.1147/JRD.2019.2942288}
}
@inproceedings{torresarias2019intoto,
  title     = {in-toto: Providing Farm-to-Table Guarantees for Bits and Bytes},
  author    = {Torres-Arias, Santiago and Afzali, Hammad and Kuppusamy, Trishank Karthik and Curtmola, Reza and Cappos, Justin},
  booktitle = {28th USENIX Security Symposium},
  pages     = {1393--1410},
  year      = {2019}
}
```

Before committing, please check the author lists against the publisher pages. I confirmed the titles, venues, years and first authors; I wrote the full author lists from memory.

## 2. P5 text swaps to make after the bib entries land (P5-owned files; listed here so they are applied together)

- `sections/p5_stack.tex`: `(e.g.\ Yue et al., 2025, arXiv:2504.13837)` → `\citep[e.g.][]{yue2025does}`
- `sections/p5_related.tex`: `FactSheets (Arnold et al., 2019, \emph{IBM J.\ Res.\ Dev.})` → `\emph{FactSheets} \citep{arnold2019factsheets}`
- `sections/p5_verifier_impl.tex`: `in-toto (Torres-Arias et al., USENIX Security 2019)` → `in-toto \citep{torresarias2019intoto}`

## 3. `sections/_shared_author.tex` (no change requested)

Finding 312 calls the `\thanks{Project guide.}` footnote an internal note. It is a normal supervisor credit, so I am not requesting a change. The footnote is shared across all papers, so the decision belongs to the coordinator.
