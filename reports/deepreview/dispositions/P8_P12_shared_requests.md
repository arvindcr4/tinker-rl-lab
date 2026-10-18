# Shared-file change requests from the P8_workshop / P12_signal_starvation cluster

All requests are for `platform_hybrid/paper/references.bib` (shared; not edited by this cluster).
P12 uses its own local `unified_signal_starvation/references.bib`, which was edited directly.

## R1 (finding 684, MAJOR): strip internal tracking metadata from the Tinker entry
`@misc{thinkingmachines2024tinker,` (line ~1152)

old:
```
  note      = {Tinker technical blog post --- not a peer-reviewed venue; this worktree uses the Tinker API as its primary post-training substrate (P7 iter-134 row 152)}
```
new:
```
  note      = {Technical blog post}
```
Reason: worktree/iteration identifiers leak into every rendered bibliography that cites this key.

## R2 (finding 705, MINOR): xLAM entry — remove inline license tag
`@misc{liu2024xlam,`

old:
```
  note         = {\url{https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k}; CC BY 4.0}
```
new:
```
  note         = {\url{https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k}}
```
Also protect the acronym in the title: `Empower AI Agent Systems` -> `Empower {AI} Agent Systems`.

## R3 (finding 706, MINOR): inline editorial notes in notes fields
- `krakovna2020specification`: note `...; not a peer-reviewed venue --- retained for traceability of the specification-gaming term` -> keep only the `\url{...}`.
- `lee2023platypus`: note `arXiv:2308.07317; Open-Platypus CC BY-NC 4.0` -> `arXiv:2308.07317`.
- `li2024numinamath`: note `Apache 2.0; \url{...}` -> `\url{...}`.
- `rafailov2024direct`: note `arXiv:2305.18290; NeurIPS 2023 (proceedings vol. 36)` -> `arXiv:2305.18290` (venue already in booktitle).
- `zhang2024gsm1k`: note `arXiv:2405.00332 (GSM1k)` -> `arXiv:2405.00332`.

Venue upgrades suggested by the reviewer (coordinator should verify before applying; this cluster did not verify them):
- `patterson2024empirical` -> JMLR 25(318):1--63, 2024 (reviewer claim).
- `sheng2024verl` / `verl2024` -> EuroSys 2025 (reviewer claim).
- `riddell2024contamination` -> reviewer says TACL 12:1408--1426; this cluster could not confirm (may be ACL 2024). Leave unless verified.

## R4 (finding 688, optional): Bay & Yearick (2026) group-standard-deviation identity
Verified to exist (arXiv abs page read 2026-10-18): "GRPO, Dr. GRPO, and DAPO Are Three Operations on One Number: The Group-Standard-Deviation Identity", Yong Yi Bay and Kathleen A. Yearick, arXiv:2607.00152, 2026. Not cited by P8 in this pass (DAPO, Dr. GRPO, and Nie et al. 2026 already cover the point and are in the shared bib); add only if a paper wants it:
```
@article{bay2026groupstd,
  author  = {Bay, Yong Yi and Yearick, Kathleen A.},
  title   = {{GRPO}, {Dr. GRPO}, and {DAPO} Are Three Operations on One Number: The Group-Standard-Deviation Identity},
  journal = {arXiv preprint},
  year    = {2026},
  doi     = {10.48550/arXiv.2607.00152},
  note    = {arXiv:2607.00152}
}
```
