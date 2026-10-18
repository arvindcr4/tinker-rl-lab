# Bibliography verification and citation plan

Historical preparation note, reviewed 18 October 2026. The recommendations below describe an earlier 20-entry working plan, including 19 scholarly references and one platform announcement; they are not the current bibliography or a separate delivered file. The retained bibliography is `../refs.bib`, with expanded author metadata in `full-reference-metadata.bib`. The current manuscript cites all 37 entries: 28 `article`, `inproceedings`, or `techreport` records and nine `misc` records. See `citation-review-2026-10-18.md` and `reference-addendum.md` for the current state.

## Historical recommended citation placements

1. Introduction, after reproducibility/disclosure literature: add `\cite{kapoor2024agentsmatter}` to a sentence such as: “Recent work on agent evaluation also identifies inadequate holdout sets, cost-blind comparisons and a lack of standardized evaluation practices.” Replace the unsupported novelty/generalization “The evaluation layer ... has not yet been held to the same standard” with a narrower claim that this campaign operationalizes these concerns through receipts, explicit coverage and fail-closed scope boundaries.
2. Table 1, CORE-Bench row: `CORE-Bench~\cite{siegel2024corebench}`
3. Table 1, WebArena row: `WebArena~\cite{zhou2024webarena}`
4. Table 1, MLE-bench row: `MLE-bench~\cite{chan2025mlebench}`
5. Table 1, BALROG row: `BALROG~\cite{paglieri2025balrog}`
6. E11 complete-scope paragraph, first mention: `VerilogEval~\cite{liu2023verilogeval,pinckney2025revisiting}`. Call the denominator “312 task instances across two 156-problem framings”; it is not 312 distinct underlying hardware problems. The official harness says the problems are identical and only task format changes.
7. Externally blocked lanes paragraph, AgentHarm first mention: `AgentHarm~\cite{andriushchenko2025agentharm}`. Alternatively cite it in the discussion distinguishing harmful-task completion from AgentDojo benign utility. Do not imply AgentHarm is wholly private: this campaign's original private-task contract was blocked, while the benchmark also has a public release.

The initial plan retained 13 historical citation placements and proposed seven more. This was a placement recommendation, not evidence that a separate 20-entry bibliography was delivered or validated. The current manuscript has since changed: it cites all 37 keys in `../refs.bib`, does not use `\nocite{*}`, and does not include DeepSeek-R1. The source-specific notes below retain the historical review context; they do not imply that every reviewed source appears in the current manuscript.

## Main corrections

- SWE-Bench Pro: original title and arXiv identifier were correct, but the author field was incomplete. The current 2025 revision has 22 authors, including Vijay Bharadwaj, Jeff Holm and Raja Aluri, who are absent from some cached first-version search metadata. Use the current arXiv record. DOI: `10.48550/arXiv.2509.16941`. The benchmark's overall dataset size is not interchangeable with the campaign's receipt-bound 731-task contract.
- VerilogEval: complete the ICCAD 2023 record with “Invited Paper” in the publisher title, pp. 1–8, IEEE, and DOI `10.1109/ICCAD57390.2023.10323812`. Include the revised benchmark reference because the campaign uses both code-completion and specification-to-RTL framings.
- Revisiting VerilogEval: its title changed. The 2025 v2 title is “A Year of Improvements in Large-Language Models for Hardware Code Generation.” The older 2024 title “Newer LLMs, In-Context Learning, and Specification-to-RTL Tasks” is still shown in the repository citation block. The supplied entry explicitly cites the 2025 revision and states first posting in 2024.
- Omni-MATH: use the ICLR 2025 published version, full 20-author list, and proceedings pp. 100540–100569. Retain the historical key `gao2024omnimath` only for compatibility; its bibliographic year is 2025. Normalize publisher HTML's “runxin xu” and inverted “Wang Benyou” against the paper/arXiv byline.
- LAB-Bench: preserve the arXiv preprint designation; an arXiv comment saying “Submitted to NeurIPS ... in review” is not evidence of acceptance. Full nine-author list and DOI `10.48550/arXiv.2407.10362` are supplied.
- AgentDojo: NeurIPS 37 (2024), pp. 82895–82920, DOI `10.52202/079017-2636`. Cite the paper for benchmark design and distinction between utility and adversarial security, not as evidence of this campaign's exact receipt outcome.
- DeepSeekMath: restore omitted authors Xiao Bi and Haowei Zhang; the primary record lists 11 authors.
- DeepSeek-R1: upgrade the cited record to the Nature version of record, 645(8081), pp. 633–638 (2025), DOI `10.1038/s41586-025-09422-z`. Do not mix the preprint title/author sequence with Nature metadata.
- Pineau et al.: JMLR 22(164), pp. 1–20 (2021). The JMLR publisher's own BibTeX has no DOI; remove the historical `10.5555/...` ACM-index identifier rather than presenting it as a publisher DOI.
- MLE-bench, BALROG and AgentHarm: use ICLR 2025 versions. MLE-bench byline order is taken from the published paper, where Weng precedes Madry; proceedings HTML reverses those final two names. AgentHarm's published 12-author byline differs from the arXiv listing, which also includes Eric Winsor and Jerome Wynne.
- CORE-Bench: use the computational-reproducibility benchmark (arXiv:2409.11363), not the unrelated 2026 code-retrieval benchmark with the same name. The current arXiv author metadata spells Nitya Nadgir correctly; older search snippets contain “Nagdir”.
- Tinker: identify it accurately as a dated technical announcement, “Announcing Tinker,” 1 October 2025. It is not counted toward the 19 scholarly references.

## Primary-source verification URLs

- DeepSeekMath: https://arxiv.org/abs/2402.03300
- DeepSeek-R1 Nature: https://www.nature.com/articles/s41586-025-09422-z ; publisher-registered metadata: https://api.crossref.org/works/10.1038/s41586-025-09422-z
- Henderson: https://ojs.aaai.org/index.php/AAAI/article/view/11694 ; publisher PDF verifies pp. 3207–3214: https://ojs.aaai.org/index.php/AAAI/article/download/11694/11553
- Pineau: https://www.jmlr.org/papers/v22/20-303.html ; publisher BibTeX: https://www.jmlr.org/papers/v22/20-303.bib
- Model Cards: https://dl.acm.org/doi/10.1145/3287560.3287596 ; publisher-registered metadata: https://api.crossref.org/works/10.1145/3287560.3287596
- Datasheets: https://dl.acm.org/doi/10.1145/3458723 ; author-institution record: https://www.microsoft.com/en-us/research/publication/datasheets-for-datasets/ ; publisher-registered metadata: https://api.crossref.org/works/10.1145/3458723
- AI Agents That Matter: https://arxiv.org/abs/2407.01502
- LoRA: https://openreview.net/forum?id=nZeVKeeFYf9 ; author metadata: https://arxiv.org/abs/2106.09685 ; official ICLR 2022 listing: https://iclr.cc/Downloads/2022
- SWE-Bench Pro: https://arxiv.org/abs/2509.16941
- LAB-Bench: https://arxiv.org/abs/2407.10362
- AgentDojo: https://proceedings.neurips.cc/paper_files/paper/2024/hash/97091a5177d8dc64b1da8bf3e1f6fb54-Abstract-Datasets_and_Benchmarks_Track.html
- VerilogEval: https://ieeexplore.ieee.org/document/10323812/ ; publisher-registered metadata: https://api.crossref.org/works/10.1109/ICCAD57390.2023.10323812 ; author record: https://arxiv.org/abs/2309.07544
- Revised VerilogEval: https://arxiv.org/abs/2408.11053 ; official harness and task-format explanation: https://github.com/NVlabs/verilog-eval/blob/main/README.md
- Omni-MATH: https://proceedings.iclr.cc/paper_files/paper/2025/hash/f9e1e8b56c7e363985ebeb0e9dd1a85c-Abstract-Conference.html ; author names cross-check: https://arxiv.org/abs/2410.07985
- MLE-bench: https://proceedings.iclr.cc/paper_files/paper/2025/hash/7e3767db483c942b883eb4f8cfb74e31-Abstract-Conference.html ; byline: https://openreview.net/pdf?id=6s5uXNWGIh
- CORE-Bench: https://arxiv.org/abs/2409.11363
- BALROG: https://proceedings.iclr.cc/paper_files/paper/2025/hash/f0b1515be276f6ba82b4f2b25e50bef0-Abstract-Conference.html ; author-provided citation: https://balrogai.com/
- WebArena: https://proceedings.iclr.cc/paper_files/paper/2024/hash/4410c0711e9154a7a2d26f9b3816d1ef-Abstract-Conference.html
- AgentHarm: https://proceedings.iclr.cc/paper_files/paper/2025/hash/c493d23af93118975cdbc32cbe7323f5-Abstract-Conference.html
- Tinker: https://thinkingmachines.ai/news/announcing-tinker/

No publisher DOI was shown for the ICLR records used. Their verified proceedings URLs are supplied instead; do not attach an arXiv DOI to a conference entry as though it identified the proceedings edition. arXiv-only entries use their actual DataCite-issued arXiv DOIs.

## Current artifact checks

Checked against the retained files on 18 October 2026:

- `../refs.bib` and `full-reference-metadata.bib` each contain the same 37 unique keys; all 37 resolve to citations in `../paper_camera.tex`
- `refs-review-additions.bib` contains verification copies of 15 keys already present in `../refs.bib`; it must not be appended to that file
- Of the 28 `article`, `inproceedings`, or `techreport` records, 16 are dated 2024 or 2025; across all 37 records, 18 are dated 2024 or 2025
- No validation claim is made for an unretained working bibliography. The final package build and `../verify_manuscript.py` provide the rendered-manuscript checks
- Direct IEEE and OpenReview page fetches were intermittently blocked during the initial review. That review used public publisher proceedings pages, publisher-registered Crossref metadata and primary arXiv/paper pages
