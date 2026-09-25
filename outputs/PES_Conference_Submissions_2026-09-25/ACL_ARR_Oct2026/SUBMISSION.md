# ACL Rolling Review — October 2026 cycle: submission package

**Deadline:** 12 Oct 2026 (AoE). Confirm this at https://aclrollingreview.org/dates before submitting.
**Where to submit:** OpenReview, ACL Rolling Review October 2026 cycle. The user must submit it themselves; nothing has been submitted.

## Files
- `paper_anon.pdf`: anonymised review version to upload. 4 pages in total; the content before Limitations ends on page 3, so it is under the 4-page short-paper limit.
- `paper_camera.pdf`: camera-ready version with author names. Only for use after acceptance.
- `source_anon.zip`: anonymised LaTeX source (`paper_anon.tex`, `body.tex`, `refs.bib`, `.bbl`, and the ACL `acl.sty`/`acl_natbib.bst` downloaded from github.com/acl-org/acl-style-files).

## Form fields
- **Title:** The Stack Is Part of the Result: Negative Results on Attributing GRPO-Family Differences
- **Paper type:** Short paper
- **Keywords:** reinforcement learning from verifiable rewards; GRPO; reproducibility; negative results; reporting standards; LLM post-training
- **Research area (suggested):** Machine Learning for NLP, or Resources and Evaluation
- **Contribution type (suggested):** Reproduction study / negative results; also a resource (reporting standard and registry)
- **Abstract:** paste the abstract from `body.tex`, which is the same as the one in the PDF.

## Responsible NLP checklist (suggested answers; the user must confirm each one)
- A1. Limitations section? **Yes**, in the section "Limitations".
- A2. Potential risks discussed? **Yes**: the Ethics Statement covers over-reading underpowered nulls.
- B. Artifacts used or created? **Yes**: public models (Qwen, Llama), the GSM8K benchmark, RL libraries (TRL, veRL, OpenRLHF), and a managed training API. Suggested: cite each and state that its licence permits research use. *Confirm licences.*
- B (created artifacts): the registry schema and CLI. *Decide whether to release them, and under which licence, before answering.*
- C. Computational experiments? **Yes**: model sizes (~0.5B–35B) and step budgets are stated. *The user must add compute and GPU-hour details if the form asks for them. The paper states short horizons, not total GPU hours.*
- C (stats): **Yes**: tests, effect sizes, CIs, power and MDE are reported.
- D. Human annotators or participants? **No.**
- E. AI assistants used in research or writing? *The user must answer truthfully. An AI assistant helped prepare this manuscript, so disclose it according to ACL policy.*

## Checklist before submitting
- [x] Page count: content ends on page 3, under the 4-page short-paper limit. Limitations, Ethics and references come after.
- [x] Anonymisation: the PDF text has zero hits for "Arvind", "PES", "Guledgudd", "arvindcr4", "tinker-rl-lab", "M.Tech", "Bengaluru", "outputs/" and "platform_".
- [ ] No self-identifying links in supplementary material. None is attached. If you add code, host it on an anonymous repository (e.g. anonymous.4open.science).
- [ ] **No dual submission:** ARR does not allow the same work to be under review at another archival venue at the same time. This paper is deliberately scoped to stack attribution. The IEEE (RAIT) and Springer (ICICC) papers cover the ZVF/controller and the evaluation campaign; the user must confirm the overlap stays below the venues' thresholds.
- [ ] **Reviewer registration:** every author must register as an ARR reviewer (OpenReview profile + reviewer form) within 48 h of the deadline. Both authors need complete OpenReview profiles.
- [ ] Both authors agree to the author list and order (confirm with Ramesh Prakash Guledgudd).
- [ ] **The user must submit it themselves** on OpenReview. Nothing has been submitted, and no account was created.
