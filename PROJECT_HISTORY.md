# Project History and Academic Ownership

This repository contains two connected academic phases. The codebase is shared, but the deliverables and authorship are not.

## Semester 3 — Group 6

Semester 3 was completed by six students: Arvind C R, Sandhya Jeyaraj, Madhu Kumara L, Mohammad Rafi, Dhruva N Murthy, and Arumugam Chetty K. Anwesh Reddy Paduri and Narayana Darapaneni were the project guides.

This phase established the literature foundation, multi-framework RL post-training environment, first experiments, group benchmark paper, capstone report, and NeurIPS main-track submission. The blind-review manifest is dated April 19, 2026. The frozen boundary is tag `capstone-final-2026-04-25` at commit `21a99ef7`.

Review this phase in [`platform_hybrid/sem 3 work/`](sem%203%20work/).

## Semester 4 — Solo continuation

After the capstone boundary, Arvind C R continued the project individually with Ramesh Prakash Guledgudd as project guide. This phase expands the experiment and evidence base, develops the P1–P8 paper series covering scaling, ZVF, group size, length bias, reporting standards, a GRPO registry, an adaptive controller, and an applied fraud study, and produces the NeurIPS workshop submission first committed on June 21, 2026.

Review this phase in [`platform_hybrid/sem 4 work/`](sem%204%20work/).

## Why the implementation stays shared

The second phase deliberately builds on the first. Experiment drivers, result tables, figures, LaTeX sections, and audit scripts reference one another across the repository. Physically moving those files into semester trees would create duplicate sources or break the working artifact. The two semester folders are curated academic views:

- Semester 3 contains immutable historical deliverables, the NeurIPS main-track submission, and its original citation record.
- Semester 4 contains the NeurIPS workshop submission, freshly built current papers, a source/evidence map, and the solo-continuation provenance.
- Shared code and raw evidence remain at the root, with Git preserving exact history.

This arrangement lets a professor distinguish ownership immediately while retaining a reproducible research repository.
