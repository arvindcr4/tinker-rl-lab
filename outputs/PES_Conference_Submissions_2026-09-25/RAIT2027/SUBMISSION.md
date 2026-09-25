# RAIT 2027 submission package

- **Venue:** RAIT 2027, IIT (ISM) Dhanbad, 12-13 Mar 2027. Official site: https://people.iitism.ac.in/~rait/ (deadlines on dates.php, author guidelines on guidelines.php)
- **Deadline:** 2 Oct 2026 (check the dates page again before submitting; extensions are common)
- **Publisher:** IEEE Xplore ("submitted for possible inclusion")
- **Format:** IEEE two-column conference template. Up to 6 pages free; extra pages up to 8 at $50 each. Double-blind review.

## Title
Signal Starvation in GRPO: Measuring and Acting on Zero-Variance Groups in RL Post-Training of LLMs

## Abstract (plain text for the form)
Group Relative Policy Optimization (GRPO) standardises each reward against the statistics of a group of G completions for the same prompt. When all completions in a group receive the same verifiable reward, every advantage is exactly zero and the group contributes no reward-driven gradient, even though it was sampled and scored at full cost. We study this signal starvation directly. We define the Zero-Variance Fraction (ZVF), recompute it from stored reward tensors, and find that on our main configuration roughly 0.72--0.77 of each batch is starved, mostly because prompts are already solved rather than because they are too hard. A simple signal model, S = p(1-p)(1-h_G(p)) with h_G(p) = p^G + (1-p)^G, makes three falsifiable predictions; two reproduce at scale, including a U-shape of ZVF across a 368-run audit, and the third holds only directionally at toy scale. Because ZVF cannot tell mastery from incapacity and collapses under a 1e-4 reward jitter, we promote the pairwise contrast density (PCD) as the control signal. Finally, we audit an adaptive group-size controller: it ties the best fixed recipe (+0.575 held-out gain at 186 rollouts), but 92.3% of its escalations fire on all-correct groups, where no group size can restore contrast. Curriculum and re-baselining interventions return nulls. We conclude that ZVF and PCD are useful monitoring diagnostics, not objectives to optimise.

## Keywords
reinforcement learning; GRPO; large language models; verifiable rewards; advantage estimation; group size

## Author metadata (for the submission form only, never in the PDF)
1. Arvind C R, Dept. of Computer Science and Engineering, PES University, Bengaluru, India. arvindcr4@gmail.com (corresponding author)
2. Ramesh Prakash Guledgudd, Dept. of Computer Science and Engineering, PES University, Bengaluru, India. Email: fill in before submission.

## Files
- `paper_anon.pdf`: upload this for review (4 pages)
- `source_anon.zip`: anonymised LaTeX source (paper_anon.tex, body.tex, refs.bib, paper_anon.bbl, figures/)
- `paper_camera.pdf` / `paper_camera.tex`: camera-ready version with authors, for use after acceptance

## Checklist
- [x] Page count: 4 pages including references (limit 6)
- [x] Anonymisation: PDF text grep for "Arvind", "PES", "Guledgudd", "arvindcr4" and "tinker-rl-lab" returns 0 hits in paper_anon.pdf; no repository paths or programme names
- [x] References: every entry comes from the verified refs.bib (arXiv IDs checked against the arXiv API on 25 Sep 2026)
- [ ] Confirm the co-author and their email; get the guide's approval to submit
- [ ] Presentation must be IN PERSON at IIT (ISM) Dhanbad (a proxy presenter is allowed; remote presentation is not)
- [ ] Registration fee: TBA on the site; check before committing
- [ ] Check that no other venue has this paper under review at the same time (it is a different paper from the ICICC and ARR submissions, but keep it that way)
- [ ] **The user must create the account and submit personally**; nothing has been submitted
