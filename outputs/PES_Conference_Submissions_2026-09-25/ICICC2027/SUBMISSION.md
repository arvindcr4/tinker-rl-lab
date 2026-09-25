# ICICC-2027 submission package

**Venue:** 10th International Conference on Innovative Computing and Communication (ICICC-2027), Shaheed Sukhdev College of Business Studies, University of Delhi, 5–6 Feb 2027. Proceedings: Springer LNNS (Scopus-indexed). Double-blind review.

## What was verified on the site (25 Sep 2026)

Sources: icicc-conf.com homepage, the site's JS bundle (Important Dates and Author Guidelines text), /call_for_papers, /registrations, and the CMT page.

| Item | Finding |
|---|---|
| Paper submission deadline | **30 September 2026**. It has not moved. |
| Notification of first review | 30 Oct 2026 |
| Revised manuscript | 25 Nov 2026 |
| Acceptance / rejection | 20 Dec 2026 |
| Registration deadline | 30 Dec 2026 |
| Final manuscript | 10 Jan 2027 |
| Page limit | The CFP says a **maximum of 10 pages**. The Author Guidelines say the "standard length is 10–12 pages (including references and appendices). Additional pages may be subject to extra charges." This package is **9 pages**, within both limits. |
| Template | Springer LaTeX/Word templates from Springer's official site. This package uses `llncs.cls` + `splncs04.bst`, the LNCS/LNNS format. |
| Review | Double-blind. The CMT upload is the PDF in Springer format. |
| Plagiarism | The Guidelines say **< 15 %** similarity (Turnitin/iThenticate), rejected without review otherwise. The CFP page says ≤ 20 %. **Use the 15 % bar.** |
| Camera-ready | PDF + LaTeX source and a signed Springer copyright form (CTP). The paper must be presented, in person or virtually, to appear in the proceedings. |
| Registration fee (Springer track) | Research scholar/student: **INR 12,000** early (before 30 Oct 2026), INR 13,000 late. Academician: INR 13,000 / 14,000. Charged only after acceptance. At least one author must register. |
| CMT | https://cmt3.research.microsoft.com/ICICC2027 is **now live**: the login page shows "10th INTERNATIONAL CONFERENCE ON INNOVATIVE COMPUTING AND COMMUNICATION (ICICC 2027)". However, the homepage still shows "CMT Submission Link will be coming shortly!" I could not confirm whether the submission form is open without logging in. |

**Caution:** the homepage banner still says the CMT link is "coming shortly" five days before the deadline. Expect the deadline to be extended, but plan for 30 Sep. If the CMT form isn't accepting submissions on 29 Sep, email info@icicc-conf.com.

## Files

| File | Purpose |
|---|---|
| `paper_anon.pdf` | **Upload this to CMT.** Anonymised, 9 pages. |
| `paper_anon.tex`, `source_anon.zip` | Anonymised source (tex, bbl, bib, 2 figures) |
| `paper_camera.pdf`, `paper_camera.tex` | Named camera-ready version, for use after acceptance |
| `refs.bib` | 24 references, every one verified against arXiv or OpenAlex (10 are cited) |
| `figures/` | Receipt-pipeline figure and lane-status figure, both cropped so no repo paths remain |

## CMT form fields

- **Title:** Receipt-Bound Held-Out Evaluation of a GRPO-Trained Language Model Across Fourteen Agentic and Reasoning Suites
- **Track:** Innovative Computing (Machine Learning / Artificial Intelligence)
- **Authors (enter in CMT, not in the PDF):**
  1. Arvind C R, Department of Computer Science and Engineering, PES University, Bengaluru, India, arvindcr4@gmail.com (corresponding)
  2. Ramesh Prakash Guledgudd, Department of Computer Science and Engineering, PES University, Bengaluru, India (email needed; CMT requires one for every co-author)
- **Keywords:** Large language models; Reinforcement learning; Benchmark evaluation; Agentic evaluation; Reproducibility; Evaluation governance
- **Abstract:**

> Benchmark results for reinforcement-learning post-trained language models are usually reported as single headline numbers. The report rarely says how much of each suite was actually graded, whether the original benchmark or a substitute was run, or whether the number can still be traced to the run that produced it. We report a held-out evaluation of one Group Relative Policy Optimization (GRPO)-trained actor against fourteen agentic and reasoning benchmark suites under a receipt-bound, fail-closed governance regime. The actor is fixed across every lane, each suite is graded only by its own native evaluator at a pinned revision, and every reported figure is bound to a write-once receipt that a deterministic ledger check recomputes. Four scopes closed complete: LAB-Bench public split (1,967/1,967 items, accuracy 0.2288), AgentDojo benign utility (97/97 episodes, utility 0.9072), VerilogEval (312/312 problems, 41.35% pass@1) and Omni-MATH (4,428/4,428 dispositions, official accuracy 51.31% reproduced). SWE-bench Pro carries a full original-contract score of 2/731 = 0.274% pass@1 with every non-native outcome retained in the denominator. The remaining lanes are reported at their true terminal states: partial with coverage, blocked on cloud quota, or closed as externally blocked with a named reopen condition. A separately authorised base-model rerun of BankerToolBench scored 0.0 over 100/100 trials, which an audit traces to tool-dialogue collapse rather than domain reasoning. Eleven of eleven deterministic ledger checks pass. Original-contract and replacement-scope numbers are never pooled, no cross-suite aggregate is computed, and no improvement over any baseline is claimed.

## Pre-submission checklist

- [x] **Page count:** `paper_anon.pdf` = 9 pages, `paper_camera.pdf` = 9 pages (limit 10; 10–12 per guidelines).
- [x] **Anonymisation:** a text search of `paper_anon.pdf` for "Arvind", "PES", "Guledgudd", "arvindcr4", "tinker-rl-lab", "pavlov" and "outputs/" found **0 hits**. The same search over the `.tex`/`.bbl`/`.bib` in `source_anon.zip` also found 0 hits. The adapter is given as "a GRPO LoRA adapter (identifier withheld for review)".
- [x] **Figures:** no "Source: outputs/…" footnotes, repo paths or usernames.
- [x] **Numbers:** every figure was copied verbatim from thesis ch09 (full results) and ch10. None was invented.
- [ ] **Plagiarism check < 15 %.** The text overlaps heavily with the thesis and the other two conference papers. Run it through Turnitin/iThenticate before submitting, and paraphrase if the score is high (self-overlap counts).
- [ ] **No dual submission.** This paper (evaluation campaign) is written to be distinct from the RAIT and ACL-RR papers, but all three come from the same project. Check each venue's policy. ACL Rolling Review forbids the same work being under review elsewhere at the same time, so make sure the overlap between this paper and the ACL paper is small.
- [ ] **Co-author consent** from Ramesh Prakash Guledgudd, plus their email for CMT.
- [ ] **Fee:** INR 12,000 (student, early) is due only if the paper is accepted, plus presentation (virtual is allowed).
- [ ] **Submit it yourself:** create or log into a CMT account, choose ICICC 2027, upload `paper_anon.pdf`, and enter authors, keywords and abstract from above. Nothing has been submitted, registered or emailed.
