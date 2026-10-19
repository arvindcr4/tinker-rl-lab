# 06 — Public thesis revision: Must-fix status (19 October 2026)

Panel report: `reports/ars_review_2026-10-19/06_public_thesis_2026-10-11.md` (§4 Revision Roadmap) and `00_portfolio_editorial_synthesis.md` §5 item 06. Corrected copy: `reports/public_revision_2026-10-19/`; the frozen `reports/public_revision_2026-10-11/` is unchanged. `T/` = `reports/public_revision_2026-10-19/thesis/`. Line numbers are from the committed sources.

No number was invented. Every added figure traces to the 11 October thesis body or to `reports/public_revision_2026-10-11/PUBLIC_CHECKOUT_NUMBER_AUDIT.json`. That covers 970/828/22/120, 33 arithmetic and 501 token-presence passes, and the per-chapter split of the 120 missing occurrences, which comes from the JSON `groups`.

## Muse round-2 verification (commit 0052ee7db)

| Muse change | Verdict |
|---|---|
| Saturated exact sign-flip p = 0.625 placed before paired-t 0.374 | Kept. 0.625 is in the 11 Oct body (`ch06` §6.2.3, §6.4.2; Table E.1). |
| Unsaturated +0.050 demoted; exact p 0.0625 before paired-t 0.016; untuned critic, 8× prompts, 190–196 tokens under the 200 cap | Kept. All values are in `ch06` §6.5.5 and `samestack_scope_check.json`. |
| 828/22/120/970 and 33 arithmetic passes | Kept. Matches the audit JSON. The §1.8 path was wrong (the file is not in the 10-19 dir) and is fixed. |
| 17 finite / 9 synthetic, 41/52 synthetic safe rows, 7 imputed anchors, 45/60 rows, 461/640, 92%, 1,259/1,967, 8-gram over 3 suites, "30–50 steps" | Kept. Each is in the body (`ch06:65,85`, `ch07:59`, `ch09:80`, `ch04:124`, `ch10:38,51` T2). In `ERRATA.md`, "simulation rows" was corrected to "projection rows". |
| Terminology box | Fixed. Its clipped-GRPO examples included TRL rows, which are "defaults, not set in code". It now assigns every study from Table 4.2, and the group-size sweep and controller trace are REINFORCE group-mean. |
| "Sole RQ2 evidence" = §6.5.5 only | Fixed. The unsaturated Dr. GRPO and variant-audit comparisons are also RQ2 evidence, so the wording is now "unsaturated comparisons". |
| E14 erratum paragraph | Fixed a duplicated sentence. 4,426 is declared canonical. |
| Abstract, revision record, AI-declaration AUTHOR_TODO flag, claim map, §6.3 and §6.3.4 retitles, §7.3 status box, §9.2 disclosure, Appendix F quarantine, C1 framing, E.4 errata, READMEs | Kept, with the edits listed below. A stale reference to the nonexistent `fixes/06_…changes.md` now points to this file. |
| Thesis 29069cdfb corrections (75-run P8 manifest, 368-run zvf-audit, P8 configs) | Present. `ch04` and `ch07` differ from 10-11 only in Muse's lines. `ch_back_run_registry.md` was identical to 10-11 before this pass; this pass added one caption sentence. |

## Must-fix items

| ID | Status | What / where |
|---|---|---|
| DA-CRITICAL-1 (abstract to body tier) | **done** | `T/frontmatter.tex:200-310`. Exact p first (`:239-253`), loss-form labels inline (`:209-217`, `:238-247`), harness and decontamination qualifiers (`:261-291`), C1 framed as an audit (`:294`), claim map (`:298`). |
| EIC-W2 (independent review, AI declaration, institutional front matter) | **author-owed**, partly mitigated | An editing pass cannot do independent review or PES-policy confirmation. Explicit review-copy status is at `T/frontmatter.tex:24,68-86`, the AI-declaration flag at `:173`, and the items in `AUTHOR_TODO.md` #1–3. |
| EIC-W3 (contribution matrix, collaborator authorisation) | **author-owed** | Needs signatures or consent (`AUTHOR_TODO.md` #4). |
| R2-W1 (terminology box in all headlines) | **done** | Box at `T/ch01_introduction.md:15`. Labels in the abstract (`T/frontmatter.tex:209-247`), Table 1.1 rows (`ch01:145-163`), `ch06:3`, the `ch09:3` actor-loss banner and the E.4 bullet. |
| R1-W3 (P2 to instrumentation; quarantine) | **done** | `T/ch06_results_core.md:3,49,51,79`; `T/ch04_methodology.md:96`; abstract `(1)` at `T/frontmatter.tex:219`. Measured-only held-out validation needs a new run and is listed as future work. |
| R3-W1 (828/22/120 disclosure; "not publicly verifiable" labels) | **done** (labels); deposit **author-owed** | Disclosed in the abstract (`T/frontmatter.tex:276`), revision record (`:77`), §1.8 with a per-chapter location of the 120 (`T/ch01_introduction.md:135`) and §9.2 (`T/ch10_synthesis_conclusions.md:29`). Labels are in Table 1.1 (`ch01:163`), `ch07:23,39`, the Table C.3 caption (`T/ch_back_notation.md:100`) and the Table A.8 caption (`T/ch_back_run_registry.md:307`). Controlled-access deposit is `AUTHOR_TODO.md` #7. |
| DA-CRITICAL-3 (stored-evidence labels; unrecoverable items stated once) | **done** | `T/ch09_results_campaign.md:3` (banner: loss unknown, adapter revision withheld, 80 consumed prompts unknown); lane paragraphs `ch09:82-94`; §8.8 `ch09:200`; §9.2 `T/ch10_synthesis_conclusions.md:29`; abstract `(4)` at `T/frontmatter.tex:261`. The labels make no claim about missing source; rerunning is not required. |
| R3-W2 (abstract campaign scores carry stored-evidence and decontamination scope) | **done** | `T/frontmatter.tex:261-291`: stored-evidence, 8-gram envelope over 3 of 14 suites, receipts as file consistency only. |
| R1-W1 (saturated contrasts to pilot; RQ2 on unsaturated; future design) | **done** | Abstract `:238-247`; `T/ch01_introduction.md:31,39,85,150-153`; `T/ch06_results_core.md:3,195`; `T/ch10_synthesis_conclusions.md:76`. The powered item-paired design is stated as required future work, not run. |
| R1-W2 (demote untuned-PPO contrast; exact p first) | **done** | Abstract `:248-253`; `T/ch06_results_core.md:185,187`; `ch01:31,85,151`; `ch10:76,80`. Renaming the arm "batch-centred baseline" was not done; the arm is labelled "minimal untuned-critic PPO" throughout instead. Tuning the critic would need a new run. |
| R2-W3 (controller unevaluated; plug-in tautology; ranking to appendix) | **done** (ranking moved to an archival subsection, not a separate appendix) | Status box at `T/ch07_results_infra.md:51`; heading `:49`; refinements and eight-controller ranking under `### 7.3.1 Archival detail … (not controller evidence)` at `:67`; abstract `:255`. Moving them to a lettered appendix was skipped to avoid renumbering the appendices. |
| R3-W3 (harness-score labels at first mention; E14 erratum) | **done** | Abstract `:261-291`; lane paragraphs E8/E10/E11/E14/E1 `T/ch09_results_campaign.md:82-94`; §8.8 `:200`; Table 1.1 `ch01:163`; Table C.3 rows and caption `T/ch_back_notation.md:85,94,98,100`; Table A.8 caption; figure captions (campaign design, harness, lane status, E8, E11) in `T/build_thesis.py`. The E14 erratum is at `ch09:88` and Table E.1 row `T/ch_back_errata.md:37`, with 4,426 canonical; the fig_lane_status caption now leads with 51.31%. |
| R3-W4 (rename flip-risk; schema-only downgrade) | **done** | `T/ch03_requirements.md:39,168`; `T/ch04_methodology.md:102,104`; `T/ch01_introduction.md:92`; §7.2 heading and status box `T/ch07_results_infra.md:25,27`; `T/figures/fig_research_questions.tex:114,115,138`; abstract `(2)` at `:231`. External-corpus validation needs new work, so the registry is downgraded to a schema proposal instead. |
| R3-W5 (blind human rating; C1 as audit) | framing **done**; rating **author-owed** | `T/frontmatter.tex:294`; `T/ch07_results_infra.md:109`; `T/ch10_synthesis_conclusions.md:88`. Human rating cannot be produced by an editing pass (`AUTHOR_TODO.md` #5). |
| DA-CRITICAL-2 (independent vs closed-system checks) | **done** | §9.2 `T/ch10_synthesis_conclusions.md:29` (tensors and bootstrap independent; registry, same-step null r = 0.92 and 501 token-presence passes closed-system); §1.8 `ch01:135`; abstract `(2)` and `(4)`; §7.2 box `ch07:27`. |

Should-fix items touched: DA-MAJOR-1 (`ch10:7` scope limit and the conclusion sentence at `ch10:84`), DA-MAJOR-3 (abstract `:204-207`), EIC-W1 claim map (`frontmatter.tex:298`) and DA-MINOR-1 (Appendix F quarantine, `T/ch08_results_fraud.md:3`). Not touched: R1-W4, R1-W5, R2-W2, R2-W4, DA-MAJOR-2 and DA-MINOR-2.

## Build

`python3 compile_figures.py` (24/24), then `python3 build_thesis.py`, using pandoc and Tectonic 0.17.0. The Tectonic cache was first warmed with one uncached compile because `report.cls` was missing from the cache. Result: exit 0, **326 pages**, 0 undefined references or citations, 0 `??` in the extracted text, and 14 overfull hboxes of at most 9.3 pt. PDF `T/Thesis_Report_ArvindCR_2026-10-19_ARS_CORRECTED_REVIEW.pdf`, SHA-256 `3e278e30811a8d16b0a4d88db8b645caf2515146976a32862fb94f523786586d`. This is a compile check only: there is no quality receipt and no independent review. `python3 -m pytest -q tests/test_audit_thesis_numbers.py`: 158 passed.
