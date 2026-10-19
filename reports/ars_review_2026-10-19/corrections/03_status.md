# ARS corrections: 03 PES IEEE paper, status

Manuscript: `papers/pes_ieee_attribution/ArvindCR_GRPO_Attribution_IEEE.tex` (abbreviated `tex:` below) + `refs.bib`, worktree branch `ars/p03`.
Inputs: `reports/ars_review_2026-10-19/03_pes_ieee_stack_conditioned.md` (roadmap §5), `00_portfolio_editorial_synthesis.md` §§3B, 3C, 3F, 4, 5(03).
Path chosen: **A** (provenance/ledger method, retitled). The roadmap row DA-CRITICAL-3/R3-W4 explicitly recommends "Retitle/reframe to provenance/ledger method", and the decision letter recommends Path A for the inferred venue.

## Build

| Version | Pages | Undefined refs/cites | Overfull |
|---|---|---|---|
| pre-Muse (`0cc717ea2`) | 6 | 0 | 0 |
| Muse WIP (`0052ee7db`/`7eccfea05`) | 8 | 0 | 1 table (53.8pt) + appendix |
| this revision | **6** | **0** | **0** |

Built with `tectonic --keep-logs --outdir /tmp/ars03/post`. The only warnings are font-shape substitutions, and the `.blg` has no warnings. Cited keys equal the bib keys (29 = 29).

## Verification of Muse's edits (0052ee7db)

| Muse claim | Verdict | Evidence |
|---|---|---|
| Tensor SHA-256 prefixes 475c717f / d3885dd6 / 6fa60ff8 / 5c1e6a89 | correct | `shasum -a 256 platform_hybrid/experiments/results/n2_reward_tensor_resume/*_s0_tensors.jsonl` |
| GRPO/AERO rewards differ in 225/640 groups; summaries coincide (0.7203, 461, 427, 34); G=8 in all 40 AERO steps | correct | recomputed from the tensor files; matches `review_2026-10-18/PES_IEEE_Attribution/recompute.json` |
| "at most three distinct summary profiles" + **deduplicated replay 1,406/1,920, 1,296 (92.2%), 110** | **wrong premise, removed** | The files are four *distinct runs*. Per-step uniform counts agree on only 12/40 steps, and total rewards are 4,271 (GRPO) vs 4,237 (AERO). Only the aggregate totals coincide, so the pooled 2,560 does not double-count a run. The arithmetic was right (1867−461 etc.) but deduplicating was wrong. |
| W&B bins "post-hoc **deciles**", "207 **zero-reward** runs" | wording wrong, fixed | `PES_IEEE_Attribution_zvf_audit.py` uses fixed 0.1-wide bins (`int(rew*10)/10`), so bin 0.0 means reward <0.1. Counts are correct: 207/2/3/1/1/3/11/26/44/52 = 350 (`zvf_audit_recompute.json`). The 350 binned runs are exactly the finished runs, and the 18 crashed/failed runs have no summary values (`P5_minreport/zvf_audit_368_wandb_export.tsv`). |
| Qwen3-32B 0.33/0.18/0.08 (2, 2, 1 runs); Qwen3-4B-Instruct 0.89/0.84/0.92 | correct | `zvf_audit_recompute.json` grid_mean_zvf |
| Variant per-seed table, Δ, CI, t, p, BH, MDE, means | correct | `e1_confirmatory_wandb_export.tsv` (recomputed per seed) and `p11_recompute.json` |
| Confound table "TRL-style" stack for sweep and PPO/GRPO | **wrong, fixed** | Both come from a self-contained minimal trainer on Modal: `platform_hybrid/experiments/modal/modal_groupsize_zvf_sweep.py`, `modal_samestack_ppo_grpo.py` (Transformers, A10). The PPO/GRPO GRPO arm has G=8 (`GROUP = 8`), not "---". |
| PPO/GRPO and sweep scored "on the split reserved from training" (pre-existing wording that Muse kept) | **wrong, fixed** | Both scripts sample evaluation items from the global RNG after training. The `modal_samestack_ppo_grpo.py` docstring says the stored results used different items per arm, overlapping training draws by about 40% (PPO) and 7% (GRPO), paired by seed only. |
| PPO/GRPO samestack path `modal_registry/...` | incomplete, fixed | The real location is `platform_hybrid/experiments/results/modal_registry/modal_artifacts/tinkerrl-results/samestack/` (the root is now stated) |
| RAIT sibling title | truncated, fixed | `papers/rait2027/paper_camera.tex:11` |
| ICICC sibling title quoted verbatim | unstable, replaced | The title is under revision (`papers/icicc2027/ars_revision_2026-10-19/paper_camera.tex:4,17` reverts to the accepted title), so the paper is now described without a title |
| Retained r=+0.71 as an "unlogged directional observation" | replaced by full withdrawal | The task says remove or recompute only from an artifact. No artifact with n exists, so the number is removed (tex:285-287), matching the 05/06 disposition (§3F) |
| E1/E14 lane-table states "complete (2/731)", "complete, terminal note" | inconsistent with the taxonomy, fixed | Now "full-suite, denom. note" and "terminal" (tex:457-480) |
| Abstract, contribution bullets, per-estimator prose, timeline, availability/overlap appendix, S(p,G) demotion | kept, rewritten and condensed to 6 pp | see the table below |

New finding while verifying (now in the paper): under the GIFT flag, all 455 all-correct uniform groups (of 493) carry a constant advantage of +0.5, and the 38 all-wrong groups carry 0. So ZVF is not a zero-advantage count for that arm. Source: the `advantages` field of `gift_s0_tensors.jsonl` (tex:155-170, 256-258).

## Must-fix items

| ID | Status | What / where |
|---|---|---|
| DA-CRITICAL-3 / R3-W4 (title) | **done** | Author approved the new title on 2026-10-19. Retitled "Provenance-Bound Ledgers for GRPO-Family Post-Training: Zero-Variance Bookkeeping and a Fourteen-Lane Case Study on a Null Actor". The old title is recorded in a comment (tex:19-23). No causal stack language, and attribution is disclaimed in the abstract and conclusion (tex:39-62, 505-521). |
| R1-W6 (abstract taxonomy) / §3C | **done** | One taxonomy (strict / terminal-complete / full-suite with denominator note) in the abstract (tex:55-60), §IV-F (tex:404-424), Table IV class column (tex:432-447) and Table V states (tex:457-480). "Five complete" is gone. |
| EIC-W1 (paper type, bullets) | **done** | Paper type stated in the abstract and §I. The four bullets are rescoped to the delivered evidence, and the harness is demoted to infrastructure (tex:96-120). |
| R2-W1 (confounding table, causal language) | **done** | Table II with corrected stacks/G (tex:330-346). Causal stack phrases removed. |
| R3-W1 (per-estimator table) | **done** | New Table I covers what a uniform group contributes for each estimator and which runs used it, including the GIFT finding (tex:155-170). |
| R3-W2 (S(p,G)) | **done** | S(p,G) deleted outright. No gradient prediction is made or tested (tex:188-201). |
| R2-W2 (§III-E: pool composition, decontamination) | **author-owed** | The 512-row pool manifest, the 80 drawn prompts and the training code are not recovered in the repo. Only `zvf-program/flagship/pavlov_api_bank_rlvr_train_adapter.py` exists and gives no composition. The non-claim and an author-todo stay in the text (tex:210-226). Contract-vs-replacement taxonomy and paired-rerun foregrounding: **done** (tex:220-226, 483-490). |
| R2-W2 (Tables I–II: Omni-MATH denominator) | **done** | 4,426 judged is canonical, and the source-ledger conflict is noted (tex:413-417, Table IV). The SWE-bench Pro denominator note is kept. |
| R1-W1 / DA-MAJOR-3 (identical GRPO/AERO arms) | **done** | Hash-compared: four byte-distinct files, distinct runs, coincident totals only. AERO is relabelled a second realisation of the baseline (its flag never changed G). GIFT, AERO and AReaL are defined as internal flags from `platform_hybrid/experiments/variance_mitigation_integration.py:68-122`. Muse's wrong dedup is removed (tex:232-262, 309-318). Hashes are in Appendix A (tex:528-531). Archival renaming of the flags is **author-owed** (todo at tex:240-241). |
| R1-W3 (W&B 350-run accounting) | **done** | All 350 finished runs binned with per-bin n. The 18 crashed/failed runs are explained, and the post-hoc fixed-width bins are disclosed (tex:273-287). |
| R1-W4 / DA-CRITICAL-1 (r=0.71) / §3F | **done** | Number removed and association withdrawn (tex:285-287). Every "signal" claim is now bookkeeping/identity: "Signal Diagnostics" was dropped from the title, and §III-C says agreement with h_G is consistency, not prediction (tex:188-201). |
| R1-W5 (per-seed table, exact stats) | **done** (verified Muse) | Table III (tex:374-391); source TSV and script in Appendix A. |
| DA-CRITICAL-2 (ledger as null-actor case study) | **done** | Title, abstract, §III-E heading, §IV-F heading, Table IV caption ("null actor") and §V (tex:22-23, 50-54, 210, 404, 432-435, 483-490). Held-out language follows 02/06 (§3B): "not described as held-out" (tex:219), and the subsection heading no longer says held-out. |
| EIC-W3 / DA-MINOR-1 (grey refs, years) | **done** for refs; **author-owed** for external verification | The 17 citations used only for ungraded/closed/exploratory lanes, including all six 2026 grey sources, were moved out of the archival refs into `papers/pes_ieee_attribution/lane_provenance.bib`, which the build does not use (tex:424-428, Table V caption). DPO/TÜLU-3 refs were cut for space. Remaining key/year items: the `thinkingmachines2024tinker` key says 2024 but prints 2025, and `le2025rlzvp` prints ICLR 2026. These are cosmetic key-name mismatches, not printed errors. Online verification of the 2025–2026 arXiv entries was not done (no web check in this pass). |
| EIC-W4 (availability/compute/platform) | **done** as truthful statements; contents **author-owed** | Appendix A "Availability" (tex:549-553): no release claimed, compute platforms named. An author-todo covers funding, acknowledgements, version pins, conflict note and AI-use declaration, which cannot be verified from records. |
| DA-MAJOR-1 (artifact appendix) | **done**, partial by nature | Paths verified at commit `7eccfea05`, tensor hashes, scripts and outputs for Fig. 1, §IV-A/C, §IV-B, §IV-D (tex:524-548). The non-reproducible list is explicit (tex:544-548). An archival commit/DOI is **author-owed**. Tinker run IDs and the adapter hash are not in the repo. |
| DA-MAJOR-2 (overlap statement) | **done** (text); enumeration **author-owed** | Names the ICICC 2027 paper, thesis, ACL ARR and RAIT siblings with what is shared and what is unique. It says shared tables are not independent confirmation (tex:554-566). A per-sibling shared-text enumeration and submission status need collaborator input. |
| DA-CRITICAL-1 (global "signal" claims) | **done** | See R1-W4. No held-out predictive test exists, so this takes the downgrade route. The new conclusion rule (4) also covers hash-comparing arms (tex:505-521). |

Should-fix items handled along the way: DA-MINOR-2 (no PCD; GU is a plotting axis only, tex:181-187), R2-W4 (replay is arithmetic only and recommends nothing, tex:309-318), R1-W2 (§IV-D titled protocol demonstrations), R1-W7 (length bias condensed into §V), DA-MINOR-3 (timeline, tex:226-229), and EIC-W2 (five practice rules, tex:510-521). Not done: R3-W3 (dispersion/effective-G decomposition) and R2-W3 (ZVF-beyond-mean-reward head-to-head). Both need new analysis.

## Unresolved / flagged (not changed)

- **Phase-1 model provenance conflict.** The paper and `tools/review_ledger/PES_IEEE_Attribution_recompute.py:19` say Qwen3.5-4B GSM8K on Tinker. `platform_hybrid/docs/p5p8_improvements/101_p5_n2_unpacking.md:14` describes the N2 panel as "Qwen2.5-0.5B-on-MATH". The tensor files carry no model field. **Author-owed.**
- These numbers are kept from the pre-Muse text and were not re-traced in this pass: the sampling-only cohort (0.130/0.190/0.155), the length contrast (195.0→180.3, etc.), the label-collision exhibits (0.00/0.58, 0.856/0.050, 0.710–0.744), LAB-Bench 450/1,967, AgentDojo 88/97, and the paired-rerun McNemar p=0.25.
