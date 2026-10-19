# ICICC 2027 paper 243: revised claim-to-source map

Audit date: 18 October 2026. This revision changes manuscript/source-package material only. It does not train, sample, execute benchmarks, rerun judges, alter stored results or modify v2 work. Revision concerns were reconciled with the current manuscript snapshot. Claims below are based on retained source evidence.

## Evidence levels

1. Retained structured evidence: JSON/CSV in this checkout can be read and values recounted; this does not independently authenticate the execution or grader
2. Retained check record: records what its author checked historically; reading a PASS is not rerunning its original inputs
3. Retained narrative/recovery report: describes artifacts or audits not all supplied here; unavailable primary artifacts remain unavailable
4. Printed-count arithmetic: verifies division/reconciliation, not authenticity of the underlying observations
5. Public primary metadata: verifies benchmark/model identity and source availability at audit time, not historical execution or tensor equivalence

## Manuscript identity and title

Accepted title: Receipt-Bound Held-Out Evaluation of a GRPO-Trained Language Model Across Fourteen Agentic and Reasoning Suites

Printed title: the accepted title (author decision 2026-10-19). The proposed revision "Receipt-Bound Evaluation Governance: A Fourteen-Lane Language-Model Case Study" was withdrawn.

Author and venue approval of the change remain required. It has not been changed in any portal. Historical source revision `a03d0677`, path `outputs/PES_Conference_Submissions_2026-09-25/ICICC2027/paper_camera.tex`; identified-source SHA256 `71684f2ccc8e5f4d4082dc5fe18bf854119967d894bf2d9916538a2aa06dba1c`; anonymous-source SHA256 `01976b09c7f59599dcb00cedc4e48f94fdfa04ca637fdaa072fc6928b49f9cc1`. Arvind C R and Ramesh Prakash Guledgudd, PES University, and the sole historically corroborated corresponding email are preserved.

## Actor and training

- Official base: `Qwen/Qwen3.6-35B-A3B`, immutable revision `995ad96eacd98c81ed38be0c5b274b04031597b0`
- Exact adapter repository: `arvindcr4/pavlov-portfolio-qwen36-seed809-stepfinal-tinker-cf0ad8c1-1f1b-5ff-9f777c4018b6`, revision `64444133c55d88c3f1bf0df8a2f5d7ac646125c8`
- The shorter adapter name in the critique is incomplete. The pinned public API identifies both SHAs; model metadata and adapter config corroborate base name, LoRA rank/alpha 32/32 and all-linear targets. No tensors were downloaded; reported LFS hash is server metadata. Adapter config's `revision=null` does not independently bind base SHA
- Sources: `zvf-program/flagship/pavlovs_domain_contract.json`; `zvf-program/e5_successor27/resource_request_2026-09-19.json` immutable_identity; `verification/huggingface-public-metadata-2026-10-18.json`; full public URLs in `verification/citation-review-2026-10-18.md`
- Training table source: `reports/public_revision_2026-10-11/thesis/ch09_results_campaign.md` section 8.1.1; provenance qualifications in `ch_back_campaign_detail.md` D.2 and `ch_back_reproducibility.md` B.11
- Reported API-Bank-RLVR and SWE-Gym; 512-row pool; 4 completions per group, 2 prompts per step and 40 steps yield 80 prompt draws/320 completions; learning rate 1e-5; Adam betas (.9, .95), epsilon 1e-8; prompt/output caps 1,536/384 tokens; temperature .7; top-p .95; seed 809
- Original training identity receipt/logs are described as recovered/hash-matched in the retained report, but missing dirty source and exact consumed-row inventory remain unresolved. The declared source-list is not a complete consumed corpus or content-level E1–E14 overlap audit
- No causal GRPO benefit, canonical-objective reconstruction, training-seed variance estimate or blanket held-out claim follows

## Original/replacement identity and dated states

- Original registry: `zvf-program/flagship/pavlovs_domain_contract.json`; replacement definitions: `pavlovs_open_portfolio_overrides.json`; exact benchmark public sources: `verification/citation-review-2026-10-18.md`
- Original E1–E14: SWE-bench Pro, FrontierSWE v1, SDAB, BankerToolBench, APEX-Agents, WebBench, BinaryAudit, LifeSciBench, MLE-bench, AgentHarm, VerilogEval, App-Bench, OpenReward games, FrontierMath
- Reported replacements: E1 SWE-bench Multilingual; E2 CORE-Bench; E5 tau3 banking_knowledge; E6 WebArena; E8 LAB-Bench; E9 ML-Dev-Bench; E10 AgentDojo benign; E13 BALROG; E14 Omni-MATH
- E3/E7/E12 have proposed-only overlay replacements; these are not fabricated completed scopes in the figure
- E6 original missing runtime/verifier prerequisites and E13 missing held-out game-family/seed/verifier contract are detailed in `verification/empirical-review-findings.md`. Public project existence does not close those historical contracts
- Table 3 has all 14 original rows plus the 3 scored primary replacement rows. Table 4 has 6 later replacements, with separately dated states. Figure 1 represents all 23 reported O/R scope identities and never substitutes R state for O state
- Historical six launch/quota blocks were followed by retained 27 September execution records. They are not presented as current quota blockers or publishable model findings

## Historical scores, diagnostics and corrections

Unless a direct retained file is named, the support is narrative in `ch09_results_campaign.md` sections 8.2–8.6 and `ch_back_campaign_detail.md` D.3–D.5, not an independent recount of absent raw payloads.

| Scope/claim | Evidence and interpretation |
|---|---|
| E1 public731, 2 resolved | Historical ledger check plus narrative;2/731=.002735978112. Public731 is not entire SWE-Bench Pro dataset |
| E1 dispositions |713 native evaluations +14 generation failures +4 lost artifacts=731; 593 corrupt patches +120 parseable=713; 476 managed +255 vLLM generations=731. Both successes among120 parseable. Full candidates unavailable here |
| E1 timeout | Direct earlier `outputs/e1_swe_bench_pro/lane_receipt_2026-08-09.json` is harness_validation, not model score. Narrative D.5 says full scoring used native linux/amd64, 731official scripts unmodified.1200vs120 belongs only to earlier one-task arm64-QEMU gold check. Prior manuscript/source-map conflation corrected |
| E1 residual gap | CPU model/container-runtime version/full log breakdown of timeout/assertion/compilation failures unavailable. Patch counts cannot substitute |
| E2 |1/17,0.8628 replay-normalized partial. Official v1 has17; v2's34 is not substituted |
| E4 |1/100,0.3115 recovery metric; not37/128 or suite accuracy. Separate100 trial base zero is secondary-record-only confounded serving diagnostic |
| E5 |7/480 mean.079365; attempted-prefix.050505 is different denominator |
| E7 |1/46 attempted, no native grade; historical28/current33 discrepancy remains. Current public count is not substituted |
| E8 |450/1967=.2287747839; 1259 unparsed, 1086 length finishes.1259diagnostic membership not reconstructed;450/708=.6355932203 is conditional sensitivity. Joint category table/nontruncated-only accuracy unresolved |
| E9 |41 grade rows over40unique of75; 53.3333% coverage; suite score null. Separate merged-vLLM arm retained separately |
| E10 |97completed, 88utility passes,88/97=.9072164948;v1.2.2 no upstream SHA; training inventory absent; benign-only scope |
| E11 |67/156+62/156=129/312=.4134615385; 156shared designs; 150extraction failures;129/162=.7962962963 conditional; 33other nonpassing extracted candidates |
| E11 uncertainty |`platform_hybrid/experiments/scripts/e11_design_cluster_bootstrap.json` is present (prior source map said absent incorrectly).156 pair outcomes,54 both-pass/81 both-fail/13 code-only/8 spec-only support stored [34.3%,48.7%]. No bootstrap run during this revision |
| E11 diagnostic limit |312622/312≈1002 tokens does not establish cap-hit count. Response text/finish reasons/per-design extraction indicators absent; reasoning-draft extraction possible. No causal category allocation or conditional CI |
| E11 reference sensitivity |129/311=.4147909968 after excluding one reference-test mismatch is non-canonical only |
| E14 |2271/4426=.5131043832 native accepted;2271/4428=.5128726287 all dispositions. Two judge outputs omit equivalence verdict. Historical inconsistent equation retained unchanged in original record |
| E14 cap association |Reported1029/2982=.3450704 capped versus 1242/1446=.8589212 uncapped; difficulty confounds causal budget inference |

### Eleven historical checks

`outputs/verification/LEDGER_CODE_CHECK_2026-09-19.json`, SHA256 `7193cc30dfcdaacfaca736e246a971f83c1f7e06419e99337d8383e6bff3b3a2`, records11 PASS/0 FAIL. Its actual key is `strict_completed_suites_are_E8_E10_E11`; it does NOT call all three replacements. Correct interpretation: E8-R/E10-R strict replacements; E11-O strict original; E14-R terminal-complete. No historical receipt was rewritten. The new editorial regression separately checks those classifications.

## Later replacements

All6 retained `outputs/finish_pending_2026-09-27/<lane>/result.json` files support:
E1attempt190/grade57/pass1; E2 45/45/27; E5 97/72/8; E6 812/704/90; E9 34/34/10; E13 255/255,26.1% mean progression.

E1's earlier110 tasks/35 grades/4 passes stay unpooled because context65536 versus 32768 and collector changed. E6's108ungraded imply logical bound90/812 to198/812. E5's8/72 conditional=.1111 is not8/97. ML-Dev-Bench's34 are campaign configurations at pinned upstream SHA, whereas publication describes30 tasks; verified inventory in `verification/mldev-pinned-config-inventory.json`. E13 later 6 environments differs from pilot5. Lead-issued override is not independent approval. Later protocol departures remain visible.

## Paired pilot and protocol contrast

Every lane's direct `outputs/e1_e14_small_scale_2026-09-26/E*/paired.json` was read; manuscript Table 5 reports all 17 primary outcome contrasts. Companion `paired-outcomes-inventory.json` indexes source fields. Binary exact fractions are printed, avoiding earlier ambiguous .312 rounding. E4 is fractional mean reward. E12's151 rubric items cluster in6 apps, base-model judge, cluster sign-flip p = .25. E13's26 episodes/5 environments yield25.5062 vs23.6310%, unrounded difference 1.8752 pp and retained interval [−2.727,7.091]. No model/judge rerun occurred.

E2 separately records a nominal nonzero conditional runtime interval on21jointly passed items; therefore the manuscript only says17 primary outcome contrasts inconclusive, not every metric in every file. Timing is noisy and selection-conditional.

The reported E11 original-protocol versus later-trained-vLLM comparison23/50→33/50, discordances11/1,p≈.006, is a historical narrative comparison. The trained/base pilot is33/50versus33/50. Engine, thinking, temperature and cap all change in the former; it is not controlled sampler divergence, a full 312 rerun, or a corrected headline score.

## Inference, citations and build

`verification/inference-configuration-inventory.json` and `.csv` enumerate historical attempted/scored and paired scopes; null = not recovered, never a default. Primary settings are retained narratives; later result fields and narrative shared-context claims are labeled separately. Current source-code defaults do not authenticate missing historical settings.

`verification/citation-review-2026-10-18.md` verifies all original/replacement/pilot identities, exact model naming, official count/version caveats and Tinker live/archive URLs. `refs.bib` uses consistent first3+et-al display for lists longer than6; `verification/full-reference-metadata.bib` retains complete available verified metadata. All references in main bibliography are cited; publisher/arXiv/project entries are distinguished. Human reference review remains required by venue policy.

`verify_manuscript.py` checks actual compiled pages, citations, fonts, tables, scopes, arithmetic and unchanged historical ledger hash. Its quality report does not certify scientific truth, similarity or publication-policy compliance. All pages must also be visually inspected after the final build.

## Publication-review statistical clarification

E12’s exact two-sided test uses equal-app pass-rate differences, with 16/64 equally likely sign assignments at least as extreme (p = .25). A raw-pass-count test instead gives .50; it is a different statistic, not a two-sided correction to the rate test. Per-app counts, denominators, source paths and both exact calculations are retained in verification/e12-signflip-arithmetic.json; verify_manuscript.py recomputes the rate test using rational arithmetic. No original result receipt was edited.
