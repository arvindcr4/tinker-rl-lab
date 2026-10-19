# ARS corrections, round 2: paper 05 (Reward Contrast compendium) status

Root: `platform_hybrid/paper/paper.tex`. Review: `reports/ars_review_2026-10-19/05_reward_contrast_compendium.md` (Revision Roadmap) and section 5 of `00_portfolio_editorial_synthesis.md`. Worktree branch `ars/p05`. Muse's partial pass is commit `0052ee7db`; the shared `references.bib` was restored in `7eccfea05` and is untouched here.

## Build

- `cd platform_hybrid/paper && tectonic -X compile --outdir /tmp/ars05 paper.tex` builds with exit 0: **23 pages** (main text pp. 1–13, ethics 13–17, references 18–19, appendix 20–21, checklist 22–23). 0 undefined references, 0 undefined citations, 0 overfull boxes, 42 bibliography entries printed. PDF sha256 `63b2e19a…5b47`.
- The anonymized mode (a scratch wrapper that defines `\paperauditsubmission`, then `\input{paper}`) builds at 22 pp. with "Anonymous Author(s)". The Hub handle `arvindcr4` (4 occurrences) and "PES University" (acknowledgments) still appear in the text.
- No `\nocite` anywhere in the inclusion closure, so only cited entries print. The review's "108 uncited refs" counts entries in the shared `.bib`, not entries in the compiled list. Nothing was deleted from `references.bib`.
- Before this pass the root did not build: `tikz/*.pdf` and `figures/v2/kl_proxy.pdf` are gitignored build products. It now uses only tracked figures (`figures/wave6_sensitivity.png`). The taxonomy, pipeline and reward-flow diagrams, the KL-proxy figure, the ZVF heatmap and the TRL-seeds figure were removed along with the displays they belonged to (see below).
- Gates: `python3 platform_hybrid/paper/scripts/p05_ledger_window_gate.py` reports PASS (28 rows recomputed from artifacts). `python3 platform_hybrid/paper/scripts/reviewer_36320_corpus_check.py`, DESC section:
  - Before this pass it had 2 errors ("derived Qwen PPO comparison statistic"; "unsafe local 82.0-to-83.3 statement"). Both are now gone.
  - One error remains: "live Qwen PPO numeric comparison". It is a false positive. The pattern `qwen3-8b.{0,300}35\.0` matches the audited-runner Qwen3-8B G=2 row (tail 35.0%), not a PPO value. I did not edit the shared checker.
  - The P1–P12 failures in that script predate this pass and are unchanged.

### Other roots rebuilt (they share the files Muse touched)

| Root | Result |
|---|---|
| `ethics_wrapper.tex` (consumes `ethics_statement.tex`) | builds, 7 pp.; text is byte-identical to the pre-Muse (`0cc717ea2`) build |
| `paper_P2_zvf.tex` | builds, 28 pp. |
| `neurips_2026_variants/paper_P8_workshop.tex` | builds, 12 pp. |
| `archive/absorbed/U01_main_compendium/main.tex` (consumes the shared abstract/intro/ethics) | fails with `\DeclareUnicodeCharacter` under XeTeX. The same failure occurs at `0cc717ea2`, so it predates this pass. |

## Verification of Muse's edits (0052ee7db)

**Consumers of the files Muse edited:**
- `sections/abstract.tex`, `sections/intro.tex` and `ethics_statement.tex` are also used by the archived U01 roots, and `ethics_statement.tex` by `ethics_wrapper.tex`. All three were **reverted to `0cc717ea2`**. Muse's content now lives in paper-05-only forks: `sections/p05_abstract.tex`, `sections/p05_intro.tex`, `ethics_statement_p05.tex`.
- `sections/diagnostic_method.tex` and `sections/evidence_identity_audit.tex` are used only by `paper.tex`, so they were edited in place.
- No P1–P12 root inputs any of these files.

**Muse's numbers and labels:**
- Correct and kept:
  - Prompt-paired held-out counts (15/13, 16/15, 16/19, 16/12, 16/7). Recomputed exactly from `platform_tinker/reports/final/gsm8k_heldout_seed*.json` against `gsm8k_base_control_200.json`, with identical prompt order.
  - Qwen3-235B "20 planned / 15 completed / 4 logged" (`frontier_gsm8k_qwen3-235b.json`).
  - Tool-cell run IDs.
  - The GU rename.
  - The Wave-6 softening.
  - Triage-language removal.
- Incomplete:
  - Window relabels missed the 3–5-point partial rows (Qwen3.5-27B, Qwen3-32B, both Qwen3-30B-A3B).
  - The campaign-v2 "26 completed / 6 logged" rows are snapshot records. A complete run-ID-bearing file supersedes them (see item 14).
- Not acceptable under one standard: Muse kept untraced percentages (`tab:task_grpo`, SI/PTD, the lagged table) as "demoted historical record". They are now withdrawn (item 15).

## Must-fix items (Revision Roadmap)

| # | Item | Status | What / where |
|---|---|---|---|
| 1 | DA-CRITICAL-1 retitle | **done** | New title at `paper.tex:49–50`: "An Evidence Audit and Correction Record for Critic-Free Group-Relative RL Runs: Homogeneous-Reward Groups, Window Labels, and Artifact Identity". The abstract (`sections/p05_abstract.tex:21–22`) and the claims list (`paper.tex:527+`) state that the reward-contrast thesis is not tested. |
| 2 | R3-W4 cold-reader framing / anonymized mode | **done (text); author-owed (anonymity)** | #36320 genealogy moved out of the abstract into `sections/p05_overlap_map.tex:5–8`. Withdrawal narration consolidated into one "Withdrawn Displays" list (`paper.tex:456+`) and the Archival Correction Log appendix (`paper.tex:679+`). Anonymized mode builds (see Build). An anonymized artifact mirror and certified anonymity are for the authors. |
| 3 | EIC-W3 overlap disclosure | **done** | `sections/p05_overlap_map.tex`, Table `tab:overlap`. Each display is mapped to the sibling roots and thesis chapters that contain the same values (string scan of each P1–P12 inclusion closure, portfolio 01–04, and the thesis chapters); unique content is marked. Submission status of the siblings is declared author-owed in the text. |
| 4 | DA-MAJOR-3 overlap map | **done** | Same table. It also states that no display counts as independent of the same rows elsewhere. |
| 5 | EIC-W1 retarget / contribution scale | **author-owed** | Choosing the venue and its page budget is the authors' decision. The text is scoped as an audit/correction note (`paper.tex:5–8` header, abstract, intro contributions `sections/p05_intro.tex:14–26`). A validated prospective result would need new experiments. |
| 6 | DA-CRITICAL-2 remove validated-triage/predictor language | **done** | Every "triage", "predict" and "stopping" use left in the PDF is a negation (checked in the compiled text). ZVF is described as a reporting field: `paper.tex:406+` and the claims list. The lagged-correlation "prediction" framing was removed. |
| 8 | R2-W1 reposition vs ACR / dynamic sampling | **done** | `sections/p05_intro.tex:28–40`. New paper-05 related work, `sections/p05_related_work.tex:8–28`, covering ACR, Gradient Starvation, DAPO, VCRL, GRESO, CPPO, NGRPO, recycling and AERO; it claims no new measurable. |
| 10 | R1-W4 actual treatment in table bodies | **done** | The ledger column "Stack" reads "Audited runner". The caption spells out prompt-token loss, shifted targets, and no ratio/KL/mask (`paper.tex:202–215`). `tab:tasks` was relabelled. `sections/diagnostic_method.tex:16–22, 66–71` adds that the candidate runner for the reviewed cells uses the same loss form. |
| 11 | R3-W2 immutable manifest | **done (partial binding); not-possible (rest)** | `platform_hybrid/paper/manifests/p05_reviewed_cells_manifest.json`, built by `scripts/p05_build_reviewed_cells_manifest.py`, contains: result-file SHA-256 and git blobs, run IDs, `tinker://` URIs, config, Hub repos at fixed revisions with byte-identical `results.json`, step-level recomputation (30/30 steps reward 0, ZVF 1), candidate runner `c68110a2:experiments/tinker-runs/scripts/tinker_parallel_runner.py` (blob `c5f3a67e`), and the prompt-set, system-prompt and reward-function hashes. Text: `sections/evidence_identity_audit.tex:3–68`. **Not possible:** raw per-group rewards and completions were never stored, the execution-commit binding is unprovable, and no evaluator exists. These are recorded as `UNBOUND`. |
| 12 | DA-MAJOR-2 exact refs instead of wildcards | **done** | `arvindcr4/tinker-rl-bench-*` replaced by 12 exact repos with revisions in `platform_hybrid/paper/manifests/p05_hub_repositories.json`, cited at `ethics_statement_p05.tex:26–31`. Correction: 10 of the 12 hold only result metadata; only the two Modal PPO repos hold LoRA adapters. The earlier "LoRA adapters published" claim is fixed. |
| 13 | EIC-W2 single ledger / restructure / PDF | **done (structure, PDF); author-owed (page budget)** | The duplicate regroupings `tab:cross_arch_scaling`, `tab:dense_moe`, `tab:frontier` and `tab:ppo_grpo` are deleted. One ledger remains, `tab:main_results`, with prose observations at `paper.tex:386+`. Also removed: config inventories, artifact-scope table, BrowserGym table, compute table, and the 427-line ethics (trimmed). The PDF compiles at 23 pp. Meeting a venue's page budget depends on the venue choice (item 5). |
| 14 | R1-W1 window labels + generation gate | **done** | Every ledger row now shows planned/completed/logged and a window ("last 10" or "all n"). The gate is `scripts/p05_ledger_window_gate.py`: 28 rows are recomputed from artifacts, and it fails if any row with fewer than 10 points is labelled last-10. The audit also found that the campaign-v2 snapshot rows (master, no run ID, 6 points) are superseded by the complete 30-step run-ID records in `tinker-runs/results/campaign_v2_20260419_111623.json`. Snapshot values withdrawn (e.g. G=4 52.1%, now 28.7%; Kimi-K2-Thinking 95.8%, now 83.1%). The apparent G=4 maximum disappears. Qwen3-8B-Base has two executions (85.6% / 98.8%), both listed. The Nemotron "interrupted" dagger was wrong: the artifact has 20/20 steps and a final checkpoint. Notes at `paper.tex:267–291`. |
| 15 | DA-CRITICAL-3 one quarantine standard | **done** | Rule stated at `paper.tex:188–200`. Withdrawals listed at `paper.tex:456–490`: SI/PTD 28-run analysis, lagged table and heatmap, team percentages, schema percentages, two-phase percentages, snapshot values. Conflicts are in `tab:identity-conflicts` (`sections/evidence_identity_audit.tex:76–97`). |
| 16 | R2-W2 `tab:task_grpo` | **done** | Table removed, percentages withdrawn, "multiple seeds" claim deleted (`paper.tex` withdrawn list). |
| 18 | R1-W2 SI/PTD numbers | **done** | Withdrawn. Only the definitions are kept, in the appendix. The KL-proxy figure was removed. The `r=-0.517` in the discussion was deleted. |
| 19 | R1-W3 lagged table: bind 33 run IDs or remove | **done (removed); not-possible (bind)** | The output JSON has no run IDs, and the generator reconstructs 61 runs at HEAD and 58 at `ddcbb868`, so the 33-run cohort cannot be bound. The table and heatmap were removed. The G=2 counterexample is kept as the negative illustration (`paper.tex:445–452`). |
| 20 | DA-MAJOR-1 circular PTD/ZVF validation | **done** | PTD is stated to be definitional with the outcome. Both analyses are withdrawn (`paper.tex:462–469`). |
| 22 | R3-W1 operator decision or reporting-field framing | **done (framing); not-possible without new data (decision analysis)** | Signed ZVF is presented strictly as a reporting field. The so-what test is stated in the Discussion (`paper.tex:607+`). A retrospective early/late decision analysis needs held-out whole runs with positive/mixed/recovery cases, which the corpus lacks. |
| 24 | R2-W4 same-stack comparator | **author-owed (new experiment)** | The retitle (item 1) resolves the title dependency. The fixed-completions reward-intervention check is listed as not executed (`paper.tex` Limitations and Discussion). |

## Should-fix items also handled

| # | Item | Status | Note |
|---|---|---|---|
| 7 | DA-MINOR-1 GU rename | done | "Mixed-group fraction"; GU kept only as a historical alias. |
| 9 | R2-W3 related work / external effect sizes / uncited bib | done (paper-05 scope) | New 3-paragraph `sections/p05_related_work.tex` with no quoted external effect sizes. The shared `related_work_v2.tex` is unchanged. The printed bibliography is 42 entries. Shared `.bib` entries were not deleted (other papers cite them). |
| 17 | R1-W6 Wave-6 peak headlines | done | `paper.tex:356–384`. Muse's softening was verified against `wave6_ablations.json`. |
| 21 | DA-MINOR-2 BrowserGym / artifact-scope padding | done | Tables removed; one line remains in the appendix. |
| 23 | R1-W5 prompt-paired held-out + ethics uplift bound | done | `paper.tex:337–353`; `ethics_statement_p05.tex:43–50`. |
| 25 | R3-W3 compute/carbon ledgers | done | Spend, carbon and compute tables removed and no totals reported (`ethics_statement_p05.tex:105–138`). The historical entries remain only in the shared `ethics_statement.tex`. |

## New factual corrections found while binding (for siblings and authors)

- **Tool-cell confounders.** In the candidate source, any JSON-parseable output scores 0.3. Reward 0 at every step therefore means that no completion passed a strict whole-string `json.loads`. Both models received Qwen ChatML prompts, including Llama. Parser and template failure cannot be separated from policy failure because completions were not stored.
- **Tool corpus.** The ethics text's `data/synthetic_tools/` corpus does not exist. The actual prompts are 10 in-source requests over calculator/weather/search/reminder/translate.
- **SDK version conflict.** The paper text says `tinker==0.16.1`; the Hub model cards say v0.18.1.
- **Wildcard refs in siblings.** `tinker-rl-bench-*` wildcard references remain in P1–P3. They are not edited here because those roots are outside this paper's scope.

## Changed files

- `platform_hybrid/paper/paper.tex` (rewritten root)
- `sections/{p05_abstract,p05_intro,p05_overlap_map,p05_related_work,p05_checklist}.tex` (new, paper-05 only)
- `ethics_statement_p05.tex` (new fork)
- `sections/{diagnostic_method,evidence_identity_audit,audit_regression_cases}.tex` (paper-05-only consumers)
- `manifests/p05_{reviewed_cells_manifest,hub_repositories}.json`
- `scripts/p05_{ledger_window_gate,build_reviewed_cells_manifest}.py`
- Reverted to `0cc717ea2`: `sections/abstract.tex`, `sections/intro.tex`, `ethics_statement.tex`
