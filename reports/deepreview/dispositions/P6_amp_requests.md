# P6 (registry + submission): edit requests for the Amp session that owns P6

From the fix-P6 Deep Review pass, 2026-10-18. Base commit 75a500589. No P6 source was changed on this branch.
Every number below was recomputed by a script on this branch:

- `tools/review_ledger/P6_registry_n2_recompute.py` → `platform_hybrid/experiments/results/review_2026-10-18/P6_registry/n2_descriptive_deltas.tsv`, `n2_truncation_share.json`
- `tools/review_ledger/P6_registry_numbers.py` → `.../P6_registry/p6_registry_numbers.json`, `p6_registry_generated_proposed.tex` (a drop-in replacement for `sections/p6_registry_generated.tex` with the zvf130 fixtures excluded; it adds the macros `\NSTACKALL`, `\NFIXTURE`, `\NCKPT`, `\KLFULL`/`\KLPARTIAL`/`\KLUNREP`, `\NCIMETHOD`, `\LEAFCOUNTS`, `\NLEAVES`, `\BADGEA..\BADGEALL` and an Exec. column in `\PSIXENTRYROWS`, so the table header needs one more column)
- `tools/review_ledger/P6_submission_matched_truncation.py` → `.../P6_submission/matched_truncation.json`

Both builds currently compile with 0 errors. `paper_P6_registry` has two undefined refs, `sec:benchmark` and `sec:setup`, left behind when `_shared_methods` was dropped in 75a500589 (see R3).

---

## R0. §5.2 "below base" / matched TRL-vs-Tinker rerun: truncation verdict

**Verdict: truncation is demonstrable from existing artifacts.** The below-base held-out result cannot be read as training degrading the model or as backend equivalence. Whether training actually hurts held-out accuracy can only be settled by re-evaluating with a larger generation cap (needs compute).

Evidence (`platform_hybrid/experiments/results/publication_fixes/matched_trl_tinker/eval_heldout.json`, summarised in `matched_truncation.json`):
- Decoding was greedy with `max_new_tokens=512` on the first 200 GSM8K test items. 97.5–99.5% of generations hit the 512-token cap, for the base checkpoint and for every adapter: base 0.985, TRL 0.985/0.985/0.980, Tinker 0.985/0.995/0.975. Mean generated length is 511.4–511.95 tokens.
- Regex accuracy on the truncated text: base 0.345. TRL seeds 0.310/0.285/0.270, Tinker seeds 0.280/0.285/0.335. **All six trained adapters score below base**, by −0.010 to −0.075. Boxed-only accuracy is 0.105–0.135 throughout, so most "correct" answers come from the last-number fallback applied to cut-off reasoning.
- Training rollouts were also truncation-dominated. Tinker `trunc_by_seed` is 0.994/0.983/0.983; TRL truncation was not logged (NaN).
- Seed-level gaps (n=3, df=2). Last-10 train reward, Tinker−TRL: +0.050, t-CI [−0.488, +0.588]. Held-out: +0.012, t-CI [−0.109, +0.132]. Neither establishes equivalence, and no TOST margin was pre-stated.

The current submission source already withdraws the result (`p6s_worked_examples.tex`: "No historical twelve-stack denominator or matched-backend equivalence result is retained"). That is the correct state. If the rerun is mentioned at all, use:

> A matched TRL-vs-Tinker rerun (Qwen3-8B, three seeds, 30 steps; `experiments/results/publication_fixes/matched_trl_tinker/`) is not usable as validation: 97.5–99.5\% of held-out generations, base and trained alike, hit the 512-token cap, and every trained adapter scores below the base checkpoint (0.270–0.335 vs.\ 0.345 regex accuracy). The evaluation measures truncation and parser fallback, not backend equivalence; settling it requires re-evaluation with a larger generation budget.

Never reinstate "the 17× nominal gap was entirely an artifact … disappears into noise".

## R1. zvf130 fixtures: rule 2, unsourced, remove from the catalog
The 11 `registry/entries/zvf130_*.json` stack records were written from `experiments/results/zvf_iter130_method_risk.tsv`. Its variance-mitigation rows are a declared simulation projection, and its tool-use rows are anchors fixed by `platform_modal/scripts/zvf_diagnostic_iter130.py`. The thesis ch06 says so explicitly, and the current P6 text already calls them a "synthetic risk-index fixture". No training run stands behind them.

The abstract, intro and conclusion still say "28 seed stack entries derived from … real artifacts … and a five-seed risk-index batch". Proposed changes:
- abstract (ii)/(iii) and intro contributions: "28 seed stack entries derived from this benchmark's released config dumps, an open Colab trainer audit, a managed-runtime head-to-head, a same-stack four-method run, and a five-seed risk-index batch" → "17 seed stack entries derived from this benchmark's released config dumps (including two dry-run placeholders), an open Colab trainer audit, a managed-runtime head-to-head, a same-stack four-method run, and two W&B PPO/REINFORCE runs".
- p6_population §Family E / p6s_population: delete the family. Replace with: "The released directory also holds 11 `zvf130_*` records built from a simulation-projection risk-index fixture; they describe no executed training run and are excluded from every count, table and `stackdiff` statistic in this paper."
- Replace `sections/p6_registry_generated.tex` with `p6_registry_generated_proposed.tex`. Recomputed values on the 17 records: 136 pairs; R0 6, R1 0, R2 1, R3 1, R4 23, R5 105; 136/136 underspecified; median coverage 0.20; 12 generated / 5 hand-entered; same-checkpoint 33 pairs (R0 6, R2 1, R3 1, R4 23, R5 2); identical-signature class {openrlhf, verl} → 16 entries, 120 pairs (R0 4, R2 1, R3 1, R4 23, R5 91); same-checkpoint after collapse 30 pairs (R0 4, R2 1, R3 1, R4 23, R5 1); generated-only 66 pairs (R0 1, R3 1, R4 11, R5 53); backend pair coverage 0.48; DAPO pair 0.73. Badge ranges: A 51–78, B 84, C 78, D 58–59, W&B 40, all 40–84.
- p6_query_auditor: "R0: 42 … all 42 R0 pairs"; "16 of the 28 stack entries are still hand-entered"; "378/378"; "median coverage 0.14" → use the macros above. "the entries span eight base models" → "the 17 entries span five normalized checkpoint ids (Qwen2.5-0.5B-Instruct, Qwen3.5-4B, Qwen3-8B, Qwen3-8B-Base, Llama-3.1-8B-Instruct)". The *current* 28-entry corpus has 7 normalized strings, not eight: `llama-3.1-8b-instruct` and `meta-llama/llama-3.1-8b-instruct` fail to normalize to one id, and `qwen3-32b` adds the seventh.
- p6_conclusion "28 seed stack entries derived from real artifacts" → "\NSTACK{} seed stack entries derived from executed runs and released configs".
- Shared request (registry artifact, used by P5 too): move `zvf130_*.json` to `registry/excluded_unsourced/` with a README. Coordinator decision.

## R2. 17× exhibit wording (abstract, intro, synthesis)
Source: `experiments/results/framework_comparison.json` gives Tinker last10 0.85625 on `Qwen/Qwen3-8B-Base` and TRL 0.050 on `Qwen/Qwen3-8B`. `master_results.json` `wave6_w6_base_r32_t0.8_b2`: Qwen3-8B (Instruct) on Tinker, last10 0.3375. Checkpoint share = (0.85625−0.3375)/(0.85625−0.05) = 0.643.
- abstract: "64% of that gap is an unreported switch from Qwen3-8B (Instruct) to the Base checkpoint" → "the 85.6% run trained Qwen3-8B-Base on Tinker and the 5.0% run the Instruct Qwen3-8B on TRL; an Instruct run on the same Tinker harness reaches 33.8%, so the checkpoint accounts for 64% of the gap".
- intro: "They used different checkpoints (Base vs. Instruct; the checkpoint alone explains 64% of the gap on a same-harness rerun)" → "The Tinker run trained Qwen3-8B-Base and the TRL run the Instruct Qwen3-8B; Qwen3-8B (Instruct) on the same Tinker harness reaches 33.8% (W&B run 01848b9b, temperature 0.8), so the checkpoint alone accounts for 0.519 of the 0.806 gap (64%)". Both runs used lr 1e-5 per `framework_comparison.json`.
- intro: "0.567 → 0.578 (closed)" → "(managed)"; "a closed-stack arm labelled" → "a managed-runtime arm labelled".

## R3. Counts, dangling refs, internal notes
- p6_results_intro: "28 seed stack records and 20 variant-delta records (including two amendment sidecars)" → "\NSTACK{} artifact-backed stack records (plus \NFIXTURE{} excluded fixtures), 18 variant-delta records, and two amendment sidecars (not variant-delta records)". Remove both "Section~\ref{sec:benchmark}" and "Section~\ref{sec:setup}" (undefined): "(Section ??)" → "", and "the benchmark infrastructure of Section ??" → "the benchmark infrastructure described in companion work".
- p6_schema §Variant-Delta Records: "We ship three, each verified against its source paper rather than against folklore:" → "The registry ships 18 variant-delta records; three of them, described here, are verified component by component against their source papers:". Verified on disk: 28 stack, 18 variant_delta and 2 sidecar files.
- p6_appendix_artifact Listing: after the 3 delta files add the line `    ...  (17 stack + 11 excluded zvf130_* fixtures; 18 delta_*.json; 2 *.amendment.json)`.
- Remove the internal notes: abstract "Submission target is a position-artifact or datasets/resources track; living-catalog growth is post-acceptance community work, not a gate on releasing the schema and seeds."; the "Scope. … former condensed catalog statement is retired …" box; p6_related_resources "Companion catalog statement" paragraph; p6_conclusion "Vehicle packaging" paragraph. Keep the "August 2026 evidence boundary" sentence but move it into the Introduction.

## R4. Listing 1 and Family B/C prose are stale against the entries
Actual `variant_deltas_applied`, from the generated entries:
- tinker_dapo: clip_higher **absent** ("advantage clip [-2,3]; no ratio exists to clip"), dynamic_sampling absent, token_level_loss **absent** (sequence-sum loss), overlong absent, kl_removed **implemented**.
- colab-open_dapo: clip_higher implemented, dynamic_sampling **surrogate** (≤2 resamples, batch not refilled), token_level_loss **absent**, overlong absent, kl_removed implemented.
- tinker_drgrpo: both components **implemented**. tinker_gspo: both **absent**.

Changes:
- p6_schema Listing 1 and its lead-in: replace "clip_higher: surrogate, dynamic_sampling: absent, and token_level_loss: unknown (managed_by_tinker)" with the five statuses above for tinker_dapo, and regenerate the listing from `registry/entries/tinker_dapo_qwen3.5-4b_gsm8k.json`. Keep "one component out of five", which is still true (kl_removed).
- Family B: "DAPO (clip-higher, dynamic sampling, token-level loss, and KL removal implemented; overlong shaping recorded as absent …)" → "DAPO (clip-higher and KL removal implemented; dynamic sampling a surrogate — at most two resamples per dead group, batch not refilled; token-level loss and overlong shaping absent)".
- Family C: "every claimed delta on the closed stack is surrogate or unknown, never fully implemented" → "on the managed stack Dr. GRPO's two removals are implemented in the client-side loss, the 'dapo' arm implements only KL removal, and the 'gspo' arm implements neither sequence-level component".

## R5. N2 (Family D, Table, §7–8, synthesis): seed-level rigor and what actually executed
N2 is one seed per method, 40 steps. Step-bootstrap CIs and † marks treat steps as replicates (rule 4). Replace the Table with descriptive point deltas (variant − grpo) from `n2_descriptive_deltas.tsv`:

| metric | aero last10 | gift last10 | areal last10 | aero full40 | gift full40 | areal full40 |
|---|---|---|---|---|---|---|
| reward_mean | −0.014 | +0.016 | −0.020 | −0.007 | +0.011 | −0.006 |
| zvf | −0.025 | +0.125 | −0.056 | 0.000 | +0.050 | −0.014 |
| mean_len | +31.08 | −3.89 | +30.13 | +14.16 | −0.74 | +14.60 |
| cv_len | −0.034 | +0.002 | −0.038 | −0.013 | −0.001 | −0.018 |
| loss | +56.6 | −1.67×10⁴ (var. mean −1.65×10⁴) | +116.2 | −46.2 | −1.79×10⁴ | −6.3 |

The old per-prompt rows (−0.007/+0.011/−0.006) are not reproduced by either window. Drop them.

What the N2 arms executed (`experiments/tinker-runs/scripts/n2_reward_tensor_20260704.py` + `experiments/variance_mitigation_integration.py`):
- **aero** is an adaptive-G controller with `g_max=min(16,G)=8`. `n2_metrics.tsv` logs group_size 8 at all 40 steps, so the arm ran the GRPO update rule. Its deltas are a run-to-run noise floor between two identical update rules.
- **areal** is GRPO with sampler weights refreshed every 4 steps (a staleness surrogate).
- **gift** is a "simplified" blend, adv' = 0.75·a + baseline/2, a constant shift: a surrogate of the cited method.

Text changes:
- p6_population Family D: "so the per-method delta against GRPO is guaranteed to isolate the label, not the stack. Measured last-10 reward/ZVF land in the same paired-bootstrap envelope as GRPO (Section 7)" → "One seed per method, so the per-method deltas are descriptive. The aero arm's adaptive-G controller was capped at G=8 and never changed G, so aero−GRPO differences (reward −0.014, length +31 tokens over the last 10 steps) measure run-to-run variation between identical update rules, and the areal (+30 tokens) and gift deltas are of the same order."
- Table caption: delete "AERO and AREAL show small but statistically significant reward decreases" and every †.
- p6_measured_claimed: delete "its step-bootstrap interval [−0.0625, +0.0125] includes zero".
- p6s_synthesis "two of the shifts are small but statistically detectable decreases—while completion length moves more; a raw-data recompute reproduces the registry's measured-delta block" → "differences that, for the aero arm, are run-to-run variation between identical update rules; our recompute (tools/review_ledger/P6_registry_n2_recompute.py) reproduces the last-10 point deltas".
- Registry data request: tinker_aero `advantage_guided_evolution: implemented` → `absent`. Change tinker_areal and tinker_gift to `surrogate`. The governance gate already requires open evidence for `implemented`.

## R6. Smaller text fixes
- ci_method paragraph: "populated on the 7 entries derived from the N2 same-stack four-method run" → "populated on 7 entries: the four Family C head-to-head records and the three Family D N2 records. On the Family C records the declared source (the N2 per-step bootstrap) does not match their head-to-head provenance and must be corrected". Registry data request for the same.
- Badge leaf counts (Eq. 1): add "Leaf denominators per item are 10, 3, 4, 2, 3, 2, 2 (26 leaves; `clip_eps_*` drop out when `clipping_applied=false`)", from `registry/schema.json` via `query.item_score`.
- Governance "Schema evolution": add "Adding a leaf inside `min_report` changes the badge denominator, so it is a badge-version change even when schema-additive; badges are only comparable within one badge version". Add "`expected_effects`/`measured[]` blocks on variant-delta records are evidence annotations; updating them does not create a new record id, but each row must name its source artifact".
- Delta field namespace (`sampling.dynamic_sampling`, `reward.overlong_shaping` in Listing delta_dapo): the shipped `delta_dapo.json` notes say the fields are "now anchored on iter-41 schema leaves", but the paper listing still prints the old prefixes. Regenerate the listing from the file.
- CLI flags in prose: `query –item`, `stackdiff –all`, `–exit-level` → `\texttt{-{}-item}`, `\texttt{-{}-all}`, `\texttt{-{}-exit-level}`.
- Delete "the manifest-emitting trainer plugin and LeverTrace check remain specifications" (both undefined).
- registry Listing 4 / "Seven of twelve stacks fully report KL handling" → on the 17 records the census is FULL 11, PARTIAL 5, UNREPORTED 1. Either print all 17 or say the listing is the 12 generated entries.
- Registry Listing 3 note: "The `loss.objective`/`loss.clip` strings come from the generated `run_config` (cell_runner.py@b0ac85bfb), not from the hand-entered note".
- Submission: "(full per-entry listing in Table 2, Appendix B) (regenerated with …)" → one parenthetical. Move the "B Full Seed-Entry Table" heading above the table (`\begin{table}[h]` after the heading, or a lead sentence).
- Entry table: add an Exec. column (run / toy run / dry-run, with seed count). This is already in `\PSIXENTRYROWS` of the proposed tex.
- `-Base` vs `-base` model strings come from the entries. Drops out with R1.

## R7. Bibliography (references.bib is shared; coordinator)
Verified via arXiv search 2026-10-18. First author and ID checked; copy the full author list from the abs page.
- MO-GRPO: currently "Shiki Ichihara et al., Automatic variance-based reward reweighting…, arXiv:2510.17711". Should be **Yuki Ichihara et al., "MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems", arXiv:2509.22047**.
- GRPO-LEAD: currently "Yingqian Lu et al., Length-aware reward shaping for GRPO". Should be **Jixiao Zhang et al., "GRPO-LEAD: A Difficulty-Aware Reinforcement Learning Approach for Concise Mathematical Reasoning in Language Models", arXiv:2504.09696**.
- GIFT, arXiv:2510.23868 (Z. Wang): the title and ID are correct.
- AERO (new): **Z. Zhang et al., "Train Less, Learn More: Adaptive Efficient Rollout Optimization for Group-Based Reinforcement Learning", arXiv:2602.14338**. Registry `delta_aero.json` cites `le2025rlzvp` (RL-ZVP, arXiv:2509.21880), which is the wrong paper; it should cite 2602.14338.
- AReaL (arXiv:2505.24298) and ES (arXiv:2509.24372) are cited by the registry entries. Add them to the paper if aero/areal/es stay in the text.
- Minor: Limozin "E. Durech" → full first name from the arXiv page; "Tricks or traps? a deep dive" → "{A} deep dive"; Le et al. venue string → "ICLR 2026".
