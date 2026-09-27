# Peer Review: M.Tech Thesis "Tinker RL Lab: A Multi-Framework Benchmark and Study of GRPO-Style RL Post-Training of LLMs"

Working draft, 2026-09-27. Written for the author's pre-defense self-review. It is AI-assisted: local reading of the compiled PDF (209 pp) plus targeted reads of the repository. Page numbers are the printed page numbers. The author must check every location before acting on it.

**Scope limits.** The Chapter 9 replacement-lane numbers for E1, E2, E5, E6, E9 and E13 (§9.7, Tables 9.A/9.B) are being rerun, so their values are not reviewed here. Only the text that depends on them is flagged (Major 4). The title-page degree, course code and period are correct as stated and are not reviewed.

**Checked against the repository.** Two claims were checked against files in the repo:
- `platform_hybrid/experiments/results/samestack_ppo_grpo.json`
- the two E11 receipts, `outputs/modal_e1_e14/2026-08-16/e11_full_receipt.json` and `outputs/e11_verilog_eval/e11_trained_step40_receipt.json`

Nothing was re-run.

---

# Comments to authors

## Summary

The thesis reports four things:
1. A dispatch harness meant to run one GRPO configuration across several RL frameworks.
2. Four controlled studies (P1–P4): a cross-scale null, the ZVF diagnostic, a group-size sweep, and a length-bias null.
3. A reporting standard, a registry and a controller audit (P5–P7), plus an off-topic fraud case study (P8).
4. A held-out evaluation of one Tinker-trained Qwen3.6-35B-A3B LoRA actor against 14 benchmark suites (E1–E14), run under receipt and terminal-state rules.

The main contribution is methodological: attribution discipline, receipts, and refusing to pool numbers. The empirical findings are mostly nulls.

## Assessment

Strengths:
- It reports honestly on its limits: it withdraws claims, keeps negatives, and never pools results.
- It is unusually careful about provenance.
- The reporting standard and registry (P5/P6) are a concrete, reusable artefact.

Weaknesses, as an examiner would see them:
1. The flagship nulls were measured on a saturated toy task.
2. The optimiser labelled "GRPO" is not described consistently.
3. The E1–E14 actor's training recipe is undocumented.
4. Numbers and lane statuses contradict each other across chapters.
5. The "multi-framework benchmark" in the title rests on two completed runs.

Most of these are fixable with text changes and one canonical-numbers table. Only Major 1 really needs new compute, and even there careful re-scoping is an acceptable alternative.

**Verdict.** Defensible, after substantial revision of framing and internal consistency. The attribution and reporting work will hold up at the defense. The empirical claims in the abstract and the Chapter 6 "central claim" will not hold up in their current wording.

## Major comments

### Major comment M1
- Location: Abstract p. iii; §1.2 p. 8; §4.1 Table 4.1 and p. 50; §6.2.3 p. 89; §6.4.1 pp. 96–97; §6.5.1 p. 100; §6.6 p. 105 ("A reader looking for the project's central empirical claim should take this one").
- Observation: The main de-confound results are all nulls on an arithmetic task that is already at ceiling:
  - The same-stack PPO vs GRPO contrast (paired Δ = −0.002, p = 0.374) comes from `samestack_ppo_grpo.json`. That file records Qwen2.5-0.5B, 40 steps, held-out accuracy 0.990 (GRPO) vs 0.992 (PPO), and training reward near 0.9 by step 4.
  - The P3 group-size sweep sits at 0.978–0.990.
  - The Dr. GRPO control sits at 0.987/0.992, with completions averaging 4.7 tokens.
  - The abstract and §6.6 present these as the thesis's central empirical finding ("algorithm label matters much less than sampling configuration") and never mention the ceiling.
- Evidence or criterion: When every arm scores about 0.99, a null cannot separate "no effect" from "no headroom". The thesis's own P3 abstract says so ("no detectable G effect under saturation", p. 96), but that caveat never reaches the PPO/GRPO headline. The one unsaturated regime (Qwen2.5-1.5B GSM8K-CoT, §6.5.4 p. 104) does show +5–6 pp pre-to-post gains, so headroom-sensitive contrasts were available.
- Why it matters: This is the first thing an examiner will ask about when a result slide shows "p = 0.374". As worded, the conclusion does not follow from the evidence.
- Requested action: Choose (a), or better, (a) plus (b).
  - (a) Re-scope everywhere. Say "on a saturated 0.5B arithmetic task (held-out ≈ 0.99), no difference was detectable". Add the held-out accuracies next to every p-value. Drop "central empirical claim" from §6.6, or make it conditional on saturation.
  - (b) Rerun PPO vs GRPO (and G ∈ {2, 8}) on the unsaturated 1.5B GSM8K-CoT configuration with 5 paired seeds. Report the difference, its CI, and an equivalence margin in accuracy points.

### Major comment M2
- Location: §1.1 p. 4 and Fig. 1.1 p. 3; §4.3 p. 54; §5.3 p. 73; §6.3.1 p. 90; Table 4.1 p. 50.
- Observation: The optimiser being studied is described in contradictory ways:
  - §1.1 and §5.3 define GRPO as a clipped, KL-regularised surrogate, and say a zero-variance group still gets "a pure pull toward the reference".
  - §4.3 says the runner actually used "omits the PPO ratio/clip, the frozen reference policy, and the completion-only token mask". With that runner, ZVF = 1 does entail a zero gradient.
  - §6.3.1 then says a zero-variance group "does not mean the total gradient is zero".
  - The reader cannot tell which objective each study (P1–P7, the Tinker runs, the Colab runs, the Modal sweeps) actually ran. Nor is it clear what "PPO" means in the same-stack contrast (critic? clipping? which runner?).
- Evidence or criterion: NFR-1 and P5 item 1 (loss form tuple) are the thesis's own standard. The thesis does not apply that standard to its own runs.
- Why it matters: The ZVF interpretation depends on the loss form, and so does the claim that GRPO and PPO are "the only difference". Without that information the same-stack contrast cannot be attributed.
- Requested action: Add one table in Chapter 4 (or Appendix A) with one row per study or run family. Columns: runner file, ratio/clip, KL/reference, token mask, advantage normalisation, critic (PPO arms), LoRA rank, lr, G, steps. Reconcile §1.1, §5.3 and §6.3.1 with that table. Where the runner is REINFORCE with a group baseline, say so and use that name.

### Major comment M3
- Location: §4.5 pp. 61–66; §9.1 pp. 129–130; Appendix A.4 pp. 175–176 ("The training step count behind that adapter survives in no artefact"); §4.5 p. 65 (decontamination cannot reconstruct "the exact 512 consumed training records").
- Observation: The campaign evaluates a "Tinker-trained actor" (pavlov-portfolio seed809 stepfinal). Nowhere does the thesis say how that actor was trained: algorithm, reward, data mix, G, steps, lr. The only hints are the training-envelope datasets named in the decontamination section (API-Bank-RLVR, SWE-Gym) and the `pavlov_portfolio` CLI preset (§5.3).
- Evidence or criterion: A thesis about GRPO post-training has to connect its main evaluation to the post-training it studies. Otherwise the fourteen scores are properties of an unspecified checkpoint.
- Why it matters: At the defense, "what did the GRPO step contribute to these scores?" is the obvious question. The only paired evidence is the new §9.7 Table 9.B, and it is still being rerun.
- Requested action: Add §9.1.1, "How the actor was trained". Fill it from whatever survives: W&B run `bsv8vx04` (already cited on p. 65), the `pavlov_portfolio` preset config, and `PAVLOV_EXPERIMENT_PROTOCOL_2026-08-09.md`. List each unknown field explicitly as unrecoverable. In §1.4 (O7) and §10.3, describe the campaign as an evaluation-governance contribution, not evidence about GRPO. Once §9.7 is final, make Table 9.B the campaign's only post-training delta.

### Major comment M4
- Location: Abstract pp. iii–iv; §9.3 p. 136 ("No lane here has a matched base-model run"); §9.8 p. 144 ("no baseline comparison of any kind"); §10.1 p. 148 ("no per-suite base-model comparator appears in the campaign ledger"); §10.4 p. 156 (future work: "Adding a per-suite base-model arm"); §9.7 caveat bullets p. 143 vs §9.1 p. 129.
- Observation: §9.7 adds a paired trained-vs-base comparison on the same vLLM engine (Table 9.B). At least five other passages still say no such comparison exists.
  - The §9.7 caveats call the seed809 adapter "lost" (E1, E2, E8, E10 bullets).
  - §9.1 says the adapter is preserved on an HF branch and was verified present on 26 September; only the Tinker sampler route is gone.
- Evidence or criterion: The thesis contradicts itself about which evidence exists.
- Why it matters: An examiner who reads the abstract and then §9.7 will see the contradiction immediately. It also undercuts the "no improvement claimed" stance, because Table 9.B is exactly the test that could support or refute an improvement.
- Requested action: Once the E1/E2/E5/E6/E9/E13 reruns finish, update the abstract, §9.3, §9.8, §10.1, §10.3 and §10.4. Say that one small-n paired comparison exists and that none of its differences excludes zero (if that still holds). Use "Tinker sampler route purged" rather than "adapter lost" throughout. Remove the future-work item that §9.7 has already done.

### Major comment M5
- Location: §9.5 pp. 138–139; §5.8 p. 80; §9.1 p. 130; §9.2 pp. 132–133.
- Observation: The E4 base-model zero is traced to the serving bridge sending no stop sequences. The thesis does not say which other trained-actor lane results used the same bridge path. Candidates include E10 AgentDojo 0.9072, E5, E7, E13 and E2.
- Evidence or criterion: Serving conditions also vary between lanes and within them:
  - E1's 731 generations mix 476 Tinker and 255 Modal-vLLM generations within one run (p. 130).
  - E11 had 150 extraction failures out of 312 (p. 133).
  - For E8, 1,259 of 1,967 responses were unparsed and 1,086 were length-truncated (p. 132).
  - E1 used a gold-resolve timeout of 1,200 s against the official 120 s (p. 135).
- Why it matters: If the bridge defect or truncation affected other lanes, their numbers measure the harness, not the model. §9.5 already applies this reasoning to E4. For E8 and E11, format and truncation failures dominate the scores.
- Requested action: Add a per-lane serving table:

  | Column | Content |
  |---|---|
  | Serving path | bridge / Tinker sampler / vLLM |
  | Stop sequences | set or not |
  | Chat template | thinking on or off |
  | Decoding | max tokens, temperature |
  | Failure rates | parse-failure rate, truncation rate |

  For each lane, state whether the stop-sequence defect could apply. Report E8 and E11 as "end-to-end harness scores". Next to each, give the score over parseable responses as a sensitivity figure, not the headline. List the E1 timeout as a non-native deviation in the result line itself.

### Major comment M6
- Location: §1.1 p. 4 and §1.2 p. 5 vs Fig. 1.1 p. 3; §4.3 pp. 55–57; §6.3 pp. 91–94; §10.1 p. 146; §10.3 p. 154; §10.2 p. 150; Appendix B.3 p. 182; Table A.2 p. 165; §1.2 p. 7 vs Fig. 1.2 p. 6 and §4.1 pp. 50–51.
- Observation: The same quantity carries different numbers in different places, with nothing to reconcile them.
  - **"Typical" ZVF.** 0.72–0.77 (Phase 1). That value is used as the headline in the §10.3 conclusion. The measured values are 0.16 (Qwen3-8B, Fig. 4.4), 0.481 (GRPO panel), 0.130/0.190/0.155 (P2 tensors), and 0.838→0.631 (sweep).
  - **ZVF decline.** 0.207 (p. 4) vs 0.214 (Fig. 1.1).
  - **Qwen3-8B GSM8K GRPO headline.**
    - Last-10 0.856 (Table A.2).
    - Last-10 34.4% / peak 62.5% (B.3).
    - "Tinker 99.9% vs TRL 73.4%" (p. 150).
    - Mean 0.285 (P1, p. 88).
  - **ZVF–outcome association.**
    - ρ ≈ 0.27.
    - r = −0.769 (N = 15).
    - r = +0.22 (n = 23).
    - Residualised r = +0.80.
    - r ≈ 0.09 (n = 12).
  - **Motivating example.** It says the contrast was run "on Qwen3-8B" and quotes d = −0.14 with a CI and power. Fig. 1.2 labels the same numbers Qwen2.5-0.5B, and §4.1 says the Qwen3-8B row is "not estimable".
- Evidence or criterion: A reader cannot tell which number is canonical. The conclusion reports a Phase-1 figure that the Semester-4 measurements do not reproduce.
- Why it matters: Examiners probe inconsistent numbers first, and each one costs defense time.
- Requested action: Add a one-page "Canonical numbers" table (in Chapter 1 or Chapter 10). Columns: quantity, value, model/task/G, n seeds, source file. Point every text mention to a row. In §10.3, say ZVF is regime-dependent (0.16–0.84 across the measured runs). Remove the non-estimable statistics from §1.2 and fix the model name.

### Major comment M7
- Location: Abstract p. iv; Fig. 4.6 p. 62; §4.5 p. 64; Fig. 5.3 p. 78; Fig. 9.1 p. 128; §9.1 p. 130; §9.2 p. 133; §9.3 p. 136; §9.6 p. 139; §10.2 p. 152; §10.4 p. 156.
- Observation: Lane scope and status labels contradict each other:
  - **VerilogEval (E11).** The abstract lists it both among the "three replacement scopes … strictly complete" and among the "two [that] carry full verified scores under their original contracts". Fig. 4.6 tags it "orig".
  - **E1.** Called a "full-suite result" (§9.3), yet Fig. 4.6 and Fig. 9.1 show E1 as launch-pending. That status belongs to the Multilingual replacement, but the figures do not say so.
  - **Fig. 5.3 is stale.** It is dated 2026-08-29 and reads "2 scored exact · 5 partial · 7 blocked externally". It lists E8, E10 and E14 as having no path.
  - **CLOSED_EXTERNAL counts.** §4.5 lists six lanes; Fig. 4.6 counts three. §10.4 says "seven" blocked, adding E13-original; §9.6 says six.
  - **§9.1 p. 130.** It says the E10 run used "1,967 native requests". 1,967 is LAB-Bench's count; E10 has 97 episodes. The same passage calls E10 and E11 "the two lanes that produced complete scores".
  - **§9.2 p. 133.** It says E11 is "the campaign's only complete suite that produced both full coverage and a native model score". E8, E10 and E14 also qualify.
  - **§10.2 p. 152.** It cites `e11_trained_step40_receipt.json` (sampler `step_seed809_40`) as the provenance of the retained 129/312 result. The repo's `e11_full_receipt.json` records sampler `seed809_final`. The step-40 receipt is the 4-task pilot described in §9.1.
- Evidence or criterion: The thesis's main claim is that every number is bound to the correct receipt. These errors break that chain.
- Why it matters: The campaign's value rests entirely on provenance, so a provenance error hurts it more than a numerical error would.
- Requested action: Four fixes, listed below.
  - Make Table C.3 the single source of truth: lane, scope type (orig / repl), terminal state, figure, receipt.
  - Regenerate Fig. 5.3 from that table, or delete it.
  - Correct the sentences listed above.
  - Point the §10.2 E11 threat at `e11_full_receipt.json`.

### Major comment M8
- Location: Title; Abstract p. iii; §1.4 O1 p. 10; Fig. 1.2 p. 6; §5.2 pp. 71–73 and Fig. 5.2; Table A.2 p. 165.
- Observation: "Multi-Framework Benchmark" rests on two completed runs, one on Tinker and one on TRL. They are not matched, because Tinker used Qwen3-8B-Base while TRL used Qwen3-8B Instruct (§5.2 p. 73). The veRL and OpenRLHF entries are seeded dry-run placeholders, yet Fig. 1.2 still draws them as bars (0.553, 0.479).
- Evidence or criterion: The title claims more than the evidence supports. Plotting placeholder values is also inconsistent with the thesis's own never-report-an-unmeasured-number rule.
- Why it matters: Examiners read the title first. "Which frameworks did you actually benchmark?" has an awkward answer right now.
- Requested action: Choose one:
  - (a) Complete one more framework (e.g. veRL) on the canonical config with the same base checkpoint, over at least 3 seeds.
  - (b) Retitle to reflect a multi-framework harness plus GRPO studies. Replace the dry-run bars in Figs. 1.2 and 5.2 with "not run".

  In either case, state the Base-vs-Instruct confound in the caption of every figure that shows 0.856 vs 0.050.

### Major comment M9
- Location: References pp. 158–161 (31 entries); roughly 500 "Source:" footnotes throughout; Chapter 2 pp. 16–32; §6.3.3 p. 93.
- Observation: Citations fall short of what a thesis needs:
  - **Too few references.** The bibliography has 31 entries.
  - **Uncited primary work.** Many named methods and all fourteen benchmark suites have no bibliographic citation. Examples: GSPO, VAPO, GRESO, CPPO, NGRPO, Scaf-GRPO, AERO/RL-ZVP, GVPO, MC-GRPO, "GRPO is secretly DPO", Kaplan/Hoffmann/Snell, HELM, LM Eval Harness, SWE-bench Pro, LAB-Bench, AgentDojo, VerilogEval, Omni-MATH, MLE-bench, BALROG, AgentHarm.
  - **Internal footnotes instead of citations.** Claims are "sourced" to internal files (e.g. `related_work_v2.tex`) that an examiner cannot open.
  - **AI attribution of theory.** §6.3.3 attributes the closed-form ZVF result to "ChatGPT Pro Extended and Gemini Deep Think". There is no AI-assistance declaration in the front matter.
- Evidence or criterion: Standard thesis practice is that literature claims cite primary sources. AI use should be declared under PES policy.
- Why it matters: Chapter 2 currently reads as a synthesis of the author's own documents, not a literature survey. The AI attribution will draw questions if it is not disclosed properly.
- Requested action: Three citation fixes and one derivation, listed below.
  - Add BibTeX entries and bracketed citations for every named method and suite.
  - Move repository paths into an appendix evidence map (claim ID → file), and cut the footnotes to one per paragraph at most.
  - Add an AI-assistance declaration after the Declaration.
  - Derive h_G(p) = p^G + (1−p)^G directly: under i.i.d. Bernoulli, it is the probability that all G rewards are equal. Keep the provenance note, but state the derivation.

### Major comment M10
- Location: §1.3–1.4 pp. 8–10; Chapter 3 pp. 33–46; §5.9 pp. 80–83; Chapter 8 pp. 119–127; §7.1 pp. 108–110.
- Observation: The thesis has no small set of testable research questions with pre-stated outcomes. Its eight objectives mix engineering, scoping and reporting.
  - **P8 is off-topic.** It uses synthetic `make_classification` data. The LLM's AUC of 0.48 comes from a different split, and its 0.792 accuracy is roughly the majority-class rate at 20% prevalence. It contributes no RL evidence, yet takes a full chapter.
  - **Operational detail crowds out results.** Chapter 3 and §5.9 spend many pages on schedule offsets, HMAC domains and reservation IDs.
  - **Degenerate statistics in §7.1.** It reports η² = 1.0000 for group size, R² = 0.993, and stack-to-seed ratios up to 96,128×. These figures indicate degenerate (one-observation-per-cell) designs, but they are presented as findings.
- Evidence or criterion: The defense runs to a 20-minute cap covering results and a demo. The written thesis should make the RQ → result mapping obvious.
- Why it matters: Examiners judge whether the conclusions answer the questions asked. At present the questions are implicit.
- Requested action: Restructure around explicit research questions, as below.
  - State three or four RQs in §1.4. For example: (RQ1) is ZVF recomputable and informative; (RQ2) do algorithm labels separate once the stack is fixed; (RQ3) can a stack manifest flag label-flip risk; (RQ4) what does governance-first evaluation of one actor yield. Map each RQ to a results table and a one-line answer in §10.3.
  - Move P8 to an appendix.
  - Condense §5.9 and Chapter 3 detail into tables.
  - In §7.1, report cells, observations per cell and df alongside every η²/R², or drop the saturated figures.

## Minor comments

### Minor comment m1
- Location: §1.1 p. 4 (equation) vs §4.3 p. 54 and Fig. 4.4 p. 55.
- Observation: σ_g is defined with 1/G in Chapter 1 but with the unbiased 1/(G−1) in §4.3 and Fig. 4.4, where σ ≈ 0.535.
- Evidence or criterion: The notation should be consistent (Appendix C).
- Why it matters: The advantage magnitudes quoted in Fig. 1.1 depend on which definition is used.
- Requested action: Pick one definition, state it in §4.2, and note which runner uses which.

### Minor comment m2
- Location: §4.3 p. 54.
- Observation: The reference `zvf()` code has rendered as run-on prose, with an "[]" artefact.
- Evidence or criterion: Code should be legible.
- Why it matters: It is the diagnostic's defining implementation.
- Requested action: Render it as a verbatim or listings block.

### Minor comment m3
- Location: Fig. 4.1 p. 49 and §4.1 p. 51 vs §5.2 p. 72 and §1.1 p. 5.
- Observation: The text says "Held fixed throughout the programme: LoRA rank 32 … lr 1e-4". But the canonical matrix uses rank 16 and lr 1e-6, the stack probe uses lr 1e-5, and the P4 extension uses rank 4.
- Evidence or criterion: "Held fixed" should be true as written.
- Why it matters: Otherwise it overstates how controlled the design was.
- Requested action: Change to "headline configuration", and point to the per-study table requested in M2.

### Minor comment m4
- Location: §2.5 p. 26; §4.1 p. 51; §10.2 p. 154.
- Observation: The scale range is given variously as 0.6B–1T, 0.6B–671B and 0.6B–235B, and is deliberately left unharmonised.
- Evidence or criterion: The thesis needs one range, with a stated rule.
- Why it matters: The mismatch reads as uncertainty about the thesis's own roster.
- Requested action: Use "0.6B–1T (anchors above 235B single-seed, descriptive)" everywhere.

### Minor comment m5
- Location: Fig. 2.2 p. 29 and §4.4 p. 60 vs §2.4 p. 24 and §7.1 p. 108.
- Observation: The reporting standard is called "seven-field" in some places and "eight-item" in others.
- Evidence or criterion: The count should be consistent.
- Why it matters: P5 is the thesis's strongest contribution.
- Requested action: Use "seven manifest fields + one evaluation item (eight items)" at first use, and a consistent short form after that.

### Minor comment m6
- Location: Fig. 6.2 p. 91 and §6.3.2 p. 91.
- Observation: The figure is titled "Mean ZVF by backend library", and the text calls the nine-row panel "the headline measurement". But eight of the nine rows are a simulation projection, and the x-axis shows methods, not libraries.
- Evidence or criterion: The caption should describe what the figure actually shows.
- Why it matters: The caption's "empirical case" wording overstates the evidence.
- Requested action: Split the figure into a measured panel and a simulated panel, retitle it, and change "headline measurement" to "sensitivity projection (one measured row)".

### Minor comment m7
- Location: §6.3.4 p. 93–94.
- Observation: The residualised r = +0.80 has a CI that runs to +1.00. Several within-method AUROCs are below 0.5 (0.073, 0.335, 0.293, 0.396).
- Evidence or criterion: An AUROC below 0.5 means the ranking is inverted; a CI touching 1.00 suggests truncation.
- Why it matters: As written, these figures read as weak signal, not as inverted signal.
- Requested action: Explain both, and state the CI method.

### Minor comment m8
- Location: §6.3.3 p. 93.
- Observation: The "theory" ZVF values (0.964, 0.954, 0.923, 0.704) are not smooth in G for a single p.
- Evidence or criterion: For a fixed p, h_G(p) is smooth and monotone.
- Why it matters: A reader will suspect an error.
- Requested action: State the p used at each G, or the per-step averaging behind these values.

### Minor comment m9
- Location: §6.4.1 p. 97 vs §6.4.3 p. 98 and Fig. 6.3.
- Observation: One passage says the evidence "supports G = 2 on the measured easy sweep". The next page says G = 2 "under-trains with a 50% batch collapse". Fig. 6.3 puts the apex at G = 8.
- Evidence or criterion: The thesis should give one recommendation.
- Why it matters: The recommendations contradict each other.
- Requested action: Delete the G = 2 recommendation, or scope it explicitly to the 0.5B arithmetic task.

### Minor comment m10
- Location: §6.5.4 p. 104 and §6.6 p. 106.
- Observation: The text says "the only matched base-versus-RL held-out control" is Qwen3-8B, but the same section reports 1.5B pre/post McNemar tests.
- Evidence or criterion: "Only" should be accurate.
- Why it matters: This is an internal contradiction.
- Requested action: Change to "the only multi-seed matched control on Qwen3-8B".

### Minor comment m11
- Location: §1.5 p. 11 vs §3.8 p. 45 and §5.5 p. 76.
- Observation: Held-out sets are described as "200 to 500 prompts in most cases" in one place and "n between 8 and 20" in another. The layer-freeze result rests on n = 8.
- Evidence or criterion: The denominator discipline claimed in NFR-4.
- Why it matters: A reader cannot tell which results are small-n.
- Requested action: Add an n column to the canonical-numbers table (M6).

### Minor comment m12
- Location: §2.2 p. 20.
- Observation: The text says "Four limitations are documented", then lists three.
- Evidence or criterion: The count should match the list.
- Why it matters: It is a small error, but a visible one.
- Requested action: Fix the count or add the missing limitation.

### Minor comment m13
- Location: §10.2 p. 150.
- Observation: The Henderson ten-seed sentence ("met only for the TRL baseline and the held-out GSM8K evaluation … then forced that evaluation to five seeds") contradicts itself.
- Evidence or criterion: A sentence should not contradict itself.
- Why it matters: It weakens the statistical-power paragraph.
- Requested action: State plainly how many seeds each result family has.

### Minor comment m14
- Location: Appendix A pp. 163–164.
- Observation: The registry says runs "span 8 January to 4 July 2026", which excludes the August–September campaign. Run totals are 1,708 / 1,662 / 70+ / 79 / 368 / 790 in different places, with no reconciliation.
- Evidence or criterion: The appendix calls itself "the auditable index of every … run".
- Why it matters: An examiner will ask how many runs there were.
- Requested action: Add a reconciliation table (corpus → count → used by which chapter), and extend the date range or state the cut-off.

### Minor comment m15
- Location: §4.5 p. 63 vs §9.1 p. 130 and §5.8 p. 80.
- Observation: The text says "Sampling is frozen … maximum of 128 response tokens". But the lanes used 1,024, 4,096 and 8,192 tokens at temperatures 0, 0.1 and 0.2.
- Evidence or criterion: The protocol description should match the per-lane receipts.
- Why it matters: A reader could think Omni-MATH or SWE-bench was run with 128 tokens.
- Requested action: Say the 128-token setting applies to the training-selection protocol only, and refer to the per-lane serving table (M5).

### Minor comment m16
- Location: Chapter 9 pp. 132–134.
- Observation: Scores are printed to 16–17 digits (e.g. 0.22877478393492628).
- Evidence or criterion: Reporting precision should match the measurement's precision.
- Why it matters: The extra digits imply spurious precision and hurt readability.
- Requested action: Report 4 significant figures, and keep the exact fractions (450/1967).

### Minor comment m17
- Location: §9.7 Table 9.B p. 142 (E12 row).
- Observation: McNemar is computed over 151 rubric items drawn from only six applications. The text itself says this "overstates the evidence".
- Evidence or criterion: McNemar assumes independent pairs.
- Why it matters: It reports a test the text says is invalid.
- Requested action: Drop the p-value, or cluster by application (n = 6).

### Minor comment m18
- Location: Figs. 1.2, 2.2, 4.6, 5.1, 9.4, 10.1.
- Observation: The figures are very text-dense: paragraphs inside figures and multi-sentence provenance boxes.
- Evidence or criterion: Readability, and whether the figures will work on defense slides.
- Why it matters: They will not read at slide scale within a 20-minute talk.
- Requested action: Move the provenance text into captions or the appendix. Make simplified slide versions of Figs. 4.6, 6.3 and 9.1.

### Minor comment m19
- Location: Throughout (e.g. pp. 13, 57, 101, 154, 157).
- Observation: "honest", "honestly" and "candour" appear dozens of times. Many paragraphs defend the reporting discipline rather than report results.
- Evidence or criterion: Academic register.
- Why it matters: The repetition reads as defensive and adds length.
- Requested action: Cut these self-descriptions to a single statement in §1.5.

### Minor comment m20
- Location: §5.2 p. 72 and §5.9 pp. 81–83.
- Observation: Inline JSON, hashes, reservation IDs and integer offsets appear in running prose.
- Evidence or criterion: Readability.
- Why it matters: They slow down the implementation chapter.
- Requested action: Move them to a listing or an appendix table.

### Minor comment m21
- Location: §6.2.1 p. 87 and §6.2.2 pp. 88–89.
- Observation: Five of the twelve P1 anchors are interrupted runs averaged over 3–5 steps, and the recipe changes with scale. Even so, the section reports slopes, bootstraps and changepoint tests on them.
- Evidence or criterion: Inference is not valid on collinear, heterogeneous anchors. The text half-admits this.
- Why it matters: The statistics suggest more identifiability than the data allow.
- Requested action: Keep the descriptive table and the "constant model wins by AIC" point. Move the bootstrap and changepoint detail to an appendix.

### Minor comment m22
- Location: §1.6 p. 13, §2.7 p. 32 and §4.5 p. 66 vs §9.5 p. 139.
- Observation: Several early passages still present the E4 zero first as "tool-dialogue collapse in the base model" before correcting it.
- Evidence or criterion: Superseded readings should not lead the text.
- Why it matters: Examiners may quote the superseded reading back.
- Requested action: Lead with the bridge finding everywhere, and mention the earlier reading once, in §9.5.

## Suggested defense preparation (outside the review proper)

1. One slide: the RQ → answer table (M10).
2. One slide: the canonical-numbers table (M6), with held-out accuracy shown next to every null (M1).
3. One slide: the E1–E14 lane table with scope type and serving path (M5, M7), plus Table 9.B once the reruns are final (M4).
4. Demo: `registry/query.py stackdiff` on the open-vs-closed "DAPO" pair (R5 verdict), and `p2_collapse_analysis.py` recomputing 0.130/0.190/0.155 from stored tensors. Both are quick, deterministic, and show the thesis's strongest contributions.

## Three-persona ensemble scoring (self-review skill, NeurIPS form)

The NeurIPS form scores research-paper novelty; it is not an M.Tech pass criterion. These scores are indicative only.

| Dimension | R1 harsh-fair | R2 harsh-critical | R3 open-minded | Avg |
|---|---|---|---|---|
| Overall (/10) | 4 | 3 | 5 | 4.0 |
| Soundness (/4) | 2 | 2 | 3 | 2.3 |
| Presentation (/4) | 2 | 2 | 2 | 2.0 |
| Contribution (/4) | 2 | 2 | 3 | 2.3 |
| Originality (/4) | 2 | 2 | 3 | 2.3 |
| Quality (/4) | 2 | 2 | 3 | 2.3 |
| Clarity (/4) | 2 | 1 | 2 | 1.7 |
| Significance (/4) | 2 | 2 | 3 | 2.3 |
| Confidence (/5) | 4 | 4 | 3 | 3.7 |

Weighted score (AgentLaboratory weights) is about 4.9/10, against a NeurIPS bar of about 5.9.

What the three reviewers agree on:
- **Strengths.** The provenance and no-pooling discipline; P5/P6 as a usable artefact; willingness to withdraw claims.
- **Weaknesses.** Ceiling-bound nulls (M1); inconsistent numbers (M6, M7); an undocumented actor (M3); length and sprawl (M10).

R3 credits the registry and the audit tooling as genuinely novel. R2 argues the nulls are uninformative, so the empirical contribution is small.

All required sections are present: Abstract, Introduction, Methods, Results, Conclusion.

## Limitations of this review

- Figures were read from extracted text, not visually inspected.
- The appendix tables (A.3–A.7, C) were skimmed, not checked row by row.
- No experiment was re-run.
- The Chapter 9 lanes under rerun were not assessed.

# Confidential comments to editor

Not applicable. This is an author-requested self-review with no editor channel. No conflicts to declare. AI assistance: prepared with local LLM reading of the author's own thesis at the author's request. No manuscript text was sent to external services.
