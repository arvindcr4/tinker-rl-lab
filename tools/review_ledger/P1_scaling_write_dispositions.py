"""Write dispositions JSONL + claim ledgers for P1_scaling and P1_identifiability_note (review 2026-10-18)."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "reports/deepreview/dispositions"
L = ROOT / "reports/deepreview/ledgers"
D.mkdir(parents=True, exist_ok=True)
L.mkdir(parents=True, exist_ok=True)
S = "platform_hybrid/paper/sections/"
RV = "platform_hybrid/experiments/results/review_2026-10-18/P1_scaling/"
SYN = (
    "platform_modal/scripts/scaling_law_iter33.py:143-190 def synth() builds traces by np.linspace(early,late) "
    "from scaling_law_extended_frontier.tsv summaries; iter37/37b/37d/41/45 reuse it; iter49/53 consume iter33/45 outputs; "
    "iter65 TSVs (non-v2) come from the synthetic-trace version (paper text said 'evaluates the 12-anchor pool on the 6-statistic "
    "synthetic step-trace reconstructions')"
)
REMOVED = [S + f"scaling_law_iter{i}.tex (no longer \\input)" for i in (33, 37, 41, 45, 49, 53, 65)]

P1 = {
    1: (
        "fixed",
        "Ward label is now stated to be computed from the response, so it is descriptive only. The pre-update R(1)>=0.5 indicator AICc was recomputed on raw traces (n=5: -23.08, n=7: -41.61, matching iter133; n=12: -42.35, R2=0.795). The earlier unsourced text (-32.39, R2=0.608, 'Qwen3-235B switches classes') was wrong: the raw Qwen3-235B R(1)=1.0 and the switcher is Qwen3.5-27B. Caveats added that R(1) is the same rollout process and the threshold is post hoc. Script: tools/review_ledger/P1_scaling_rawtrace_refit.py -> exogenous_R1_indicator_aicc.tsv",
        [S + "scaling_laws.tex", S + "p1_conclusion.tex"],
    ),
    2: (
        "fixed",
        "Table 26 replaced by a joint bounded nested fit (c in [0,1], c<=Rmax<=1, lambda in [1e-4,10]). RSS3p<=RSS2p on all 5 anchors and dAICc is in [+1.91,+3.17]. The two-stage estimator (dAICc +18.18/+14.77, Rmax 1.05) is withdrawn. The iter137 cross-scale slopes were recomputed (-0.63+-0.87 log t80; -0.13+-0.20 Rmax) and the capability-propagation claim retracted (Qwen3-8B goes to the Rmax bound). Table 23 piecewise was recomputed with Rmax<=1, and Table 8/Fig 4 bootstrap with Rmax<=1 (tools/review_ledger/P1_scaling_elevated_bounded.py). Outputs: core5_offset_fit_bounded.tsv, core5_piecewise_bounded.tsv, elevated_bounded/",
        [
            S + "scaling_laws.tex",
            "platform_hybrid/paper/figures/review_2026-10-18/scaling_law_elevated.png",
        ],
    ),
    3: (
        "already-fixed",
        "Current text gives +1.170+-0.254 for log10(t80*N) and labels the unlogged regression separately (commit 9b3e4f394). The unlogged value was recomputed with bounded bootstrap t80: +383+-341.",
        [S + "scaling_laws.tex"],
    ),
    4: (
        "unsourced-removed",
        "Confirmed. "
        + SYN
        + ". Sections 4.3-4.8 (iter33/37/41/45/49/53) and iter65 removed from the build. The iter61 jackknife (Table 40, P3/P4, figure) also used summary-reconstructed traces and is removed. Replaced by sections/p1_rawtrace_reanalysis.tex, recomputed on the released per-step traces; three-segment fits use only the 7 traces with T>=20.",
        REMOVED
        + [
            S + "p1_rawtrace_reanalysis.tex",
            S + "scaling_law_iter61.tex",
            "platform_hybrid/paper/paper_P1_scaling.tex",
        ],
    ),
    5: (
        "unsourced-removed",
        "The Rmax=2.000 values came from [0,2]-bounded fits to synthetic traces (iter41/45). All dependent tables (31-37, 40) are removed. All remaining/recomputed fits constrain Rmax to [0,1]. Short probes (T<=5) are no longer imputed; they are listed descriptively in Table 30 without fits.",
        REMOVED + [S + "p1_rawtrace_reanalysis.tex"],
    ),
    6: (
        "fixed",
        "The tables with Nemotron=dense (32/33/35/44) are removed with their sections. The raw-trace re-analysis uses the 7 MoE / 5 dense split and reports slopes on both log10 N_tot and log10 N_act, plus an active-parameter compute proxy (Table 31). Table 38/iter61 shows moe.",
        [S + "p1_rawtrace_reanalysis.tex"],
    ),
    7: (
        "already-fixed",
        "Section 7 (AntiVibe mechanisms) is absent from the current source and build; grep for 'antivibe' in the P1 sections finds nothing.",
        [],
    ),
    8: (
        "already-fixed",
        "Section 7.3 verification gates are absent from the current source/build.",
        [],
    ),
    9: (
        "already-fixed",
        "Title is now 'Limits of Scaling Laws ... Execution-Substrate Variance'. Abstract/Sec 2.2 separate the two seven-library rosters and say only Tinker and TRL completed LLM runs.",
        [],
    ),
    10: (
        "already-fixed",
        "Abstract/intro now state GSM8K, 5-choice arithmetic and multi-hop tool use. The Table 1 caption scopes the tracks analysed. Figure 1 is a shared tikz artifact (see shared request R4).",
        [],
    ),
    11: (
        "already-fixed",
        "Sec 3.1 now lists Nemotron-3-Super-120B-A12B as LatentMoE (120B total / 12B active) and notes the dense mislabel correction.",
        [],
    ),
    12: (
        "already-fixed",
        "Kimi-K2-Thinking is in the frontier tier (1T total / 32B active). The roster range is 0.5B-1T.",
        [],
    ),
    13: (
        "already-fixed",
        "Sec 2.4 Reward Verification is populated (GSM8K extraction, arithmetic regex, 199-action env). Sec 2.5 includes GSM8K floor/ceiling.",
        [],
    ),
    14: (
        "fixed",
        "GRPO hyperparameters are now in Sec 2.3, but the shared-methods values (G=16/8, temperature 0.7, lr 1e-5-2e-5, frontier LoRA r=32) contradict the trace JSON/W&B configs (group 8x2 prompts, lr 3e-5, r32; frontier 4x2, lr 1e-5, r16; temperature 0.8 in the driver). The correct values are stated in sections/p1_rawtrace_reanalysis.tex. The shared-methods correction is filed as P1_scaling_shared_requests.md R1/R2.",
        [
            S + "p1_rawtrace_reanalysis.tex",
            "reports/deepreview/dispositions/P1_scaling_shared_requests.md",
        ],
    ),
    15: (
        "already-fixed",
        "Variance-decomposition paragraph explicitly labels the four eta^2 as marginal, overlapping and non-additive (sum 1.897), and says a factorial/mixed model would be needed.",
        [],
    ),
    16: (
        "already-fixed",
        "Table 5 is labelled a legacy audit trail, the double dagger is defined (per-step pseudo-replication), and the restricted Tier-A/B F3 result is given in text with F, p and BH-p. A request to demote the table and correct Rank 1 is in shared_requests R3.",
        ["reports/deepreview/dispositions/P1_scaling_shared_requests.md"],
    ),
    17: (
        "already-fixed",
        "The 'headline positive result' sentence was replaced. The withdrawal table, 7-vs-5 corrected p=0.178 (now recomputed exactly: 141/792) and Levene on both splits are reported; Fig 5(c) is captioned withdrawn.",
        [S + "p1_rawtrace_reanalysis.tex"],
    ),
    18: (
        "already-fixed",
        "Iter 21 (H) now reports a flat Rmax slope (+0.077+-0.125, p=0.549) and the Sec 4 intro and Iter 25 say the marginal regression is flat. Consistent with the raw recompute (long-7 Rmax2p slope +0.028, perm p=0.84).",
        [],
    ),
    19: (
        "already-fixed",
        "Table 16 caption now explains the 4-vs-1 partition. Iter125/133 indicators are explicitly 'relaxed' and distinguished from Nimmaturi. Fig 9 caption corrected.",
        [],
    ),
    20: (
        "fixed",
        "Eq. 5 continuity was already stated. Added: the AICc with t_peak counted as a parameter (dAICc +4.30..+6.78) vs fixed (+1.40..+3.16), recomputed with bounded fits, and a caveat that AIC regularity fails for boundary estimates (iter129 text and the Table 13 caption).",
        [S + "scaling_laws.tex"],
    ),
    21: (
        "refuted",
        "The minus sign is intentional and documented in the text and the Table 22 caption: planted Rmax(N)=0.85-beta*log10(N/8) stays inside [0,0.85] for N>=8B, and the R^2>=0.5 recovery criterion is sign-invariant, so power depends only on |beta|. (scaling_law_iter121.py)",
        [],
    ),
    22: (
        "already-fixed",
        "Pools are now labelled n=11 consistently (Table 24/25). The iter13 list includes Qwen3-30B-MoE-Inst. Additionally fixed: the per-anchor horizons in the iter13 paragraph were wrong (gpt-oss T=24, Qwen3.5-27B T=4, Qwen3-32B T=5, Qwen3-235B T=3) and were corrected from the trace JSONs to 30/3/3/4.",
        [S + "scaling_laws.tex"],
    ),
    23: (
        "already-fixed",
        "Paragraph (F) now says 7 of 12 at bound and 5 interior. The '(estimated)/(estimate)' placeholders are replaced by values.",
        [],
    ),
    24: (
        "unsourced-removed",
        "The contradictions were between synthetic-trace classifiers (iter33/49/65) and raw Table 10; all synthetic-trace sections are removed. The raw recompute flags Nemotron as the sole collapse; its raw post-peak slope is -0.0036/step (elevated_bounded/scaling_law_nemotron_rootcause.tsv).",
        REMOVED,
    ),
    25: (
        "unsourced-removed",
        "iter65 (Tables 42-44, Fig 16) was computed on summary-reconstructed traces and is removed. Raw three-segment fits on 7 full traces are in p1_rawtrace_reanalysis.tex.",
        [S + "scaling_law_iter65.tex (no longer \\input)", S + "p1_rawtrace_reanalysis.tex"],
    ),
    26: (
        "unsourced-removed",
        "Table 43 belonged to synthetic-trace iter65 and is removed.",
        [S + "scaling_law_iter65.tex (no longer \\input)"],
    ),
    27: (
        "unsourced-removed",
        "PCI and P3 verdict belonged to synthetic-trace iter65 and are removed.",
        [S + "scaling_law_iter65.tex (no longer \\input)"],
    ),
    28: (
        "unsourced-removed",
        "Root cause confirmed: berkeley/sweagent_passk_aci.py:171-190 bootstraps synthetic Gaussian draws (np.random.normal(mean, sd, n_steps)), not the traces. That is why p_hat=0.416 vs R=0.285; the 'post-warmup window' explanation in the text was not true. Table 27 and the straddle test were recomputed on the raw per-step traces (p_hat = trace mean; CIs 0.115-0.213 wide). The text now says these are batch-to-batch training-reward intervals, not held-out Pass@1, and that p_hat and Rmax are different estimands. The unsourced 'n>=100 steps' claim was removed.",
        [S + "scaling_passk_ci.tex", RV + "core5_step_bootstrap.tsv"],
    ),
    29: (
        "already-fixed",
        "Text states the RQS ratio is an endogenous diagnostic bound and cannot separate parser failure from policy deficit without a relaxed-parser audit.",
        [],
    ),
    30: (
        "already-fixed",
        "Method item (e) now has T*=(z+z)^2(sigma/Delta)^2 and (b) says LCM.",
        [],
    ),
    31: (
        "fixed",
        "Verified samestack_ppo_grpo.json comes from experiments/modal/modal_samestack_ppo_grpo.py on two-operand addition ('What is a + b?'), which explains held-out 0.99 vs GSM8K 0.734. Sec 4.2 now says so. The Hill-n=2 '5/5 PPO' claim (inconsistent with iter37's 2/5 and dependent on the removed iter37 section) was removed. The critic-warm-up statement is reframed as an untested hypothesis and the ranking is scoped to the 3-model candidate set.",
        [S + "scaling_law_iter29.tex"],
    ),
    32: (
        "fixed",
        "Section 4.9 moved to an appendix titled '(Conjectures)' with a disclaimer that no equation was fitted and several variables were unmeasured. In-text 'Frontier synthesis' appeals in iter21/25/29 are relabelled as external conjectures pointing to the appendix. Eq. 12 KL_t is indexed with an explicit sum. RLVR baselines (DAPO, Dr. GRPO, online filtering) are already cited in Related Work and Limitations.",
        [
            S + "frontier_synthesis_scaling.tex",
            S + "scaling_laws.tex",
            S + "scaling_law_iter29.tex",
            "platform_hybrid/paper/paper_P1_scaling.tex",
        ],
    ),
    33: (
        "fixed",
        "Seven iteration sections were removed (iter33/37/41/45/49/53/65), which eliminates the shifting lambda<=5 and Rmax<=2 bounds; the remaining fits all use Rmax in [0,1]. The 'Recommended action' directive is already gone. Full narrative consolidation of the remaining iter-ordered paragraphs is not done (scope).",
        REMOVED,
    ),
    34: (
        "already-fixed",
        "Figures 17-26 after the references are absent from the current build.",
        [],
    ),
    35: (
        "already-fixed",
        "Companion paper cited as [5]; F3 is defined at first use in Sec 2.1.",
        [],
    ),
    36: (
        "wontfix-scope",
        "Figure 1 is shared tikz/taxonomy (currently not built: renders as a missing-artifact placeholder). Request R4 filed.",
        ["reports/deepreview/dispositions/P1_scaling_shared_requests.md"],
    ),
    37: ("already-fixed", "'verl' is used consistently.", []),
    38: (
        "fixed",
        "Tier-A subset is already stated as 0.5B-32B. Added a Table 7 caption note that Tables 6-7 use DeepSeek 671B while the 12-anchor analyses use 685B, and verified the slopes are unchanged (-0.025 vs -0.024 on mean).",
        [S + "scaling_laws.tex"],
    ),
    39: ("already-fixed", "Sec 3.4 reconciles 14+6+10+12=42 core runs and 70+ total.", []),
    40: (
        "already-fixed",
        "TTestIndPower equal-variance caveat is present in the Table 4 caption and the power paragraph.",
        [],
    ),
    41: (
        "wontfix-scope",
        "Shared tikz figure; request R4.",
        ["reports/deepreview/dispositions/P1_scaling_shared_requests.md"],
    ),
    42: (
        "wontfix-scope",
        "Shared tikz figure (N vs G); request R4.",
        ["reports/deepreview/dispositions/P1_scaling_shared_requests.md"],
    ),
    43: (
        "wontfix-scope",
        "Shared tikz figure; request R4.",
        ["reports/deepreview/dispositions/P1_scaling_shared_requests.md"],
    ),
    44: ("already-fixed", "Hedges and Olkin is reference [11].", []),
    45: (
        "already-fixed",
        "Reads 'an internally recorded'. (The sentence was later rewritten.)",
        [S + "p1_results_intro.tex"],
    ),
    46: ("already-fixed", "Sentence ends 'reported in Table 7'.", []),
    47: (
        "already-fixed",
        "Dangling colons replaced by table references (e.g. Table 8, 9, 10).",
        [],
    ),
    48: (
        "fixed",
        "'Frontier synthesis (ChatGPT Pro Extended, R1)' replaced with an external-conjecture pointer to the appendix. The other 'frontier synthesis' appeals were relabelled.",
        [S + "scaling_laws.tex"],
    ),
    49: ("already-fixed", "Paragraph (H) reports +0.077+-0.125, p=0.549.", []),
    50: (
        "already-fixed",
        "Nimmaturi capitalised and a single bib entry ([26]). Inline script paths remain (reproducibility choice; wontfix for that sub-point).",
        [],
    ),
    51: (
        "already-fixed",
        "'four of five' in the caption is fixed. The PNG supertitle cannot be changed without regenerating figures/scaling_law_fit.png (not done).",
        [],
    ),
    52: ("already-fixed", "Model names in braces are rendered as text.", []),
    53: (
        "already-fixed",
        "Lecture shorthand and the 'Recommended action' directive are gone; Eureka is cited as ma2024eureka.",
        [],
    ),
    54: (
        "unsourced-removed",
        "Table 34 (P*=3.46B) belonged to the synthetic-trace iter45/49 section; removed.",
        [S + "scaling_law_iter45.tex (no longer \\input)"],
    ),
    55: (
        "unsourced-removed",
        "The orphaned subheadings were in the removed iter41/45/49 sections; the remaining 'Results.' headings have content.",
        REMOVED,
    ),
    56: ("already-fixed", "Artifacts paragraphs no longer reference missing figure PDFs.", []),
    57: (
        "unsourced-removed",
        "Table 30 (iter33 battery) removed. The raw recompute reports Nemotron uniqueness as a descriptive count, not a test.",
        [S + "scaling_law_iter33.tex (no longer \\input)"],
    ),
    58: (
        "unsourced-removed",
        "iter33 phase-count bullets removed with the section.",
        [S + "scaling_law_iter33.tex (no longer \\input)"],
    ),
    59: (
        "unsourced-removed",
        "iter37d holdout paragraph removed with the synthetic-trace section.",
        [S + "scaling_law_iter37.tex (no longer \\input)"],
    ),
    60: (
        "unsourced-removed",
        "iter41 truncation section (synthetic traces) removed.",
        [S + "scaling_law_iter41.tex (no longer \\input)"],
    ),
    61: (
        "fixed",
        "The jackknife table/bullets with 'residuall' and float literals are removed (synthetic-trace iter49 fit). Remaining iter61 floats are rounded. The ZVF tilde comes from shared preamble rendering (left).",
        [S + "scaling_law_iter61.tex"],
    ),
    62: (
        "unsourced-removed",
        "Eq. 9 belonged to the iter65 section, removed.",
        [S + "scaling_law_iter65.tex (no longer \\input)"],
    ),
    63: (
        "unsourced-removed",
        "Figure 16 belonged to the iter65 section, removed.",
        [S + "scaling_law_iter65.tex (no longer \\input)"],
    ),
    64: (
        "fixed",
        "N0, (x)+ and GU were already defined. KL_t is now indexed by an explicit sum over t in Eq. 12 (moved to the conjecture appendix).",
        [S + "frontier_synthesis_scaling.tex"],
    ),
    65: ("already-fixed", "Single Kaplan entry in the current bibliography.", []),
    66: ("already-fixed", "Kimi K2 / Kimi K2 Thinking capitalised with URLs.", []),
    67: ("already-fixed", "Single Nimmaturi entry with five authors.", []),
    68: ("already-fixed", "Internal iteration self-citation removed from the bibliography.", []),
    69: ("already-fixed", "OpenHands entry has the full author list.", []),
    70: (
        "already-fixed",
        "Related Work cites DAPO, Dr. GRPO, online filtering, Tan et al. 2025 and connects zero-variance starvation to the null. ScaleRL not added (would need a new bib entry; not verified).",
        [],
    ),
    71: (
        "already-fixed",
        "Limitations item 1 distinguishes GSM8K step-1 pre-saturation from intrinsic scaling and calls for AIME/OlympiadBench. R(1) range corrected to [0.875,1.00] from raw traces.",
        [S + "p1_conclusion.tex"],
    ),
    72: (
        "fixed",
        "Limitations item 2 rewritten: synthetic-trace batteries withdrawn and replaced, the five console-transcribed probes are excluded from fits, and inference is restricted to the 7 full traces.",
        [S + "p1_conclusion.tex"],
    ),
    73: (
        "already-fixed",
        "Limitations item 3 cites DAPO, Dr. GRPO and online difficulty filtering for the roadmap.",
        [],
    ),
}

NOTE_F = "platform_hybrid/paper/paper_P1_identifiability_note.tex"
AMP = "reports/deepreview/dispositions/P1_identifiability_note_amp_requests.md"
DEF = "deferred: owned by concurrent Amp session"
NOTE = {
    588: (
        "wontfix-scope",
        DEF
        + ". Verified: the current note enumerates all 12 anchors with horizons and gives the repo URL, but lacks recipe/config and a frozen archive. Proposed panel-spec text (configs from W&B, console-probe flag, run_id mislabel) is in "
        + AMP,
        [AMP],
    ),
    589: (
        "wontfix-scope",
        DEF
        + ". Verified the reviewer is right that leave-Nemotron is a deterministic subset of checked-in traces. Recomputed: R slope on log10 N_tot +0.118+-0.118 (n=11, perm p=0.36), +0.073+-0.102 (n=6 full runs, p=0.62). The 'reversal' was the MoE-dense 6v5 one-sided p=0.048 (post hoc). Replacement row text is in "
        + AMP,
        [AMP, RV + "cross_scale_regressions.tsv"],
    ),
    590: (
        "already-fixed",
        "Current note calls plateau/drift/collapse descriptive labels, not an established taxonomy, and names ceiling effects and co-varying recipe/horizon/capability. Optional R(1) ceiling sentence is in "
        + AMP,
        [AMP],
    ),
    591: (
        "already-fixed",
        "Section 3 is populated with Table 1; workflow directives ('Do not submit the long PDF') are removed; repo URL given and frozen archive disclaimed.",
        [],
    ),
    592: (
        "already-fixed",
        "Note states params_B is nominal total parameters and Nemotron is 120B total / 12B active.",
        [],
    ),
    593: (
        "already-fixed",
        "Historical p-values are 'not retained as evidence of flatness'; Table caption says analyses are 'identified, not endorsed as tests'.",
        [],
    ),
    594: (
        "already-fixed",
        "Note states fitted boundary values are not exchangeable replicates and withdraws the lambda/R_inf permutation tests as evidence.",
        [],
    ),
    595: (
        "already-fixed",
        "Functional form and bounds (R0 in [-1.5,1.5], R_inf in [0,1.5], lambda in [1e-4,50]) are stated, with the nonphysical-bounds caveat; the tests are withdrawn.",
        [],
    ),
    596: (
        "already-fixed",
        "AICc comparisons withdrawn as headline evidence because of short per-family panels.",
        [],
    ),
    597: (
        "wontfix-scope",
        DEF
        + ". Recomputed 7v5 gap +0.188 (exact p=141/792=0.178), 6v5 excluding Nemotron +0.278 (one-sided p=0.048, two-sided 0.093), full-runs-only +0.020 (p=0.57). Proposed row text in "
        + AMP,
        [AMP, RV + "moe_vs_dense_permutation.tsv"],
    ),
    598: (
        "wontfix-scope",
        DEF
        + ". Verified: the note still has no bibliography. Six verified keys from references.bib are proposed in "
        + AMP,
        [AMP],
    ),
    599: (
        "already-fixed",
        "Note uses Nemotron-3-Super-120B-A12B with 'historically labelled Nemotron-120B'.",
        [],
    ),
    600: (
        "already-fixed",
        "Note now refers to 'fitted rates and asymptotes'/'saturation-rate fits'; no conflicting terms.",
        [],
    ),
    601: ("already-fixed", "AICc is expanded on first use; 'elevation suite' jargon removed.", []),
    602: (
        "already-fixed",
        "Row now reads 'Size regressions on fitted rates and asymptotes'; the lambda-N notation is gone.",
        [],
    ),
    603: (
        "already-fixed",
        "p-values are formatted consistently; the permutation counts are withdrawn rows. Exact enumeration verified (141/792) in moe_vs_dense_permutation.tsv.",
        [],
    ),
    604: ("already-fixed", "Table 1 now sits under its Section 3 heading with [ht] placement.", []),
}

for paper, dd in (("P1_scaling", P1), ("P1_identifiability_note", NOTE)):
    with open(D / f"{paper}.jsonl", "w") as f:
        for k in sorted(dd):
            st, ev, files = dd[k]
            f.write(json.dumps({"id": k, "status": st, "evidence": ev, "files": files}) + "\n")

GEN = "tools/review_ledger/P1_scaling_rawtrace_refit.py"
GENB = "tools/review_ledger/P1_scaling_elevated_bounded.py"
rows = [
    (
        "p1_intro.tex (i)",
        "5-anchor mean-reward slope",
        "-0.024 +- 0.205 /dec (was +0.056+-0.130, unsourced)",
        RV + "cross_scale_regressions.tsv",
        GEN,
        "recomputed",
    ),
    (
        "p1_intro.tex (i)",
        "12-anchor mean-reward slope",
        "+0.081 +- 0.129 /dec",
        RV + "cross_scale_regressions.tsv",
        GEN,
        "verified",
    ),
    (
        "p1_intro.tex (i)",
        "p range",
        "all p > 0.5 (was p >= 0.22, unsourced)",
        RV + "cross_scale_regressions.tsv",
        GEN,
        "recomputed",
    ),
    (
        "Table 8 Nemotron row",
        "bootstrap Rmax / lambda / P(bound) / t80 CI",
        "0.271 / 3.27 / 0.278 / 12.95 [0.16,104.4]",
        RV + "elevated_bounded/scaling_law_bootstrap_ci.tsv",
        GENB,
        "recomputed",
    ),
    (
        "Fig 4",
        "elevation panels with Rmax<=1",
        "figures/review_2026-10-18/scaling_law_elevated.png",
        RV + "elevated_bounded/",
        GENB,
        "recomputed",
    ),
    (
        "compute-proxy paragraph",
        "unlogged t80*N slope",
        "+383 +- 341 (was +482+-490, R2 0.244, p 0.398 from Rmax<=1.5 fit)",
        RV + "elevated_bounded/scaling_law_compute.tsv",
        GENB,
        "recomputed",
    ),
    (
        "iter13 paragraph",
        "horizons gpt-oss/Qwen3.5-27B/Qwen3-32B/Qwen3-235B",
        "T=30/3/3/4 (were 24/4/5/3)",
        "platform_hybrid/experiments/tinker-runs/results/*_gsm8k_*.json",
        GEN,
        "recomputed",
    ),
    (
        "Table 23",
        "piecewise Rmax_pw / dAICc (5 anchors)",
        "0.816,0.255,0.923,0.842,1.000*; +2.68,+1.40,+1.47,+3.16,+2.64",
        RV + "core5_piecewise_bounded.tsv",
        GEN,
        "recomputed",
    ),
    (
        "iter129 text",
        "dAICc range, F p",
        "[+1.40,+3.16] fixed t_peak; [+4.30,+6.78] counted; F p>=0.29",
        RV + "core5_piecewise_bounded.tsv",
        GEN,
        "recomputed",
    ),
    (
        "iter133 (d)",
        "n=12 R(1)>=0.5 indicator AICc / R2",
        "-42.35 / 0.795 (was -32.39 / 0.608, unsourced); switcher Qwen3.5-27B (was Qwen3-235B)",
        RV + "exogenous_R1_indicator_aicc.tsv",
        GEN,
        "recomputed",
    ),
    (
        "iter133 (d)",
        "n=5/n=7 R(1) indicator AICc",
        "-23.08 / -41.61",
        RV + "exogenous_R1_indicator_aicc.tsv",
        GEN,
        "verified",
    ),
    (
        "Table 26",
        "joint bounded offset fit c/Rmax/lambda/AICc2p/AICc3p",
        "see table; dAICc +1.91..+3.17, nested OK 5/5",
        RV + "core5_offset_fit_bounded.tsv",
        GEN,
        "recomputed",
    ),
    (
        "Table 26 (old)",
        "two-stage offset fit dAICc +18.18/+14.77",
        "withdrawn",
        "platform_hybrid/experiments/results/scaling_law_iter137_offset_fit.tsv",
        "platform_modal/scripts/scaling_law_iter137.py",
        "removed",
    ),
    (
        "iter137 (b)",
        "cross-scale slopes log t80 / Rmax",
        "-0.63+-0.87 / -0.13+-0.20 (were +0.507+-0.718 / -0.172+-0.198)",
        RV + "core5_offset_fit_crossscale.tsv",
        GEN,
        "recomputed",
    ),
    (
        "Table 27",
        "step-bootstrap p_hat and 95% CI (5 anchors)",
        "0.869[0.808,0.923] 0.844[0.781,0.900] 0.817[0.735,0.890] 0.285[0.225,0.348] 0.175[0.075,0.288]",
        RV + "core5_step_bootstrap.tsv",
        GEN,
        "recomputed",
    ),
    (
        "Table 27 (old)",
        "p_hat on synthetic Gaussian draws",
        "withdrawn",
        "platform_hybrid/experiments/results/berkeley/sweagent_passk_per_anchor.tsv",
        "platform_modal/scripts/berkeley/sweagent_passk_aci.py",
        "removed",
    ),
    (
        "passk audit para",
        "H5 slope t-CI",
        "-0.023, t-CI [-0.67,+0.62]",
        RV + "cross_scale_regressions.tsv",
        GEN,
        "verified",
    ),
    (
        "Table 30",
        "12-anchor raw-trace table",
        "see table",
        RV + "rawtrace_anchor_table.tsv",
        GEN,
        "recomputed",
    ),
    (
        "Table 31",
        "cross-scale slopes (7 rows)",
        "see table",
        RV + "cross_scale_regressions.tsv",
        GEN,
        "recomputed",
    ),
    (
        "Sec 4.3 architecture",
        "MoE-dense gaps and exact p",
        "+0.188 p=0.178; -Nem +0.278 p=0.048 (2s 0.093); long7 +0.020 p=0.57; long7-Nem +0.188 p=0.25",
        RV + "moe_vs_dense_permutation.tsv",
        GEN,
        "recomputed",
    ),
    (
        "Sec 4.3 three-phase",
        "peak_frac>0.10 3/7; late>early 4/7; template 1/7 (Llama); collapse Nemotron only",
        "counts",
        RV + "long7_three_phase.tsv",
        GEN,
        "recomputed",
    ),
    (
        "Table 33 (iter61 strata)",
        "lambda at bound HIGH/LOW/ALL",
        "4/6, 1/1, 5/7; RMSE 0.186/0.170/0.184 (was 7/9, 2/3, 9/12)",
        RV + "long7_zvf_strata.tsv",
        GEN,
        "recomputed",
    ),
    (
        "iter61 P1 bullet",
        "HIGH more bound",
        "FAIL on full traces (was PASS)",
        RV + "long7_zvf_strata.tsv",
        GEN,
        "recomputed",
    ),
    (
        "Table 40 + P3/P4 (iter61)",
        "jackknife of iter49 fit",
        "withdrawn",
        "platform_hybrid/experiments/results/scaling_law_iter61_jackknife.tsv",
        "platform_modal/scripts/scaling_law_iter61.py (synthetic-trace version)",
        "removed",
    ),
    (
        "Secs 4.3-4.8 old + iter65",
        "all numeric claims (Tables 30-37, 42-44)",
        "withdrawn",
        "platform_hybrid/experiments/results/scaling_law_iter{33,37,41,45,49,53,65}_*.tsv",
        "platform_modal/scripts/scaling_law_iter33.py synth()",
        "removed",
    ),
    (
        "p1_conclusion item 1",
        "R(1) range of instruction-tuned anchors",
        "[0.875,1.00] (was [0.75,1.00])",
        RV + "rawtrace_anchor_table.tsv",
        GEN,
        "recomputed",
    ),
    (
        "Sec 4.3 pool para",
        "run configs (group x prompts, rank, lr)",
        "8x2, r32, 3e-5; frontier 4x2, r16, 1e-5",
        "tinker-runs/results/*.json + W&B wandb_runs.json config",
        "-",
        "verified",
    ),
    (
        "Table 6/7 caption",
        "DeepSeek 671 vs 685 slope invariance",
        "mean slope -0.025 vs -0.024",
        "platform_hybrid/experiments/tinker-runs/results/*_gsm8k_*.json",
        "inline check (numpy polyfit)",
        "verified",
    ),
]
with open(L / "P1_scaling.tsv", "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["location", "claim_text", "value", "source_artifact", "generator_script", "status"])
    w.writerows(rows)
nrows = [
    (
        "Table 1 MoE-dense row (proposed)",
        "7v5 / 6v5 / long7 gaps",
        "+0.188 p=0.178; +0.278 p=0.048; +0.020 p=0.57",
        RV + "moe_vs_dense_permutation.tsv",
        GEN,
        "recomputed",
    ),
    (
        "Table 1 leave-Nemotron row (proposed)",
        "slope n=11 / n=6",
        "+0.118+-0.118 p=0.36; +0.073+-0.102 p=0.62",
        RV + "cross_scale_regressions.tsv",
        GEN,
        "recomputed",
    ),
    (
        "Table 1 Nemotron row",
        "peak 0.875@3, last-10 0.1625, zero 11/20, 5/20 > 0.25",
        "verified against trace",
        "platform_hybrid/experiments/tinker-runs/results/frontier_gsm8k_nemotron-120b.json",
        GEN,
        "verified",
    ),
    (
        "Table 1 size regressions",
        "historical p=0.7382/0.5061",
        "verified present in TSV; not retained as evidence",
        "platform_hybrid/experiments/results/scaling_law_iter109b_permtest.tsv",
        "platform_modal/scripts/scaling_law_iter109b.py",
        "verified",
    ),
]
with open(L / "P1_identifiability_note.tsv", "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["location", "claim_text", "value", "source_artifact", "generator_script", "status"])
    w.writerows(nrows)
from collections import Counter

for paper, dd in (("P1_scaling", P1), ("P1_identifiability_note", NOTE)):
    print(paper, len(dd), Counter(v[0] for v in dd.values()))
