"""Write reports/deepreview/dispositions/P5_minreport.jsonl (one line per finding id 269-348)."""

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RM = (
    "Removed from the compiled paper in 2a808b576 (inputs of _shared_methods and all p5_iter* sections "
    "dropped from paper_P5_minreport.tex; p5_evidence.tex cut to Exhibits 1-3 with \\endinput; "
    "post-\\endinput text in p5_stack/p5_toolchain is not compiled). "
)
S = "platform_hybrid/paper/sections/"

D: dict[int, dict[str, int | str]] = {}


def put(ids, status, evidence, files=""):
    for i in ids:
        D[i] = {"id": i, "status": status, "evidence": evidence, "files": files}


put(
    [269, 270, 284],
    "already-fixed",
    RM
    + "Sections 2-3 (42-experiment one-way ANOVAs, Table 4 power, legacy 20-test BH Table 5) came from sections/_shared_methods, which P5 no longer inputs. p5_results_intro.tex states that the pooled benchmark tests are withdrawn.",
    "platform_hybrid/paper/paper_P5_minreport.tex",
)
put(
    [271, 272, 273, 286, 287, 290, 291, 292, 304, 328, 329, 330, 331, 333],
    "already-fixed",
    RM
    + "The mega_20260704 corpus and the cross-corpus eta^2 ratio analyses are gone. The p5_evidence.tex 'Evidence boundary' paragraph now says the 98-cell corpus is training-free sampling, that marginal eta^2 values are not additive, and that cross-population ratios do not rank stack vs algorithm.",
    S + "p5_evidence.tex",
)
put(
    [
        274,
        275,
        288,
        289,
        299,
        305,
        306,
        321,
        323,
        324,
        325,
        326,
        327,
        332,
        335,
        336,
        337,
        342,
        343,
        344,
        345,
    ],
    "already-fixed",
    RM
    + "These auditor-coverage, badge, sub-field, cross-corpus and schema exhibits (Ex. 6-21, Sec. 6.9-9.11) are no longer in the compiled paper.",
    S + "p5_evidence.tex",
)
put(
    [276, 277, 282, 300, 301, 302, 303, 307, 308, 340, 341],
    "already-fixed",
    RM
    + "The N2 / N10 / zvf130 panels and their eta^2 tables and Cohen's d values are no longer in the paper; Exhibit 3 (h2h2 Tinker panel) is reported on its own with seed-level statistics.",
    S + "p5_evidence.tex",
)
put(
    [278, 293],
    "already-fixed",
    RM
    + "The live-coupling 11.4-bit and placebo-replacement paragraphs sit after \\endinput in p5_stack.tex, so they are not compiled.",
    S + "p5_stack.tex",
)
put(
    [279, 280, 281, 297, 334, 338, 339],
    "already-fixed",
    RM
    + "Items 13-17 and the delta_div sections (p5_iter80/81/113) are no longer input. p5_results_intro.tex states that no CI gate for the retracted Items 13-17 is endorsed.",
    S + "p5_results_intro.tex",
)
put(
    [283, 309],
    "already-fixed",
    RM
    + "The entropy-based PLACEBO deprecation table is gone. p5_results_intro.tex states the paper does not deprecate model or temperature because a binary-factor entropy is below one bit.",
    S + "p5_results_intro.tex",
)
put(
    [294],
    "already-fixed",
    "Table tab:p5-minreport row 3 now lists 'Base-checkpoint identity and revision/hash' (added in 2a808b576), and the abstract calls it out. The Table 20 sub-field decomposition is removed.",
    S + "p5_stack.tex",
)
put([298], "already-fixed", RM + "The N2 panel no longer appears.", "")
put(
    [313, 314, 315, 316, 318],
    "already-fixed",
    RM
    + "Sections 2.1/2.4, Table 3, the Hedges-Olkin citation and Table 5 belonged to _shared_methods, which is no longer input.",
    "",
)
put(
    [285],
    "fixed",
    "Sections 2-3 were removed in 2a808b576. The remaining part (Sec. 1 'four exhibits ... plus Pillar-2 cross-library ZVF') did not match the evidence section's 'Only Exhibits 1-3'; the intro contribution list now says three exhibits.",
    S + "p5_intro.tex",
)
put(
    [295],
    "fixed",
    "Item 7 now defines PCD as the mean over groups of the ddof=0 within-group reward variance, equal to p_hat(1-p_hat) for binary rewards with expectation (G-1)/G p(1-p). It states the data (600 stored GSM8K groups, 3 sampling seeds x 200, G=8), gives the exact ZVF change 95/600 -> 0/600, PCD +2.98e-7, and the source TSV, and notes that a tolerance-based ZVF (ddof=1 var <= 1e-6) is unchanged. Source: experiments/results/pcd_vs_zvf_recomputed_summary.tsv, pcd_vs_zvf_tolerance_README.md.",
    S + "p5_stack.tex",
)
put(
    [296],
    "fixed",
    "Removed the unsourced 'roughly two-thirds of papers report only pass@1'. Added Yue et al. 2025 (arXiv:2504.13837, verified) as an inline prose reference, with a bib entry requested in P5_minreport_shared_requests.md. Stated the panel provenance: Qwen3-8B Tinker GRPO LoRA, GSM8K, 30 steps, lr 1e-5, one training seed per G; eval on 200 test problems x 32 samples, T=1.0, 512 tokens; W&B zvf-experiments-next; experiments/results/w2_groupsize_passk_wandb_export.tsv. On recompute the -0.5..+1.9 pass@1 and +1.5..+2.5 pass@32 deltas hold only against base eval seed s42. The text now reports the base eval-seed range (pass@1 26.6-30.6, pass@32 86.5-89.0), and the 'G=2 regression' reading is shown to sit inside evaluation noise.",
    S + "p5_stack.tex",
)
put(
    [310],
    "fixed",
    "Threat-model grades R0-R5 were rewritten to the field-severity scale that registry/query.py stackdiff actually computes (R1 nuisance, R2 runtime, R3 hyperparameter/budget, R4 algorithm, R5 outcome-critical incl. checkpoint; unknown fields go to coverage + underspecified flag). Toolchain, conclusion, limitations and the worked-example verdict now use the same single rubric, and the 'different mapping' caveat is removed.",
    S
    + "p5_threatmodel.tex;"
    + S
    + "p5_toolchain.tex;"
    + S
    + "p5_conclusion.tex;"
    + S
    + "p5_limitations.tex;"
    + S
    + "p5_worked_example.tex",
)
put(
    [311],
    "fixed",
    "Removed the 'August 2026 evidence boundary' paragraph after the abstract. Exhibit dates now match the run JSON (h2h2 runs started 2026-06-21; W&B upload 2026-07-11; zvf-audit June 2026).",
    S + "p5_abstract.tex",
)
put(
    [312],
    "fixed",
    "The 'Absorbed satellite (retired)' note was already gone (2a808b576). I removed the orphan 'August 2026 evidence boundary' paragraph. The \\thanks{Project guide.} footnote lives in shared _shared_author.tex and is a normal supervisor credit; it is noted in the shared-requests file with no change requested.",
    S + "p5_abstract.tex",
)
put(
    [317],
    "fixed",
    "Sec. 3.4/Table 5 were removed (2a808b576). In the remaining tab:p5-gsize, 'Llama-3.1-8B' is now 'Llama-3.1-8B-Instruct', matching the W&B zvf-audit config (meta-llama/Llama-3.1-8B-Instruct).",
    S + "p5_stack.tex",
)
put(
    [319],
    "fixed",
    "Recomputed from raw run JSON (tools/review_ledger/P5_minreport_exhibit3.py). The reviewer is right that +/-0.19 fails: the all-pairs unpaired TOST margin is 0.193 with SD pooled over the four arms (df=8), or 0.224 with pair-specific variances (df=4). The text now gives both, says the paired-contrast 90% CIs are the one-sided TOST bounds and are unadjusted for 6 comparisons, gives the 95% CIs that include zero ([-0.001,+0.067], [-0.003,+0.065]), and notes that contrasts use unrounded means. The spread is corrected from 0.034 to 0.033 (unrounded) in the abstract, intro, evidence and conclusion.",
    S + "p5_evidence.tex;" + S + "p5_abstract.tex",
)
put([320], "already-fixed", RM + "Table 18 is removed.", "")
put(
    [322],
    "fixed",
    "The unrounded last-10 values (0.85625, 0.3375, 0.05; framework_comparison.json, master_results.json wave6_w6_base_r32_t0.8_b2) give -0.51875 -> -0.519, so the reviewer's -0.518 comes from rounded inputs. The text now prints the unrounded values, and adds that the Instruct reference run is a T=0.8 ablation while the Base run's temperature is unrecorded.",
    S + "p5_evidence.tex",
)
put(
    [346],
    "wontfix-scope",
    "The related work already cites hochlehnert2025sober, li2026oxrl and liu2025tricks in the controlled-comparison paragraph. The Sec. 5/7.7/8.1 exhibits the suggestion targets were withdrawn, so there is no main-text framing left to qualify.",
    S + "p5_related.tex",
)
put(
    [347],
    "fixed",
    "The 'reporting-standards lineage (iter 109)' paragraph, which mapped Model Card/Datasheet fields onto non-existent MIN-REPORT items 9-14 and the retracted zvf_yield_residual, is replaced with a short documentation-standards paragraph that adds FactSheets (verified, inline prose; bib request filed). Yue et al. 2025 was added to Item 8. EvalCards was not added because I could not verify it.",
    S + "p5_related.tex;" + S + "p5_stack.tex",
)
put(
    [348],
    "fixed",
    "Sec. on the signer now cites in-toto (Torres-Arias et al., USENIX Security 2019; verified; inline prose, bib request filed). While checking, I found the verifier section described an older implementation (A-F badge, HMAC fallback, a self-test, an 'A (100%)' demo). I rewrote it to match the current minreport.py/sign.py and re-ran both demonstrations (tools/review_ledger/P5_minreport_verifier_demo.py): both repo records are REJECTED at 16% completeness; the seed rule passes campaign and fails sweep; the Ed25519 signature verifies pristine and fails on a forged git_sha. The 'A (100%)' claim is withdrawn in the text.",
    S + "p5_verifier_impl.tex",
)

con = sqlite3.connect(ROOT / "reports/deepreview/findings.sqlite")
ids = [r[0] for r in con.execute("select id from findings where paper='P5_minreport' order by id")]
missing = set(ids) - set(D)
extra = set(D) - set(ids)
assert not missing and not extra, (missing, extra)
out = ROOT / "reports/deepreview/dispositions/P5_minreport.jsonl"
with open(out, "w") as f:
    for i in ids:
        f.write(json.dumps(D[i]) + "\n")
from collections import Counter

print(Counter(d["status"] for d in D.values()), len(D))
