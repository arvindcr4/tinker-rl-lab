#!/usr/bin/env python3
"""Write P6_registry / P6_submission dispositions + ledgers (Deep Review 2026-10-18).

Scope note: P6 sources are owned by a concurrent Amp session; every edit this pass would
have made is recorded as wontfix-scope ("deferred: P6 owned by concurrent Amp session")
with the exact old->new text in reports/deepreview/dispositions/P6_amp_requests.md.
"""

import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
D = ROOT / "reports/deepreview/dispositions"
L = ROOT / "reports/deepreview/ledgers"
DEF = "deferred: P6 owned by concurrent Amp session; "
REQ = "reports/deepreview/dispositions/P6_amp_requests.md"
SH = "reports/deepreview/dispositions/P6_shared_requests.md"
SEC = "platform_hybrid/paper/sections/"
RM_REG = (
    "already-fixed",
    "75a500589 removed the section from paper_P6_registry.tex (iteration logs / _shared_methods no longer \\input); text absent from current build",
)
RM_SUB = (
    "already-fixed",
    "2a808b576 rewrote p6_measured_claimed.tex / p6_iter190_raw_recompute_audit.tex; reconciliation tables, claim tallies and Finding A-C no longer in build",
)


def af(ev):
    return ("already-fixed", ev)


def dfr(ev, req):
    return ("wontfix-scope", DEF + ev + f" (proposed fix: {REQ} {req})")


reg = {
    349: af(
        "75a500589: _shared_methods.tex (Table 5 legacy BH table, ANOVA, power table) no longer \\input in paper_P6_registry.tex"
    ),
    350: af(
        "75a500589 removed Sections 10-11; current §9 (p6_iter190) states the -0.261 vs -0.148 contrasts came from different summaries and retains neither. zvf130 source is a simulation projection: see R1"
    ),
    351: (
        "wontfix-scope",
        DEF
        + "paper text: delta_div sections not compiled since 75a500589 (already fixed in paper); residual registry schema/CLI antiherding blocks filed in "
        + SH,
    ),
    352: af("75a500589: p6_joint_controller_extension.tex not \\input"),
    353: af("75a500589: p6_measured_coverage.tex / iter74 not \\input"),
    354: dfr(
        "iteration logs removed in 75a500589, but current p6_results_intro says '20 variant-delta records (including two amendment sidecars)' vs p6_population '18 + 2 sidecars'; disk has 28 stack/18 delta/2 sidecar",
        "R3",
    ),
    355: af("75a500589: ANOVAs were in _shared_methods.tex, no longer \\input"),
    356: dfr(
        "benchmark boilerplate removed in 75a500589; residual dangling 'Section ??' refs (sec:benchmark, sec:setup undefined in current build)",
        "R3",
    ),
    357: dfr(
        "verified: framework_comparison.json Tinker Qwen3-8B-Base 0.85625, TRL Qwen3-8B 0.05, lr 1e-5 both; master_results wave6_w6_base_r32_t0.8_b2 Instruct-on-Tinker 0.3375 -> 64.3%; wording still ambiguous in current abstract/intro",
        "R2",
    ),
    358: dfr(
        "recomputed n2_metrics.tsv: last-10 deltas reproduced (gift loss var mean -1.65e4, delta -1.67e4); single seed, step-bootstrap CIs/daggers are pseudoreplication; Family D 'same envelope' still contradicts caption; aero arm executed GRPO (G fixed 8)",
        "R5",
    ),
    359: af("75a500589: p6_iter86_cross_stack_matrix.tex not \\input"),
    360: af("75a500589: p6_schema_extension.tex not \\input"),
    361: af(
        "current p6_measured_claimed.tex §8 states AERO ZVF delta is -0.025 and the OPPOSE verdict was a sign error (2a808b576); recompute -0.0250 confirms"
    ),
    362: af("75a500589: Table 19 (p6_registry_health/antiherding) not \\input"),
    363: af("75a500589: Tables 25/26 not \\input"),
    364: dfr(
        "window sections removed (75a500589) but last-10 reward significance still asserted in Table caption/synthesis; recomputed full40 deltas aero -0.007, gift +0.011, areal -0.006",
        "R5",
    ),
    365: af("75a500589: p6_iter74_drgrpo.tex not \\input"),
    366: af("75a500589: p6_registry_health.tex not \\input"),
    367: af(
        "75a500589: iter106 section not \\input; current Table has areal +30.13, gift -3.89 reproduced from n2_metrics.tsv"
    ),
    368: af("2a808b576/75a500589: claim tallies withdrawn in current §9"),
    369: af("2a808b576/75a500589: claim tallies withdrawn in current §9"),
    370: (
        "wontfix-scope",
        DEF
        + "MCTS/continuity text not in current build (already fixed in paper); registry delta_mcgrpo 'mcts_rollout' / delta_cppo 'continuity_penalty' component names remain, filed in "
        + SH,
    ),
    371: af("75a500589: iter126 tier section not \\input"),
    372: (
        "wontfix-scope",
        DEF
        + "paper text gone (75a500589) but registry delta_aero.json still cites le2025rlzvp; correct AERO is arXiv:2602.14338 (Z. Zhang et al., verified via arXiv search); filed "
        + SH
        + " and "
        + REQ
        + " R7",
    ),
    373: (
        "wontfix-scope",
        "shared references.bib: verified MO-GRPO = arXiv:2509.22047 (Y. Ichihara), GRPO-LEAD authors J. Zhang et al.; request in "
        + SH,
    ),
    374: (
        "refuted",
        "arXiv:2510.23868 is 'GIFT: Group-Relative Implicit Fine-Tuning Integrates GRPO with DPO and UNA' by Z. Wang (verified via arXiv search) -- reference correct. Mechanism mismatch is real but lies in the N2 implementation (variance_mitigation_integration.py 'simplified' baseline blend), recorded under R5",
    ),
    375: af("75a500589: power table in _shared_methods.tex, not \\input"),
    376: af("75a500589: _shared_methods.tex not \\input"),
    377: dfr(
        "current text retains internal 'Scope…retired' box, 'Submission target' sentence, 'Vehicle packaging'",
        "R3",
    ),
    378: af("75a500589: _shared_methods.tex not \\input"),
    379: af("75a500589: _shared_methods.tex not \\input"),
    380: af("75a500589: _shared_methods.tex not \\input"),
    381: af("75a500589: _shared_methods.tex not \\input"),
}
for i in range(382, 403):
    reg.setdefault(
        i,
        af(
            "75a500589: the cited section (iter logs / registry_health / schema_extension / iter190 tables) is no longer \\input in paper_P6_registry.tex"
        ),
    )
reg[403] = (
    "wontfix-scope",
    "suggestion; new citations need verification and shared bib edits; not required for correctness",
)
reg[404] = (
    "wontfix-scope",
    "suggestion; current related-resources already contrasts Engstrom, Liu (Tricks or Traps) and Hochlehnert; catalog expansion is future work",
)

sub = {
    761: dfr(
        "disk: 28 stack, 18 variant_delta, 2 sidecars; §2.2 'We ship three' and Listing layout (no ellipsis) still inconsistent",
        "R3",
    ),
    762: dfr(
        "R07 banner removed; Appendix C now 'retired'; residual internal notes ('Submission target', 'Scope' box)",
        "R3",
    ),
    763: (
        "wontfix-scope",
        "shared references.bib: verified arXiv:2509.22047 MO-GRPO (Y. Ichihara); request in " + SH,
    ),
    764: (
        "wontfix-scope",
        "shared references.bib: verified arXiv:2504.09696 GRPO-LEAD first author J. Zhang; request in "
        + SH,
    ),
    765: af(
        "2a808b576: hard-coded Listing 7 removed (p6_appendix_artifact.tex A.3 now points to regenerated table); p6_registry_generated.tex reproduced exactly by analysis_publication_fixes/p6_registry_numbers.py"
    ),
    766: af(
        "2a808b576: current §8 states AERO delta -0.025 and the OPPOSE verdict was a sign error; recompute confirms -0.0250"
    ),
    767: af(
        "2a808b576: Table 4/5 removed; current §9 says the two zvf130 summaries are not one estimand and neither is retained. Source is a simulation projection (R1)"
    ),
    768: (
        "wontfix-scope",
        DEF
        + "AERO arXiv:2602.14338 verified; AReaL arXiv:2505.24298, ES arXiv:2509.24372 named in registry entries; shared bib request in "
        + SH,
    ),
    769: dfr(
        "verified 64% = (0.85625-0.3375)/(0.85625-0.05) from framework_comparison.json + master_results.json wave6 run; Tinker=Base, TRL=Instruct",
        "R2",
    ),
    770: dfr(
        "verified: N2 'aero' ran with G fixed at 8 (controller capped g_max=8) so 'implemented' is false; areal/gift are surrogates; governance text lacks badge-version and measured[] rules",
        "R5, R6",
    ),
    771: dfr(
        "badge outcome-blindness already fixed (text now says excluded); 7 ci_method entries = 4 Family C + 3 Family D, whose Family C source string wrongly cites the N2 bootstrap",
        "R6",
    ),
    772: dfr(
        "listing prints sampling./reward. prefixes; delta_dapo.json notes say fields re-anchored to schema leaves",
        "R6",
    ),
    773: dfr("leaf counts from schema: 10,3,4,2,3,2,2 (26)", "R6"),
    774: dfr("Exec. column computed in proposed generated tex", "R6"),
    775: dfr(
        "matched rerun withdrawn from current build (75a500589). Truncation DEMONSTRATED: 97.5-99.5% of held-out generations hit 512-token cap for base and all adapters; all 6 trained adapters below base (0.270-0.335 vs 0.345); n=3 seeds, t-CIs [-0.488,+0.588] and [-0.109,+0.132]. Whether training hurts needs re-eval at larger cap (compute)",
        "R0",
    ),
    776: dfr(
        "verified: current 28-entry corpus has 7 normalized checkpoint ids (llama id normalizes inconsistently), not eight; 17 retained records have 5; N2 entries hand-entered (no run_config) although run script exists",
        "R1, R5",
    ),
    777: RM_SUB,
    778: af(
        "2a808b576: current §8 states component declarations must not be counted as claimant stacks; claimant counts withdrawn"
    ),
    779: dfr(
        "single-seed step bootstrap; current synthesis still says 'statistically detectable decreases'",
        "R5",
    ),
    780: dfr("intro still says '(closed)'", "R2"),
    781: ("wontfix-scope", "shared references.bib; request in " + SH),
    782: ("wontfix-scope", "shared references.bib; request in " + SH),
    783: ("wontfix-scope", "shared references.bib; request in " + SH),
    784: dfr("heading still after floated table", "R6"),
    785: dfr("strings come from zvf130 fixture entries, removed under R1", "R1"),
    786: af("2a808b576: singular 'variant_delta_applied' occurrences were in removed Appendix E"),
    787: dfr("en-dashes still render in prose flags", "R6"),
    788: dfr("double parenthetical still present", "R6"),
    789: af("2a808b576: Listing 7 removed"),
    790: af("75a500589: submission §5.2 KL example no longer prints a twelve-stack listing"),
    791: dfr("LeverTrace/trainer plugin still undefined", "R6"),
    792: af(
        "75a500589: submission no longer prints Listing 3 (registry-paper counterpart noted in R6)"
    ),
    793: RM_SUB,
    794: RM_SUB,
    795: RM_SUB,
    796: RM_SUB,
    797: RM_SUB,
    798: RM_SUB,
}
files_req = [REQ]
D.mkdir(parents=True, exist_ok=True)
L.mkdir(parents=True, exist_ok=True)
for paper, m, rng in (
    ("P6_registry", reg, range(349, 405)),
    ("P6_submission", sub, range(761, 799)),
):
    assert set(m) == set(rng), (paper, set(rng) ^ set(m))
    with (D / f"{paper}.jsonl").open("w") as f:
        for i in rng:
            st, ev = m[i]
            files = [REQ] if st == "wontfix-scope" else ([SEC] if st == "already-fixed" else [])
            f.write(json.dumps(dict(id=i, status=st, evidence=ev, files=files)) + "\n")

R = "platform_hybrid/experiments/results/"
RV = R + "review_2026-10-18/"
ledger = [
    (
        "abstract/intro",
        "17x same-label gap: Tinker final reward",
        "0.856",
        R + "framework_comparison.json",
        "experiments/results/aggregate_framework_comparison.py",
        "verified",
    ),
    (
        "abstract/intro",
        "17x same-label gap: TRL final reward",
        "0.050",
        R + "framework_comparison.json",
        "experiments/results/aggregate_framework_comparison.py",
        "verified",
    ),
    (
        "abstract/intro",
        "checkpoint switch share of gap",
        "64% ((0.85625-0.3375)/0.80625=0.643)",
        "platform_hybrid/experiments/master_results.json:wave6_w6_base_r32_t0.8_b2",
        "tools/review_ledger (hand arithmetic, recorded)",
        "recomputed",
    ),
    (
        "§4/Table 1",
        "seed stack entries",
        "28 (17 artifact-backed + 11 zvf130 fixtures)",
        RV + "P6_registry/p6_registry_numbers.json",
        "tools/review_ledger/P6_registry_numbers.py",
        "recomputed",
    ),
    (
        "Family E / Table 1",
        "zvf130_* stack records (11)",
        "simulation-projection fixture",
        R + "zvf_iter130_method_risk.tsv",
        "platform_modal/scripts/zvf_diagnostic_iter130.py",
        "unsourced",
    ),
    (
        "§5.3",
        "stackdiff distribution (28 records)",
        "R0 42,R2 1,R3 10,R4 23,R5 302; 378/378 underspecified",
        "platform_hybrid/paper/sections/p6_registry_generated.tex",
        "platform_hybrid/analysis_publication_fixes/p6_registry_numbers.py",
        "verified",
    ),
    (
        "§5.3",
        "stackdiff distribution (17 records, fixtures excluded)",
        "136 pairs: R0 6,R2 1,R3 1,R4 23,R5 105; median cov 0.20",
        RV + "P6_registry/p6_registry_numbers.json",
        "tools/review_ledger/P6_registry_numbers.py",
        "recomputed",
    ),
    (
        "§5.3",
        "base models spanned",
        "text says 8; 7 normalized ids (28 recs), 5 (17 recs)",
        RV + "P6_registry/p6_registry_numbers.json",
        "tools/review_ledger/P6_registry_numbers.py",
        "recomputed",
    ),
    (
        "§6.2",
        "reference_kl census",
        "text 7/12 FULL; 17 recs: FULL 11, PARTIAL 5, UNREPORTED 1",
        RV + "P6_registry/p6_registry_numbers.json",
        "tools/review_ledger/P6_registry_numbers.py",
        "recomputed",
    ),
    (
        "§3.3",
        "ci_method populated entries",
        "7 (4 Family C + 3 Family D)",
        RV + "P6_registry/p6_registry_numbers.json",
        "tools/review_ledger/P6_registry_numbers.py",
        "recomputed",
    ),
    (
        "Eq.1",
        "badge leaf denominators",
        "10,3,4,2,3,2,2 (26)",
        "platform_hybrid/registry/schema.json",
        "tools/review_ledger/P6_registry_numbers.py",
        "recomputed",
    ),
    (
        "§2/A.1",
        "variant-delta records on disk",
        "18 + 2 sidecars",
        "platform_hybrid/registry/entries/",
        "platform_hybrid/registry/query.py load()",
        "verified",
    ),
]
for metric, vals in (
    ("reward_mean", "-0.014/+0.016/-0.020"),
    ("zvf", "-0.025/+0.125/-0.056"),
    ("mean_len", "+31.08/-3.89/+30.13"),
    ("cv_len", "-0.034/+0.002/-0.038"),
    ("loss gift", "-1.67e4 (var mean -1.65e4)"),
):
    ledger.append(
        (
            "Table N2 (last-10)",
            f"{metric} delta aero/gift/areal",
            vals,
            R + "n2_reward_tensor_resume/n2_metrics.tsv",
            "tools/review_ledger/P6_registry_n2_recompute.py",
            "recomputed",
        )
    )
ledger += [
    (
        "Table N2",
        "95% step-bootstrap CIs and significance marks",
        "removed (single seed)",
        R + "n2_reward_tensor_resume/n2_metrics.tsv",
        "tools/review_ledger/P6_registry_n2_recompute.py",
        "removed",
    ),
    (
        "Table N2",
        "per-prompt pooled reward deltas -0.007/+0.011/-0.006",
        "not reproducible (full40 -0.005/+0.011/-0.006)",
        RV + "P6_registry/n2_descriptive_deltas.tsv",
        "tools/review_ledger/P6_registry_n2_recompute.py",
        "unsourced",
    ),
    (
        "Table N2 (full-40)",
        "reward_mean delta aero/gift/areal",
        "-0.007/+0.011/-0.006",
        RV + "P6_registry/n2_descriptive_deltas.tsv",
        "tools/review_ledger/P6_registry_n2_recompute.py",
        "recomputed",
    ),
    (
        "Family D",
        "N2 aero group size",
        "8 at all 40 steps (controller g_max=8)",
        R + "n2_reward_tensor_resume/n2_metrics.tsv",
        "platform_hybrid/experiments/tinker-runs/scripts/n2_reward_tensor_20260704.py",
        "verified",
    ),
]
for p in ("P6_registry", "P6_submission"):
    rows = list(ledger)
    if p == "P6_submission":
        rows += [
            (
                "§5.2 (reviewed)",
                "matched rerun held-out truncation share",
                "0.975-0.995 at 512 tokens",
                R + "publication_fixes/matched_trl_tinker/eval_heldout.json",
                "tools/review_ledger/P6_submission_matched_truncation.py",
                "recomputed",
            ),
            (
                "§5.2 (reviewed)",
                "held-out acc base vs trained adapters",
                "0.345 vs 0.270-0.335 (all below base)",
                R + "publication_fixes/matched_trl_tinker/eval_heldout.json",
                "tools/review_ledger/P6_submission_matched_truncation.py",
                "recomputed",
            ),
            (
                "§5.2 (reviewed)",
                "last-10 train gap Tinker-TRL (n=3 seeds)",
                "+0.050, t-CI [-0.488,+0.588]",
                R + "publication_fixes/matched_trl_tinker/matched_trl_tinker_result.json",
                "tools/review_ledger/P6_submission_matched_truncation.py",
                "verified",
            ),
            (
                "§5.2 (reviewed)",
                "held-out gap Tinker-TRL (n=3 seeds)",
                "+0.012, t-CI [-0.109,+0.132]; equivalence claim withdrawn",
                R + "publication_fixes/matched_trl_tinker/matched_trl_tinker_result.json",
                "tools/review_ledger/P6_submission_matched_truncation.py",
                "removed",
            ),
        ]
    with (L / f"{p}.tsv").open("w") as f:
        f.write("location\tclaim_text\tvalue\tsource_artifact\tgenerator_script\tstatus\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
print("ok")
