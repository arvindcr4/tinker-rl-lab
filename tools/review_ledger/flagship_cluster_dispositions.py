"""Write dispositions JSONL and claim-ledger TSVs for the Flagship cluster (2026-10-18 deep review).

Papers: Flagship_Main, Flagship_TMLR, Next_Submission, Spectral_Entropy.
Usage (repo root): python3 tools/review_ledger/flagship_cluster_dispositions.py
"""
import json
from pathlib import Path

FM = "zvf-program/flagship/paper/main.tex"
FT = "zvf-program/flagship/paper/tmlr_submission/main.tex"
FB = "zvf-program/flagship/paper/flagship.bib;zvf-program/flagship/paper/tmlr_submission/flagship.bib"
NS = "zvf-program/next-submission/paper/main.tex"
NB = "zvf-program/next-submission/paper/refs.bib"
SE = "zvf-program/flagship/paper/spectral_entropy_paper_kdense_revision.tex"
R = "platform_hybrid/experiments/results/review_2026-10-18"
T3 = f"{R}/Flagship_Main/table3_recompute.json (tools/review_ledger/Flagship_Main_table3_recompute.py)"
S1 = "review_bundle.zip frozen zvf-program/flagship/s1/{reference.py,fixtures.py,receipt.py,results/*_receipt.json}"
SUP = "zvf-program/flagship/pilot/launch-v2-corpus-resume-r4-2/{supervisor_state.json,execution-notes.md 2026-07-23 root-cause}"
PRE = "zvf-program/flagship/pilot_preregistration.json runtime.execution_contract; pilot/remote_unit.py (hash matches acceptance manifests)"
TMLR_NOTE = "; TMLR main.tex regenerated from canonical main.tex via tmlr_submission/build_submission.make_tmlr_source"

shared = {  # flagship fixes applied identically to Main and TMLR
    "cite": ("fixed", "Added verified citations (arXiv API: 2503.20783 Dr.GRPO, 2507.18071 GSPO, 2602.14338 AERO; TRL repo) and stated that each S1 arm is a fixture-level encoding (DAPO = upper clip only; DrGRPO = no std scaling but per-completion reduction; AERO = posterior zero-variance branch only), verified in frozen s1/reference.py", f"{FM};{FT};{FB}"),
    "units": ("fixed", "Caption (WIP) plus renamed columns 'Intended fixture cases / Native material arms / Native not-tested arms'; Appendix C now states one native fixture per arm and that GRPO-selected has no native verdict (verified: trl/verl receipts native_cases has 5 rows, fixtures base,dapo_clip,base,base,aero_posterior)", f"{FM};{FT}"),
    "spec": ("fixed", "Specified S as positive-mask rows, exclusion of unselected/zero-mask rows, empty-S rejection, singleton s_g:=0, per-completion reduction for all arms incl. DAPO/DrGRPO, GSPO max{1,.} guard, S1 gradient wrt logps, explicit L_global (new Eq.), epsilon_only floor branch, s^all notation; all verified against frozen s1/reference.py and pilot/provenance/r4-2-objective.py", f"{FM};{FT}"),
    "grad": ("fixed", "All Table 3 numbers recomputed exactly from the six acceptance records. Added: fixed counterfactual pair (g_intended,g_native) at unit's own params; rel-l2 = ||g_int-g_nat||/||g_int|| (training.py _relative_l2); equal active length (4096 tokens in 98/100 steps per unit) => intended==reduction and epsilon==native objectives; analytic scalar bound 2.8e-4 vs stored 0.0177 (unexplained, not interpreted; bf16 backward noted); per-unit joint counts 69,69,70,69,70,69; all nonzero failures rel-l2 only; cosine bound non-binding (sin bound); registered epsilon_only gate 69/100 and 70/100 (selected-vs-intended, new, recomputed); reduction_only selected==intended on 200/200. Reviewer's inference that lengths vary within groups is refuted by active_tokens", f"{FM};{FT};{T3}"),
    "cv": ("fixed", "Added frozen decoding/optimizer/LoRA contract (temperature 1.0, top-p 1.0, top-k 0, thinking mode on, 512 tokens, rev 70d244cc, LoRA r16/a32, AdamW lr 1e-6, wd 0, clip 1.0, KL beta 0) and CV=0 root cause from execution notes (completions fill 512 cap); only seed-11 filtered corpus executed (attempts=1), seeds 23/37 corpora and 12 units attempts=0", f"{FM};{FT};{SUP};{PRE}"),
}

dispo = {
    "Flagship_Main": [
        (830, *shared["cite"]),
        (831, "fixed", "Claim discipline: WIP added negative labels; now maps Table 6 abbreviations (Sourced/Proved/Verified) to the four types and names the only proposed item; Table 6 labelled and introduced. Provenance: verify_claims.py re-hashes every acceptance-manifest source against the bundle (verified by running bundle verifier: PASS); Discussion and Reproducibility now state that checkpoints, raw corpora and predictions are withheld and only format-checked. Note: verifier fails on the live working tree (s1 sources drifted 2026-10-18) but passes on review_bundle.zip", FM),
        (832, *shared["units"]),
        (833, *shared["spec"]),
        (834, *shared["grad"]),
        (835, *shared["cv"]),
        (836, "fixed", "WIP had partially rephrased; abstract now 'Near a binary reward boundary, a larger fixed group size reduces...'", FM),
        (837, "fixed", "WIP defined s_R and epsilon; renamed stabilizer to varepsilon_std and stated additive-vs-floor distinction", FM),
        (838, "fixed", "Stated unresolved = {noisy, missing, delayed} -> recheck with 0 rollouts; Wilson z=1.959964 on pooled history+group counts with n=2 and no history; drop branches unexercised (UB(0,2)=0.658, LB(2,2)=0.342 from trl_receipt.json); targets 16 / default 8", FM),
        (839, "fixed", "WIP identified beta as KL coefficient; added AdamW (beta1,beta2)=(0.9,0.999) PyTorch default (remote_training.py passes none) and corrected rationale (zero-grad AdamW step would still move params via m)", FM),
        (840, "fixed", "Total row now '10/14/1/2/4 (Table 2 order)'", FM),
        (841, "fixed", "Added introductory sentence and label tab:status to Appendix E", FM),
        (842, "wontfix-scope", "Scope critique; paper already states corollaries are elementary and the witness restricted (Contributions item 2, Sec 4.2/4.3 closing paragraphs, Limitations). No change needed", FM),
        (843, "fixed", "he2026advantage -> ICML 2026 (arXiv:2605.21125 comment 'Accepted at ICML 2026'); PMLR volume/pages not verified so not added", FB),
        (844, "refuted", "arXiv metadata lists 'Zhang Zijian' (family-name-first form); bib 'Zhang, Zijian' already has family=Zhang, consistent with other authors. No change", FB),
        (845, "fixed", "yan2025verifybench -> ICLR 2026 (arXiv:2505.15801 comment 'ICLR 2026: openreview JfsjGmuFxz')", FB),
        (846, "already-fixed", "Commit 2a808b576 (2026-10-18 20:04, after review) changed 'its' to 'their'", FM),
        (847, "fixed", "'base (rows 0,1,3)'", FM),
    ],
    "Flagship_TMLR": [
        (904, "fixed", "WIP defined s_R/epsilon; renamed stabilizer varepsilon_std (distinct from success probability epsilon) and stated additive-vs-floor distinction" + TMLR_NOTE, FT),
        (905, shared["cite"][0], shared["cite"][1] + "; TRL cited (vonwerra2020trl)" + TMLR_NOTE, shared["cite"][2]),
        (906, shared["spec"][0], shared["spec"][1] + TMLR_NOTE, FT),
        (907, shared["units"][0], shared["units"][1] + TMLR_NOTE, FT),
        (908, shared["grad"][0], shared["grad"][1] + "; also replaced TMLR's incorrect 'discrepancy driven by scaling' sentence" + TMLR_NOTE, f"{FT};{T3}"),
        (909, shared["cv"][0], shared["cv"][1] + TMLR_NOTE, FT),
        (910, "fixed", "Abstract rephrased (boundary regime) and Delta introduced as net expected value of the prepaid batch under the solvable state" + TMLR_NOTE, FT),
        (911, "fixed", "Same as Flagship_Main 838: unresolved set, z, (k,n), targets 16 / default 8, unexercised drop branches" + TMLR_NOTE, FT),
        (912, "fixed", "epsilon_only floor branch explicit; differentiation targets (S1: logps; campaign: adapter params) stated; beta = KL coefficient with AdamW betas (0.9,0.999)" + TMLR_NOTE, FT),
        (913, "fixed", "O_G subscripts in proof of Proposition 2" + TMLR_NOTE, FT),
        (914, "fixed", "Claim-discipline mapping to Table 6 labels plus Appendix E intro sentence" + TMLR_NOTE, FT),
        (915, "fixed", "Total row reordered and labelled" + TMLR_NOTE, FT),
        (916, "fixed", "Stated S1 uses its comparison tolerance 1e-8 as zero threshold (reference.py: std > ATOL) and that for binary G=8 rewards nonzero std >= sqrt(7/56)=0.354, so both thresholds select the same branch" + TMLR_NOTE, FT),
        (917, "wontfix-scope", "Scope critique; restricted nature already stated in contributions and Sec 4", FT),
        (918, "fixed", "native_trl uses s^all_g notation" + TMLR_NOTE, FT),
        (919, "fixed", "'base (rows 0,1,3)'" + TMLR_NOTE, FT),
        (920, "fixed", "Appendix E introductory sentence" + TMLR_NOTE, FT),
        (921, "fixed", "Added Engstrom et al. ICLR 2020 (arXiv:2005.12729) and Zheng et al. 2025 GRESO (arXiv:2506.02177), both verified via arXiv API" + TMLR_NOTE, f"{FT};{FB}"),
    ],
    "Next_Submission": [
        (922, "fixed", "Added batch geometry from frozen artifacts: 16 completion slots = 2 prompt groups per update, 60 groups per unit (A004); adapter (bound version, commit 81eb211c) keeps 8 slots per prompt, skipped groups = 8 inactive single-EOS slots with zero mask and no reward, no replacement prompt. Loss-denominator and all-inactive optimizer.step semantics are not fixed by any frozen artifact; disclosed as pre-execution requirements", NS),
        (923, "fixed", "Wrote A003 objective: scale_rewards=group (std normalization, so 'centered only' was incomplete), loss_type grpo, symmetric clip 0.2, token-level IS, beta 0, num_iterations 1, mask_truncated False; optimizer/precision/LoRA values; defined data-policy telemetry as consistency check under one inner iteration. Source: protocol_amendment_003 change block", NS),
        (924, "fixed", "Reassessment text now 'intended to'; Limitations states per-pair sign flips estimate sigma^2+mu^2 and requires consistent orientation inside a routine releasing only the variance and seed count; mechanism still unspecified in protocol (disclosed)", NS),
        (925, "fixed", "Recomputed exact noncentral-t: power 0.820 at n=19, 0.793 at n=18; inflation multiplies seed count ceil(19*1.2)=23 (verify_design.py line 198); power/precision gate defined as the required power receipt (no numeric precision gate in protocol); cost alternative disclosed as unspecified (WIP Limitations)", NS),
        (926, "fixed", "Stated joint rule is an IUT across tasks so Holm is conservative; listed missing bootstrap variant/resamples, uniform vs step-down level, and bootstrap-vs-t adjudication as pre-execution requirements", NS),
        (927, "fixed", "Table 3 caption discloses unresolved baseline-row, CI-level, cost-scale and power-receipt conventions (results_contract.json main_table_columns has none)", NS),
        (928, "already-fixed", "Commit 2a808b576 (after review) added joint-power (0.8^4, dependence caveat) and update-sparsity paragraphs to Limitations", NS),
        (929, "fixed", "Added stop-rate/cost relations (1-0.75a, a>=4/15; iid agreement >=1/2 => >=37.5% completion saving); ZVF-under-intervention note; disclosed that false-homogeneity denominator and parser-disagreement comparator are not frozen in any artifact", NS),
        (930, "fixed", "Clipping field defined as cumulative truncated/generated (remote_preflight.py); confirmatory scope/status disclosed as unfixed; PRNG: seeded per run (seed=data_seed=run seed), probes not coupled across arms after first expansion (bound adapter generation order)", NS),
        (931, "fixed", "WIP added geometric-mean note; now states log-scale bound <= ln0.8 = -0.2231 equivalently -0.20 after exp(.)-1 and the 18.1% pitfall", NS),
        (932, "already-fixed", "Commit 2a808b576 (after review) stated familywise 0.025 and described E1; now also names zvf-program/audit and 'worst' SD", NS),
        (933, "fixed", "A001 amended_rule quoted (final unbraced numeric atom after \\boxed whitespace; rows 888,1011); amendment gives no lexical grammar, stated explicitly", NS),
        (934, "fixed", "Table 2 row label 'MATH (MATH-500 eval)'", NS),
        (935, "fixed", "Corrected two wrong prefixes (exec authorization 8f13c393 -> bound/actual 43e7d3cb; remote preflight dafa185f -> bound 2cc7b68f, dafa185f marked superseded per A004), added preregistration 4c0c318b and canonical e3403617, A004 row, Qwen3-8B revision b968826d, gsm8k config main; held-out manifest hashes are not bound anywhere (stated). Also disclosed binding drift: 9 bound files differ at HEAD and verify_design.py fails ('power implementation hash drift'); bound versions found in git history", f"{NS};{R}/Next_Submission/binding_drift.json"),
        (936, "wontfix-scope", "Pre-results preregistration skeleton by design; no change", NS),
        (937, "fixed", "'are all identical'", NS),
        (938, "fixed", "Stated E1 seeds and that seeds+reserve are the 24 consecutive primes 211..349 (verified by script)", NS),
        (939, "fixed", "Artifact table now a numbered, captioned table (tab:artifacts)", NS),
        (940, "fixed", "Labelled as Hugging Face Hub Git commit identifiers", NS),
        (941, "fixed", "Added Kim et al. 2026 Pilot-Commit (arXiv:2605.26606, verified; description from its abstract)", f"{NS};{NB}"),
    ],
    "Spectral_Entropy": [
        (878, "fixed", "Partly refuted: repository objective (pilot/objective.py spectral_legendre_grpo/entropic_givens_grpo) DOES couple lambda*A_aux explicitly at the advantage level (detached), and eliminate_noise_components already raises unless 1<=N_noise<D. Repair list rewritten as 'Implementation status and required changes' with verified defects only (padded grid; unnormalized per-row gate tau=0.5 that never fires; batch-mean gate in GivensEntropyGate); gate formula defined; Conclusion now lists defects", SE),
        (879, "fixed", "Abstract adds conditional-independence qualifier", SE),
        (880, "fixed", "Contribution 4 and Sec 3 state conditional independence; correlated sampling added to procedural-change list", SE),
        (881, "fixed", "Identified superseded July 2026 draft (spectral_entropy_paper.tex body after \\endinput) and listed its withdrawn claims (64.2%->1.5% ZVF, benchmark/sample-efficiency gains, gradient-fidelity figure, attention-level Givens gate as mitigation)", SE),
        (882, "unsourced-removed", "The 77-case audit bundle (SHA-256 dbd1386a...) is absent from repo and disk (find over /, grep; also flagged in autoresearch/.../statistical_refutation.md sec 6 and claim_inventory J5). Hash/bundle claims removed; Appendix A points only to repository files; no public DOI exists (stated)", SE),
        (883, "fixed", "Added analytic mechanism (2/L weight vs 2/(L-1) spacing, endpoint weights => first-order; padded-grid domain contraction; rank <= L_eff), defined L_eff,i, wrote active-token trapezoid formula. Table 1 (from absent bundle) removed", SE),
        (884, "fixed", "Specified input (per-token feature vector, last dim), index map q_k=D-1-k, p_k=max(0,q_k-N_noise), sequential angles, non-interference, raise outside [1,D-1] (entropic_gating.py eliminate_noise_components); gate entropy formula", SE),
        (885, "fixed", "Code-level cause measured: per-row softmax entropy 6.24-7.62 >> tau=0.5 on all 9 fixtures, so Givens branch never fires and gated==ungated gradients bit-identical; surrogate and differentiation target (fixture logps) written out; forward-path cosines (absent bundle) removed", f"{SE};{R}/Spectral_Entropy/lambda_sweep.json"),
        (886, "unsourced-removed", "Table 2 values came from the absent bundle; replaced by a recomputed sweep from repository code (6 lambdas incl. 1e-3; G=8,L=512 fixture, seed 42; reward-gradient norm 0.014640; slope 0.014267; linearity <3e-15 over 9 configs). Old 3.7351 slope and cosines removed", f"{SE};{R}/Spectral_Entropy/lambda_sweep.json (tools/review_ledger/Spectral_Entropy_lambda_sweep.py)"),
        (887, "fixed", "Partly refuted: exact noncentral-t MDE at n=5, alpha=.05 is 1.68 s_d (MC check power 0.801), not reviewer's 1.79; power at 1.62 is 0.77 not 0.721. Reviewer's 2.48 s_d at alpha/4 confirmed. Text now gives 1.62 (Eq.8), 1.68 exact, 2.48 at Holm level; states four primary contrasts are not yet enumerated (not invented)", SE),
        (888, "fixed", "H1 gate rewritten: completion-level labels (branched pass@k or within-group-varying verifier-corrected success), within-group residualization on length and entropy, one-sided AUC lower bound >0.5 separately for all-correct and all-wrong groups", SE),
        (889, "wontfix-scope", "Novelty critique; paper already claims no novelty for constituents and makes no policy-learning claim", SE),
        (890, "fixed", "pass@$k$", SE),
        (891, "fixed", "alpha=0.05", SE),
        (892, "fixed", "lambda=0.1 in Appendix table", SE),
        (893, "fixed", "Braced Wasserstein/Legendre/Givens/Group Relative Policy Optimization/GSPO in flagship.bib; He et al. ICML 2026 (PMLR pages unverified, not added); Nie author order refuted (arXiv 'Zhang Zijian' is family-first)", FB),
        (894, "unsourced-removed", "2e-17 / 9.6e-4-6.2e-2 / 1e-17 numbers came from the absent bundle and were removed; text now states exact-arithmetic energy statement (projection removes zero energy; truncation removes sum x_q^2)", SE),
        (895, "unsourced-removed", "Table 1 (Gram/operator/coefficient errors) from absent bundle removed with the 0.885%/0.600%/0.00203%/296x/rank-4/1.1e19/153% prose; replaced by analytic statements", SE),
        (896, "fixed", "s_r and s_s Bessel-corrected; epsilon=0 with 1e-12 zero branch, epsilon_s=1e-4 with floor; 'active rows' defined; detachment stated at Eq.3 (objective.py)", SE),
        (897, "fixed", "Appendix A table rewritten: one row per measurement family pointing to existing repository files and generators; literature row and absent-bundle pointers removed", SE),
        (898, "fixed", "Hyphenation fixed within rewritten H1", SE),
        (899, "fixed", "GRPO capitalized", SE),
        (900, "refuted", "Source text of 'methods proposal' is plain ASCII and the tectonic-rebuilt PDF text extracts cleanly; artifact not reproducible from source", SE),
        (901, "fixed", "Feature coordinates renamed p,q and rotation R(p,q,theta); atan2 upright", SE),
        (902, "unsourced-removed", "Float32 absolute-error sentence (absent bundle) removed", SE),
        (903, "unsourced-removed", "Table 1 removed (absent bundle)", SE),
    ],
}

ledgers = {
    "Flagship_Main": [
        ("tab:runs", "Table 3 all cells (start/final/joint0/min cos/max rel-l2/tokens)", "as printed", T3, "tools/review_ledger/Flagship_Main_table3_recompute.py", "verified"),
        ("Sec5.3", "intended-full s23 joint criterion", "69/100 (65 joint-zero + 4/35 nonzero)", T3, "tools/review_ledger/Flagship_Main_table3_recompute.py", "verified"),
        ("Sec5.3", "joint intended/native counts per row", "69,69,70,69,70,69", T3, "tools/review_ledger/Flagship_Main_table3_recompute.py", "recomputed"),
        ("Sec5.3", "nonzero failures due to rel-l2 only; cos fails", "31 rel-l2, 0 cos (s23 intended)", T3, "tools/review_ledger/Flagship_Main_table3_recompute.py", "recomputed"),
        ("Sec5.3", "epsilon_only selected-vs-intended gate", "69/100 (s11), 70/100 (s23), max rel-l2 0.0168", T3, "tools/review_ledger/Flagship_Main_table3_recompute.py", "recomputed"),
        ("Sec5.3", "reduction_only selected==intended", "200/200 receipts (cos 1, rel-l2 0)", T3, "tools/review_ledger/Flagship_Main_table3_recompute.py", "recomputed"),
        ("Sec5.3", "active tokens 4096 in 98/100 steps per unit", "98/100", T3, "tools/review_ledger/Flagship_Main_table3_recompute.py", "recomputed"),
        ("Sec5.3", "analytic scalar rel-l2 bound", "1e-4/(sqrt(7/56)+1e-4)=2.8e-4", "analytic", "-", "recomputed"),
        ("Sec5.1", "decoding/optimizer/LoRA contract", "T=1.0 top-p 1.0 top-k 0 thinking on 512; lr 1e-6; LoRA r16 a32; rev 70d244cc", PRE, "-", "verified"),
        ("Sec5.2", "filtered CV root cause and attempts", "CV 0.000000; s11 attempts 1; s23/s37 attempts 0", SUP, "-", "verified"),
        ("Sec3.1", "Wilson bounds UB(0,2), LB(2,2); z", "0.658, 0.342; 1.959964", S1, "-", "verified"),
        ("Sec3.1", "min nonzero binary std G=8", "sqrt(7/56)=0.354", "analytic", "-", "recomputed"),
        ("tab:s1", "intended 14 cases; native 4/1 and 1/4 arms", "14; 4/1; 1/4", S1, "review_bundle verify_claims.py (PASS)", "verified"),
        ("tab:crosswalk", "state counts", "10/14/1/2/4", SUP, "-", "verified"),
    ],
    "Next_Submission": [
        ("Sec4.3", "exact-t power at n=19 / n=18 (sd 0.0128285, margin 0.01, alpha 0.0125)", "0.820 / 0.793", "zvf-program/next-submission/preregistration.json power_plan", "scipy noncentral t (review recompute)", "recomputed"),
        ("Sec4.3", "inflated seed count", "ceil(19*1.2)=23", "verify_design.py expected_planning_seed_count", "-", "verified"),
        ("Sec3.3", "completion ratio / stop-rate threshold / iid bound", "1-0.75a; 4/15; >=37.5%", "analytic", "-", "recomputed"),
        ("Sec4.2", "ln 0.8; exp(-0.2)-1", "-0.2231; -18.1%", "analytic", "-", "recomputed"),
        ("Sec3.2", "batch geometry", "2 groups/update; 60 groups/unit", "protocol_amendment_004 implementation_path; preregistration treatment", "-", "verified"),
        ("Sec3.4", "Qwen3-8B revision", "b968826d", "protocol_amendment_002 source_basis; remote_preflight.py MODEL_REVISION", "-", "verified"),
        ("Sec4.1", "seeds are consecutive primes 211..349", "24 primes", "preregistration paired_seed_plan", "review script", "recomputed"),
        ("AppA.1", "artifact SHA-256 prefixes", "see table", "preregistration.json bindings; files", "tools/review_ledger/Next_Submission_binding_drift.py", "verified"),
        ("AppA.1", "binding drift at HEAD", "9 files drift; verify_design fails", f"{R}/Next_Submission/binding_drift.json", "tools/review_ledger/Next_Submission_binding_drift.py", "recomputed"),
        ("Sec4.4", "A001 rows 888,1011; 7500 retained", "2 rows", "protocol_amendment_001", "-", "verified"),
    ],
    "Spectral_Entropy": [
        ("tab:lambda", "homogeneous grad norm / cosine sweep (G=8,L=512)", "0.014267*lambda; cos 0.995248@0.1, 0.696798@1", f"{R}/Spectral_Entropy/lambda_sweep.json", "tools/review_ledger/Spectral_Entropy_lambda_sweep.py", "recomputed"),
        ("Sec6", "slope range, linearity, cosine ranges over 9 configs", "0.0052-0.0188; <3e-15; 0.9952-0.9954; 0.683-0.716", f"{R}/Spectral_Entropy/lambda_sweep.json", "tools/review_ledger/Spectral_Entropy_lambda_sweep.py", "recomputed"),
        ("Sec6", "gate row entropy range; gated==ungated", "6.24-7.62; True", f"{R}/Spectral_Entropy/lambda_sweep.json", "tools/review_ledger/Spectral_Entropy_lambda_sweep.py", "recomputed"),
        ("Sec6", "nine-condition ledger FLOP ratios; identical gated/ungated", "4.48x, 5.48x", "zvf-program/flagship/pilot/spectral_benchmark_results.json", f"pilot/benchmark_spectral_harness.py (rerun {R}/Spectral_Entropy/spectral_benchmark_rerun.json identical)", "verified"),
        ("Sec6", "16 focused tests pass", "16 passed", "pilot/test_spectral_attention.py etc.", "pytest", "verified"),
        ("Sec7", "MDE at n=5", "1.62 (Eq.8), 1.68 exact, 2.48 at 0.0125", "analytic", "scipy noncentral t + MC (review recompute)", "recomputed"),
        ("old Tab1", "Gram/operator/coefficient/trapezoid errors, 296x, rank 4, 1.1e19, 153%", "-", "absent 77-case bundle dbd1386a", "-", "removed"),
        ("old Sec5", "norm errors 2.61e-16, 1.52e-8, 1.20e-7, 11/12, 2e-17, 9.6e-4-6.2e-2, >3", "-", "absent 77-case bundle", "-", "removed"),
        ("old Tab2", "3.7351*lambda sweep and cosines; forward-path cosines 0.9504/0.2620; 77 cases", "-", "absent 77-case bundle", "-", "removed"),
        ("Abstract", "1e-17 vs 1e-3-1e-2", "-", "absent 77-case bundle", "-", "removed"),
    ],
}
ledgers["Flagship_TMLR"] = [(loc, c, v, s, g, st) for (loc, c, v, s, g, st) in ledgers["Flagship_Main"]]

ddir = Path("reports/deepreview/dispositions")
ldir = Path("reports/deepreview/ledgers")
ddir.mkdir(parents=True, exist_ok=True)
ldir.mkdir(parents=True, exist_ok=True)
for paper, rows in dispo.items():
    with (ddir / f"{paper}.jsonl").open("w") as f:
        for i, st, ev, files in rows:
            f.write(json.dumps({"id": i, "status": st, "evidence": ev, "files": files}) + "\n")
for paper, rows in ledgers.items():
    with (ldir / f"{paper}.tsv").open("w") as f:
        f.write("location\tclaim_text\tvalue\tsource_artifact\tgenerator_script\tstatus\n")
        for r in rows:
            f.write("\t".join(x.replace("\t", " ") for x in r) + "\n")
print({p: len(r) for p, r in dispo.items()})
