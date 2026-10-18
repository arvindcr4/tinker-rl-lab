"""Write dispositions JSONL for the Sem3 cluster (Group 6 report + NeurIPS main track).

Usage: python3 tools/review_ledger/Sem3_write_dispositions.py <findings.sqlite>
Asserts every finding id of both papers is covered exactly once.
"""
import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports/deepreview/dispositions"
G6 = "platform_tinker/reports/combined/base_final.tex"
G6B = "platform_tinker/reports/combined/refs.bib"
NT = "platform_hybrid/paper/paper.tex"
NTR = "platform_hybrid/paper/sections/related_work_v2.tex"
MR = "platform_hybrid/experiments/results/review_2026-10-18/Sem3_Group6_Original_Report/measured_runs.json"
FROZEN = ("Reviewed PDF is the frozen Apr-2026 deliverable (no LaTeX source in repo; committed as binary grup6.pdf in 8695a1d22). "
          "Fixes applied to the closest maintained source, " + G6 + ". ")
NFROZEN = ("Reviewed PDF is the frozen Apr-23 submission; maintained descendant is " + NT + ". ")

g6 = {
1015: ("fixed", "Source now builds with tectonic (bibtex run); 0 undefined citations, References chapter present.", [G6]),
1016: ("fixed", "Citations in Table 1.1 resolve after bibliography build (0 undefined cites).", [G6]),
1017: ("fixed", "Worked example now labels 3.3/3.6/4.5 as surrogate objective values; gradient wrt r_t is A=3 unclipped and 0 when clipped (A>0, r>1+eps).", [G6]),
1018: ("fixed", "Proposition and Algorithm 2.1 now keep KL as a separate loss term (not folded into rewards before normalization), match DeepSeekMath single-update form, note ratio/clip omitted; derivation notes mu_G includes own reward (biased) and cites RLOO.", [G6, G6B]),
1019: ("unsourced-removed", "Fig 3.4 SFT bars (30.2/19.4/32.5) had no source and contradicted Table 3.1; figure removed.", [G6]),
1020: ("fixed", "Sec 3.3 paragraph rewritten: SWE-bench 1.7% (original) vs 49% (Verified, scaffold) flagged as different splits/no ratio; RL-isolating pairs (WebAgent-R1 8.5->44.8, arXiv:2505.16421; DeepSeek-R1) identified.", [G6]),
1021: ("fixed", "Takeaway now uses 72.4% as the human reference; Mobile-Agent-v3 OSWorld corrected to 37.7% (73.3% is AndroidWorld; verified arXiv:2508.15144); SWE-bench takeaway reworded.", [G6]),
1022: ("fixed", "Table 3.1 rebuilt from verified rows only; unattributed rows removed; 'projected 2026' caption removed; CI notes corrected to 1.96*SE (+-4.4 SWE-bench Verified n=500, +-14 AIME n=30).", [G6]),
1023: ("unsourced-removed", "Table 5.2 (7%->100%) has no released trace; measured W&B tinker-rl-scaling Qwen3-8B runs give last-10 0.275-0.350. Replaced with measured-runs table labelled training reward.", [G6, MR]),
1024: ("unsourced-removed", "Fig 5.2 was hand-entered pgfplots coordinates; removed. Measured runs table substitutes.", [G6, MR]),
1025: ("unsourced-removed", "Table 5.3 bootstrap over reconstructed pooled binary outcomes removed (no trace; pseudoreplication). Held-out result reported at seed level: 83.3% vs 82.0%, t(4)=1.32, p=0.26.", [G6, MR]),
1026: ("fixed", "Chapter 5 now reports only the W&B-tracked runs (table of 10 runs) consistent with Table 5.10 of the PDF; contradicting Tables 5.2/5.3 removed.", [G6, MR]),
1027: ("unsourced-removed", "HumanEval 'Qwen3-3.8B' 32->40% result has no artifact (and no such Qwen3 size); removed.", [G6]),
1028: ("unsourced-removed", "Generalization-to-code claim removed with the unsourced HumanEval result.", [G6]),
1029: ("fixed", "Sec 5.1 reworded as a proposed, unexecuted programme with explicit scope note.", [G6]),
1030: ("fixed", "57-run suite and Stage 3 RAG described as proposed and not executed.", [G6]),
1031: ("unsourced-removed", "z-tests and pooled bootstraps over 128 rollouts/steps removed; seed is the unit (n=5 seeds).", [G6]),
1032: ("unsourced-removed", "Argmax Chow F-tests removed (no trace; invalid fixed-breakpoint p-values).", [G6]),
1033: ("unsourced-removed", "Post-hoc Mann-Whitney/Cohen d at data-selected breakpoint removed.", [G6]),
1034: ("fixed", "Eq 1.1 now puts per-step KL at realized history inside the trajectory expectation, sums t=0..H-1, uses r(s,a) and beta_KL; h_t indexing fixed.", [G6]),
1035: ("fixed", "Abstract and C1 distinguish six topical domains (discussion) from 19 categories in five clusters (taxonomy).", [G6]),
1036: ("fixed", "'four-cluster' corrected to five-cluster with pointer to Fig 2.3 and Appendix E.", [G6]),
1037: ("fixed", "'single-model training' replaced by 'policy plus frozen reference; no critic or learned reward model'; 'one model' sentence removed with the catalog paragraph.", [G6]),
1038: ("fixed", "DAPO (clip-higher, dynamic sampling, token-level loss), ReMax (greedy baseline REINFORCE, predates GRPO), Flow-GRPO (AgentFlow, arXiv:2510.05592, verified) corrected; unverifiable FoldGRPO bullet removed.", [G6]),
1039: ("unsourced-removed", "80-variant catalog counts, 39%/22%/10%/29%, kappa=0.85 footnote, pie chart, and 52/28/20% architecture shares have no retained catalog; removed everywhere (abstract, C2, methods, takeaways, conclusion).", [G6]),
1040: ("fixed", "RLOO (Ahmadian et al. 2024, ACL; arXiv:2402.14740, verified) added as leave-one-out baseline in the GRPO derivation. Dr.GRPO/REINFORCE++ not added (no verified local bib entry).", [G6, G6B]),
1041: ("fixed", "Fig 3.3 and Table 3.1 now both use OSWorld 12.2% (original paper best model, verified abstract); unsourced 18.4% removed; SWE-bench human bar (no human baseline) removed.", [G6]),
1042: ("fixed", "'4-29x/2-29x' multipliers removed from abstract, C4, highlights, timeline; C4 rewritten without a multiplier.", [G6]),
1043: ("fixed", "DeepAgent now described as memory folding + ToolPO (verified arXiv:2510.21618).", [G6]),
1044: ("fixed", "VerlTool now described as agentic RL training framework extending veRL (arXiv:2509.01055).", [G6]),
1045: ("fixed", "Toy model now states non-distraction and pessimistic retrieval-failure assumptions under which the bound follows.", [G6]),
1046: ("unsourced-removed", "Illustrative synthetic Fig 5.1 (PPO/DPO/GRPO curves) removed.", [G6]),
1047: ("fixed", "Statistical framework relabelled as a proposed plan; no tests claimed as run; seeds as unit.", [G6]),
1048: ("unsourced-removed", "Table 5.7 within-run Mann-Whitney comparison removed.", [G6]),
1049: ("fixed", "'summed to 2.0' reward statement removed with the old section; reward described as online training reward in [0,1].", [G6]),
1050: ("fixed", "Protocol now separates planned seeds (42,123,456) from executed seeds (42,137,256,512,999) and W&B-logged lr 3e-5/rank 32.", [G6, MR]),
1051: ("fixed", "Block C confound (rank 16, lr 6e-5/5e-5 for 0.6B/1.7B vs rank 32, lr 4e-5 for 8B; verified in atropos configs) stated explicitly; study marked planned.", [G6]),
1052: ("fixed", "Table E.2 recounted from the mapping table by tools/review_ledger/Sem3_Group6_Original_Report_taxonomy_counts.py: 59 papers (8/0/10/31/10); empty categories listed. (Source mapping has 59 rows; reviewed PDF had 67.)", [G6]),
1053: ("fixed", "Appendix D checklist now marks unexecuted items as not built/not trained; repo URL corrected to pes-llm-research/tinker-rl-lab.", [G6]),
1054: ("fixed", "Notation table adds O(o|s,a).", [G6]),
1055: ("fixed", "Graphical abstract node relabelled 'Safety (Ch. 3 & 6)'.", [G6]),
1056: ("fixed", "Note now reads 'preprints through early 2026'.", [G6]),
1057: ("fixed", "Placeholder logo box no longer printed when logo file absent.", [G6]),
1058: ("fixed", "Graphical Abstract heading moved inside the figure float.", [G6]),
1059: ("fixed", "First ToC page uses empty page style (no duplicate 'Contents' header).", [G6]),
1060: ("fixed", "Fig 2.1 caption now 'Pretraining and two-stage post-training (Base -> SFT -> RL)'.", [G6]),
1061: ("fixed", "Mapping longtable now has a numbered caption (Table E.1).", [G6]),
1062: ("fixed", "'Executive Summary' -> 'Abstract'; Chapter dependencies heading kept with its flowchart in a minipage.", [G6]),
1063: ("fixed", "DPO expansion corrected; stray space before comma removed.", [G6]),
1064: ("fixed", "Chapter 2 intro now points PPO-style RLHF to the RLHF-mechanisms and PPO sections.", [G6]),
1065: ("fixed", "Llama-2-Chat moved from DPO to PPO/RLHF column.", [G6]),
1066: ("wontfix-scope", FROZEN + "TikZ label overlap is a frozen-PDF layout issue; not re-verified in the maintained build.", [G6]),
1067: ("fixed", "Table 2.1 and its kappa footnote removed (unsourced catalog).", [G6]),
1068: ("fixed", "Algorithm has its caption; Ensure/return now consistent (returns pi_theta*).", [G6]),
1069: ("fixed", "beta_DPO used in Eq 2.2, beta_KL in Algorithm 2.1; TRPO constraint typeset as expected KL.", [G6]),
1070: ("fixed", "OS-Harm described as 150 tasks across three harm categories (verified arXiv:2506.14866).", [G6]),
1071: ("fixed", "Heuristic now states O(k) hops, O(log n) only for balanced trees.", [G6]),
1072: ("fixed", "Chapter 3 overview pointers match section titles.", [G6]),
1073: ("wontfix-scope", FROZEN + "TikZ label overlap in Fig 3.1; cosmetic, not re-verified.", [G6]),
1074: ("wontfix-scope", FROZEN + "TikZ label overlap in Fig 3.2; cosmetic, not re-verified.", [G6]),
1075: ("unsourced-removed", "Fig 3.5 trajectory plot removed: intermediate points (e.g., 12, 15, 34.2, 44) had no source.", [G6]),
1076: ("fixed", "FireAct described as single-agent trajectory fine-tuning.", [G6]),
1077: ("fixed", "'which includes most interesting tasks'.", [G6]),
1078: ("fixed", "Taxonomy sentence states the two-axis framing is adopted from the cited survey, not claimed as new.", [G6]),
1079: ("wontfix-scope", "SUGGESTION to add BFCL/tau-bench/VinePPO; out of scope for a correction pass (no new literature added without a reviewed need).", []),
}

nt = {
1080: ("unsourced-removed", NFROZEN + "15-run ZVF correlation (r=-0.769, rho=-0.784), task means (8.5%), scale r, and family ANOVA have no per-run table; repo verifier (platform_modal/scripts/verify_claims_offline.py) reports no cohort contract. Removed; the 22-run precision/recall rule was already withdrawn.", [NT]),
1081: ("fixed", "Per-seed records added: gsm8k_heldout_seed{042..999}.json 83.0/82.5/80.5/84.0/86.5% vs base 164/200; t(4)=1.32, p=0.256 recomputed; adapters match W&B tinker_run_ids.", [NT, MR]),
1082: ("already-fixed", NFROZEN + "GDPO and placeholder IDs no longer present in current related work/bibliography.", [NT]),
1083: ("already-fixed", NFROZEN + "Ahmadian et al. now described as REINFORCE/leave-one-out, explicitly 'not Dr. SAC'.", [NT]),
1084: ("already-fixed", NFROZEN + "Statistical policy section withdraws step-iid tests and significance decisions; historical inferential tables excluded.", [NT]),
1085: ("already-fixed", NFROZEN + "Table 27 (rigor pass) excluded from the root; step-iid bootstraps withdrawn.", [NT]),
1086: ("already-fixed", NFROZEN + "Cross-run ANOVAs withdrawn; remaining family ANOVA removed with 1080.", [NT]),
1087: ("already-fixed", NFROZEN + "Table 7 BH family removed; Sec 5.3 withdraws the historical families.", [NT]),
1088: ("already-fixed", NFROZEN + "Table 19 effect sizes removed; power section now illustration only.", [NT]),
1089: ("already-fixed", NFROZEN + "Table 30 removed; Llama PPO tail marked unresolved (Table 7).", [NT]),
1090: ("already-fixed", NFROZEN + "Cross-library comparison quarantined; Tinker row identified as summary-statistics-only/reconstructed, intervals withheld. Generator dict CROSS_LIBRARY in compute_statistics.py remains the evidence.", [NT]),
1091: ("already-fixed", NFROZEN + "Table 28 synthetic-seed tests withdrawn (Sec 6.4, checklist item 7).", [NT]),
1092: ("already-fixed", NFROZEN + "Task 4 Tinker tail marked unresolved (85.6 vs 84.4); verl/OpenRLHF identified as dry runs.", [NT]),
1093: ("already-fixed", NFROZEN + "Historical four-bar chart excluded; Table 10 readiness only.", [NT]),
1094: ("already-fixed", NFROZEN + "Top-10 checkpoint table/aggregate withdrawn; selection analysis only.", [NT]),
1095: ("already-fixed", NFROZEN + "Arithmetic learning-curve panels withheld (Sec 6.4).", [NT]),
1096: ("already-fixed", NFROZEN + "Performance profiles withheld (Sec 6.4).", [NT]),
1097: ("already-fixed", NFROZEN + "Current figures/wave6_sensitivity.png is the three-panel T/rank/batch sweep matching wave6 rows in master_results.json; caption line styles corrected.", [NT]),
1098: ("already-fixed", NFROZEN + "P1-A diagnostic claims no longer in the paper.", [NT]),
1099: ("already-fixed", NFROZEN + "Abstract formula removed; Sec 6.18 defines h_G(p), notes Jensen gap and per-prompt averaging.", [NT]),
1100: ("fixed", "Nimmaturi entry already corrected; 'Liu et al.' for the RL-subnetworks paper corrected to Mukherjee et al. (arXiv:2505.11711).", [NTR]),
1101: ("already-fixed", NFROZEN + "Anonymous URL removed; Sec 7/11.5 state private mirrors do not establish public access.", [NT]),
1102: ("already-fixed", NFROZEN + "Sec 5.3 states independent resampling is unpaired and steps are not independent.", [NT]),
1103: ("already-fixed", NFROZEN + "Llama PPO 97.5 vs 95.0 listed as unresolved in Table 7.", [NT]),
1104: ("already-fixed", NFROZEN + "Sec 5.3 withdraws all four historical correction families.", [NT]),
1105: ("unsourced-removed", "GSM8K scale summary (Table 11, source of Table 29 rows) removed: values only in hand-entered GSM8K_SCALING dict (platform_hybrid/experiments/compute_statistics.py, n=10 not 5); all W&B Qwen3-0.6B/14B GSM8K runs failed/crashed.", [NT]),
1106: ("already-fixed", NFROZEN + "Table 31 / Appendix G family table excluded from root.", [NT]),
1107: ("already-fixed", NFROZEN + "Power section states TTestIndPower is pooled equal-variance, not Welch.", [NT]),
1108: ("fixed", "Figure 1 caption now describes the drawn task-family diagram and flags node model counts as historical labels; 'of of' already gone.", [NT]),
1109: ("fixed", "Table 2 rows without reported results (chat SFT, DPO, distillation) removed; caption states scope.", [NT]),
1110: ("fixed", "Sec 4.2 text and Table 3 caption state PufferLib/rl_games/d3rlpy are configuration mappings without runs; Table 12 now '4 with reported runs' and 'Offline RL: config only'.", [NT]),
1111: ("already-fixed", NFROZEN + "Training protocol states 1e-4 nominal and 1e-5 for Task 4/Wave 6; double punctuation gone.", [NT]),
1112: ("fixed", "Fig 3 caption already corrected; Sec 5.4 now states 14/6 counts are the original suite and no reconciled run total is claimed; 44-experiment count removed with Sec 6.7.", [NT]),
1113: ("fixed", "'32+ models' removed; Nemotron identified from run record as nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 (sparse, 12B active); Table 16 Arch -> MoE; Table 9 size 120B (12B).", [NT]),
1114: ("already-fixed", NFROZEN + "Table 12 checkpoint aggregate withdrawn.", [NT]),
1115: ("already-fixed", NFROZEN + "Mean/MAD text withdrawn.", [NT]),
1116: ("already-fixed", NFROZEN + "Text now shows expected mixed fraction is independent of batch size.", [NT]),
1117: ("fixed", "Model x task heatmap removed: Llama cell 0.97 is the Modal PPO run and Qwen3-8B 0.35 is not the max; Table 9 carries values.", [NT]),
1118: ("unsourced-removed", "Scale correlations (r=0.468/0.533/-0.088) disagree with Fig 9 titles and no released fit output reproduces either; removed with Figs 5-6.", [NT]),
1119: ("unsourced-removed", "Saturation-fit summaries (44 vs 28 traces, R2 0.210/0.170 vs 0.214/0.228, 53.6%, 73%, 62%) removed; released fit_saturation_model.py fits a different form and reports no R2. Eq kept as descriptive form with clamp.", [NT]),
1120: ("unsourced-removed", "Group-size: no campaign_v2_w2 G=8 run exists in master_results.json; the G=8 row (100%/84.4%) removed; G=32 (11 logged points, 45.3/43.8) added; pooled historical panel already excluded.", [NT]),
1121: ("fixed", "Sec 6.20 group-size text rewritten from released runs: G in {2,4,16,32}, 37.5/52.1/38.0/43.8%, no optimum or inverted-U claimed.", [NT]),
1122: ("already-fixed", NFROZEN + "Sec 6.18 states h_G is symmetric and U-shaped, not monotone.", [NT]),
1123: ("already-fixed", NFROZEN + "Interventions replaced by prospective hypotheses; G=2 stated not to preserve contrast.", [NT]),
1124: ("already-fixed", NFROZEN + "Intervention 2 heading no longer present.", [NT]),
1125: ("already-fixed", NFROZEN + "Paper now claims no priority, cites AVSPO/RL-ZVP/DAPO/online filtering; RLOO cited.", [NT]),
1126: ("already-fixed", NFROZEN + "Spend/GPU-hour/carbon tables labelled unreconciled, arithmetic discrepancy stated (USD 147-177 vs 130-140).", [NT]),
1127: ("wontfix-scope", "Author email cannot be verified from repository evidence (one source lists @gmail.com, others @mail.com); left for the author.", []),
1128: ("already-fixed", NFROZEN + "Abstract no longer contains the 'lr=' expression.", [NT]),
1129: ("already-fixed", NFROZEN + "Phrase no longer present in current abstract/intro.", [NT]),
1130: ("already-fixed", NFROZEN + "Phrase no longer present.", [NT]),
1131: ("already-fixed", NFROZEN + "Appendix D.2 (Hedges g) excluded from root.", [NT]),
1132: ("already-fixed", NFROZEN + "Figure 2 now indexes completions by G (tikz/reward_flow); vector PDF.", [NT]),
1133: ("fixed", "Empty Sec 4.4 Reward Verification now has text; empty Sec 4.2 also filled.", [NT]),
1134: ("already-fixed", NFROZEN + "TRL bootstrap CI no longer reported.", [NT]),
1135: ("fixed", "Appendix B now states the table lists configured classic-RL PPO sweep ranges with no reported results and gives the actual Wave 6 sweep values; broken figure ref removed.", [NT]),
1136: ("fixed", "Eq now R(t)=Rmax(1-exp(-k max(0,t-t0))).", [NT]),
1137: ("already-fixed", NFROZEN + "Cross-Library Comparison section has body text.", [NT]),
1138: ("fixed", "Sec 6.6 now has text (and its unsourced scatter figure removed).", [NT]),
1139: ("fixed", "Text now references the adjacent expanded evidence table and the earlier one.", [NT]),
1140: ("fixed", "Unused double-dagger and star footnotes removed.", [NT]),
1141: ("unsourced-removed", "Scale scatter (Fig 7) removed: duplicate labels, mixed reward/accuracy, values not matching Table 9.", [NT]),
1142: ("fixed", "'cross-model heatmap above' removed with the heatmap.", [NT]),
1143: ("fixed", "Table 13 caption defines Func./Impr. as unscored qualitative reports.", [NT]),
1144: ("fixed", "Unused '---' caption sentence replaced; caption states rows repeat Table 9.", [NT]),
1145: ("fixed", "'8 larger-model experiments' -> the three larger-model runs with recorded traces.", [NT]),
1146: ("already-fixed", NFROZEN + "M4 G=32 sentence no longer present.", [NT]),
1147: ("already-fixed", NFROZEN + "Sec 6.18 now gives the exact G 8->16 gain at p=0.2 (13.96 pp, recomputed: 0.13963).", [NT]),
}

db = sqlite3.connect(sys.argv[1])
for paper, d in (("Sem3_Group6_Original_Report", g6), ("Sem3_NeurIPS_Main_Track", nt)):
    ids = {r[0] for r in db.execute("select id from findings where paper=?", (paper,))}
    assert ids == set(d), (paper, sorted(ids ^ set(d)))
    with open(OUT / f"{paper}.jsonl", "w") as f:
        for i in sorted(d):
            st, ev, files = d[i]
            f.write(json.dumps({"id": i, "status": st, "evidence": ev, "files": files}) + "\n")
    from collections import Counter
    print(paper, dict(Counter(v[0] for v in d.values())))
