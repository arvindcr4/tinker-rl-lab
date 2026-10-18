"""Write reports/deepreview/dispositions/Sem4_P8_Fraud.jsonl (ids 1148-1204)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = "platform_hybrid/paper/archive/absorbed/P08_fraud/paper_P8_fraud.tex"
VER = "platform_hybrid/experiments/results/review_2026-10-18/Sem4_P8_Fraud/verify.json"
GONE = ("The reviewed 88-page PDF was built from the pre-2a808b576 source (sections/p8_*.tex); "
        "the current source is an 84-line corrected descendant that \\inputs no section files. ")
WD = "Text withdrawn in Sec.3 'Withdrawal of downstream conclusions'. "

S = {}
for i in (1148, 1151):
    S[i] = ("fixed", "Verified: _p8_common.py fits on fraud_data.csv (TRAIN) and evaluates on test_data.csv; "
            "verify.json shows all 10,000 test rows occur in fraud_data.csv while train_data/test_data share 0 rows. "
            "Downstream in-sample tables were already withdrawn (2a808b576); this pass adds the verified overlap "
            "evidence and script path to Sec.3 and the disjoint-refit requirement to Sec.4.", [TEX, VER])
S[1158] = ("already-fixed", GONE + WD + "Recall@K tables come from the same in-sample pipeline; no K-metrics retained.", [TEX])
S[1149] = ("fixed", "Sec.2 already states non-comparability (500-row 20%-enriched vs 10,000-row natural) and that "
           "SFT accuracy 0.792 < constant-negative 0.800. Added model/run identity (Qwen/Qwen3.5-4B, LoRA r16, "
           "W&B ek1b2cxn, auc 0.482675) and subset composition (100/400 vs 144/9,856) from "
           "qp8_fraud_sft_20260704.py and wandb summary.", [TEX])
for i in (1150, 1164, 1167, 1171):
    S[i] = ("already-fixed", GONE + "Sec.1 states the four aggregates are deterministic row functions (re-verified: "
            "aggregates_deterministic_of_V1_V20=true in verify.json), not VLM/LLM outputs; Sec.3 states they need no "
            "per-row LLM invocation and that the study cannot support sensor/scribe effectiveness.", [TEX, VER])
S[1152] = ("already-fixed", "Inverted threshold removed. Sec.3 gives p>=c/L (cost on every alert) and c/(L+c) "
           "(cost only on false alerts); both derivations checked (pL>=c; pL>=(1-p)c).", [TEX])
S[1153] = ("already-fixed", GONE + "Sec.3: hybrid modes charged a fee without reranking; cost disadvantage is not sensor utility.", [TEX])
for i in (1154, 1156, 1157, 1159, 1160, 1162, 1165, 1166, 1168, 1169, 1170, 1172):
    S[i] = ("already-fixed", GONE + WD + "The contradictory tables/definitions are not reproduced; Sec.3 notes they "
            "are not reconciled by picking an iteration (incl. ECE 0.99 being worse than 0.17-0.21, bootstrap tie rules, "
            "D-label definitions, per-cohort call totals).", [TEX])
S[1155] = ("already-fixed", GONE + "Sec.3: unrelated RL telemetry and cross-paper density tables are excluded.", [TEX])
S[1161] = ("already-fixed", "Sec.2 reports infer_time_sec_per_10k=0.006184816360473633 s as a per-10,000-row batch "
           "duration, matching train_xgboost.py ((infer_time/len(X_test))*10000 with len=10000); per-transaction "
           "cost figures removed.", [TEX, "xgboost_results.json", "platform_local/train_xgboost.py"])
S[1163] = ("already-fixed", GONE + "Cost/latency table removed; Sec.3 says token prices are scenario assumptions, not measured.", [TEX])
S[1173] = ("already-fixed", GONE + "Sec.3: 'No ... 85-fold cost reduction is validated'.", [TEX])
S[1174] = ("fixed", "Added make_classification parameters from platform_local/train_xgboost.py and the realised "
           "positive rate 719/50,000=1.44% (verify.json) with the flip_y explanation.", [TEX, VER])
for i in range(1175, 1204):
    S[i] = ("already-fixed", GONE + "The flagged sentence/table/caption no longer exists in the source.", [TEX])
S[1204] = ("wontfix-scope", "Descendant makes no comparative or LLM-advantage claim; adding a TabPFN literature "
           "discussion would need new verified citations and is out of scope for this withdrawal note.", [])

assert sorted(S) == list(range(1148, 1205))
out = ROOT / "reports/deepreview/dispositions/Sem4_P8_Fraud.jsonl"
out.write_text("".join(json.dumps({"id": i, "status": s, "evidence": e, "files": f}) + "\n"
                       for i, (s, e, f) in sorted(S.items())))
from collections import Counter
print(Counter(s for s, _, _ in S.values()))
