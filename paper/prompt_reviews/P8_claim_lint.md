# P8 (Fraud Side-Probe) Claim–Evidence Lint

**Contract:** `research_prompts/writing/claim-evidence-linter.md` — label each major claim in the
abstract + results + conclusion of `paper/paper_P8_fraud.tex` as **supported / weakly supported /
unsupported**, checked against the only on-disk evidence for this paper: `train_xgboost.py` and its
committed output `xgboost_results.json` (both added in commit `921e6ed`), plus the previously
web-verified citation set. Lint run 2026-07-04.

**Sections linted:** `p8_abstract.tex`, `p8_setup.tex`, `p8_scorer.tex` (results),
`p8_taxonomy.tex`, `p8_limitations.tex`, `p8_future.tex` (closing/conclusion).

**Headline tally: 21 claims — 10 supported, 8 weakly supported, 3 unsupported-as-reproducible
(retained only as explicitly scoped internal single-run records, per the program contract).**

**Worst offender (now disclosed in the paper):** the paper's headline **XGBoost AUC 0.975** is
contradicted by the paper's own released artifact — `xgboost_results.json`, written by the released
`train_xgboost.py` and committed alongside it, records **AUC 0.794** (f1 0.356, precision 0.723,
recall 0.236). The **LLM 0.948** number has *no* artifact anywhere in the repo (no fine-tuning
script, no results file; the only fraud code is `train_xgboost.py`). Both numbers survive only as
21-June-2026 internal single-run records. The fix applied is disclosure, not deletion: the rerun
discrepancy is now stated in §Setup (scorer-arm paragraph) and in a new Limitations paragraph
("Headline numbers are records, not reproductions"), and every assertive use of the head-to-head
outcome was rewritten into internal-records mood.

Support-level key:
- **S** supported — reproduces from an on-disk artifact or a web-verified primary source.
- **W** weakly supported — direction/qualitative content holds (literature-cited or artifact-adjacent) but the specific number or scope does not reproduce from local evidence.
- **U** unsupported — no artifact backs it; retained only because the program contract mandates presenting 0.975/0.948/85x as scoped internal single-study results, with the gap disclosed in-paper.

---

## 1) Claim table

### Abstract (`sections/p8_abstract.tex`)

| ID | Claim | Label | Evidence / check |
|----|-------|-------|------------------|
| A1 | Custom synthetic dataset: 50,000 transactions, 1% fraud rate | **S** | `train_xgboost.py`: `n_samples=50000`, `weights=[0.99, 0.01]`. |
| A2 | XGBoost held-out AUC 0.975 (internal single-run record, 21-Jun-2026) | **U** | Contradicted by the committed artifact: `xgboost_results.json` (output of the released script) = **AUC 0.794**. No repo artifact contains 0.975. Kept only as an explicitly scoped internal record; discrepancy disclosed in §Setup and §Limitations. Never stated as a general truth. |
| A3 | Fine-tuned LLM AUC 0.948 (internal single-run record) | **U** | No LLM fine-tuning script or results file exists anywhere in the repo (only fraud code is `train_xgboost.py`). Kept only as a scoped internal record; flagged in §Setup ("Per our internal experimental records") and §Limitations ("Single LLM family, single recipe" + rerun paragraph). |
| A4 | Tree keeps scorer seat for latency, cost, calibration, injection exposure | **W** | Argued qualitatively; calibration leg cited to Guo et al. ICML'17, injection leg to Greshake et al. 2023 (both web-verified). Latency leg partially artifact-backed (see R4). Table caption labels operational columns "qualitative rankings, argued in the text." |
| A5 | Four capability gaps (perception, narration, cold-start, agentic) where LLM adds value | **W** | Literature- and regulation-grounded (FakeShield, Forensics-Bench, FinCEN, Reg B, TabLLM, Co-Investigator, Pirmorad), not measured by this paper; §Limitations "Capability gaps argued, not all measured" says exactly this. |
| A6 | Agentic triage ≈85× cheaper than human analyst | **U** | No cost-model artifact on disk. Presented as "by our internal estimate," one significant figure; §Gap-4 and §Limitations call it "internal, assumption-laden," "not validated against any production deployment," "read as one to two orders of magnitude." |
| A7 | Hybrid sensor/scorer/scribe architecture with post-score triage | **S** | Design contribution, no empirical claim; §Limitations states integrated performance is future work. |
| A8 | Study is side-probe of reproducibility program | **S** | Program-context statement, non-empirical. |

### Setup (`sections/p8_setup.tex`)

| ID | Claim | Label | Evidence / check |
|----|-------|-------|------------------|
| S1 | Dataset config: 20 features (10 informative, 2 redundant), 2 clusters/class, class_sep 0.8, flip_y 0.01, seed 42; +4 aggregate features = 24 | **S** | Matches `train_xgboost.py` line-for-line (`make_classification(...)`, `V_mean/V_std/V_max/V_min`). |
| S2 | 80/20 stratified split, seed 42 → 40,000 train / 10,000 test, ~100 test positives | **S** | `train_test_split(test_size=0.2, random_state=42, stratify=y)`; 1% of 10,000 = ~100. |
| S3 | XGBoost config: 200 estimators, depth 6, lr 0.05, subsample/colsample 0.8, scale_pos_weight 7, eval auc; no hyperparameter search | **S** | Matches `xgb.XGBClassifier(...)` exactly. |
| S4 | Script released with paper, emits train + per-10k-row inference times | **S** | `train_xgboost.py` in repo root; writes `train_time_sec`, `infer_time_sec_per_10k`. |
| S5 | (NEW) Rerun of released script yields AUC 0.794, not 0.975 | **S** | `xgboost_results.json` (committed, `921e6ed`): `"auc": 0.7942...`. Disclosure added this pass. |
| S6 | Data is synthetic stand-in for single-institution feature view; supports role comparison, not production AUC claims | **S** | Honest scoping statement; consistent with the generator. |

### Results (`sections/p8_scorer.tex`)

| ID | Claim | Label | Evidence / check |
|----|-------|-------|------------------|
| R1 | Internal records place XGBoost 0.975 vs LLM 0.948 | **U/W** | Same evidence status as A2/A3. Reworded this pass from "the outcome is unambiguous" to internal-records mood with an explicit cross-reference to the rerun caveat; table caption now says "single-run internal results … rerun caveat in §Setup." |
| R2 | A 2.7-pt AUC gap at 1% base rate would mean materially more missed fraud at fixed alert budget | **W** | Counterfactual operational reasoning; moved to conditional mood ("would not be cosmetic … would translate") this pass since the gap itself is a single-run record. |
| R3 | Result consistent with tabular literature (trees ≥ DL / LLM serialization at 40k labels) | **S** | Grinsztajn 2207.08815, Shwartz-Ziv 2106.03253, TabLLM 2210.10723 — all previously arXiv-verified; TabLLM re-verified this pass. |
| R4 | Committed run measures ≈6 ms per 10,000 rows inference | **S** | `xgboost_results.json`: `infer_time_sec_per_10k = 0.00618 s`. Rewritten this pass to cite the artifact and the concrete number. |
| R5 | Marginal cost per transaction ≈0 for tree, token-linear for LLM | **W** | Qualitative economics; not benchmarked; §Limitations flags cost/latency contrasts as "operational characterizations, not benchmarked service-level measurements." |
| R6 | LLM-verbalized confidences inherit NN miscalibration | **W** | Guo et al. ICML'17 (verified) covers modern NN miscalibration; extension to verbalized LLM confidence is argued, not measured here. |
| R7 | Reading attacker-authored text creates indirect prompt-injection surface; trees have none | **S** | Greshake et al. arXiv:2302.12173 (re-verified this pass); the tree half is definitional. |

### Taxonomy (`sections/p8_taxonomy.tex`)

| ID | Claim | Label | Evidence / check |
|----|-------|-------|------------------|
| T1 | Fraud flows through unstructured evidence (receipts, IDs, checks) | **W** | Was "a substantial fraction of fraud losses" — an uncited quantitative claim; softened this pass to "a persistent channel of fraud." Qualitative version is common knowledge + consistent with the VLM-forensics literature cited. |
| T2 | VLMs can detect/localize forgeries with textual justification; far from solved | **S** | FakeShield (ICLR'25, arXiv:2410.02761) + Forensics-Bench (arXiv:2503.15024, re-verified this pass; 112 forgery types matches its abstract). Paper correctly confines VLM to feature extraction. |
| T3 | Agentic systems target SAR drafting with human-in-the-loop | **S** | Co-Investigator AI (arXiv:2509.08380, re-verified this pass — title/authors match). Was "report substantial analyst-time reductions" — softened this pass since only title/abstract-level metadata was verified, not their measured reductions. |
| T4 | 85× triage cost estimate | **U** | Same as A6; §Gap-4 carries the full caveat block in-line. |
| T5 | LLM few-shot cold-start competitiveness | **W** | TabLLM few-shot regime (verified) + Pirmorad arXiv:2507.14785; the specific fraud-typology workflow is argued, not evaluated. |

### Compliance (`sections/p8_compliance.tex`) — spot-check

| ID | Claim | Label | Evidence / check |
|----|-------|-------|------------------|
| L1 | 31 CFR §1020.320: SAR for ≥$5,000, "knows, suspects, or has reason to suspect," ~30 days | **S** | Verified against LII primary source in the drafting pass. |
| L2 | FinCEN 2003 guidance: narrative must cover who/what/when/where/why | **S** | Verified against fincen.gov PDF in the drafting pass. |
| L3 | 12 CFR §1002.9 + CFPB Circular 2022-03: specific principal reasons; no black-box defense | **S** | Verified against consumerfinance.gov primary sources in the drafting pass. |

### Conclusion / closing (`sections/p8_future.tex`)

| ID | Claim | Label | Evidence / check |
|----|-------|-------|------------------|
| C1 | "If the program thesis holds anywhere outside RL post-training, it should hold here" | **W** | Explicitly conditional/hedged opinion; acceptable as closing stance. |
| C2 | Future-work plan replaces the 85× estimate with measured costs | **S** | Self-referential commitment; correctly treats 85× as unvalidated. |

---

## 2) Missing evidence list

1. **LLM arm (0.948):** no fine-tuning script, no serialization code, no results JSON anywhere in
   the repo. The claim rests entirely on the 21-Jun-2026 internal records. → To upgrade from U:
   release the fine-tuning/eval script and its results artifact, or rerun the arm.
2. **XGBoost 0.975:** the released script's committed output says 0.794. → To upgrade: reconstruct
   the internal run's environment (library versions, any config delta) and commit a reproducing
   artifact; until then the paper now discloses the discrepancy in two places.
3. **85× cost estimate:** no cost-model worksheet (token prices, analyst comp, per-alert minutes)
   is on disk. → Commit the assumption sheet, or keep the current one-significant-figure,
   order-of-magnitude framing.
4. **Sensor / scribe / cold-start seats:** no own evaluations (declared future work in-paper) —
   acceptable as long as §Limitations "Capability gaps argued, not all measured" stays.

## 3) Rewrites applied this pass (semantic-preserving softenings + disclosures)

| File | Change |
|------|--------|
| `p8_abstract.tex` | "find, in our internal experiments, that XGBoost reaches…" → "in our internal single-run records (21 June 2026), XGBoost reached… numbers we report strictly as internal single-study results." |
| `p8_intro.tex` | "reaches a held-out AUC of 0.975 (an internal result…)" → "recorded … in a single internal run (21 June 2026 records; see the reproducibility caveat in §Setup)"; "it loses the head-to-head" → "it scores below the tree… in our internal records"; "the loss" → "this gap"; "one of them favors a 2016-era tree, and four favor the LLM" → "favored … in our internal records, and we argue the other four…"; contribution (i) "winning the scorer seat on accuracy" → "keeping the scorer seat on our internal accuracy records"; "internal results" → "single-run internal results". |
| `p8_setup.tex` | Added the rerun disclosure: released script's committed rerun = AUC 0.794 (`xgboost_results.json`); 0.975 is a record of one internal run whose environment could not be reconstructed. |
| `p8_scorer.tex` | Opening "the outcome is unambiguous" → internal-records phrasing + rerun cross-reference; "is not cosmetic… translates" → "would not be cosmetic… would translate"; table caption → "single-run internal results… rerun caveat in §Setup"; latency claim now cites the committed ≈6 ms/10k artifact number; closing "the LLM lost" → "our internal records score the LLM below the tree". |
| `p8_taxonomy.tex` | "A substantial fraction of fraud losses flows through" → "A persistent channel of fraud runs through"; Co-Investigator claim softened from "report substantial analyst-time reductions" to "target SAR-narrative drafting specifically, with human-in-the-loop review". |
| `p8_limitations.tex` | "We believe the ordering is robust" → "matches what the tabular literature predicts… but we have not independently re-established even the ordering"; added new paragraph "Headline numbers are records, not reproductions" (0.794 rerun, LLM arm not rerun, seat assignment insulated because operational arguments hold at accuracy parity). |

## 4) Citation re-verification (5 random external, this pass)

Selected by seeded `shuf` over the 12 external scholarly keys; all 5 **passed** (title + authors +
venue match the bib entry); none removed.

| Key | Source checked | Result |
|-----|----------------|--------|
| `greshake2023injection` | arxiv.org/abs/2302.12173 | PASS — title + 6 authors exact. |
| `hegselmann2023tabllm` | arxiv.org/abs/2210.10723 | PASS — title + 6 authors exact (cited as arXiv preprint, consistent). |
| `wang2025forensicsbench` | arxiv.org/abs/2503.15024 | PASS — title + 9 authors exact. |
| `pedregosa2011sklearn` | jmlr.org/papers/v12/pedregosa11a.html | PASS — title, authors, JMLR vol. 12, pp. 2825–2830. |
| `naik2025coinvestigator` | arxiv.org/abs/2509.08380 | PASS — title + 5 authors exact. |

## 5) Build after fixes

`pdflatex → bibtex → pdflatex ×2` on `paper_P8_fraud.tex`: **0 errors, 0 undefined
citations/references, 0 BibTeX warnings, 0 overfull boxes.**
