# Appendix J. Number-to-Artifact Provenance

This registry covers selected numeric token occurrences, not every number or scientific claim in the thesis. `tools/audit_thesis_numbers.py` regenerates this table and `provenance_audit.json`, which lists each registered token, its check and the cited files' full paths and SHA-256 hashes. A PASS means the token appears verbatim in its chapter and satisfies its named check; it does not establish independent scientific verification. Methods: R recomputed from rows (counts, means, SDs, Wilson, Fisher, t, TOST, ANOVA, McNemar, sign-flip, cluster bootstrap, `git apply --stat`); S stored summary field; C hyperparameter matched in the runner code; T numeric transcription in the cited source report; P selected cited-prose tokens matched to a source or another registered check; A arithmetic from printed counts, without independently checking those inputs against an artifact; W withheld source, not checked. CONTEXT includes definitions and unverified printed inputs; these tokens are not independently checked and are excluded from passing-check counts. Σ marks the SHA-256 of a group's sorted `sha256  path` lines. Hashes bind the author's checkout; Appendix I states which files the public package includes.

Number-to-artifact registry for Chapters 1, 4 and 6–9 and Appendices A and F. Of 967 registered token occurrences, 945 checks pass and 22 are context only, not independently checked. The passing checks include 34 calculations from printed counts only. 0 occurrences rest on withheld sources and are not checked; 0 checks fail, 0 are unbound and 0 lack required artifacts. Passing this registry does not verify every numerical thesis claim or authenticate its sources.

\newpage

| Location | Values | M | Artifacts | SHA-256 (prefix) | Result |
|----------------------|-----:|:----:|------------------------------------------|---------------|-------------|
| Table 1.1 | 19 | RT | `35 files` | `Σd136e3050c0` | 19 pass |
| Table 4.1 | 2 | R | `2 files` | `Σa5f4cf0dddc` | 2 pass |
| Table 4.2 | 18 | C | `17 files` | `Σ95031ae100e` | 18 pass |
| §6.1 | 6 | PA | `2 files` | `Σ01447712b59` | 6 pass |
| §6.2 | 80 | P | `6 files` | `Σ64d731df3ca` | 80 pass |
| §6.3 | 110 | P | `14 files` | `Σ59fbf15a925` | 110 pass |
| §6.4 | 112 | RSP | `15 files` | `Σ27b2056e098` | 112 pass |
| §6.5 | 207 | RP | `18 files` | `Σ25dd7f0669b` | 204 pass; 3 context |
| Table 6.1 | 11 | R | `samestack_gsm8k_cot.json` | `ed3022290a87` | 11 pass |
| §6 recomputations | 43 | RT | `21 files` | `Σ7c553cace5c` | 43 pass |
| §7.1 | 36 | RST | `5 files` | `Σfa6c68c5ebe` | 36 pass |
| §7.2 | 32 | RA | `2 files` | `Σ8429f41bf5b` | 25 pass; 7 context |
| §7.3 | 34 | TA | `p7_controller.tex` | `a5caaab4b855` | 29 pass; 5 context |
| §7.4 | 14 | A | — | `—` | 11 pass; 3 context |
| §7.5 | 11 | R | `c1_case_ledger.csv` | `70f48e55a442` | 10 pass; 1 context |
| Table 8.1 | 5 | RA | `2 files` | `Σd7797af0cf8` | 4 pass; 1 context |
| Table 8.2 | 14 | R | `10 files` | `Σ10e8cbb1b7b` | 14 pass |
| §8.2 | 21 | R | `5 files` | `Σ388c689fdb0` | 21 pass |
| §8.3 | 10 | RTA | `3 files` | `Σ19c1dd75631` | 9 pass; 1 context |
| §8.4 | 3 | R | `3 × result.json` | `Σf6a7ff9171a` | 3 pass |
| Table 8.C | 18 | R | `6 × result.json` | `Σa13e3d43680` | 18 pass |
| Table 8.D | 6 | S | `6 × result.json` | `Σa13e3d43680` | 6 pass |
| §8.5 | 2 | S | `result.json` | `3603158797cd` | 2 pass |
| Table 8.A | 14 | R | `14 × result.json` | `Σe3c416e24a2` | 14 pass |
| Table 8.B | 18 | R | `14 × paired.json` | `Σ023282562fb` | 18 pass |
| §9.1 | 23 | RA | `14 files` | `Σ5eafca0389e` | 23 pass |
| §9.2 | 6 | T | `LIMITATIONS_AND_IMPACT.md` | `c44062ee5dbd` | 6 pass |
| Table 9.1 | 3 | RC | `6 files` | `Σe9e99772009` | 3 pass |
| Table A.1 | 1 | R | — | `—` | 1 pass |
| Table A.2 | 4 | S | `framework_comparison.json` | `454b9f3db7c4` | 4 pass |
| Table A.3 | 7 | R | `modal_results_all.json` | `522119d906da` | 7 pass |
| Table A.4 | 4 | R | `groupsize_zvf_sweep.json` | `4aefae380f2c` | 4 pass |
| Table A.5 | 21 | RTA | `23 files` | `Σ78fd633186c` | 20 pass; 1 context |
| Table A.6 | 19 | RS | `9 files` | `Σ401a5d3495c` | 19 pass |
| Table A.7 | 3 | R | `samestack_gsm8k_cot.json` | `ed3022290a87` | 3 pass |
| Table A.8 | 18 | RTA | `14 files` | `Σ201037072c8` | 18 pass |
| Table A.10 | 6 | RTA | `5 files` | `Σc1f5919db9f` | 6 pass |
| Table F.2 | 2 | S | `qp8_fraud.tsv` | `179d1f36cc88` | 2 pass |
| Table F.4 | 4 | R | `p8_headline_cis.tsv` | `1468df0df3fb` | 4 pass |

: Selected numerical token occurrences and their recorded checks.

