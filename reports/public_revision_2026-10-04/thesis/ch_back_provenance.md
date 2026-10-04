# Appendix J. Number-to-Artifact Provenance

This is a public derivative of verification against locally recovered original files. All 915 numerical checks pass; 22 registered values are context only. 31 source patterns are withheld from this package. Public-source reproducibility remains incomplete: the result and hashes do not supply those original bytes. Opaque withheld-source labels conceal private locations and provider identities. For a glob, the retained hash binds the original sorted source-hash/path manifest, whose private paths are not published. No raw response bodies, patches, provider archives or training-run identities are included here.

Each row binds the values printed at one location to the bytes they were checked against. `tools/audit_thesis_numbers.py` performs the underlying local checks. A separately preserved private projection replaces unpublished paths with source-hash labels, removes runner-pattern details, and adds availability fields. It produces this table and `provenance_audit.json`, which lists every value, its check, source availability, public path or opaque private-source label, and SHA-256. A value passes only if it appears verbatim in its chapter and agrees at its printed precision. Methods: R recomputed from rows (counts, means, SDs, Wilson, Fisher, t, TOST, ANOVA, McNemar, sign-flip, cluster bootstrap, `git apply --stat`); S stored summary field; C hyperparameter matched in the runner code; T verbatim in the cited source report; P prose sweep, every decimal and percentage in a cited span found in a file that span cites or verified by an R row (22 values stated as thresholds or definitions are exempt); A arithmetic from printed counts, artifact absent from this checkout; W withheld source, not checked. Σ marks the SHA-256 of a group's sorted source hashes and listed public paths or opaque labels. Individual source hashes bind the locally checked originals; Appendix I states which files the public package includes.

\newpage

| Location | Values | M | Artifacts | SHA-256 (prefix) | Result |
|----------------------|-----:|:----:|------------------------------------------|---------------|-------------|
| Table 1.1 | 19 | RT | `35 files` | `Σd136e3050c0` | agree |
| Table 4.1 | 2 | R | `2 files` | `Σa5f4cf0dddc` | agree |
| Table 4.2 | 18 | C | `17 files` | `Σ3fac2be0211` | agree |
| §6.1 | 6 | RP | `2 files` | `Σef6dbd7be9d` | agree |
| §6.2 | 79 | P | `6 files` | `Σd519f7e5ab0` | agree |
| §6.3 | 115 | P | `13 files` | `Σd2ebbf5ae78` | agree |
| §6.4 | 111 | RSP | `15 files` | `Σ8adf6b0a02c` | agree |
| §6.5 | 174 | P | `15 files` | `Σ68b18a3ffd4` | agree |
| Table 6.1 | 11 | R | `samestack_gsm8k_cot.json` | `ed3022290a87` | agree |
| §6 recomputations | 43 | RST | `20 files` | `Σ4c812e92bc7` | agree |
| §7.1 | 36 | RST | `5 files` | `Σf51fc73ea89` | agree |
| §7.2 | 32 | RA | `2 files` | `Σ3dc3ccc3d12` | agree |
| §7.3 | 34 | TA | `p7_controller.tex` | `a5caaab4b855` | agree |
| §7.4 | 14 | A | — | `—` | agree |
| §7.5 | 11 | R | `c1_case_ledger.csv` | `70f48e55a442` | agree |
| Table 8.1 | 5 | RA | `2 files` | `Σ82d376d8289` | agree |
| Table 8.2 | 14 | R | `10 files` | `Σac8c5a0edcc` | agree |
| §8.2 | 21 | R | `5 files` | `Σ23283b8034c` | agree |
| §8.3 | 10 | RTA | `3 files` | `Σ54e3062a380` | agree |
| §8.4 | 3 | R | `3 × result.json` | `Σf6a7ff9171a` | agree |
| Table 8.C | 18 | R | `6 × result.json` | `Σa13e3d43680` | agree |
| Table 8.D | 6 | S | `6 × result.json` | `Σa13e3d43680` | agree |
| §8.5 | 2 | S | `result.json` | `3603158797cd` | agree |
| Table 8.A | 14 | R | `14 files` | `Σ3f9a5a2ae7e` | agree |
| Table 8.B | 18 | R | `14 × paired.json` | `Σ023282562fb` | agree |
| §9.1 | 23 | RA | `14 files` | `Σ12718b4d143` | agree |
| §9.2 | 6 | T | `LIMITATIONS_AND_IMPACT.md` | `c44062ee5dbd` | agree |
| Table 9.1 | 3 | RC | `6 files` | `Σed5e76646e8` | agree |
| Table A.1 | 1 | R | — | `—` | agree |
| Table A.2 | 4 | S | `framework_comparison.json` | `454b9f3db7c4` | agree |
| Table A.3 | 7 | R | `modal_results_all.json` | `522119d906da` | agree |
| Table A.4 | 4 | R | `groupsize_zvf_sweep.json` | `4aefae380f2c` | agree |
| Table A.5 | 21 | RTA | `23 files` | `Σ78fd633186c` | agree |
| Table A.6 | 19 | RS | `9 files` | `Σ401a5d3495c` | agree |
| Table A.7 | 3 | R | `samestack_gsm8k_cot.json` | `ed3022290a87` | agree |
| Table A.8 | 18 | RTA | `14 files` | `Σ6f740f1a69c` | agree |
| Table A.10 | 6 | RTA | `5 files` | `Σ7d9e79de031` | agree |
| Table F.2 | 2 | S | `qp8_fraud.tsv` | `179d1f36cc88` | agree |
| Table F.4 | 4 | R | `p8_headline_cis.tsv` | `1468df0df3fb` | agree |

: Number-to-artifact provenance for Chapters 1, 4 and 6–9 and Appendices A and F. 915 of 937 registered non-withheld values pass; 22 are context only; see row results for unavailable or unbound evidence.
