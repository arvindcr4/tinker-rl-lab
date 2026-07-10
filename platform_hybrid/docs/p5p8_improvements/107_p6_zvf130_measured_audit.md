# 107 — P6 zvf130 measured-vs-claimed audit (iter 90)

**Pillar:** P6 (Pillar 2 — GRPO-Registry)
**Vein (fresh):** closes brief vein (a) on the 9-method 5-seed zvf130 panel.
The registry already carried five `zvf130_<method>.json` entries (CPPO, ES,
MCGRPO, NGRPO, SCAFGRPO), but **all five had `outcomes.zvf_risk_mean = null`**
even though `experiments/results/zvf_iter130_method_risk.tsv` measured every
method on a 5-seed panel. Iter 90 (i) closes that 100% gap, (ii) adds a
paired-bootstrap measured delta vs GRPO for every variant, (iii) cross-checks
the registry's *claimed* delta-component count against the measured risk rank.

## Falsifiable headlines (all measured)

- **H1** — **100% of the five `zvf130_<method>` stack entries had `outcomes.zvf_risk_mean = null` before iter 90** (gap = `null` on every entry, even though the measured value was sitting in `zvf_iter130_method_risk.tsv` row by row). All five entries are now patched via `scripts/p5p8/p6_iter90_zvf130_measured_vs_claimed.py`.
- **H2** — **Every named variant is significantly below GRPO on `zvf_risk`** (paired bootstrap, $B{=}4000$, seed 20260705, n=5 seeds). All 8 non-GRPO methods have a CI that excludes 0 in the negative direction. Ranked: SCAFGRPO $\Delta{=}{-}0.352$ → ES $\Delta{=}{-}0.273$ → GIFT $\Delta{=}{-}0.263$ → AREAL $\Delta{=}{-}0.246$ → MCGRPO $\Delta{=}{-}0.174$ → CPPO $\Delta{=}{-}0.151$ → AERO $\Delta{=}{-}0.148$ → NGRPO $\Delta{=}{-}0.131$. SCAFGRPO has the lowest measured risk ($0.2253$), GRPO has the highest ($0.5777$).
- **H3** — **26/36 (72.2\%) pairs SIG** at $\alpha{=}0.05$ on the $\binom{9}{2}{=}36$ pairwise matrix (paired bootstrap, $B{=}4000$). The 10 NS pairs cluster in the mid-risk middle (AERO$\leftrightarrow$AREAL/CPPO/MCGRPO/NGRPO, AREAL$\leftrightarrow$CPPO/MCGRPO/NGRPO, ES$\leftrightarrow$GIFT/SCAFGRPO, CPPO$\leftrightarrow$NGRPO).
- **H4** — **Spearman $\rho$(claim_delta_count, zvf_risk_mean) = −0.483** (point estimate, $n{=}9$ methods). Bootstrap CI $[-0.767, +0.617]$ brackets zero: at $n{=}9$ the correlation is suggestive but underpowered; need $n{\gtrsim}20$ methods for a conclusive negative trend.
- **H5** — **No `zvf130_<method>` entry lacks measured data.** All 5 `zvf130_*` entries map to a measured method in `zvf_iter130_method_risk.tsv`. The 4 N2 same-stack methods (GRPO, AERO, AREAL, GIFT) are recorded as `tinker_<method>_qwen3.5-4b_gsm8k.json` instead; the iter-86 cross-stack matrix already validated those.

## Operational recommendation

Registry consumers should now prefer `outcomes.zvf_risk_mean` on
`zvf130_<method>` entries as the canonical *single-batch same-stack risk
index* field. The `outcomes.delta_vs_grpo_mean / _ci_lo / _ci_hi / _sig`
block added by iter 90 gives a uniform per-variant significance filter.

Future audits should treat `outcomes.zvf_risk_mean = null` on a
`zvf130_<method>` entry as a **red-flag gap**, not as
``reported-as-absent'' — the existing `null`-vs-`false` convention of
`schema.json` does not catch this case.

## Cross-paper coupling

1. **P5 row 101/106 (iter 89)** — iter 89 isolated GIFT as the largest
   algorithm-axis variance carrier on N2. Iter 90 measures GIFT as the
   *third-lowest-risk* method on the zvf130 panel ($0.3145$, behind only
   SCAFGRPO and ES). The two findings are complementary: registry
   `zvf_risk_mean` is a single-batch risk; algorithm-axis $\eta^2$ is a
   same-stack *differential*. GIFT being a high-contrast method on N2 and
   a low-risk method on zvf130 jointly characterises the GIFT-vs-rest gap.
2. **P6 row 102 (iter 86)** — iter 86's 4-method same-stack matrix is the
   cell-level analog of iter 90's 9-method single-batch matrix. Together
   they cover both axes (same-stack cross-method and single-batch
   cross-method).
3. **P6 row 92 (iter 78)** — iter 78's field-coverage audit scored
   `measured_coverage: 0.0` on every `zvf130_*` entry. Iter 90 closes that
   gap to `measured_coverage: 1.0` on the 5 patched entries.

## Reproducibility

- Script: `scripts/p5p8/p6_iter90_zvf130_measured_vs_claimed.py` (~280 LoC, stdlib only)
- Outputs:
  - `experiments/results/p5p8/p6_iter90_zvf130_measured_audit.tsv` (9 rows)
  - `experiments/results/p5p8/p6_iter90_zvf130_measured_pairs.tsv` (36 rows)
  - `experiments/results/p5p8/p6_iter90_zvf130_claim_vs_measured.tsv` (9 rows)
  - `experiments/results/p5p8/p6_iter90_zvf130_measured_audit.json` (machine-readable)
- Patched entries (now carry `outcomes.zvf_risk_mean` + `delta_vs_grpo_*` block):
  - `registry/entries/zvf130_cppo.json`
  - `registry/entries/zvf130_es.json`
  - `registry/entries/zvf130_mcgrpo.json`
  - `registry/entries/zvf130_ngrpo.json`
  - `registry/entries/zvf130_scafgrpo.json`
- Paper section: `paper/sections/p6_iter90_zvf130_measured_audit.tex`
- Paper rebuild: `paper/paper_P6_registry.pdf` → **46 pages / 0 errors / 0 undefined citations** (was 45, +1 page from new subsection)
- Citation: every `delta_<method>.json` entry references an arXiv paper
  already in `paper/references.bib` (verified via
  `verified_citation_present = True` on every patched claim).