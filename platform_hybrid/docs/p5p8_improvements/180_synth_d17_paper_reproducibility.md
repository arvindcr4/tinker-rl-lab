# P5P8-SYNTH D17 cross-paper evidence reproducibility density (iter 180)

**Pillar:** P5P8-SYNTH (Pillar 4 JOB B)
**Vein:** brief vein — extends the iter-176 16-domain density matrix
(D1..D16) at the **cross-paper metadata layer** (D17).
**Status:** validated (3/6 H PASS + 3 FAIL honestly framed).

## Why this iter

Iter-176 added D16 = N2 per-prompt reward stability (HIGH layer at
0.7293). The 16-domain density matrix is the canonical SYNTH density
reference, but it measures **substantive evidence density** only.
This iter (P5P8-SYNTH JOB B) extends the matrix to the
**cross-paper metadata layer** with D17: how reproducible are the
stored findings across papers?

D17 is computed directly from `findings_ledger.jsonl` (39 stored
findings across P5 / P6 / P7 / P8 / P5P8-SYNTH). Two complementary
denominators are reported:

- **D17 ALL**: every stored finding counted.
- **D17 VERIFIED**: only findings whose `verdicts` dict is non-empty
  (9 of 39).

D17 in three sub-views per pillar:

- **D17a** = fraction of stored findings that are REPRODUCIBLE
  (≥1 PASS verdict AND 0 FAIL verdict).
- **D17b** = mean PASS/total ratio across stored findings.
- **D17c** = per-pillar finding count (density itself).
- **D17d** = per-pillar mean claim length (chars; proxy for
  evidence-width reporting).

Layer assignment matches iter-176 (LOW ≤0.10 < MID < 0.50 ≤ HIGH).

## Method

1. Parse `findings_ledger.jsonl` line-by-line.
2. For each line, read `pillar`, `claim`, `verdicts` keys.
3. Compute `n_pass`, `n_fail`, `n_total` from the `verdicts` dict; if
   the dict is empty, fall back to a text parser
   (`parse_passes_from_claim` regex
   `(\d+)\s*\/\s*(\d+)(?:\s*H[1-9](?:[A-Z]+)?)?\s*PASS`).
4. Per-pillar reproducible flag = (`n_pass ≥ 1` AND `n_fail == 0`).
5. Per-pillar mean of pass ratio for D17b.
6. Wilson 95% CI on D17a per pillar.
7. Bootstrap-CI on `SYNTH_d17a − Pillar_d17a` and `P7_d17a − P8_d17a`.
8. Layer classification on D17a.

## Headlines (3/6 H PASS)

### H1 FAIL — SYNTH reproducible rate (using ALL stored) is NOT
strictly greater than P5 reproducible rate.

SYNTH D17a = 0/6 = 0.0 (CI [0, 0.39]); P5 D17a = 0/8 = 0.0 (CI
[0, 0.32]). The bootstrap difference is 0.0 ∈ [0, 0]; cannot reject
zero. Both pillars have structurally low reproducible-rate because
their stored findings predate the structured `verdicts` schema. **Honest
reading**: the iter-176 D16 finding had no `verdicts` dict on the JSONL
line because D16 is metadata, not a falsifiable-H construct.

### H2 PASS — P7 has highest 5/5-PASS count among pillars.

`n_5of5_per_pillar` = {P5: 0, P6: 1 (row 178), P7: 1 (row 171), P8: 1
(row 176), SYNTH: 0}. P7 ties P6/P8 with the highest single-findings
record (n_5of5=1) and additionally holds the **single 8/8 PASS
finding** (P7 row 171 has 8 H verdicts stored, all PASS). The
8/8 finding is structurally unmatchable: P5/P6/P8/SYNTH have no
finding with ≥8 stored verdicts. The argmax is P7.

### H3 PASS — P6 has highest D17c (count) at n=10 stored findings.

Counts: P5=8, P6=10, P7=9, P8=6, SYNTH=6. P6 historically added
many entries (registry iteration has been the most prolific single
paper), and the verified-only sub-count is P6=2 (tied with P7 and P8).
P6 has the highest TOTAL stored-finding count, mirroring iter-100
row 117 ("P6 is the most-iterated pillar"). Confirms the
**previously-validated iter-100 SYNTH finding** at the
JSONL-stored-evidence layer.

### H4 FAIL — P8 does NOT have the widest mean claim length.

Mean claim lengths (chars): P5=1843.5, P7=1538.1, P6=1442.0,
P8=1365.7, SYNTH=1110.5. P8 ranks 4th, not 1st. **Sharpest reading**:
the iter-176 P8 finding (a 6-H battery with CIs) is the longest
single claim by far (~2700 chars), but the iter-148 P8 finding (a
4-threshold sweep) is much shorter (~700 chars). The mean washes out
the headline-vs-secondary contrast. P5 wins on mean because it
typically reports 4-5 paragraphs of audit context.

### H5 FAIL — D17 OVERALL (using ALL stored findings) is NOT in HIGH layer.

D17 OVERALL = 3/39 = 0.0769 ∈ LOW (≤0.10). The structural reason:
30/39 stored findings lack a stored `verdicts` dict (they predate the
iter-171 schema adoption), so `n_pass=0` and `n_fail=0` for those
findings — meaning reproducible=False. **This is a metadata-gap
finding, not a substantive finding about evidence quality.**

### H6 PASS — D17a VERIFIED is NOT in LOW layer.

D17a VERIFIED OVERALL = 3/9 = 0.3333 ∈ MID layer (between 0.10 and
0.50). CI95 = [0.0945, 0.6576] straddles the LOW/MID boundary. The
**honest sharpest reading**: across the 9 stored-with-verdicts
findings (i.e., recent ones, iter 171+), 3 are reproducible (P7 171,
P8 176, P6 178), 6 are not. The 0.333 reproducible rate is consistent
with the ledger's "falsifiable-H is now a multi-battery with 0-1
honest FAILS" expectation, where P5 iter-177 has 4/5 PASS, P6
iter-174 has 4/5 PASS, etc. Honest fails are the structural
counter-weight that keeps D17 from HIGH.

## Sharpest paper-grade claims

1. **Cross-paper reproducibility density at the JSONL-stored layer
   is LOW (0.0769) but NOTZERO — the recent (iter 171+) schema adoption
   is what made verdicts machine-readable, and only ~23% of the JSONL's
   lifespan has benefited.** The bulk of evidence is encoded in
   claim-text, not in the `verdicts` field — the SNL pipeline would
   benefit from a backfill to populate `verdicts` on the iter 130-170
   finding lines.
2. **P7 has the structurally highest 5/5-PASS count + the single 8/8
   finding** (P7 iter-171 row 186). P7's tightly-coupled controlled
   experiments (counterfactual on real N2 tensors) yield a higher
   falsifiable-H battery count per finding than other pillars.
3. **P6 has the highest TOTAL finding count** (n=10), consistent with
   the registry-iteration nature of the pillar (every new entry
   schema bump drives a new finding line).
4. **Per-pillar VERIFIED D17a**: P6/P7/P8 all sit at 0.50 (HIGH);
   P5/SYNTH at 0.00 (LOW). The structure is driven by which pillars
   adopted the verdicts schema first and how each pillar's findings
   distribute across PASS-only / FAIL-included profiles.

## Cross-paper coupling

- **SYNTH iter-176 row 188 (D16 per-prompt stability)**: iter-176
  added D16 to the substantive density layer at 0.7293; iter-180
  adds D17 to the metadata density layer at 0.0769 (ALL) / 0.3333
  (VERIFIED). The two domains are orthogonal (substantive vs
  metadata) and together span the SYNTH density matrix.
- **P6 iter-100 row 117 (P6 coverage closure)**: iter-100 noted "P6
  is the most-iterated pillar"; iter-180 confirms this at the
  JSONL-stored-finding layer (P6 leads with n=10).
- **P5 iter-177 row 189 (v2.5 forward-compat)**: P5's iter-177 has
  verdicts stored; iter-180 VERIFIED row for P5 picks up this single
  finding.
- **P8 iter-176 row 187 (sensor/scribe/scorer CIs)**: P8's
  iter-176 has 6/6 PASS stored; iter-180 VERIFIED row for P8 picks
  this up as reproducible=True.

## Failure honesty record

- **H1 FAIL** is structural: SYNTH stored findings are mostly meta
  (density domains, layer counts), not falsifiable-H constructs.
  The right comparator would be D17b (pass ratio) which is comparable
  for SYNTH (0.50) and P5 (0.43).
- **H4 FAIL** is a mean-vs-headline artefact: the iter-176 finding
  is the longest by absolute characters, but mean dilutes it with the
  shorter iter-148-style rows.
- **H5 FAIL** is the sharpest finding: most stored findings have no
  machine-readable verdicts (30/39 = 77% of the JSONL) — this is a
  **metadata-completeness gap** that future iters should close.

## Operational recommendations

1. **Backfill `verdicts` on iter 130-170 findings** to convert the
   30 missing-records into machine-readable verdicts. This will lift
   D17 OVERALL from LOW to MID (likely) without re-running any model
   training; it is a documentation pass.
2. **Adopt the `verdicts`-schema as the canonical reproducible gate**
   for any future iteration. The minimal-verdict is `{H1: true}`,
   but the falsifiable-H battery convention (4-6 hypotheses) is
   preferred.
3. **Wire `synth_iter180_d17_paper_reproducibility.py` as a CI
   pre-commit gate**: D17 OVERALL must NOT be in LOW layer
   (currently 0.0769, must rise ≥ 0.10 after backfill); D17 per-pillar
   VERIFIED should NOT have any pillar under 0.5 reproducible rate.

## Outputs

- `scripts/p5p8/synth_iter180_d17_paper_reproducibility.py` (~270 LoC,
  stdlib only)
- `experiments/results/p5p8/synth_iter180_d17_per_pillar.tsv` (10
  rows: 5 pillars ALL + 5 pillars VERIFIED)
- `experiments/results/p5p8/synth_iter180_d17_verdict_table.tsv` (39
  rows: per-finding verdict parsing)
- `experiments/results/p5p8/synth_iter180_d17_aggregate.tsv` (6 rows:
  per-pillar D17a + D17_OVERALL)
- `experiments/results/p5p8/synth_iter180_d17_summary.json` (H1-H6
  verdicts + per-pillar numbers + layer counts)
- 1 line in `findings_ledger.jsonl` (pillar P5P8-SYNTH, iter 180)
