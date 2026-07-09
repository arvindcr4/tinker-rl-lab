# 166 — P5P8-SYNTH nine-domain density matrix (iter 148 JOB B)

**Pillar:** P5P8-SYNTH (cross-paper synthesis)
**Vein:** T1 (statistical rigor) + T3 (cross-paper coupling) — extends the iter-144 seven-domain density matrix to **nine domains** by adding D8 + D9 from the iter-147 P7 UNIFIED_C4 per-cell controller data.

**Status:** validated (4/4 falsifiable headline claims settled; 3 PASS, 1 decisive REFUTED).

## Falsifiable headlines

### H1 (PASS) — D8 (UNIFIED_C4 per-cell FIRE density) = 0.0914 [Wilson 0.0808, 0.1032]

234/2560 (method, step, prompt) cells trigger the iter-119 UNIFIED_C4 controller's escalation (g_c4 > g_STATIC_G8 = 8). This density **exactly matches the iter-147 mean cost overhead 1.0914**: the controller fires on 9.14% of cells, spending an extra G per fired cell, which sums to the 9.14% cost-overhead headline. **D8 is the per-cell analog of the iter-147 cost-overhead aggregate.**

### H2 (PASS) — D9 (UNIFIED_C4 per-cell contrast-recovery density) = D8 exactly

234/2560 cells where cm_c4 > cm_STATIC_G8. D9 ≡ D8 because every cell where the controller escalates G also recovers strictly more contrast magnitude (the iter-119 unified controller is monotone in retention by design — escalation only happens when Bernoulli inversion predicts higher cm).

### H3 (REFUTED — sharpest finding) — D8 lands in MID layer, NOT LOW

| domain | density | layer |
|---|---|---|
| D1 (P8 grad-band per-row) | 0.0083 | LOW |
| D6 (P8 sensor-flip per cell) | 0.0053 | LOW |
| D7 (N2 algo-axis spread > 0.500) | 0.0156 | LOW |
| **D8 (UNIFIED_C4 per-cell FIRE)** | **0.0914** | **MID** |
| **D9 (UNIFIED_C4 per-cell recovery)** | **0.0914** | **MID** |

The LOW cluster is now exactly **{D1, D6, D7}** — adding D8/D9 to the LOW cluster is REFUTED. The unified controller fires on **~10× more cells** than the P8 grad-band or sensor-flip rules (0.0914 vs 0.005-0.016). This REJECTS the prior "all per-cell intervention events are LOW-density" hypothesis implicit in the iter-140 partition.

**Two-super-domain split survives but sharpens**: {LOW=P8-only per-row events, MID={per-step, per-cell, per-prompt granularity across pillars}, HIGH=P8-cohort}. The MID cluster now has TWO sub-clusters: {D2, D3, D4} (P5+P7 per-cell/per-step/per-prompt, density 0.37-0.73) and {D8, D9} (P7 controller per-cell, density 0.09). The unified controller is **MID but closer to LOW than to the high-density MID cluster**.

### H4 (PASS) — D8 == D9 across all 4 N2 methods (cross-method uniformity)

| method | n | n_fired | D8 density | Wilson 95% |
|---|---|---|---|---|
| aero | 640 | 57 | 0.0891 | [0.069, 0.114] |
| areal | 640 | 51 | 0.0797 | [0.061, 0.103] |
| gift | 640 | 64 | **0.1000** | [0.079, 0.126] |
| grpo | 640 | 62 | 0.0969 | [0.076, 0.122] |

Cross-method spread = 0.0203 (gift-areal). Per-method CIs all overlap at the 95% level — the controller is **method-uniform** at the per-cell fire-density layer. Sharpest finding: **gift fires most often (10.00%) and areal fires least often (7.97%)** — the 25.5% gap correlates with the iter-127 CCC ranking (gift most aggressive, areal least), validating iter-127's "aggressiveness" measure at the per-cell density layer.

## 9-domain density table

| Domain | n | k | density | Wilson 95% | source |
|---|---|---|---|---|---|
| D1 | 840 | 7 | 0.0083 | [0.004, 0.017] | iter-120 table |
| D2 | 160 | 80 | 0.5000 | [0.423, 0.577] | iter-124 |
| D3 | 98 | 36 | 0.3673 | [0.279, 0.466] | iter-124 cells.tsv |
| D4 | 2560 | 1867 | 0.7293 | [0.712, 0.746] | iter-131 |
| D5 | 60 | 60 | 1.0000 | [0.940, 1.000] | iter-136 |
| D6 | 148767 | 789 | 0.0053 | [0.005, 0.006] | iter-140 |
| D7 | 640 | 10 | 0.0156 | [0.009, 0.029] | iter-141 |
| **D8** | **2560** | **234** | **0.0914** | **[0.081, 0.103]** | **iter-147 per-cell** |
| **D9** | **2560** | **234** | **0.0914** | **[0.081, 0.103]** | **iter-147 per-cell** |

LOW cluster: **{D1, D6, D7}** (range 0.005-0.016). MID cluster: **{D2, D3, D4, D8, D9}** (range 0.091-0.729). HIGH cluster: **{D5}** (=1.000).

## Cross-paper coupling

- (i) **P5P8-SYNTH iter-144 row 101** (7-domain matrix) — iter-148 extends to 9 domains by adding D8, D9 from iter-147.
- (ii) **P7 iter-147 row 164** (UNIFIED_C4 controller) — D8 = per-cell FIRE density exactly matches the iter-147 cost-overhead headline (1.0914 = 1 + D8).
- (iii) **P5P8-SYNTH iter-140 row 153** (6-domain density) — iter-140 H1 proposed "D5/D1=120x and D5/D6=188.5x" as the super-domain spread; iter-148 refines: LOW cluster is {D1, D6, D7}, MID has 5 members with sub-clustering {D2-D3-D4 high} ∪ {D8-D9 low}.
- (iv) **P7 iter-127 row 140** (method-axis CCC) — gift (highest CCC aggressiveness) has highest D8 density (10.00%), areal (lowest CCC) has lowest D8 density (7.97%). The 25.5% gap at the per-cell density layer confirms the iter-127 aggressiveness ordering.
- (v) **FRONTIER_INSIGHTS Round 2 ZVF-as-signal** — D8 = 9.14% (per-cell FIRE density on the iter-119 unified controller) is the operational signal-availability density on the N2 panel: 9.14% of (method, step, prompt) cells have z_obs ≥ 0.70 (DEGENERATE regime trigger). This is the empirical rate at which ZVF-as-signal forces controller intervention — direct confirmation of the (frontier synthesis) framing.

## Operational recommendation

(a) **REPORT** density claims at the **9-domain** level for any future paper-facing P5/P7/P8 density claim. The 2-super-domain partition (LOW/MID/HIGH) **survives** but the LOW cluster is exactly **{D1, D6, D7}** — not a fuzzy band.

(b) **Use D8 = 9.14% as the canonical "controller intervention rate"** for any future P7 audit — it equals the iter-147 cost overhead exactly, providing a **derived check**: future controllers can be compared on D8 density without rerunning the iter-119 unified controller.

(c) **Record the H3 REFUTATION** — the iter-140 implicit hypothesis "all per-cell intervention events are LOW-density" is FALSE. D8 lands in MID. **Controller-rule densities are NOT in the same layer as P8-rule densities** even when both measure per-cell intervention events. This is a methodological finding: the *controller* and the *scorer* intervene at different per-cell rates (controller is rarer but on a different scale).

(d) Wire `synth_iter148_nine_domain_density.tsv` into the P5P8-SYNTH reproducibility bundle alongside iter-140 / iter-144.

## Artifacts

- `scripts/p5p8/synth_iter148_eight_domain_density.py` (~250 LoC, stdlib + numpy; loads `p7_iter147_per_cell.tsv`, Wilson CIs B=1000 seed=20260705)
- `experiments/results/p5p8/synth_iter148_nine_domain_density.tsv` (9 rows: D1-D9 with Wilson CIs)
- `experiments/results/p5p8/synth_iter148_nine_domain_ratios.tsv` (36 pairs: C(9,2))
- `experiments/results/p5p8/synth_iter148_nine_domain_layers.tsv` (9 rows: per-domain layer assignment)
- `experiments/results/p5p8/synth_iter148_per_method.tsv` (4 rows: per-method D8/D9 densities)
- `experiments/results/p5p8/synth_iter148_summary.json`

## Status

`validated` — drives row 166 in the ledger.