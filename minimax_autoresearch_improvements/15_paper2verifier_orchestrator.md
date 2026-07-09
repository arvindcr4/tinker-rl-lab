# Patch proposal — Paper2Verifier-orchestrator (B1 target)

**Status: proposal only.** Do NOT apply yet; iter147 prototyped the standalone
verifier in `scripts/berkeley/paper2verifier.py` (5/5 DECISIVE on Pillar-3
iter127 / iter135 / Pillar-2 ZVF). This file proposes how the
`minimax_autoresearch/` driver itself could adopt Paper2Agent-style hosting
of its own benchmark papers as MCP-verifiable artifacts.

## Why now

Iter147 shows that given a Pillar-3 paper's headline TSVs, the
Paper2Verifier pipeline can extract the recipe (12/12 fields, recall=1.000),
re-fit it on the same slice (R²=0.854), transfer it to a held-out slice
(R²=0.812), survive recipe stress (0 failures), and reuse the extractor
unchanged on a second pillar (Pillar-2 ZVF, 9/9 recall). The framework is
ready to host our paper as an artifact.

## Proposal — three additions to the orchestrator

### 1. Recipe extractor as a first-class driver state
Promote the extractor to a driver step that runs after every paper-section
patch is merged. The driver would:
- parse every `paper/sections/*.tex` and `paper/sections/_shared_*.tex` for
  headline claims and numeric anchors
- emit `paper/_recipes/<section>.json` containing the implicit (key,
  value) recipe
- run `scripts/berkeley/paper2verifier.py` against the recipe + the
  matching `experiments/results/<slice>.tsv` and emit a verification
  report (R², slope_rel_err, field recall, stress-test failure count)
- post the report to `paper/_recipes/verification_report.json`

This would mean a paper section cannot merge unless the corresponding
recipe re-verifies against the data.

### 2. MCP-server stub for the Pillar-3 paper
Create `paper/mcp/pillar3_server.py` (a ~80-line FastMCP stub) exposing:
- `get_zvf(G, T, library) -> float` — ZVF predicted from joint fit
- `get_optimal_G(T) -> int` — G* from B_optimal_G regression
- `get_joint_fit() -> dict` — full iter127 joint-fit coefficients
- `verify_recipe(slice_name) -> dict` — runs Paper2Verifier end-to-end

This is the minimal "host a paper as an MCP server" surface. Iter147 shows
each of these is a one-liner once the recipe is in place.

### 3. Per-iteration verifier gate
Add a `verifier_gate` step to `minimax_autoresearch/driver.py` that
refuses to advance if:
- recipe extraction recall < 0.80 on the latest headline TSVs
- OLS-verifier R² on the same slice falls below the published value by
  > 20%
- held-out-slice R² < 0.50

This makes "the paper is verifiable" a hard guarantee rather than a
follow-up audit.

## What this does NOT do (explicit non-goals)

- Does NOT replace the human-written scripts in `scripts/berkeley/`. The
  verifier is an *audit* of those scripts, not a substitute.
- Does NOT run real Tinker training. Iter147 uses synthetic lattices that
  match the published coefficients; a real run would invalidate the
  verifier's "same-slice" baseline (the human-built R² would shift).
- Does NOT change the 4 papers' claims. It strengthens the
  reproducibility section by giving a *machine-checkable* claim-to-data
  link.

## Risk

The biggest risk is false confidence — the OLS verifier re-fits the same
generation law and recovers the recipe tautologically. A real cross-domain
test (e.g., applying the Pillar-3 recipe to GSM8K rollouts, not the
arithmetic synthetic lattice) would distinguish "verifier memorized the
paper" from "verifier captured the scaling law". This is row 16 territory.

## Recommendation

**Defer until row 16 (cross-domain verifier test) lands DECISIVE.**
File this proposal under `minimax_autoresearch_improvements/` so the
patch is discoverable, but do NOT edit the driver.

## Cross-reads

- Row 09 (Jiao F25 L4): Paper2Verifier is the *operational* form of the
  "verifier IS the recipe" claim — the recipe is what makes the verifier
  auditable.
- Row 07 (Sida Wang F25 L8): Paper2Verifier complements the Adding
  Error Bars audit — one audits noise, the other audits extractability.
- Row 12 (B-SYNTH CDH): the CDH sharpens "what counts as a verifier
  failure" for the orchestrator gate.
- Row 14 (Hajishirzi SP25 L4): Paper2Verifier is the only artifact that
  survives the Ivison 4-axis factorization cleanly — recipes live in
  axis 3 (reward/verifier), which is the second-most-leveraged axis.