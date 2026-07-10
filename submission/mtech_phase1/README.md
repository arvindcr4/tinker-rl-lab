# Phase-1 submission handoff

This folder is the handoff index for Arvind C R's individual M.Tech Project
Phase-1 defense.

## Final artifacts

| Artifact | Canonical path | Purpose |
|---|---|---|
| Report | \`../../output/pdf/Phase1_Project_Report_ZVF.pdf\` | 49-page A4 thesis report |
| Deck | \`../../outputs/PESU_MTech_Phase1_ZVF_Defense_ArvindCR.pptx\` | 13-slide defense presentation |
| Demo | \`../demo/demo.sh\` | deterministic offline evaluator demo |
| Runbook | \`../demo/DEFENSE_RUNBOOK.md\` | 90-second defense walkthrough and fallback |
| Bundle | \`../../outputs/PESU_MTech_Phase1_ZVF_Submission_ArvindCR.zip\` | portable handoff package |

## Verified evidence boundary

- P2 is a fixed-schedule, seed-0 diagnostic across four method traces.
- The matched-token curriculum experiment is a three-seed null result.
- P3 is exploratory and does not identify a universal group-size optimum.
- P1 is a four-seed scaled verification that overturns the toy conclusion.
- P8 is a correlated row-wise method-trace classification proof of concept.
- Disputed historical campaign aggregates are excluded from the report and deck.

## Reproduce the demo

From the repository root:

\`\`\`bash
./submission/demo/demo.sh
python3 -m unittest discover -s submission/demo/tests -v
\`\`\`

The expected final demo line is \`DEMO STATUS: PASS\`.

## Before portal upload

The files are technically prepared, but the following remain human-owned:

1. guide approval and signatures;
2. declaration date and examiner fields;
3. institutional similarity/plagiarism certificate, if requested; and
4. portal-specific filename, size, slot, and deadline confirmation.
