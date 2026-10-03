# Scaffold — rl-verification (2026-10-03)

## User Prompt (VERBATIM — gospel)

Verify the DAPO / Dr. GRPO / REINFORCE++ / VAPO paper claims cited by
`tinker-rl-lab` grpo.py comments and flags against full text, fix any
mis-citations, and curate the vault.

## Run config

- vault_tag: rl-verification
- modality: collect (claim-by-claim verification verdicts)
- Method: 4 parallel citation-verifier agents, each fetching one arXiv PDF
  via `hyperresearch fetch --tag rl-verification` and checking exact
  claims (section numbers + verbatim quotes) into /tmp verdict files.

## Outcome

- 10 of 11 claims CONFIRMED; 1 DENIED (VAPO μ=0.01 → paper uses μ=0.1,
  §5.1 item 6). Code comment corrected.
- 4 source notes + 2 extract notes + 1 final report, all status=review.
- Final report: `research/notes/final_report_rl-verification.md`.
