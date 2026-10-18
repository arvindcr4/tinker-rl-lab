# P1 identifiability note: handoff resolution

The earlier proposed replacements predated the committed P1 note rewrite. They are superseded by the current manuscript, per-finding JSONL and numeric ledger.

- **589 / 597:** Resolved with the note-specific bounded three-parameter refit, full leave-Nemotron size diagnostics and exact architecture comparisons. Run python tools/review_ledger/P1_identifiability_note_refit.py from the repository root; outputs are in platform_hybrid/experiments/results/review_2026-10-18/P1_identifiability_note/refit.json. This is not the monotone-offset fit used in the longer scaling manuscript. The earlier mean-reward Monte Carlo p=0.358 used 20,000 draws and a different RNG stream; the note receipt uses 10,000 draws with seed1092026 and returns p=0.343. Neither supports a positive trend.
- **595 / 603:** Recomputed fit boundaries and interval rounding; documented exact versus Monte Carlo permutations and conditional-OLS limitations.
- **598:** Already resolved in the prior committed note revision (eight embedded references); no shared bibliography edits are needed.
- **588:** Still open for missing empirical provenance. The panel table and hashed local receipt are available, but checkpoint revisions, prompt IDs, backend/decoding/stopping metadata and a frozen archival release remain unavailable. Local trace arithmetic is not independent authentication.

The earlier claim that conflicting W&B run IDs prove mislabelled trace identities is not adopted: another audit reports matching Tinker base-model/rank records. Neither resolves the absence of exact retained rollout provenance. Do not silently relabel runs.

No GPU training, paid sampling or external publication was performed. The longer P1 scaling manuscript and Claude's P6/P8 work remain untouched.
