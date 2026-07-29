# Hugging Face evidence audit for NeurIPS submission 36320

**Audit date:** 2026-07-28  
**Access mode:** read-only Hugging Face Hub SDK `1.11.0`; the token was read through a no-echo prompt, passed only to a child process, and never installed with `hf auth login`. Downloaded manifests were held in a temporary directory and deleted when the process exited.  
**Scope:** live provenance for the 40-unit post-submission E1 audit and any discoverable Hub linkage for the claimed five-seed Qwen3-8B comparison.

## Decision

The live Hub inventory independently confirms the repository's frozen E1 verification ledger. All 40 accepted units have distinct private model repositories; every pinned commit resolves exactly; every repository contains checkpoint trainer states at steps 5, 10, 15, 20, 25, and 30, a final adapter, and a final `run_manifest.json`. All 40 remote manifests are byte-identical to the local manifests and to the SHA-256 values recorded in `campaign-verification.json`. Their arm, seed, stack and unit fingerprints, 500-item held-out trace, held-out score, and confirmatory evidence class all agree with the frozen ledger.

This strengthens E1's provenance claim. It does not turn E1 into evidence for the submitted runner, the early-collapse rule, reference-KL dependence, or a use-inspired controller.

The Hub inventory does not repair the claimed 92.6% versus 92.1% five-seed comparison. No model repository outside E1 has an identifier containing `matched`, `36320`, or `reinforce`; more importantly, the W&B backfills and Tinker records contain no Hugging Face repository or commit identifier. An artifact with no recorded link cannot be assigned to seeds 42--44 by repository-name inference.

## E1 live verification

| Check | Result |
|---|---:|
| Frozen accepted units | 40 |
| Expected repositories visible to the authenticated account | 40/40 |
| Pinned commits resolving to the recorded SHA | 40/40 |
| Repositories private | 40/40 |
| Checkpoint trainer states at steps 5, 10, 15, 20, 25, 30 | 40/40 |
| Final adapter present | 40/40 |
| Final run manifest present | 40/40 |
| Remote manifest SHA matches frozen campaign receipt | 40/40 |
| Remote manifest SHA matches local manifest | 40/40 |
| Confirmatory schema/evidence class, arm, seed, fingerprints, score, and 500-row trace agree | 40/40 |

The account contains 49 repositories with the E1 naming prefix. Nine are not members of the frozen 40-unit accepted ledger. This is expected evidence of attempts/preflights rather than a reason to expand the analysis: acceptance is governed by the immutable unit ledger, not by repository-name discovery. The aggregate contains exactly eight accepted seeds for each of GRPO, DAPO, GSPO, Dr.GRPO, and AERO.

## Sidecar discrepancy

At the pinned final commits, `evaluation/progress.json` is present in 36/40 accepted repositories. It is absent for GRPO seeds 23, 37, 53, and 71. This progress file is a resumability sidecar, not the final evaluation receipt. Each of the four affected repositories still has:

- all six checkpoint trainer states;
- the final adapter;
- a final `run_manifest.json` containing the complete 500-row held-out trace; and
- a remote manifest hash identical to the local ledger and frozen campaign receipt.

Accordingly, the missing sidecars do not alter any held-out score or accepted-unit count, but future artifact language should distinguish final evidence from optional evaluation-resume state. Do not claim that every auxiliary sidecar exists in all 40 repositories.

## Rebuttal-safe use

The existing bounded sentence is supported:

> All 40 post-submission E1 units reconcile W&B, six checkpoint states, stack/treatment fingerprints, and 500-row evaluation traces.

If provenance detail is useful in an artifact appendix, the more exact version is:

> A fresh authenticated Hub audit resolved all 40 frozen repository/commit pairs. Each pinned commit contains checkpoint trainer states at steps 5--30, a final adapter, and a final 500-row manifest whose SHA-256 matches both the local manifest and the frozen campaign receipt. Four units lack only the separate evaluation-resume sidecar.

Do not use the Hub audit to claim that the private evidence is independently visible to reviewers. Reviewer-verifiable use requires an anonymized artifact or another access path allowed by the venue.

## Security note

The credential was supplied in chat and should be rotated. It is not reproduced in this artifact.
