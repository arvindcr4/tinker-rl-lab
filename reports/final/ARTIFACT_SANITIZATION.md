# Artifact Sanitization Note

The public/submission archives are intended to be reviewable without exposing local credentials or private service state.

## Excluded Patterns

The packaging step excludes these credential and local-state patterns:

- `.env`
- `.env.*`
- `.tinker_api_key`
- `*.key`
- `*.pem`
- `*.p12`
- `*.p8`
- `*.token`
- `*.secret`
- `wandb/`
- `.modal/`
- local virtual environments, caches, and generated build artifacts

The repository `.gitignore` also contains these patterns so future public exports should not accidentally include them.

## Credential Handling

Local credential files were detected during artifact hygiene checks, but their contents were not inspected, printed, copied, or included in the submission archives. Before any public release, revoke or rotate any Tinker, W&B, HuggingFace, Modal, or other API tokens that may have existed in local files or prior private archives.

## Verification Commands

Run these checks after rebuilding the submission artifacts:

```sh
unzip -l reports/final/submission_uploads/data_and_code.zip | rg '(^|/)(\.env|\.env\.|\.tinker_api_key|.*\.key|.*\.pem|.*\.token|.*\.secret|wandb)(/|$)'
unzip -l reports/final/submission_uploads/overleaf_document.zip | rg '(^|/)(\.env|\.env\.|\.tinker_api_key|.*\.key|.*\.pem|.*\.token|.*\.secret|wandb)(/|$)'
```

Expected result: no output.

The all-files submission zip contains the nested PDF/source/data archives, so check the nested archives directly rather than relying only on the top-level listing.
