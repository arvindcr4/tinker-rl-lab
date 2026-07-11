# Security Policy

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability or exposed credential.
Use GitHub's private vulnerability-reporting or Security Advisory flow for the
repository. Include the affected path, reproduction steps, impact, and any safe
mitigation you have tested.

## Secrets and research data

- Keep API keys and tokens in `.env` or the platform's secret manager; `.env` is
  intentionally ignored.
- Treat W&B exports, browser captures, participant data, and raw external
  datasets as potentially sensitive until reviewed.
- Revoke a credential immediately if it appears in a commit, log, notebook, or
  chat transcript. Removing the file from the latest commit is not sufficient;
  rotate the credential and clean history deliberately.
- Do not execute generated model code or untrusted benchmark completions on a
  workstation. Use an isolated container or disposable VM with resource limits.

Supported security fixes target the current `main` branch. Research snapshots
and archived experiment copies are retained for provenance and may not receive
backports.
