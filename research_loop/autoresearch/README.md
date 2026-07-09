# Autoresearch harness (dormant)

The automated research-iteration machinery that produced the Semester 4 iteration ledger
(`../../AUTORESEARCH_FINDINGS.jsonl`, kept at repository root because ~68 documents and
scripts cite that path). Moved here from the repository root during the 2026-07-10 cleanup.

- `autoresearch.sh`, `autoresearch_score.sh`, `autoresearch_runnability.sh`, `autoresearch.checks.sh` — scoring/benchmark loops. Contain hardcoded origin-machine paths (`/Users/arvind/paper/tinker-rl-lab`); historical, not runnable as-is.
- `run_oracle_*.py` + `oracle_invention_queries.md` — oracle query batch runners; same hardcoded-path caveat.
- `autoresearch.md`, `autoresearch.ideas.md`, `autoresearch-dashboard.md`, `autoresearch.jsonl`, `autoresearch.config.json` — loop state, ideas backlog, and dashboard notes.
- `autoresearch_config_audit.py` — standalone audit for `autoresearch.md` (not part of `run_all_audits.py`'s suite); run from the repo root with `PYTHONPATH=.`.
- `implement_swarm.py` — one-off swarm driver.
