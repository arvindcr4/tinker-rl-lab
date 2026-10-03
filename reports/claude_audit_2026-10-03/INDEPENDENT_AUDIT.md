# Independent Audit — claude-auditor (squad inspector), 2026-10-03

**Verdict: PASS with 4 process findings.** All 9 repository audits pass, the
targeted and full test suites pass on both the working tree and a clean `HEAD`
export, and the latest commits match `SUBMISSION.md` / `FINAL_HANDOFF.md`
without needing changes to either. No defects in the committed code. The
findings concern shared-tree hygiene and a stale coordination log.

`HEAD` moved during the audit: `29bf5ee43` → `28a6c02b7`
(`test(critic): synthetic convergence + determinism measurements`, Muse CLI).
Results below are for `28a6c02b7` unless stated otherwise.

## 1. Repository audits — `platform_local/run_all_audits.py`

Run with `uv run --no-sync python platform_local/run_all_audits.py` (bare
`python` is not on PATH in this shell).

| Audit | Issues |
|---|---|
| claim | 0 |
| sync | 0 |
| anon | 0 |
| strength | 0 |
| package | 0 |
| workflow | 0 |
| export_guard | 0 |
| caveat | 0 |
| scientific | 0 |

`audits_passing=9/9`, `suite_issues=0`. Note: the run reads the working tree,
which includes another agent's uncommitted `SUBMISSION.md` edit (finding F2).

## 2. Tests

| Scope | Tree | Result |
|---|---|---|
| `test_critic.py test_grpo_loss.py test_grpo_cli.py test_grpo_module.py` | working tree | 195 passed, 1 skipped |
| same four files | clean `git archive HEAD` export (no untracked `tests/conftest.py`, no unstaged edits) | 195 passed, 1 skipped |
| `tests/` (full) | working tree | 615 passed, 2 skipped, 1 warning (scipy precision-loss warning in `test_stats_input_guards`, expected for constant inputs) |
| `ruff check platform_tinker/tinkerrl/ tests/test_critic.py tests/test_grpo_module.py` | working tree | all checks passed |

The clean-export run confirms the committed critic/GSPO code and its tests do
not depend on the untracked `tests/conftest.py` or the unstaged
`tests/test_grpo_loss.py` additions.

## 3. Commit ↔ handoff alignment

**`3787d09f4` feat(critic): value head + pretraining + loop integration.**
Adds `PromptValueCritic` (CPU hash-embedding + MLP predicting E[reward | prompt]),
critic pretraining, per-step fit with `train/critic_loss` / `train/critic_ev`,
and `.critic.pt` state with fail-closed resume. `critic_enabled` defaults to
false. `research/notes/value-track-status-2026-10-03.md` states the only
measurement is synthetic and says "Do not put this baseline on a defense slide
or in the thesis."

- `SUBMISSION.md` makes no claim about this critic. Correct: it is post-thesis,
  default-off, and has no live evidence.
- The thesis's "value head" references (`ch04_methodology.md`,
  `ch06_results_core.md`, `ch_back_run_registry.md`) are to the Modal same-stack
  PPO arm (`modal_samestack_*`), a different implementation. No thesis text cites
  `PromptValueCritic` or `critic_enabled`. **Aligned.**

**`29bf5ee43` docs(suites): decide frontier mapping, re-verify 4 pending receipts.**
Maps `frontier_swe_eval` to `Proximal-Labs/frontier-swe @422b9bb`; it stays
pending on license, and BinaryAudit / LifeSciBench / SDAB stay pending.

- `SUBMISSION.md` §limits lists E3 (SDAB), E7 (BinaryAudit), E8 (LifeSciBench)
  and E14 (FrontierMath hosted evaluation) as external-closure items. These are
  thesis experiment contracts, a different layer from the suite receipts, and
  the commit freezes none of them, so nothing in `SUBMISSION.md` is contradicted.
  `frontier_swe_eval` (SWE suite) and E14 (FrontierMath) are separate artefacts.
  **Aligned.**

**`FINAL_HANDOFF.md`** carries a "Historical handoff (April 2026), not the
current submission" banner that points to `SUBMISSION.md`. Neither commit
touches it, and it should not be updated. **Aligned.**

## 4. Findings

**F1 — Shared git index can leak other agents' changes into a commit (medium).**
At session start, `research/notes/value-track-status-2026-10-03.md` and
`tests/test_critic.py` were *staged* by another agent. Those files were later
committed as `28a6c02b7`. A plain `git add <mine> && git commit -m …` in that
window would have swept them into the wrong commit under the wrong author role.
The coordination rules ban `commit -a` but not this case.
*Recommendation:* add a rule to §4 of `PARALLEL_SESSIONS_COORDINATION.md` that
every commit uses a pathspec (`git commit -m "…" -- <owned paths>`). This
audit's own commit does so.

**F2 — Uncommitted `SUBMISSION.md` edit links to an untracked directory (medium).**
The working tree changes `SUBMISSION.md` line 29 to link
`reports/final_defense_2026-10-03/README.md`, and that directory is untracked
(`?? reports/final_defense_2026-10-03/`). If `SUBMISSION.md` is committed first,
the link dangles in a fresh checkout. `SUBMISSION.md` itself warns that "ignored
files on one machine are not proof that a fresh checkout works", and
`tests/test_thesis_submission.py` checks the Git index.
`SUBMISSION.md` is also not in any agent's declared ownership in §2.
*Recommendation:* commit the defense directory and the `SUBMISSION.md` edit
together (Grok CLI scope plus an explicit owner for `SUBMISSION.md`), then
re-run `run_all_audits.py` and `test_thesis_submission.py` against the
committed state.

**F3 — Tests rely on an untracked `tests/conftest.py` in the working tree (low).**
It pre-imports `torch._dynamo` to guard against `patch.dict(sys.modules)`
eviction. Commit `3787d09f4` already fixed the root cause with a surgical
`sys.modules` patcher, and the clean-export run passes without the conftest.
The file is therefore defensive, not load-bearing, for the audited suites. It
is still unowned, and it changes import behaviour for every test.
*Recommendation:* the owner (Antigravity, "test configuration") should either
commit it with a rationale or delete it, so CI and local runs match.

**F4 — Coordination log is stale (low).**
`PARALLEL_SESSIONS_COORDINATION.md` §3 records "193 passed tests in `tests/`
(1 skipped)". The actual figure is 615 passed / 2 skipped for `tests/`, and the
four targeted files alone give 195. The log also omits commit `28a6c02b7`. The
file itself is untracked.
*Recommendation:* update the test count and the milestone list, and track the
file, or mark it as a local scratch log.

Also uncommitted and outside this audit's scope:
`platform_hybrid/sem 4 work/submissions/mtech-final-review/README.md`
(modified), plus `platform_hybrid/experiments/{modal/modal_samestack_gsm8k_cot_v2.py,
results/samestack_scope_check.json, scripts/}` (untracked).
`ch06_results_core.md` cites `samestack_scope_check.json` under
`outputs/PES_Phase2_Third_Review_2026-09-24/analysis/`, not under
`platform_hybrid/experiments/results/`. If the new copy is meant to replace that
source, the citation path needs to be updated to match.

## 5. Reproduce

```bash
uv run --no-sync python platform_local/run_all_audits.py
uv run --no-sync python -m pytest tests/test_critic.py tests/test_grpo_loss.py \
  tests/test_grpo_cli.py tests/test_grpo_module.py
# clean-HEAD check (no working-tree state)
D=$(mktemp -d) && git archive HEAD platform_tinker tests pyproject.toml | tar -x -C "$D" \
  && (cd "$D" && PYTHONPATH="$D:$D/platform_tinker" "$OLDPWD/.venv/bin/python" -m pytest \
  -p no:cacheprovider tests/test_critic.py tests/test_grpo_loss.py tests/test_grpo_cli.py \
  tests/test_grpo_module.py -q)
```
