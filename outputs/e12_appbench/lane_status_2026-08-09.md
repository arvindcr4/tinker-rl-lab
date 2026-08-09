# E12 — `appbench_eval` lane status, 2026-08-09

**Status: BLOCKED. Score: `null`.** No AppBench task was executed, no app was
generated or deployed. What follows is data work and a license finding.

## What now runs

```bash
python3 outputs/e12_appbench/appbench_split_manifest.py build      # emit split_manifest.json
python3 outputs/e12_appbench/appbench_split_manifest.py verify     # re-derive every hash
python3 outputs/e12_appbench/appbench_split_manifest.py disjoint   # emit disjointness_proof.json
python3 -m unittest discover -s outputs/e12_appbench -p 'test_*.py' -v   # 30 tests, OK
cd zvf-program/flagship && python3 -m unittest test_pavlov_appbench_openreward_games_adapter  # 17 tests, OK
```

## The data, characterized

The pinned CSV is on disk and byte-identical to the Hub revision — its recomputed
git blob sha1 equals the `oid` at `de80d5bc…`. It holds **6 tasks and 151 rubric
items** (24/33/22/25/23/24), which exactly matches AfterQuery's published "151
rubric items" figure — good evidence the public file is the complete scored set,
though AfterQuery never says so outright.

Each row gives a markdown `Prompt` (`# Task` / `## Functionality Overview` /
numbered `## Feature Requirements`), an identical per-task CLI addendum naming a
Next.js template in an isolated Docker container with Supabase, and a `Rubric`
whose numbered items are 1:1 with the requirements. **The nine per-tool score
columns are empty**, so the CSV carries no reference answers and no gold-answer
harness validation is possible from it.

What an executor would need but the CSV does not supply: the Next.js template,
the Supabase schema, any time or token budget, the deployment procedure, and the
grading procedure.

## Blocker 3 — half closed, offline, no permissions needed

`split_manifest.json` gives each task a content-addressed 64-hex ID bound to the
dataset revision, a deterministic `split_hash`
(`ee8bb522e82505a1b9690c0b691c36a2bb73a524023ab15dff670fe7cc34601d`) that a test
asserts is byte-compatible with the contract adapter's own algorithm, and an
aggregate manifest hash. `disjointness_proof.json` shows zero intersection with
900 foreign task IDs across 41 files in this checkout. 30 unit tests cover
determinism, CRLF transport, revision binding, tamper detection, and rejection of
overlap.

**Not closed, and not closeable here:** the tasks have been public on the Hub
since 2025-12-10, so `held_out_from_model_training` stays `null`. Upstream
publishes no private partition.

## Blocker 1 — license: absent, not ambiguous

Verified against primary sources today. The HF dataset has **no `cardData` key at
all**, no license tag, an empty card ("No dataset card yet"), and 404s for
`README.md`, `LICENSE`, `LICENSE.md`, `LICENSE.txt`, `COPYING`, `NOTICE`, and
`TERMS.md` at the pinned revision — only `.gitattributes` and the CSV exist. The
leaderboard HTML contains zero occurrences of "license". `afterquery.com/terms`
(200) reserves rights over AfterQuery content and **grants users nothing**.

The absence is specific, not organizational neglect: AfterQuery declares
apache-2.0, cc-by-4.0 and mit on three of its other public datasets.

**Default copyright applies with no grant of use.** This is a policy call, not a
technical one.

## Blocker 2 — runner: spec written, upstream publishes nothing

`native_runner_spec.md` defines the environment, deployment, action interface,
artifact/side-effect capture, and verifier contract, with sources. No stand-in
was used or proposed — no BrowserGym, no Playwright.

Upstream ships no harness. The `AfterQuery` GitHub org's only App-Bench repo is
`appbench.ai-docs` @ `678a9a65632e3f146efcd879074e52f0c0e9622e` — three markdown
files, no code, no LICENSE. No paper exists.

The hard part is the verifier: **the official grader is two human full-stack
developers**, grading binary 1/0 per rubric item with no partial credit,
best-of-3 attempts. Any automated or LLM-judge verifier built locally is a
different measurement and must be labelled `harness_validation` with
`is_model_score: false`, never compared to the leaderboard.

## Adapter change (owned file)

`_validate_source` matched `appbench` as a raw substring, so the real upstream
identifiers `AfterQuery/App-Bench` and `app-bench` were **rejected** — no receipt
could cite the authoritative URL verbatim. Fixed by folding punctuation before
matching. This *strengthens* the substitution ban (`x-LAM` and
`related_benchmark` now fold onto banned forms). Two tests added; 15 → 17, all
green. Zero external importers; E13 semantics untouched.

## Cost

Zero. No Tinker, no paid API calls, no W&B runs, no image pulls, no disk added,
no git operations.

## Single next action

**Decide the license question.** Email `founders@afterquery.com` for a grant, or
accept the risk in writing. Everything downstream of that is engineering;
nothing downstream of it is legal.
