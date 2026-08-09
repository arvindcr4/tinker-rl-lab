# E13 — OpenReward game environments — access request

**Prepared 2026-08-09. Not sent.** Sending it, creating an OpenReward account,
and accepting the platform terms are the user's actions, not the agent's.

## Read this first: most of what E13 needs is already public

This request is deliberately narrow. The environment source, the seed split, and
the verifier for the OpenReward game environments are **public** and are already
pinned locally (see `lane_status_2026-08-09.md`). Asking for them would be asking
for something already in hand.

Four things are genuinely missing, and only one of them is data.

## Recipient

- `hello@openreward.ai` — the support address published on `docs.openreward.ai`.
- Platform operator: **GR Inc** (General Reasoning), per the `openreward.ai`
  footer. The game environments are owned by the `GeneralReasoning` org on the
  platform, so the operator and the environment publisher are the same party —
  one request covers both.
- No dedicated benchmark-access or data-request address is published. There is
  no application form, no gated dataset agreement, and no approval queue: the
  hosted path is self-serve sign-up plus an API key.

## What we are asking for

### 1. An explicit license grant for the environment wrapper code (blocking)

All 277 repos under `github.com/EnvCommons` are public but contain **no LICENSE
file**; the GitHub API reports `license: null` for every one. The Wordle
environment card states the license as MIT, but links to TextArena's LICENSE —
which covers the upstream game engine, not the OpenReward wrapper written by
GR Inc (the `Environment` subclass, the split definition, the reward mapping).

Request: add a LICENSE file to the `EnvCommons` environment repos, or confirm in
writing that the wrapper code is offered under MIT. Without this, the wrapper is
all-rights-reserved by default and results derived from it cannot be published.

This is the one item that blocks on policy rather than on access, and it is the
highest-value thing to resolve. It is also cheap for the provider: one file.

### 2. A definition of the held-out game suite (blocking for comparability)

The registry exposes 77 environments tagged `Decision-Making in Games`, 74 of
them under `GeneralReasoning` with `train` and `test` splits. There is no
published collection naming which of these constitute an official evaluation
suite, nor a version identifier for that collection.

Request: either a named, versioned suite manifest (environment list + revision
per environment), or confirmation that no official suite exists and that
evaluators are expected to choose and disclose their own subset. Either answer
is usable; the ambiguity is not.

### 3. An attestation binding deployed environments to public revisions

`api.openreward.ai/v1/environments/<owner>/<name>` returns
`original_github_url`, but no deployed commit SHA. We can pin the public source
(`EnvCommons/wordle@92bea32efa102e86275dedd2e0367e86d3754754`); we cannot verify
that the hosted environment serving `test` split tasks runs that same revision.

Request: expose the deployed commit SHA per environment in the public API or the
environment page — or confirm that hosted deployments track the linked repo's
default branch HEAD, and how to pin a specific revision for a reproducible
evaluation.

Without this, a hosted score is not reproducible, because the artifact that
produced it is not identified.

### 4. Confirmation on the seed split and the state verifier

We have derived both from public source and want them confirmed rather than
supplied:

- **Seed split.** Every game repo sampled uses
  `seed = seed_idx if split == "train" else seed_idx + 10000`, giving train seeds
  `[0, 50)` and held-out seeds `[10000, 10050)` per variant. Confirm this is the
  intended and universal train/held-out convention, and that the held-out seeds
  were not used in any training data GR Inc distributes.
- **Determinism.** Confirm that a given `(environment, variant, seed)` yields the
  same initial state on the hosted runtime as it does locally, so that locally
  derived manifests address the same instances the platform serves.
- **State verifier.** Confirm that game rewards are graded programmatically end
  to end, with no LLM grader anywhere in the game category, and that the hosted
  reward equals the local `ToolOutput.reward` for an identical action sequence.

A short written confirmation is sufficient. No data transfer is required.

## What we are NOT asking for

- The environment task packages — public, already pinned.
- The verifier implementation — public, already read and exercised locally.
- Any private or held-back split. We have no evidence one exists; the `test`
  split is public in source like the `train` split.
- API credits, free rollouts, or a commercial arrangement.

## What the requester has already done

- Pinned `EnvCommons/wordle@92bea32efa102e86275dedd2e0367e86d3754754`.
- Installed `openreward` 0.1.152 and `textarena` 0.7.4 into an isolated venv.
- Run the environment's own shipped self-test locally with **no credentials**.
- Verified train/held-out instance disjointness (200/200, empty intersection)
  and same-seed determinism.
- Verified the reward path end to end via a gold action (`reward=1.0`,
  `finished=True` on held-out seeds 10000–10002) — labelled `harness_validation`,
  not a model score.

## Self-serve path, for the user to decide

The hosted path needs no approval — it is sign-up plus a key:

1. Create an account at `https://openreward.ai/` and accept the platform terms.
   **User action.** Sign-up is Clerk-hosted; the agent must not create accounts
   or accept terms.
2. Mint an API key and export `OPENREWARD_API_KEY`.
3. `orwd whoami` to confirm, then `orwd list -o json` for the authoritative
   environment list.
4. A scored rollout additionally needs a paid model provider key
   (`OPENAI_API_KEY`, Anthropic, or OpenRouter). This crosses the lane's cost
   boundary and is not budgeted here.

Steps 2–4 are only required for a *hosted* score. A local score against the
pinned public environments needs none of them — only the license grant in item 1
and a model provider key.

## Documentation defects to mention in the same message

Low cost to report, and each one cost real time to work around:

- The docs footer links `github.com/OpenReward`, which is an unrelated squatted
  org (`OPENREWARD`, one empty repo, blog `openreward.vercel.app`). The actual
  source org is `EnvCommons`. This is the single biggest reason the public source
  was missed on the first pass.
- `docs.openreward.ai/api-reference/openapi.json` is an unmodified Mintlify
  placeholder ("OpenAPI Plant Store", server `sandbox.mintlify.com`). The live
  public API at `api.openreward.ai/v1` is undocumented.
- The Wordle environment card links `github.com/LeonGuertwordle/TextArena`
  (typo for `LeonGuertler/TextArena`, now `TextArena/TextArena`).
