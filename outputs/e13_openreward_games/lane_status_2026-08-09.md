# E13 — `openreward_games_eval` — lane status, 2026-08-09

**Status: PARTIAL.** Score `null`. The headline finding overturns the inventory
entry for this lane.

## The finding: E13 is not gated

`outputs/_setup/E1_E14_ASSET_INVENTORY.md:39` records E13 as
"Access: held-out game package … no public pinned revision or seed split", and
`blocked_receipt.json` asserts no public task package, revision, license, seed
split, or verifier exists. **That is wrong.** All four exist publicly and are
now pinned on disk.

The prior pass looked only at `openreward.ai` and `docs.openreward.ai`, both of
which front a sign-in wall. The catalogue is served by an **unauthenticated
JSON API** that neither page advertises:

```bash
xh GET 'https://api.openreward.ai/v1/environments?limit=100&offset=0'   # no key
xh GET  https://api.openreward.ai/v1/environments/GeneralReasoning/Wordle
```

21,812 environments; 377 GitHub-connected. Each record carries `split_types`,
`total_task_count`, `is_private`, and `original_github_url`. The per-environment
record also returns `readme_content` — the full environment card, including its
License and Tasks sections.

That field is the thread. `GeneralReasoning/Wordle` declares
`original_github_url = https://github.com/EnvCommons/wordle`, and **that repo is
public** (`"private": false`). `EnvCommons` holds **277 public repos**, one per
hosted environment, containing the environment server, the split definition, and
the reward path.

## What is public vs. what needs an account

| Asset | State |
|---|---|
| Environment catalogue + splits + task counts | **Public**, unauthenticated JSON API |
| Environment cards (license, task structure, reward design) | **Public**, `readme_content` field |
| Game environment source (server, splits, reward) | **Public**, `github.com/EnvCommons/<game>` |
| Seed split defining train vs. held-out | **Public**, in source — see below |
| Game-state verifier | **Public**, programmatic, no LLM grader |
| SDK + `orwd` CLI | **Public**, PyPI `openreward` 0.1.152, MIT |
| Game engine | **Public**, PyPI `textarena` 0.7.4, MIT |
| Stepping an environment locally | **Public**, no credential — *verified, see below* |
| Hosted rollout sessions | Needs `OPENREWARD_API_KEY` (account) |
| Platform-served authoritative task list | Needs an account |
| Run/rollout recording, leaderboard submission | Needs an account |

E13 is **one signup away for the hosted path, and zero signups away for the
local path.** No account was created and no terms were accepted.

## The game suite

77 environments carry the task tag `Decision-Making in Games`; 74 are owned by
`GeneralReasoning` and declare both `train` and `test` splits — Wordle,
Sudoku, Chess-likes, Battleship, NetHack, Settlers of Catan, Snake, 2048,
Tower of Hanoi, Secret Mafia, and so on, 100–3,000 tasks each. 1 of the 77 has a
leaderboard.

## Seed separation is provable from public source

Every game repo sampled defines its split in `env.py` the same way:

```python
NUM_TASKS_PER_VARIANT = 50
seed = seed_idx if split == "train" else seed_idx + 10000
```

Train seeds occupy `[0, 50)`, held-out seeds `[10000, 10050)`, per variant.
Verified byte-identical in 10 repos: `wordle`, `tictactoe`, `sudoku`, `snake`,
`connect_four`, `battleship`, `game_2048`, `tower_of_hanoi`, `frozen_lake`,
`taboo`, `nim`. Disjointness is checked, not assumed, by
`prove_seed_separation`.

Against the real pinned Wordle manifests: 200 train / 200 held-out instances,
`shared_instances: []`, `shared_task_ids: []`, `variant_coverage_matches: true`,
`holds: true`.

## What actually ran

Pinned: `EnvCommons/wordle@92bea32efa102e86275dedd2e0367e86d3754754`
(tree `8b768b3e3bae83381672d0596f371d60a1b0affe`, 2026-07-31), cloned to
`pinned/wordle/`. Venv at `venv/` — `openreward` 0.1.152, `textarena` 0.7.4.

```bash
cd outputs/e13_openreward_games/pinned/wordle && ../../venv/bin/python test_local.py
```

The environment's own shipped self-test passes with **no credentials**:
`Splits: ['train','test']`, `Test tasks: 200`, first task
`Wordle-v0_seed10000`, prompt rendered, `guess_word` stepped, reward returned.

Determinism and disjointness re-checked directly: same seed yields a
byte-identical prompt hash; train/test seed intersection empty.

Gold-action validation — feeding the environment's own secret word back through
the verifier — returns `reward=1.0, finished=True` on held-out seeds 10000,
10001, 10002. This proves the verifier path end to end. It is
**`harness_validation`, `is_model_score: false`, and is not a benchmark score.**

## Local machinery built

- `zvf-program/flagship/e13_openreward_games_local_runner.py` — task/seed
  manifest schema, seed-separation proof, game-state verifier interface with a
  fail-closed `ProgrammaticRewardVerifier`, and receipt emission that withholds
  a score unless every gate passes. No override path exists.
- `zvf-program/flagship/test_e13_openreward_games_local_runner.py` — **39 tests,
  all passing**, against a fixture marked `SYNTHETIC-FIXTURE-NOT-A-BENCHMARK-ARTIFACT`.

The tests caught a real modelling bug: upstream reuses the same seed index
across variants, so instance identity is `(variant, seed)`, not the bare seed.
Keying separation on bare seeds would have compared the wrong objects.

E12's `pavlov_appbench_openreward_games_adapter.py` was not touched.

## Blockers that remain

1. **No LICENSE file in any EnvCommons repo** — 277/277 report `license: null`.
   The Wordle card claims MIT by pointing at TextArena's LICENSE, but the
   wrapper code carries no grant of its own. Policy gate, not a download gate.
2. **No official `openreward_games_eval` suite definition.** The registry has 77
   game environments but publishes no named held-out collection, so which games
   and how many constitute E13 is set by us, not by the provider.
3. **Deployed revision is unverifiable** without an account. The pinned public
   commit is what OpenReward links to; that it is what OpenReward *runs* cannot
   be confirmed from outside.
4. **A model score needs a paid provider key** (OpenAI/Anthropic/OpenRouter) —
   not crossed, per the cost boundary.

## Documentation defects worth reporting upstream

- The docs footer links `github.com/OpenReward`, which resolves to an unrelated
  squatted org (`OPENREWARD`, created 2025-05-15, one empty repo, blog
  `openreward.vercel.app`). The real source org is `EnvCommons`.
- The published API reference `docs.openreward.ai/api-reference/openapi.json` is
  an unmodified Mintlify placeholder — "OpenAPI Plant Store", server
  `sandbox.mintlify.com`. The live public API it should describe is undocumented.
- The Wordle card links `github.com/LeonGuertwordle/TextArena` (typo; the repo is
  `LeonGuertler/TextArena`, now `TextArena/TextArena`).

## Tinker RL training driver (built, not run — $0 spent)

`zvf-program/flagship/e13_openreward_games_tinker_train.py` + tests (**39 more,
all passing**; 78 across the lane). Default mode is `plan`: it validates every
gate and projects cost without constructing a Tinker client. `tinker` and
`wandb` are not imported at module load — verified.

- **Model binding** goes through `assert_candidate_allowed`. Both authorized
  models pass; `Qwen/Qwen3-8B` (served but off-contract) is refused. The
  registry's own `lora_extended_context_variants` field confirms `:peft:`
  entries are extended-context builds, not a trainability flag.
- **Split firewall** — `SplitFirewall` refuses to construct over non-separated
  manifests, `assert_train_seed` rejects anything `>= 10000`, `assert_eval_seed`
  rejects anything `< 10000`, and `assert_no_leak` re-checks the admitted sets
  before any checkpoint is exported. Hard assertions, not convention.
- **Reward** is the environment's native terminal reward through the existing
  fail-closed verifier. A rejected episode contributes *no* reward rather than a
  guessed one. No shaping, no proxy.
- **W&B** online init happens before any Tinker client, project
  `tinker-rl-lab-pavlov` (enforced — a different project raises), logging
  `train/reward`, `train/loss`, `train/step`, `eval/reward`.
- **Checkpoint export** emits `sampler_path` + `hf_repo`/`hf_revision`/
  `hf_commit` shaped for `e11_paid_run_driver.py --sampler-path`. The HF fields
  are null and the record reports the probe as **blocked**: `HF_TOKEN` is absent,
  so no immutable HF revision can be created.
- **License** — emits `observed_state: absent_at_pinned_revision`,
  `claimed_spdx: null`, `proceeding_under` the owner's risk-acceptance record.
  The validator is fail-closed and points at that record: delete the record and
  the driver stops. No SPDX identifier appears anywhere in the plan output.

### Cost projection (pinned prices, conservative maxima)

Measured from a real local episode, worst case `Wordle-v0`, 7 turns: 11,242
prefill + 1,792 sample + 3,462 train tokens per episode.

| Scenario | Episodes (sampled / trained) | Cost |
|---|---|---|
| One smoke episode | 1 / 1 | **$0.0125** |
| Short pilot — 20 updates x 4 batch x 4 group | 360 / 320 | **$4.35** |
| Full pass over 200 train tasks (group 4) | 820 / 800 | **$10.20** |

Spendable is $15.00 (cap $16.50 less the $1.50 reserve). **Pilot + full pass =
$14.55, leaving $0.45 of headroom while E11 is drawing concurrently — too tight
to run both.** Cheaper shapes: 10x4x4 = $2.18, 8x2x4 = $0.89, 4x2x4 = $0.49.

Prefill is ~50% of cost because every turn is charged as uncached. If prefix
caching applies (budget lists `cached_prefill` at $0.108/M, 5x cheaper), the
pilot drops to $2.60 and the full pass to $6.22. That upside is **not** banked.

## Single next action

Decide the E13 game set (proposal: the 74 `GeneralReasoning` environments with
`train`+`test` splits, or a named subset), then pin and export their manifests
the same way Wordle was pinned. That needs no credential and no spend. The
license grant and the deployed-revision attestation are the only items that
require the provider — both are in `ACCESS_REQUEST_2026-08-09.md`.
