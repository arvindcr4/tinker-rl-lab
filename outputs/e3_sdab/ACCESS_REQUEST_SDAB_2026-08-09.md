# SDAB evaluation access request

**To:** founders@emulated.so (contact listed on https://emulated.so/sdab)
**Subject:** SDAB evaluation access — reproducible third-party run of an open-weights model
**Date:** 2026-08-09

---

## Cover note (send as-is)

Hello,

We are running a small academic evaluation campaign and would like to include the
Software Development Automation Benchmark as a primary evaluation suite. We want
to report a number that is directly comparable to your leaderboard, or not report
one at all — so we are asking for the exact evaluation artifacts rather than
approximating them.

What we are evaluating: **Qwen/Qwen3.6-35B-A3B**, open weights, pinned to Hugging
Face commit `995ad96eacd98c81ed38be0c5b274b04031597b0`, served through the Tinker
sampling API. This is a single evaluation pass; we are not training on SDAB and
we are not building a competing benchmark.

Our harness is already written and tested against the contract implied by your
public description — five categories, 80 tasks, deterministic seeded environments,
a grading harness outside the agent's reach, and a composite of behavioral tests
(60%), operational rubrics (30%), and engineering rubrics (10%). It refuses to
emit a score unless every artifact below is pinned by digest, so we cannot
accidentally publish an approximation. The four artifacts we need are listed
below with the specific fields each one has to carry.

We are happy to sign an evaluation agreement, to run entirely inside your
infrastructure, and to withhold publication until you have reviewed the numbers.
Task content never needs to leave your environment: our run receipt is
metadata-only (task IDs, hashes, and digests), and we can share the receipt
schema in advance.

Thank you,
Arvind (arvindcr4@gmail.com)

---

## Artifact 1 — The 80-task evaluation bundle

**What we are asking for:** the frozen evaluation set of 80 tasks, at one
immutable revision, under a stated license or evaluation agreement.

| Field we need | Why |
|---|---|
| `revision` — an immutable identifier, ideally a **40- or 64-character hex** git commit SHA or content digest (not a tag, not `latest`, not `main`) | Our gate rejects floating revisions. A number is only interpretable against a fixed revision. |
| The list of all **80 task IDs**, exactly as your grader names them | We hash the sorted ID list into the run receipt so anyone can confirm we ran the whole suite and not a convenient subset. |
| Per-task **category** label, from the five published categories | Lets us report per-category composite alongside the headline number. |
| Per-task **difficulty variant** label, where a task exposes variants (e.g. the four variants of the database cutover task) | Your page notes variants progressively withhold information; a composite is not comparable unless the variant selection matches your leaderboard runs. |
| **License identifier or evaluation agreement reference** — SPDX ID if one applies, otherwise the agreement name plus a receipt/reference we can cite | We record a license receipt in the run artifact; without it the run is blocked on policy, not on capability. |

**Not needed:** we do not need task prompts, solutions, or hidden tests to leave
your infrastructure. If the bundle can only be read from inside your environment,
that is fine — we need the IDs, the revision, and the license reference.

## Artifact 2 — Train / evaluation disjointness receipt

**What we are asking for:** a statement of which task IDs are held out for
evaluation and which (if any) are public, development, or tuning tasks.

| Field we need | Why |
|---|---|
| The **training / development / public task ID list**, or an explicit statement that no such split exists | We compute `train_task_id_sha256` and `task_id_sha256` and prove the two sets are disjoint, case-insensitively. |
| A **signed or otherwise immutable reference** for that statement (a dated document, a URL, a signed attestation) | The disjointness claim has to be attributable to you, not asserted by us. |
| Confirmation of whether any of the 80 evaluation tasks appear in public write-ups, blog posts, or the leaderboard methodology | Contamination check for a model whose training cutoff we do not control. |

If no train split exists — if all 80 tasks are held out and nothing was ever
released — please say so explicitly. That sentence *is* the receipt; we will
record it as such and note that the training list is empty.

## Artifact 3 — The native runtime

**What we are asking for:** the live environment, with deterministic reset, and a
programmatic entrypoint.

| Field we need | Why |
|---|---|
| **Container / image digest** for the Docker environments, in `sha256:<64 hex>` form | Pinned in the receipt; a run against a differently built image is a different experiment. |
| **Environment digest** covering the seeded state — the workspace, running infrastructure, and traffic generator configuration | Your page states environments are seeded with deterministic state; we need the identifier for that seed set. |
| For **cloudbox** tasks: the provisioning path, cloud account/quota requirements, and expected cost per task | Some tasks (e.g. the GPU Kubernetes distributed-training deployment) run on real cloud resources. We need to know what we are billed for before we start. |
| An **adapter entrypoint** in `module:factory` form exposing three methods: `list_tasks(split, limit, seed)`, `evaluate_task(task, sampler, max_actions, seed)`, `verify_result(result)` | This is the only integration surface our runner uses. If you already ship a Python driver, we will adapt to its signature instead — a pointer to it is enough. |
| The **agent scaffold / harness** you used for the leaderboard entries, or its specification (tool set, action budget, wall-clock limit) | Your tasks run up to 12 hours. A composite from a different scaffold is not comparable to Claude Opus 4.6 at 13.9%. |
| The **reset procedure** and how many attempts constitute Pass@1 | So our protocol matches the published one. |

## Artifact 4 — The native grader

**What we are asking for:** the grading harness that sits outside the agent's
reach, as you describe it.

| Field we need | Why |
|---|---|
| **Verifier identity and revision**, plus a `sha256` digest of the grader | Recorded in the receipt. A score is inadmissible for us unless the grader that produced it is identified. |
| **Behavioral test suite** digest | The 60% feature-correctness component. |
| **Engineering-quality rubric** definitions and digest, plus the **judge model and its configuration** | The 30% + 10% rubric components are LLM-judged; the judge identity and settings materially change the number. |
| **State-validation / hidden-test** digest, and confirmation these run outside the agent-visible workspace | Reward-hacking resistance is a property we assert in the receipt; we need it to be true, not assumed. |
| The **composite weighting** and the 0–100 scaling formula | To reproduce the benchmark-level score exactly rather than re-deriving it. |

We do not need the hidden test content. Digests and the grading interface are
sufficient — we call your grader and record what it returns.

---

## What happens on our side once each artifact lands

| Artifact | Unblocks |
|---|---|
| 1 — task bundle | `flagship.pavlov_sdab_eval_adapter --bundle …` validates the schema, hashes the 80 task IDs, and builds the evaluation split manifest. |
| 2 — disjointness receipt | The disjointness proof completes; `train_task_id_sha256` becomes non-null and the runtime manifest can be emitted. |
| 3 — runtime | The `exact_native_sdab_runtime` preflight gate flips from BLOCKED to PASS. This is currently the only failing gate. |
| 4 — grader | `verify_result` becomes real; without it every per-task score is discarded before it can be recorded. |

All four are required. Any three of four leaves the run blocked, and our harness
emits `status: BLOCKED, score: null` rather than a partial number.

## What we can commit to

- Single evaluation pass; no training or fine-tuning on SDAB tasks.
- No redistribution of task content, environments, rubrics, or tests.
- Metadata-only run receipt: task IDs, hashes, digests, token counts, and the
  grader's returned scores. No prompts or trajectories in anything we publish.
- Pre-publication review of any reported number, on request.
- We will cite the benchmark as *Software Development Automation Benchmark*
  (Emulated, Inc., April 2026), at the revision you provide.
