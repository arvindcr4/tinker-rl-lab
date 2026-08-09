# E12 AppBench — native executor specification

Date: 2026-08-09. Status: **specification only — nothing here is implemented.**

This document exists so that "build the AppBench runner" is a defined piece of
work rather than an open-ended one. Every requirement below is traced to a
primary source. Where AfterQuery publishes nothing, the gap is named as a gap
instead of being filled with a guess.

## 0. Why a substitute harness is not acceptable

The roadmap already rules out BrowserGym/Playwright as an AppBench stand-in, and
the upstream methodology confirms why. AppBench does not score a browser
trajectory. It scores **a deployed full-stack application**, judged by exercising
its features — including database persistence and multi-user flows — after a
one-shot generation. A DOM-automation benchmark measures a different object.

## 1. What upstream publishes, and what it does not

| Component | Published? | Locator |
|---|---|---|
| Task prompts, rubrics, app descriptions | Yes | `AfterQuery/App-Bench` @ `de80d5bcd404adee5307311571e512b5c37e6112`, file `AppBench vExternal.csv` |
| Rubric wording for 1 of 6 tasks, as markdown | Yes | `github.com/AfterQuery/appbench.ai-docs` @ `678a9a65632e3f146efcd879074e52f0c0e9622e` (3 files, no code) |
| Methodology prose | Yes | `https://www.afterquery.com/leaderboard/app-bench`, mirrored at `https://appbench.ai/blog` |
| **Evaluation harness / runner / grader** | **No** | Searched the 13 public repos in the `AfterQuery` GitHub org and the GitHub repo-search index. `anvil` is a SWE-bench-Pro runner; `harbor` is a Terminal-Bench fork. Neither mentions App-Bench. |
| **The Next.js starter template** | **No** | Referenced by every task's `Addition for CLI Tools` column but never released. `AfterQuery/FullStackBoilerplate` is a hiring take-home, not this template. |
| **Supabase project / schema** | **No** | Methodology says URL + keys were "provided"; no schema or provisioning script exists. |
| **Per-tool trajectories and per-item scores** | **No** | The 9 per-tool columns in the CSV are empty at the pinned revision. |
| Paper | **No** | The leaderboard's "Read paper" link points at `appbench.ai`, a website. arXiv returns no App-Bench result. |

Consequence: an AppBench runner must be written from scratch, and three of its
inputs (template, Supabase project, human graders) have no upstream artifact to
pin against.

## 2. Task interface — what the CSV actually specifies

6 tasks, 151 rubric items total (24 / 33 / 22 / 25 / 23 / 24). Per task the CSV
gives exactly five executable fields:

- `App Name` — label only.
- `App Description` — one-sentence summary, not given to the agent.
- `Prompt` — the full agent instruction. Structure is identical across all six:
  `# Task` (one paragraph, ending in an explicit no-follow-up-questions clause),
  `## Functionality Overview`, `## Feature Requirements` (a numbered list).
- `Addition for CLI Tools` — appended for coding-agent runs only. Identical text
  for all six tasks: Next.js template, isolated Docker container, Supabase for
  auth and database.
- `Rubric` — a numbered list that is 1:1 with the `## Feature Requirements`
  list, restated as assertions about the built application. Verified: the two
  lists have equal length for all six tasks.

The prompts forbid clarifying questions and require the agent to choose every
technical, architectural, and service-level detail itself. A runner that lets
the agent ask questions is not running AppBench.

## 3. Required components

### 3.1 Environment

- Isolated Docker container per attempt, with Node.js.
- A pre-initialized Next.js project with "basic file structure" seeded into the
  workspace. **Undefined upstream**: Next.js major version, router mode, package
  manager, TypeScript vs JavaScript, preinstalled dependencies. Any local
  reconstruction must be pinned by image digest and declared as a deviation.
- Supabase URL + anon/service keys injected as environment variables. A local
  runner needs a per-attempt Supabase instance (self-hosted container or a
  project provisioned via API) that is torn down afterwards, because rubric items
  test durable persistence across sessions.
- Outbound network access. Tasks require real third-party data — live stock
  quotes, real news articles, web search inside an AI assistant. Rubric items
  explicitly demand "real articles" and "real AI assistant responses", so an
  offline or mocked run cannot pass them.
- **Undefined upstream**: wall-clock limit, token budget, turn cap. None is
  stated. A local runner must impose one and declare it.

### 3.2 Deployment

The scored object is a running application, not a diff. The runner must:

1. Install dependencies and build the generated project.
2. Start it and wait for a health signal.
3. Expose it on a stable local URL for the grading phase.
4. Record build and boot failure as a scored zero for that attempt, not as an
   infrastructure error — upstream's own worked example treats a failed
   deployment as `0/40` for that attempt.

### 3.3 Agent action interface

Two distinct modes, per the methodology:

- **Coding-assistant mode** — agent gets a shell and filesystem inside the
  container plus the template, and receives `Prompt` + `Addition for CLI Tools`.
- **Web-builder mode** — agent gets no template and deploys on its own cloud;
  receives `Prompt` only. Not reproducible locally without each vendor's product.

Run policy: **3 attempts per task, one-shot each**, no interactive debugging, no
step-by-step assistance, no fix probes. The best attempt's score counts. The only
permitted human intervention is supplying credentials on request.

Note the scoring bias this creates: max-of-3 over 6 tasks with a human judge and
no reported error bars. Any local reproduction should report all three attempts,
not only the max.

### 3.4 Artifact and side-effect capture

Per attempt, the receipt needs:

- **Artifact**: the generated source tree, content-hashed (tar of the workspace
  excluding `node_modules`/`.next`, sha256 of the tar).
- **Build artifact**: build log, exit code, resulting image or bundle digest.
- **Runtime side effects**: the Supabase database dump after grading (proves
  durable persistence claims), plus the HTTP/websocket transcript exercised
  during grading.
- **Evidence for the real-time rubric items**: rubric items 5, 9, 23 in task 1
  and their analogues elsewhere assert *updates without manual refresh*. These
  are only checkable from a time-series observation — a screencast or a timestamped
  DOM/websocket log — not from a single page fetch.
- **Environment digest**: container image digest, Supabase image/version, Node
  version, template commit.

### 3.5 Verifier contract — and the honest problem with it

The official verifier **is two people**:

> Two experienced full-stack developers independently graded each trajectory
> against the rubric … Any criteria with contradicting grades between the two
> graders were flagged and re-evaluated jointly to reach consensus.

Scoring is binary per rubric item, no partial credit, summed across all 6 apps,
divided by total possible points. Grading is functionality-only; UI aesthetics
count only when they block a feature.

A verifier interface that matches this shape:

```
verify(task_id, deployment_url, artifact_dir, db_handle) -> {
    "task_id": <64-hex from split_manifest.json>,
    "items": [ {"index": int, "text": str, "passed": bool, "evidence": str} ],
    "passed_count": int,
    "total_count": int,     # 24/33/22/25/23/24 by task
    "grader": "human" | "llm_judge" | "automated",
    "is_model_score": bool,
}
```

**The gate:** `grader` must be `"human"` for a number to be comparable to the
published leaderboard. An LLM-judge or automated implementation is a *different
measurement*, and its output must be labelled `harness_validation` with
`is_model_score: false` rather than reported as an AppBench score. Building the
automated verifier is legitimate engineering; presenting its output as an
AppBench result is not.

## 4. Build order

1. Reconstruct and pin the Next.js template + Supabase provisioning; publish both
   digests as declared deviations.
2. Container + deployment + health-check layer, validated on a hand-written
   reference app that is known to satisfy a subset of task 1's rubric. That is
   harness validation, not a model score.
3. Artifact/side-effect capture wired to the task IDs in `split_manifest.json`.
4. Agent driver for coding-assistant mode (needs an agent model credential; none
   is available in this environment).
5. Verifier — automated first for regression use, clearly labelled; human
   grading pass only if two qualified graders are actually available.

Steps 1–3 need no credentials and no paid API calls. Step 4 is the first point
that crosses the cost boundary.

## 5. Sources

- `https://huggingface.co/datasets/AfterQuery/App-Bench` @ `de80d5bcd404adee5307311571e512b5c37e6112`
- `https://www.afterquery.com/leaderboard/app-bench` (methodology quotes above)
- `https://github.com/AfterQuery/appbench.ai-docs` @ `678a9a65632e3f146efcd879074e52f0c0e9622e`
- Local: `outputs/e12_appbench/split_manifest.json`, `outputs/e12_appbench/disjointness_proof.json`
