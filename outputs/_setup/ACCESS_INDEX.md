# How to unblock E1–E14 — consolidated access index

Date: 2026-08-09. Every route below was established from a primary source by the
lane that owns the suite; each lane's full request document is linked and written
to send as-is.

**Read this first:** four suites are not access-blocked at all. They need a
credential you can buy or a click you can make. Do those before emailing anyone.

## Tier 1 — you can do these yourself, today

| Suite | What to do | Time |
|---|---|---|
| **E5 APEX Agents** | `hf auth login`, then open <https://huggingface.co/datasets/mercor/apex-agents> and click "Agree and access repository". The repo is `gated: auto` — **automatic approval, no human review, no queue**. The token alone is not enough; terms acceptance is a separate click. | 2 min |
| **E4 BankerToolBench** | Not an access problem. Export `GEMINI_API_KEY` (Gandalf grader) and an agent key (`OPENAI_API_KEY` or swap the provider in `job.yaml`). Then `harbor run -c job.yaml`. Stop line is exactly `harbor/verifier/verifier.py:173`. | minutes + spend |
| **E7 BinaryAudit** | Not an access problem. Export `ANTHROPIC_API_KEY` or `OPENROUTER_API_KEY`. Base image is already built. Use `--include-task-name` (the README's `--task-name` does not parse on harbor 0.20.0) and set `DOCKER_DEFAULT_PLATFORM=linux/amd64`. | minutes + spend |
| **E9 MLE-bench** | 1 of 75 competitions accepted. Each remaining competition needs its own click on its own rules page, signed in as `arvindcr`. Not automatable. | 74 × ~1 min |

## Tier 2 — email a named contact, realistic chance of a yes

| Suite | Contact | The ask | Request doc |
|---|---|---|---|
| **E13 OpenReward** | `hello@openreward.ai` | **Narrowest ask in the portfolio.** The catalogue, source, splits, verifier and SDK are already public via `api.openreward.ai/v1/environments` and `github.com/EnvCommons` (277 repos). You only need a **license grant** plus confirmation of whether a private holdout exists — the public `test` split is not held out from anyone. | [doc](../e13_openreward_games/ACCESS_REQUEST_2026-08-09.md) |
| **E3 SDAB** | `founders@emulated.so` (contact on the official page) | Large but coherent: the 80-task bundle with immutable revision + license ID, the train/eval disjointness receipt, the native runtime (container + environment digest, adapter entrypoint), and the native grader. | [doc](../e3_sdab/ACCESS_REQUEST_SDAB_2026-08-09.md) |
| **E6 WebBench** | GitHub issue on `Halluminate/WebBench` (issue #2 is the existing ground-truth thread) | Live environment digest, native verifier, and — more important than either — the **grading contract**: the public CSV has four fields and *no gold answer of any kind*. Also ask whether the 78 absent task IDs are a withheld split. The request carries our pinned hashes so they can verify their copy matches. | [doc](../e6_webbench/ACCESS_REQUEST_HALLUMINATE_2026-08-09.md) |
| **E12 AppBench** | `founders@afterquery.com` | A license grant (no `cardData`, no LICENSE at the pinned revision). **But note:** even with permission there is no automatable verifier — AfterQuery's grader is *two human full-stack developers*, binary per rubric item, best-of-3. Any LLM judge is a different measurement. | — |
| **E7 license** | Upstream issue on `QuesmaOrg/BinaryAudit` | No LICENSE was ever added in any commit across all refs, while the README claims Apache-2.0. Almost certainly an oversight and cheap for them to fix. | — |
| **E1 / E2 licenses** | ScaleAI / Proximal Labs — no contact route established | E1's dataset `cardData.license` is null (the evaluator repo is MIT — that covers code, not data). E2 has no root LICENSE at the pinned revision. | — |

## Tier 3 — structurally hard; know this before spending effort

| Suite | Reality |
|---|---|
| **E14 FrontierMath** | **Never released as data, and never will be.** Epoch cannot share Tiers 1–4 without OpenAI's written permission. There is no download, no gated repo, no request form, and no official dataset on HF or GitHub. A score comes **only** from an evaluation Epoch runs themselves — `math_evals@epoch.ai`. Precedent: DeepMind's co-mathematician was evaluated blind with Epoch staff entering problems into a UI. Paid engagements appear on their transparency page. **The realistic alternative is FrontierMath: Open Problems** — Epoch-owned, and the verifiers are *purchasable* under an explicit uniform non-exclusive commitment: `math@epoch.ai`. |
| **E8 LifeSciBench** | **OpenAI publishes no benchmark-data request route.** The announcement's "Request access" button goes to GPT-Rosalind *model* access, requiring a legal entity, Org ID and government-ID check — do **not** submit that form to reach the benchmark team. Closest route is <https://openai.com/contact-sales/>. There is no public slice at all: zero tasks, rubrics, harness, or grader. Ask for the task-ID list and disjointness statement first; they are cheapest, and a documented refusal still converts an open blocker into a closed one. |
| **E10 AgentHarm** | Three documented public routes: an `inspect_evals` GitHub issue, an HF dataset discussion, or `aisi.gov.uk`. Ask for the three private-split files plus the **matching grading revision** — held-out behaviours may reference `grading_function` names absent from the public grading file, so the scorer raises without it. Also offer the alternative that **AISI run the eval themselves**; the paper scores the private set on a non-logging deployment. | 
| **E2 x86-64 host** | Not an access problem and no email fixes it. The image is amd64-only and headless Chrome **core-dumps under QEMU without Rosetta** — all 32 scene attempts failed on a timeout baked into a `chmod a-w` baseline file. **E2 needs an x86-64 Linux machine.** |

## Cheapest high-value asks, in order

1. **E5 terms click** — 2 minutes, self-service, unblocks a whole suite.
2. **E13 license grant** — everything else is already public and working locally.
3. **E7 license issue** — near-certain oversight, and the runner already works.
4. **A decontamination manifest** — not an external ask at all. It is the last gate on **both** E9 and E11, and it is what makes any train-then-evaluate result valid. Paperwork, not compute.
5. **E6 grading contract** — reframes the ask from "give us the environment" to "give us the answer key", which is the actual blocker.
