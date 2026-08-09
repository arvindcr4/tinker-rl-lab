# FrontierMath — evaluation engagement request

**Date:** 2026-08-09 · **Lane:** E14 `frontiermath_eval` · **Status:** BLOCKED, terminal

---

## 0. Read this before you send anything

**There is no dataset to request.** FrontierMath is permanently held out. Epoch
AI cannot give you the Tiers 1–4 problems even if they wanted to: their own
page states they cannot share the questions and answers with other parties
without OpenAI's written permission
([epoch.ai/frontiermath/tiers-1-4/about](https://epoch.ai/frontiermath/tiers-1-4/about)).

A message asking for the data will be declined, and asking marks you as not
having read their published terms. **The only thing that produces a
FrontierMath number is a hosted evaluation run by Epoch AI on your model.**
Epoch states plainly that they report results only for models they can
*"run evaluations ourselves"* on
([epoch.ai/benchmarks/about](https://epoch.ai/benchmarks/about)).

So the document below is a **submission / engagement** request, not an access
request.

## 1. Decide which product you want first — they have different regimes

"FrontierMath" now names two things. Conflating them will get your message
routed to the wrong inbox.

| | **Tiers 1–4** (classic) | **FrontierMath: Open Problems** |
|---|---|---|
| Launched | Nov 2024 paper; v2 correction pass since | July 2026 |
| Funder / commissioner | OpenAI | Schmidt Sciences |
| Ownership | Commissioned; OpenAI holds the problem set | **Epoch alone** |
| Size (post-v2) | **338 problems** — 295 Tiers 1–3 + 43 Tier 4 | not established |
| Can you buy in? | No | **Yes — verifiers are purchasable** |
| Contact | `math_evals@epoch.ai` | `math@epoch.ai` |

On Open Problems, Epoch states that verifier access is purchasable by any party
and commits to granting it uniformly rather than exclusively. As of July 2026
**OpenAI is the only purchaser.** If your goal is a defensible Epoch-owned math
number without a holdout-asymmetry footnote, **Open Problems is the cleaner
target.**

Epoch has also said the Tiers 1–4 arrangement will not recur — for future
benchmarks they commit to retaining ownership and providing equitable access.
Whether Tiers 1–4 ever opens to non-OpenAI parties is **not established**.

## 2. The three routes that actually exist

1. **Epoch evaluates you unilaterally.** They pick models off public APIs and
   publish to their benchmarking hub. You do not initiate this; you make your
   model publicly callable and become eligible.
2. **You grant Epoch access and they run it.** Pre-release API access, or even
   plain UI access. Precedent: Google DeepMind's AI co-mathematician was
   evaluated **blind**, with Epoch staff typing problems into the UI themselves.
   This is the route most likely to fit a research model.
3. **You commission a paid engagement** via
   [epoch.ai/about/consultations](https://epoch.ai/about/consultations). Epoch's
   transparency page lists engagements but notes it **may omit engagements under
   $30,000** — so treat ~$30k as the visible-tier signal, not a quoted price.

There is **no application form and no submission portal.**
[epoch.ai/contact](https://epoch.ai/contact) is a general contact form whose
subject dropdown includes "Math" — it is the closest thing to a request form,
but it is not a benchmark-access application. Direct email is better.

## 3. Ready-to-send — Tiers 1–4 hosted evaluation

Send to **`math_evals@epoch.ai`**. Fill the four bracketed fields and send
as-is. *(The 2024 paper lists a legacy address `math_evals@epochai.org`; prefer
the current `epoch.ai` domain.)*

> **Subject:** FrontierMath Tiers 1–4 — hosted evaluation request for an open-weights research model
>
> Hello Epoch AI math evaluations team,
>
> I am writing to ask about a hosted FrontierMath evaluation. To be explicit up
> front: I am **not** requesting the problem set. I have read that Tiers 1–4
> cannot be shared without OpenAI's written permission, and I am asking only
> for an evaluation that Epoch runs on its own infrastructure, under whatever
> holdout discipline you prefer — including fully blind, with your staff driving
> the model, as you did for the DeepMind co-mathematician engagement.
>
> **Model under evaluation**
> - Name / revision: `[MODEL_ID]` at revision `[MODEL_REVISION]`
> - Type: open-weights, non-frontier-scale
> - How you can reach it: `[pick one — (a) hosted OpenAI-compatible HTTPS endpoint I provision for you, (b) weights on Hugging Face for you to serve, (c) a UI session your staff drive]`
> - Tooling: the model is used with a Python code-execution tool, matching the
>   FrontierMath harness convention of a self-contained script that writes its
>   answer via `pickle` to `final_answer.p`.
>
> **What I am asking for**
> 1. Whether you accept externally-initiated evaluation requests for Tiers 1–4
>    at all, or whether the eligible set is limited to models you select.
> 2. If you do: your preferred access mode from the three above, the token/compute
>    budget you would run under (I understand the published harness allows a
>    1,000,000-token budget per problem), and the number of runs per problem.
> 3. Whether the result would be published on your benchmarking hub, shared
>    privately, or both — and what attribution and embargo terms apply.
> 4. Cost. If this falls under a commissioned engagement rather than your own
>    evaluation programme, please point me to the right process and a quote.
> 5. Whether an evaluation can be scoped to Tiers 1–3 only, given that 20 Tier 4
>    problems are withheld even from the commissioner.
>
> **Context**
> This is `[ACADEMIC / INDEPENDENT RESEARCH — describe in one line]`. The result
> would be reported as an Epoch-run FrontierMath evaluation with your
> methodology cited; I will not compute or publish any number labelled
> "FrontierMath" from the public sample transcripts or from any substitute
> benchmark.
>
> If a hosted Tiers 1–4 evaluation is not available to parties outside your
> selection process, please say so directly — I would rather record the lane as
> permanently blocked than leave it ambiguous. In that case I would appreciate a
> pointer to the FrontierMath: Open Problems verifier terms instead.
>
> Thank you,
> `[NAME]`
> `[AFFILIATION / EMAIL]`

## 4. Ready-to-send — Open Problems verifier (the purchasable route)

Send to **`math@epoch.ai`**. Use this **instead** if a fully-owned, uniformly
available benchmark matters more than name recognition of the classic tiers.

> **Subject:** FrontierMath: Open Problems — verifier access terms and pricing
>
> Hello,
>
> I would like to enquire about purchasing access to the FrontierMath: Open
> Problems verifiers, under the uniform-access commitment stated on your launch
> page.
>
> Specifically:
> 1. What does verifier access include — the verifier programs only, the problem
>    statements, or both? What is delivered and in what form?
> 2. Pricing and licence terms, including whether results may be published
>    independently and how Epoch must be credited.
> 3. Whether access is granted to individual academic researchers and small
>    research groups, or only to organisations.
> 4. How many problems the current Open Problems set contains, and how it is
>    versioned.
> 5. Whether Epoch will also run the evaluation on our behalf if we prefer a
>    third-party-run result, and the cost difference.
>
> Context: `[ONE LINE]`. Model: `[MODEL_ID]` at revision `[MODEL_REVISION]`,
> open-weights.
>
> Thank you,
> `[NAME]` · `[AFFILIATION / EMAIL]`

**Do not send either message from an automated process.** Send it yourself —
this lane did not create accounts, submit forms, or contact anyone.

## 5. Have these ready before you send

- A **stable model identifier and revision**. The campaign's candidate is
  `Qwen/Qwen3.6-35B-A3B` at revision `995ad96eacd98c81ed38be0c5b274b04031597b0`
  (from `zvf-program/flagship/pavlov_agentharm_frontiermath_adapter.py`).
  Confirm this is the model you actually want scored before you send.
- **One reachable access mode.** An OpenAI-compatible HTTPS endpoint is the
  lowest-friction option; public weights is the lowest-trust-required option.
- **A budget answer.** Route 3 is paid. If the answer to "what can you spend" is
  "nothing", say so in the first message rather than after a quote.

## 6. What you will not get, in any scenario

- The problems. Not under NDA, not under a data-use agreement, not a sample of
  the private set.
- The reference answers or the grader.
- A way to reproduce the score locally, ever. There is **no official
  FrontierMath dataset on Hugging Face or GitHub.** Every repository claiming
  to be FrontierMath is a solver, a scraper of the public problems, or a
  reconstruction, and **cannot produce a comparable score.** Treating one as the
  benchmark would be a fabricated result.

## 7. Holdout structure — the footnote any Tiers 1–4 result needs

From [epoch.ai/frontiermath/tiers-1-4/about](https://epoch.ai/frontiermath/tiers-1-4/about):
OpenAI commissioned 300 core problems plus 50 Tier 4, and holds all problems and
solutions **except 53 withheld solutions and 20 whole Tier 4 problems**. Epoch
retains the right to evaluate on the full set. Every other party has only the
public problems.

Two caveats that must travel with any citation of those numbers:

- The **v2 correction pass** revised 123 Tier 1–3 and 12 Tier 4 problems after
  errors were found in **42% of problems**, giving the current **338 total (295 +
  43)**. The older 300+50 = 350 figure is **superseded**.
- Epoch has **not published a restated holdout count against v2**. Whether "53
  solutions and 20 problems" still holds post-correction is **not established** —
  ask in the email rather than assuming.

## 8. What the public material actually is

Epoch publishes **12 sample problems** (10 from Tiers 1–3, 2 from Tier 4) at
[epoch.ai/frontiermath/tiers-1-4/benchmark-problems](https://epoch.ai/frontiermath/tiers-1-4/benchmark-problems).

The transcript archive this lane holds
(`sample_question_transcripts.zip`, 150 files) covers only the **original 5
public problems from the 2024 paper** — a subset of, and older than, the current
12. It contains no ground truth and no grader verdicts. See
`PUBLIC_SAMPLE_CHARACTERIZATION.md`.

**Licence for the zip specifically: not established.** Epoch's site is broadly
CC-BY, but with an explicit carve-out stating that benchmark questions and
answers remain the property of their respective creators. Do not redistribute
the archive on the assumption that CC-BY covers it.

## 9. The single next action

Pick Tiers 1–4 (§3) or Open Problems (§4), fill the bracketed fields, and send
the email yourself. Until Epoch replies, `frontiermath_eval` stays `BLOCKED`
with `score: null` — and unlike most blocked lanes, **no local work can change
that**, because the blocker is a dataset that is never released.
