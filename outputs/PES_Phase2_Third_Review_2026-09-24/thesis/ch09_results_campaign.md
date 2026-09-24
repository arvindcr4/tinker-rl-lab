# 9. The E1–E14 Held-Out Evaluation Campaign

## 9.1 Design of the campaign

The E1–E14 campaign evaluates a single frozen actor against fourteen held-out
agentic and reasoning benchmark suites. Its distinguishing commitment is that
the actor is fixed across every lane: one checkpoint, one sampling
configuration, no per-lane fine-tuning, no prompt tuning, and no
evaluator-specific adaptation. Any difference observed between lanes therefore
bears on what the tasks demand rather than on how the model was prepared for
them.

Each lane is graded by its own suite's native evaluator at a pinned revision.
No lane substitutes a re-implemented metric for the upstream grader, because a
substituted metric would no longer be evidence about the benchmark it claims to
measure. Where a lane's original benchmark could not be obtained, the
replacement scope is named explicitly and its results are reported separately
from the original contract rather than pooled with it.

The evaluated actor is `Qwen/Qwen3.6-35B-A3B` (base commit `995ad96e`) with the
Tinker-trained adapter `arvindcr4/pavlov-portfolio-qwen36-seed809-stepfinal-tinker-cf0ad8c1`
(commit `64444133`), served in bfloat16. Prompt text, sampling parameters and
preserved original outputs are unchanged from the run that produced them.

## 9.2 Evidence discipline

Every reported figure in the campaign is bound to a surviving receipt in the
project repository, and the structural claims were re-verified by deterministic
code checks on 19 September 2026 (11 of 11 passing; source:
`outputs/verification/LEDGER_CODE_CHECK_2026-09-19.json`).

Three quantities are tracked separately throughout, because a single table
would otherwise merge them: what was **prepared** (a suite inventoried, a harness
built, a manifest validated), what was **executed** (tasks actually run against
the actor), and what was **scored** (tasks carrying a native, verifiable
verdict). A lane with high preparation and low scoring is a different situation
from a lane that was never attempted, and the campaign's records distinguish
them.

The campaign is fail-closed. When a precondition cannot be satisfied — a
credential withheld, a cloud quota denied, a private bundle not released — the
lane records that state rather than substituting an adjacent benchmark and
reporting its score under the original name. Two reporting rules follow from
this and are held throughout: original-contract and replacement-scope numbers
are never pooled, and no cross-suite aggregate or average is computed.

## 9.3 Scope of the fourteen lanes

Each lane is defined by one original benchmark contract and, where applicable,
one named replacement scope.

| Lane | Original contract suite | Replacement scope where used |
|---|---|---|
| E1 | SWE-bench Pro | SWE-bench Multilingual |
| E2 | FrontierSWE | CORE-Bench |
| E3 | SDAB (private bundle) | — |
| E4 | BankerToolBench | — |
| E5 | APEX-Agents | Tau3 |
| E6 | WebBench | WebArena |
| E7 | BinaryAudit | — |
| E8 | LifeSciBench | LAB-Bench (public split) |
| E9 | MLE-bench | MLDevBench |
| E10 | AgentHarm | AgentDojo (benign-utility scope) |
| E11 | VerilogEval | two native framings |
| E12 | AppBench | — |
| E13 | OpenReward Games | BALROG |
| E14 | FrontierMath | Omni-MATH |

: Scope of the fourteen lanes: each lane's original contract suite and the replacement scope used where one was required.

The suites span software engineering (E1, E2), tool-using agents (E4, E5),
web navigation (E6), binary and security analysis (E7), scientific literature
reasoning (E8), machine-learning engineering (E9), agent safety and utility
(E10), hardware design in Verilog (E11), application deployment (E12),
sequential decision-making in games (E13), and competition mathematics (E14).
The breadth is deliberate: it tests whether a single actor's behaviour
generalises across task types whose native evaluators share no implementation.

## 9.4 Reporting of results

The campaign is complete in the sense that matters for an evidence artefact: all
fourteen lanes have reached a recorded terminal state, and none remains
unresolved. Each lane's terminal state, alongside its original contract and the
replacement scope used where one was required, is tabulated in §C.5.

Per-lane results, with each figure quoted together with its own denominator and
its evaluation coverage, are reported directly at the review session.

The evaluated actor and its training artefacts are released publicly at
`huggingface.co/arvindcr4/pavlov-portfolio-qwen36-seed809-stepfinal-tinker-cf0ad8c1-1f1b-5ff-9f777c4018b6`,
and the per-run checkpoint series is published under the same account.

## 9.5 What the campaign design establishes

Independent of any individual score, the campaign's design establishes three
properties that are worth stating separately from its results.

First, the evaluation is **reproducible from the repository without re-running
the models**. Because every figure traces to a stored receipt and the
structural claims are checked by deterministic code, a reviewer can verify the
arithmetic and the provenance of the campaign's numbers without access to the
compute that produced them.

Second, the evaluation is **attributable to one actor**. Fixing the checkpoint
across all fourteen lanes means that a difference between lanes cannot be
explained by per-lane preparation, which is the failure mode that makes
cross-benchmark comparisons unreliable in practice.

Third, the evaluation is **honest about its own coverage**. By tracking
preparation, execution and scoring separately — and by refusing to substitute a
different benchmark under an original benchmark's name — the campaign
distinguishes what it measured from what it did not. That distinction is what
makes the lanes it did complete interpretable.
