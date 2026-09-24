## 4. Discussion

### 4.1 What the campaign establishes

The campaign's primary contribution is not a headline score. It is a complete,
independently re-checkable evaluation record for a single open-weights actor
across fourteen held-out suites, in which every lane carries an explicit
terminal state and a denominator that has not been trimmed to flatter the
result.

Four lanes carry strictly complete graded scopes. E8 (LAB-Bench public split)
graded all 1,967 questions across eight categories. E10 (AgentDojo
benign-utility split) completed all 97 episodes. E11 (VerilogEval) graded all
312 tasks across two native framings. E14 (Omni-MATH) recorded all 4,428
dispositions and reproduced the official accuracy figure of 51.31% under the
upstream scorer. Two further lanes — E1 (SWE-bench Pro) and E11 — carry full
verified scores under their original contracts, with E1 reporting over a
731-generation denominator in which all eighteen non-native outcomes are
retained rather than excluded.

Taken together these describe the capability envelope of one actor under
frozen sampling conditions. They do not describe a model's ranking, and this
report does not present them as one.

### 4.2 The case for reporting null and partial results

Seven lanes did not produce a complete score, and the report states each
plainly. The temptation in a campaign of this shape is to substitute an
adjacent benchmark and present the result under the original name. That
substitution was explicitly refused. Where a replacement scope was used — E8
LAB-Bench for LifeSciBench, E10 AgentDojo for AgentHarm, E14 Omni-MATH for
FrontierMath, E2 CORE-Bench for FrontierSWE, E5 Tau3 for APEX-Agents, E1
SWE-bench Multilingual, E9 MLDevBench, E13 BALROG — the replacement is named
as such, and its numbers are never pooled with the original contract's.

This choice costs the campaign visually: a table of partial results reads
worse than a table of complete ones. It is made because the alternative
produces a document whose numbers cannot be defended under questioning. A
reviewer who asks "which benchmark is this score from" must be able to get a
single, unambiguous answer for every cell.

### 4.3 Where the failures are more informative than the scores

The E4 (BankerToolBench) rerun is the clearest example. A separately
authorised 100-trial rerun on the base model completed all 100 trials and
returned a mean reward of 0.0. Reported alone, that figure would suggest the
model cannot perform financial tool-use tasks at all. The trajectory audit
shows something more specific: agents terminate after a median of roughly six
steps without producing a `deliverables/` directory, and the verifier
short-circuits to zero on finding nothing to grade. More than a quarter of the
trajectories ended in role-token degeneration, repeating `user user assistant`
until the loop failed.

The 0.0 therefore measures a breakdown in sustained tool-dialogue, not an
absence of finance reasoning. Recording the mechanism alongside the number
converts an apparently damning result into a diagnostic about the model's
agentic-loop capacity — which is the more useful finding, and the only one the
evidence actually supports.

### 4.4 The gap between coverage and result

Several lanes illustrate that full inventory coverage is not the same as a
result. E6 (WebArena) has a fully verified canonical inventory — 65 files,
ordered episode IDs 0 through 811 — and zero graded episodes, because the lane
is blocked on AWS vCPU quota rather than on anything to do with the model.
E9 (MLE-bench) has 40 of 75 competitions natively graded, which is substantial
coverage, yet no suite score can be reported because the coverage is
incomplete and the suite's own aggregate is not defined over a subset. E13
(BALROG) has 13 of 255 episodes.

In each case the honest report distinguishes three different things that a
careless table would merge: what was prepared, what was executed, and what was
scored. This campaign tracks all three separately, and the separation is what
makes the partial rows interpretable rather than merely incomplete.

### 4.5 Practical consequences for the remaining work

The lanes that remain open are not blocked on science. E1, E2, E5 and E13 each
hold a completed technical chain — re-implemented or re-verified, with offline
tests passing — and fail only on an authorisation or credential gate held by
the project lead. E6 and E9 are blocked on cloud quota requests already filed.
The seven externally blocked lanes are blocked on third-party access that has
been requested repeatedly and not granted.

This distribution matters for planning. Roughly $108 of authorised-but-unspent
budget would close every lane that is not externally blocked, and none of that
work requires new experimental design. The campaign's remaining cost is
therefore administrative rather than scientific, which is an unusual and
favourable position for a project at this stage.
