# PES evaluation completion workflow

The user selected the highest PES project evaluation grade as the goal. A grade is not an observable completion predicate for this agent. The local exit condition is a current thesis review PDF, an accurate contribution and defense guide, a working offline demo, exact file manifests, passing relevant checks, and a recorded independent review with every material finding resolved or explicitly bounded.

## pstack source and host mapping

The requested pstack workflow comes from [Cursor's official pstack plugin](https://github.com/cursor/plugins/tree/978ca6e9c0e012047cd5058b8e829240336e704e/pstack). The autonomous-run and swarm playbooks, decision-log skill, and Prove It Works and Explain the Number principles guide this task. The skill source was read at that pinned commit. It was not installed globally. No Cursor-only commands or unavailable model names are claimed to have run.

This host allows four concurrent agents including the coordinator. Reviews run in bounded batches, with separate write scopes and read-only final reviewers. Thirty review areas are coverage requirements, not a claim that thirty agents ran simultaneously. The current foreground turn and agent completion messages provide the loop; no recurring automation or external message was scheduled.

## Autonomous-run steps

1. Define observable acceptance conditions before implementation.
2. Continue on meaningful completion events or a suitable timed wake-up.
3. Make a bounded change, test it, and retain it only when it improves the result.
4. Resolve reversible defects discovered along the way, then return to the main objective.
5. Record decisions and verification outcomes as the work progresses.
6. Stop when the original acceptance conditions are satisfied or a genuine external blocker prevents progress.

Host adaptations: step 2 uses agent completion events within this Codex turn because Cursor `/loop` is unavailable. Step 3 leaves a reviewable local diff; no PR, merge, upload, or institutional submission is part of this delivery. Earlier evidence exploration preceded the user's pstack instruction and is recorded as such in the decision log.

## Work units

| Unit | Owner | Required proof |
|---|---|---|
| Current evidence and audit semantics | Benchmark agent | Regression tests, explicit audit result counts |
| PES contribution and viva preparation | Submission agent | Source-bound claims, working links |
| Software and numerical edge cases | Validation agent | Core checks, property cases, targeted regression tests |
| Current thesis and package | Coordinator | Successful build, final PDF inspection, inventory hashes |
| Independent acceptance | Fresh review agents | Thirty-area coverage report and exact artifact identities |

The thirty review areas are thesis identity, revision date, scientific scope, inherited ownership, individual contribution, AI declaration, five reported corrections, audit status semantics, audit count semantics, source availability, frozen-file preservation, PDF text, PDF layout, references, figures, controlled comparison, inferential limits, benchmark denominators, missing outcomes, C1 endpoints, deck alignment, deck inclusion, talk timing, viva accuracy, demo execution, demo labeling, test results, numerical edge cases, package integrity, and institutional remainder.

The final review must distinguish assistant verification from human scientific review. Neither source hashes nor tests establish a high grade or independently authenticate historical provider runs.
