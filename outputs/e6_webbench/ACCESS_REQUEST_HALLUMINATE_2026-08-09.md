# Access request — Halluminate WebBench live environment, native verifier, held-out split

**Send to:** Halluminate (WebBench maintainers). Suggested channels, in order:
1. An issue or discussion on `https://github.com/Halluminate/WebBench` referencing
   the open ground-truth issue `https://github.com/Halluminate/WebBench/issues/2`.
2. The contact address listed on the Halluminate site / WebBench README at the
   time of sending. *(This document deliberately does not guess an email address —
   fill in the current one before sending.)*

**From:** Arvind (arvindcr4@gmail.com) — academic evaluation of a fine-tuned
vision-language browser agent.

**Subject:** WebBench evaluation access — live environment digest, native
verifier, and held-out split

---

## What we already have, and what we did with it

We are running WebBench as a *primary* evaluation suite, not as a proxy. We have
pinned and verified the public release:

| Item | Value |
|---|---|
| Repository | `https://github.com/Halluminate/WebBench` |
| Pinned revision | `ea7a1628443321363989f354401f0653e0cba6f4` |
| Dataset file | `webbenchfinal.csv` |
| Dataset SHA-256 | `fd5311a38bdb6f941e8f544150735656c114d76fbfb17193da973d5de0165217` |
| LICENSE SHA-256 | `96804aa272fe40cdfb8b5c8f4d1d94757bcfaf1bf5596fb829214843d2371e58` (MIT) |
| Public task count | 2,647 |

From that file we derived a deterministic task index and split manifest so that
any result we report can be tied to an exact task set:

| Aggregate | Value |
|---|---|
| Evaluation split manifest hash | `feb368884a41c994567cd7067cf22c47fc67e27158dd635df7ab4a594cf93f0c` |
| Task-ID hash (2,647 IDs, canonical JSON) | `e677af69aa5d1dc7137e54c18c41d99343a41b3e5e77377f7512f42c1a34d2b5` |
| Task-content digest hash | `061fd64b235505d3218f087c01b9b5946c6a7ef2baa5893c549b0ce8ad68a32a` |
| Task index hash | `84921790fdd4cf20af876324311ea2d6399fc6032cd4ce4ca44fa5b7fe6f85e8` |
| Newline-joined integer-ID hash (legacy form) | `22afbdd3cc47e6dba1e3c57ddbe5f762b54be5d2af6ac76bbd206c19eb83b12e` |
| Row-content manifest hash (legacy form) | `66da44a04ec48fe356b3b0d1c420c40679faa1a7ac650728e254b625bb674a07` |

Two independent implementations in our repo produce the last two values from the
same CSV, so if your copy of `webbenchfinal.csv` at that revision hashes to
`fd5311a3…`, these numbers should reproduce exactly on your side.

One observation we would like confirmed: the public CSV carries **sparse**
integer IDs spanning `0..2724` with **78 IDs absent** (2,647 rows over a 2,725-wide
ID space). The first absent IDs are 14, 18, 278, 280, 282–286, 289, 301, 342.
We currently treat the absent IDs as removed-from-public, not as a split.

## What we cannot do, and why we are writing

The public repository contains the task list and the MIT license. It does not
contain the live browser environment, the task credentials, the reset contract,
or the native ground-truth verifier. Our runner is fail-closed and refuses to
emit a score without them; it currently reports 7 of 8 gates passing and one
blocker: `official Halluminate environment/verifier access receipt is missing`.

We will not substitute BrowserGym, WebArena, MiniWoB, or a local Playwright
harness. A browser bridge is not WebBench, and we will not label a local browser
result `webbench_eval`. So the choice is: run the real thing, or report BLOCKED
with a null score. We would prefer the former.

## What we are asking for

### 1. Live environment access, with an immutable digest

We need to be able to state which environment produced the result. Specifically:

- `environment_id`
- `environment_revision` — immutable, 40-hex
- `container_image_digest` — `sha256:<64 hex>` for the exact image
- `browser_revision` — the exact browser build
- confirmation that `screenshot_capture`, `dom_capture`, and `task_reset` are
  available
- `credential_scope` — which accounts/credentials the harness may use, and on
  which sites

### 2. The native verifier

The repository's open ground-truth issue
(`https://github.com/Halluminate/WebBench/issues/2`) indicates the ground truth
and scoring script are not public. We need:

- `verifier_id`
- `verifier_revision` — immutable, 40-hex
- `verifier_sha256`
- `command` — the exact argv the verifier is invoked with
- `receipt_url` — an HTTPS URL we can cite as the access record
- confirmation that `ground_truth_available` is true for the task IDs we are
  authorized to run

Human annotation of trajectories is explicitly *not* what we are asking for. We
will not report a human-labelled pass rate as a WebBench score.

### 3. The held-out / official evaluation split

- The authoritative evaluation task-ID list (or a confirmation that all 2,647
  public IDs are the evaluation set).
- Whether any held-out tasks exist beyond the public CSV, and if so how a result
  on the public 2,647 should be labelled relative to them.
- Whether the 78 absent IDs in `0..2724` correspond to a withheld set.

### 4. Task authorization and side-effect policy

2,647 tasks run against **449 live hostnames / 448 registrable domains**, and
1,010 of them (38%) are write-class tasks — CREATE 594, UPDATE 206, DELETE 166,
FILE_MANIPULATION 44. 536 task statements reference logging in or an account.
These mutate real third-party sites, so we need explicit authorization rather
than inferred permission:

- `allowed_task_ids` (or confirmation of the full manifest)
- `credential_scope`
- `side_effect_policy` — what the harness is permitted to create, modify, or
  delete, and on which sites
- `terms_or_license_signoff` — the MIT license covers the repository contents; it
  does not authorize automated writes to third-party services

## The exact response format our harness accepts

If it is convenient, returning the following JSON object lets us wire the access
straight into our runner with no interpretation on our side. Our validator
requires every field below; it rejects the object otherwise.

```json
{
  "provider": "Halluminate",
  "benchmark": "Halluminate/WebBench",
  "approved": true,
  "access_receipt_id": "<your record id>",
  "environment": {
    "environment_id": "<id>",
    "environment_revision": "<40-hex immutable revision>",
    "container_image_digest": "sha256:<64 hex>",
    "browser_revision": "<browser build>",
    "screenshot_capture": true,
    "dom_capture": true,
    "task_reset": true,
    "credential_scope": "<which credentials, on which sites>"
  },
  "native_verifier": {
    "available": true,
    "ground_truth_available": true,
    "verifier_id": "<id>",
    "verifier_revision": "<40-hex immutable revision>",
    "verifier_sha256": "sha256:<64 hex>",
    "command": ["<argv>", "<...>"],
    "receipt_url": "https://<https url we can cite>"
  },
  "task_authorization": {
    "allowed_task_ids": ["<explicit list>"],
    "side_effect_policy": "<what may be created/modified/deleted>",
    "terms_or_license_signoff": "<reference>"
  },
  "split": {
    "evaluation_task_id_hash": "<your hash of the official eval split>",
    "heldout_exists": true,
    "heldout_relationship_to_public_csv": "<description>"
  }
}
```

## What we commit to

- We will report the exact `environment_revision`, `container_image_digest`, and
  `verifier_revision` alongside any number we publish.
- We will not publish a WebBench score produced by any verifier other than yours.
- We will respect the side-effect policy you specify, and we will not run
  write-class tasks outside the authorized scope.
- If access is declined, we will report WebBench as BLOCKED with a null score
  rather than substituting another benchmark.

Thank you.

---

### Provenance of the numbers in this request

| Claim | Artifact |
|---|---|
| Dataset / license hashes, task count | `outputs/e6_webbench/split/webbench_eval_split_manifest.json` |
| Aggregate hashes, derivation method | `outputs/e6_webbench/split/webbench_split_derivation.json` |
| Per-task IDs and digests | `outputs/e6_webbench/split/webbench_task_index.jsonl` |
| ID gaps, category and domain counts | `outputs/e6_webbench/split/webbench_task_characterization.json` |
| Train/eval disjointness | `outputs/e6_webbench/split/webbench_disjointness_proof.json` |
| Current blocker | `outputs/e6_webbench/logs/runner_preflight_2026-08-09.json` |
