# License risk acceptance — 2026-08-09

**Decision owner:** repository owner (arvindcr4). **Recorded by:** coordinator session.

The owner has directed that the assets below be treated as open for the purposes
of this work, and authorized proceeding without a per-asset license grant.

This file records that decision. It does **not** assert that a license exists.
Receipts must continue to report the observed license state factually and cite
this file as the reason work proceeded — never substitute a license identifier
that was not found at the pinned revision.

## Observed state (verified at pinned revisions, 2026-08-09)

| Asset | Pinned revision | Observed license state | Evidence |
|---|---|---|---|
| `QuesmaOrg/BinaryAudit` | `cbd86c7c` | **Absent.** No LICENSE blob added in any commit across all 4 remote refs; GitHub API `license: null`; `/license` and raw URL both 404. README:127-129 claims Apache-2.0 with a dangling link. | E7 lane + coordinator re-verification |
| `EnvCommons/*` (277 repos) | per-env, e.g. `wordle@92bea32e` | **Absent.** No LICENSE file in any of the 277 repos. Environment cards claim MIT by pointing at TextArena's license; the wrapper code carries no grant of its own. | E13 lane |
| `AfterQuery/App-Bench` | `de80d5bc` | **Absent.** No `cardData` key at all; 404 for README/LICENSE/COPYING/NOTICE/TERMS. Absence is specific — AfterQuery declares apache-2.0 / cc-by-4.0 / mit on 3 of its other 6 public datasets. | E12 lane |
| `ScaleAI/SWE-bench_Pro` | `7ab51149` | **Absent.** `cardData.license` is null. (The *evaluator* repo `scaleapi/SWE-bench_Pro-os@ca10a60` is MIT — that covers code, not data.) | E1 lane + coordinator |
| `Proximal-Labs/frontier-swe` | `422b9bb9` | **Absent.** No root LICENSE/COPYING/NOTICE; the only license files anywhere are vendored third-party fixtures inside task payloads. | E2 lane |
| RTLCoder dataset variants | various | **Absent.** `license: None` on the HF API for every variant surveyed. | coordinator |

Assets with a real, immutable license artifact are unaffected and keep it:
WebBench (MIT), BankerToolBench (CC-BY-4.0 dataset, Apache-2.0 repo),
APEX Agents (CC-BY-4.0), VerilogEval (MIT), MLE-bench code (MIT, explicitly
excluding competition data), AgentHarm (`other`, LICENSE file present),
SWE-Gym (MIT), OpenR1-Math-220k (Apache-2.0).

## Scope of the acceptance

- Applies to local training, evaluation, and internal reporting in this repository.
- Does **not** authorize redistributing any of these assets.
- Should be revisited before external publication. "No license found" is default
  copyright, which is a materially different position from a permissive grant,
  and a reviewer may ask.

## How receipts must reference this

```json
"license": {
  "observed_state": "absent_at_pinned_revision",
  "claimed_spdx": null,
  "proceeding_under": "outputs/_setup/LICENSE_RISK_ACCEPTANCE_2026-08-09.md",
  "decision": "owner_risk_acceptance_2026-08-09"
}
```

Do not emit `"license": "MIT"` (or any SPDX identifier) for an asset in the table
above. The fail-closed license validators in the lane adapters should be pointed
at this record rather than deleted, so the gap stays visible in every receipt.

## Unaffected by this decision

Kaggle per-competition rules (E9) are **not** covered. Those are an explicit
click-through agreement bound to the account holder, not a missing-license
question, and 74 of 75 competitions remain un-accepted.
