#!/usr/bin/env python3
"""Jev triage battery for the E1-E14 campaign (2026-09-19 restart).

Loads receipt-grounded lane facts, asks Jev two independent judgment
batteries over the shared state, and records one receipt per battery:

  1. next_action_class  — one choice per lane from six crisp classes
  2. research_value     — one ordered score per lane for capstone value

Arithmetic and file reads stay in this script; Jev judges only the
semantic classification and value ordering.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from jev_judge import DEFAULT_RECEIPT_DIR, run_ask  # noqa: E402

REPO_ROOT = HERE.parents[1]

# --- code-verified lane facts (sources cited per row) -----------------------
LANES: dict[str, dict[str, str]] = {
    "E1": {
        "name": "SWE-bench Pro (original) + Multilingual replacement",
        "verified_result": "2/731 = 0.274% pass@1 (single-seed, 713 native evaluations; 14 generation failures + 4 artifact losses stay in denominator)",
        "replacement_state": "Multilingual run 35/300 graded; wave10 recovery request v6 sealed ($16, 16 tasks, 10/10 targeted tests passed) BUT executable controller source was deleted with .codex-run cleanup; launch requires re-implementation + re-review + spend authorization",
        "source": "outputs/e1_e14_results_2026-09-05/results.json lane E1; outputs/PES_Phase2_Review_2026-09-12/finish/e1_recovery_v6/*",
    },
    "E2": {
        "name": "FrontierSWE (original) + CORE-Bench replacement",
        "verified_result": "original: 1/17 tasks, replay normalized 0.8628; replacement CORE-Bench: 45/45 capsule setups complete, 0 graded, direct-VM HARD adapter built",
        "blocker": "strict self-imposed lifecycle gate: requires guaranteed provider VM/disk absence by deadline, which no cloud provider guarantees; drafted amendment (report CLEANUP_UNVERIFIED/CLEANUP_LATE instead) awaits user acceptance; proposed envelope $3 target + $1 helper on GCP",
        "source": "outputs/PES_Phase2_Review_2026-09-12/finish/e2_completion/decision_v19/lifecycle_decision.md",
    },
    "E3": {
        "name": "SDAB",
        "verified_result": "No exact-suite result",
        "blocker": "private 80-task bundle, runtime/reset contract, native grader and license all absent; provider followup message sent, no reply",
        "source": "outputs/e1_e14_results_2026-09-05/results.json lane E3",
    },
    "E4": {
        "name": "BankerToolBench",
        "verified_result": "1/100 tasks, recovery metric 0.3115; exact checkout/dataset/artifacts restored, 1175 manifest checks passed",
        "blocker": "native-grader lower bound ~$59.45 exceeds remaining recorded cap; 99 attempt histories unverified (verification is local and free)",
        "source": "outputs/e1_e14_results_2026-09-05/results.json lane E4; finish/e4_completion/readiness_v3",
    },
    "E5": {
        "name": "APEX-Agents (original) + Tau3 replacement",
        "verified_result": "original: 7/480 native-scored, prefix mean 0.050505; replacement Tau3 run10: 20/97 tasks (20.62%) cleaned, v6 diagnostics LOCAL PASS",
        "blocker": "successor27 packet ($80) passed local review; needs spend admission + actor overhead/lifecycle decision; prepaid actor window ended 2026-09-12",
        "source": "outputs/PES_Phase2_Review_2026-09-12/finish/progress_report_latest.md row E5; finish/e5_runtime/successor27_v6_27",
    },
    "E6": {
        "name": "WebBench (original) + WebArena replacement",
        "verified_result": "No exact-suite result; WebArena 0/812, canonical inventory verified (65 source files + ordered IDs 0-811)",
        "blocker": "AWS EC2 Standard On-Demand quota 1 vCPU vs 16 needed (case CASE_OPENED in us-east-2); two m6a.2xlarge + 2x1000GiB plan, $12 parent reservation; readiness checker requires quota headroom + fresh termination evidence",
        "source": "outputs/PES_Phase2_Review_2026-09-12/finish/e6_continuation/REPORT_v1.md",
    },
    "E7": {
        "name": "BinaryAudit",
        "verified_result": "1/28 historical errored attempt, no grade (older ledger: 1/46, verifier reward 0.0)",
        "blocker": "private payload and native verifier missing; revision-bound authorization required for remaining tasks",
        "source": "outputs/PES_Phase2_Review_2026-09-12/finish/progress_report_latest.md row E7",
    },
    "E8": {
        "name": "LifeSciBench (original) + LAB-Bench public replacement",
        "verified_result": "LAB-Bench public split 1967/1967 complete",
        "blocker": "original private scope blocked (official task package + native grader absent)",
        "source": "outputs/PES_Phase2_Review_2026-09-12/finish/progress_report_latest.md row E8",
    },
    "E9": {
        "name": "MLE-bench (original) + MLDevBench replacement",
        "verified_result": "original: 40/75 competitions with native grades (53.33% coverage), suite score null; replacement MLDevBench 0/34 graded, 1 setup failure",
        "blocker": "runtime image incomplete; EC2 build blocked by AWS vCPU quota 1 < 4; restored dependency sources deleted with .codex-run; installed Poetry identity unverified",
        "source": "outputs/PES_Phase2_Review_2026-09-12/finish/e9_completion/HANDOFF.md",
    },
    "E10": {
        "name": "AgentHarm (original) + AgentDojo benign replacement",
        "verified_result": "AgentDojo benign utility scope 97/97 complete",
        "blocker": "private AgentHarm task files and authorized native grading route absent",
        "source": "outputs/PES_Phase2_Review_2026-09-12/finish/progress_report_latest.md row E10",
    },
    "E11": {
        "name": "VerilogEval",
        "verified_result": "312/312 complete across two native framings; canonical 129/312 = 41.35% pass@1 (67/156 code-completion, 62/156 spec-to-RTL)",
        "blocker": "none in declared scope",
        "source": "outputs/e1_e14_results_2026-09-05/E1_E14_Results.md row E11",
    },
    "E12": {
        "name": "AppBench",
        "verified_result": "No exact-suite result; admission validator work resumed earlier",
        "blocker": "official deployment, task artifacts and native grading route absent; provider request open without reply",
        "source": "outputs/e1_e14_results_2026-09-05/results.json lane E12",
    },
    "E13": {
        "name": "OpenReward Games (original) + BALROG replacement",
        "verified_result": "BALROG 13/255 episodes; 21 never-started receipt candidates mapped with full acquisition chain",
        "blocker": "strict 3600s provider-lifecycle gate BLOCKED; drafted amendment (separate hosted cancellation supervisor) awaits user acceptance; $8 H200 actor hold; full 255 scoring disabled until terminal receipts exist",
        "source": "outputs/PES_Phase2_Review_2026-09-12/finish/e13_continuation/decision_v11/DECISION.md",
    },
    "E14": {
        "name": "FrontierMath (original) + Omni-MATH public replacement",
        "verified_result": "Omni-MATH 4426/4428 accepted dispositions (2 recorded parser failures), official accuracy 51.31% reproduced by native scorer recheck",
        "blocker": "private FrontierMath hosted evaluation requires Epoch authorization",
        "source": "outputs/PES_Phase2_Review_2026-09-12/e14_scoring_recheck.json",
    },
}

BLOCKER_CLASSES = [
    "NO_NEW_RESULT_NEEDED",
    "LOCAL_FREE_STEPS_ONLY",
    "USER_SPEND_AUTHORIZATION",
    "USER_POLICY_AMENDMENT_THEN_SPEND",
    "CODE_REIMPLEMENTATION_THEN_SPEND",
    "EXTERNAL_PARTY_GRANT",
]

BLOCKER_DEFS = {
    "NO_NEW_RESULT_NEEDED": "The lane's declared replacement scope already has full verified coverage; a further native result adds nothing required.",
    "LOCAL_FREE_STEPS_ONLY": "A new native result (or required verification of an existing one) is achievable now on this machine at zero cost with no user decision.",
    "USER_SPEND_AUTHORIZATION": "Everything technical is ready; the only missing item is the user's explicit authorization to spend (quota/capacity may also be pending).",
    "USER_POLICY_AMENDMENT_THEN_SPEND": "A self-imposed lifecycle/contract gate blocks any launch; the user must first accept a drafted amendment, then authorize spend.",
    "CODE_REIMPLEMENTATION_THEN_SPEND": "The prepared execution source was lost and must be re-created and re-reviewed before a launch the user must then authorize.",
    "EXTERNAL_PARTY_GRANT": "A private dataset, hosted evaluation, or credentials from an outside party is required; money and local work alone cannot produce the result.",
}

BLOCKER_RULE = (
    "Pick the FIRST blocker in this causal chain that is not already cleared: "
    "external party grant -> code reimplementation -> policy amendment -> spend "
    "authorization -> local free steps -> none needed."
)

VALUE_LEVELS = (
    "level_0_marginal: completing this lane adds nothing citable to the capstone results; "
    "level_1_supporting: improves evidence hygiene or coverage bookkeeping but no new suite result; "
    "level_2_notable: adds a new suite result or closes a declared scope gap in the results table; "
    "level_3_headline: materially changes the headline results table or defends a core thesis claim"
)


def main() -> None:
    blocker_answers: dict[str, dict] = {}
    receipts: list[str] = []
    for lane, info in LANES.items():
        lane_state = {
            "context": (
                "PES University capstone (Tinker RL Lab) E1-E14 benchmark campaign, "
                "restarted 2026-09-19 after a clean halt on 2026-09-12. Remaining recorded "
                "spend cap ~$44; several envelopes ($16 E1, $80 E5, $4 E2, $8 E13) await "
                "user authorization. Phase-2 college review delivered 2026-09-12."
            ),
            "class_definitions": BLOCKER_DEFS,
            "selection_rule": BLOCKER_RULE,
            "lane": lane,
            "facts": info,
        }
        q = {
            "first_uncleared_blocker": {
                "type": "choice",
                "instructions": (
                    f"Lane {lane} ({info['name']}). Using only `facts` and "
                    "`class_definitions`, and applying `selection_rule`, pick the first "
                    "uncleared blocker between this lane and a NEW native suite result."
                ),
                "choices": BLOCKER_CLASSES,
            }
        }
        r = run_ask(lane_state, q, label=f"triage_blocker_{lane}")
        receipts.append(r["receipt_path"])
        blocker_answers[lane] = r["answer"].get("answers", r["answer"]).get(
            "first_uncleared_blocker", r["answer"]
        )

    state = {
        "context": (
            "PES University capstone (Tinker RL Lab) E1-E14 benchmark campaign, "
            "restarted 2026-09-19 after a clean halt on 2026-09-12. Remaining recorded "
            "spend cap ~$44 (E4 estimate alone ~$59), E5 successor27 $80 unreserved, "
            "E1 wave10 $16 sealed, E2 $4 and E13 $8 envelopes proposed. "
            "Phase-2 college review delivered 2026-09-12; final phase ahead."
        ),
        "lanes": LANES,
    }
    value_questions = {
        f"{lane}_research_value": {
            "type": "score",
            "instructions": (
                f"Score the capstone research value of FULLY completing lane {lane} "
                f"({info['name']}) given `lanes.{lane}` facts, on this ordered scale: "
                f"{VALUE_LEVELS}. Judge the marginal value of completion, not current status."
            ),
        }
        for lane, info in LANES.items()
    }
    r2 = run_ask(state, value_questions, label="triage_research_value")

    out = {
        "as_of": "2026-09-19",
        "blocker_answers": blocker_answers,
        "value_answers": r2["answer"].get("answers", r2["answer"]),
        "receipts": receipts + [r2["receipt_path"]],
    }
    dest = REPO_ROOT / "outputs/jev_receipts/TRIAGE_SUMMARY_2026-09-19.json"
    dest.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out, indent=1, sort_keys=True))
    print("summary:", dest)


if __name__ == "__main__":
    main()
