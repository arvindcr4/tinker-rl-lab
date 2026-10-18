// Sanitized paraphrases of verified feedback sources. Do not add raw email,
// reviewer identifiers, private links, or claims that revisions occurred. Scores
// require an explicit recorded value, scale provenance and dated source context.
export const FEEDBACK_VERIFIED_ON = "2026-10-18";
export const SUBMISSION_FEEDBACK = {
  "icicc-2027-243": {
    "evidence": "Organizer acceptance email dated 11 October 2026, including two complete numbered reviewer reports and final-paper guidance.",
    "coverage": {
      "reviews": "available",
      "decision": "available",
      "responses": "not_found"
    },
    "coverageNotes": {
      "reviews": "The two review reports are complete as reproduced in the acceptance email. No numerical review scores or author-response text were found in the checked correspondence. Camera-ready and registration completion remain unverified.",
      "decision": "Acceptance and requested revisions are verified from the organizer email; no separate scientific acceptance rationale.",
      "responses": "No author-response text was found in the checked correspondence."
    },
    "decisionReasons": [
      {
        "label": "Decision notification",
        "date": "2026-10-11",
        "points": [
          "The organizer accepted the paper subject to final-paper preparation and registration requirements. The notice does not give a separate scientific acceptance rationale; final publication remains unverified."
        ]
      },
      {
        "label": "Organizer final-paper requirements (not scientific decision reasons)",
        "date": "2026-10-11",
        "points": [
          "Registration is due 30 October 2026; the email does not specify a timezone. The final manuscript is requested at the earliest, without a distinct dated camera-ready deadline.",
          "Prepare the final Springer-format manuscript, send it and the registration e-receipt to the conference, and upload the revision under the existing CMT submission rather than creating another submission.",
          "Check similarity using licensed Turnitin or iThenticate. The email first says no more than 15%, then specifies below 15% in its mandatory list; use the stricter wording pending clarification.",
          "Replace low-resolution screenshots with clear, labelled, original figures; number every figure and table and use descriptive captions.",
          "Typeset and number equations rather than embedding equation images, and cite borrowed formulas.",
          "Use one consistent venue-required reference style. The email gives IEEE only as a generic example while also requiring Springer formatting; confirm against the actual template.",
          "Include at least 15 relevant, recent references with complete publication details and DOI where applicable; use self-citations only when substantively justified.",
          "Verify author names, affiliations and contact information before final submission.",
          "The organizer warns against fabricated or AI-generated content and AI-generated citations; independently verify all claims and references and follow the venue's writing policy.",
          "Cite credible, verified datasets and confirm ethical sourcing.",
          "Student authors are asked to have the final manuscript reviewed by their supervisor or mentor."
        ]
      }
    ],
    "reviews": [
      {
        "label": "Reviewer 1",
        "snapshotOn": "2026-10-11",
        "kind": "review",
        "strengths": [],
        "concerns": [],
        "requests": [
          "Explain the original contribution and how receipt-bound governance makes benchmark results more reproducible and auditable.",
          "Document the GRPO training setup, actor architecture, hyperparameters and held-out evaluation protocol in enough detail to reproduce the study.",
          "Explain why the fourteen suites were selected and which tasks and metrics can meaningfully be compared.",
          "Define partial, blocked and externally blocked evaluation states; show how coverage is calculated and what evidence would reopen each lane.",
          "Compare with suitable published baselines where a valid comparison is possible, keeping original-contract results separate from replacement-scope results.",
          "Discuss the limits of evaluating one actor, the low SWE-bench Pro pass rate, BankerToolBench behavior and how far the governance approach generalizes."
        ],
        "date": "2026-10-11"
      },
      {
        "label": "Reviewer 2",
        "snapshotOn": "2026-10-11",
        "kind": "review",
        "strengths": [],
        "concerns": [],
        "requests": [
          "Use publication-quality figures at a minimum resolution of 300 dpi and replace unclear or pixelated images.",
          "Keep tables editable and follow Springer table formatting.",
          "Typeset equations with an equation editor or LaTeX, align them consistently and number them where appropriate.",
          "Include relevant recent literature from 2024 and 2025.",
          "State the main contributions as bullets in the Introduction.",
          "Keep the manuscript within the stated 10–12-page range, with references included in the 12-page maximum.",
          "Add a dedicated discussion of practical implications, real applications and societal benefits.",
          "Apply the Springer template consistently to headings, fonts, citations and section structure."
        ],
        "date": "2026-10-11"
      }
    ],
    "authorResponses": [],
    "authorResponseReceipts": [],
    "notes": [
      "Faithful paraphrases of source feedback, not verbatim review publication. Historical author statements are attributed and do not establish that a change was implemented or that an experimental claim is currently verified.",
      "No posted author response was found in the checked ICICC messages.",
      "No numerical reviewer scores are present in the acceptance email.",
      "No independent confirmation of final submission, publication or registration completion.",
      "Reviewer 1: The review described BankerToolBench as tool-dialogue collapse. That is historical wording; the current evidence identifies a serving-bridge limitation. Preserve the corrected interpretation and do not change measurements to conform to the older review."
    ]
  },
  "neurips-2026-main-36320": {
    "evidence": "Full original reviews and initial area-chair meta-review preserved in an author-provided archived review bundle, marked as fetched from OpenReview on 24 July 2026; later official reviewer-comment and author-comment emails dated 30 July and 2 August; final decision email dated 24 September. Rating scales were independently verified against the official NeurIPS 2026 Main Track Handbook on 18 October 2026.",
    "coverage": {
      "reviews": "partial",
      "decision": "partial",
      "responses": "partial"
    },
    "coverageNotes": {
      "reviews": "Three full original reviews and the initial meta-review are available from the July 24 archive. Two later reviewer comments and three posted author comments are available in official notification emails. Three July 24 rebuttal receipts confirm posting but do not include their text. The archived draft rebuttals are not treated as the posted versions. The final meta-review and latest live portal state have not been verified.",
      "decision": "Initial area-chair assessment and later reviewer discussion are available; final paper-specific rejection rationale is missing.",
      "responses": "Three posted comments are available in full; three original rebuttal receipts confirm posting without providing the submitted text. Archived drafts are not substituted."
    },
    "decisionReasons": [
      {
        "label": "Decision notification",
        "date": "2026-09-24",
        "points": [
          "The final notification says the paper was not accepted after review by reviewers, area chairs and senior area chairs. It does not include a paper-specific final rationale."
        ]
      },
      {
        "label": "Decision coverage",
        "date": "2026-09-24",
        "points": [
          "The initial meta-review and subsequent reviewer comments below document concerns during discussion; they must not be presented as a verified final rejection rationale."
        ]
      },
      {
        "label": "Initial area-chair meta-review (archived; not final rationale)",
        "points": [
          "Determine whether two observed failures and weak ranking results can support a broad early-step diagnostic rule, with per-run results and a clear corpus/task-to-claim mapping.",
          "Establish whether conclusions from the non-canonical runner transfer to standard GRPO or must remain implementation-specific.",
          "Determine whether the ledger-summary inconsistency affects other configurations and what can be concluded from mixed-model, mixed-task and mixed-protocol analyses.",
          "Substantially improve the introduction of the diagnostics, the explanation of evaluation choices and the mapping from claims to evidence.",
          "Address the individual reviewers' substantive concerns; the chair said rebuttal and subsequent discussion would determine whether the concerns were resolved."
        ],
        "snapshotOn": "2026-07-24"
      }
    ],
    "reviews": [
      {
        "label": "Reviewer 1",
        "snapshotOn": "2026-07-24",
        "kind": "review",
        "strengths": [
          "Separating online reward, proxy reward, held-out capability and algorithm names provides methodological value.",
          "ZVF and GU are simple, interpretable and useful as early training-monitoring checks.",
          "The paper explicitly limits claims about reasoning gains, final-performance prediction and universal PPO-versus-GRPO rankings.",
          "Confidence intervals, effect sizes, multiplicity corrections and distinctions between held-out and checkpoint-selected analyses improve inspectability."
        ],
        "concerns": [
          "Evidence is fragmented across software stacks, model sizes, tasks and protocols, making clear causal conclusions difficult.",
          "Only two failure cases support the reported early-collapse rule, and consecutive-ZVF ranking is weak; broader reliability is not established.",
          "The non-canonical runner omits PPO clipping, reference KL and completion-only masking, while the GRPO-centered framing risks overstating transfer to standard GRPO.",
          "The clean held-out GSM8K comparison has only 200 questions, a 1.3-percentage-point difference and p = 0.26; it does not establish a capability gain.",
          "Organization and layout appear unfinished, including charts that extend beyond page boundaries."
        ],
        "requests": [
          "Improve coherence and presentation, narrow the reliability and transfer claims to the evidence, and address the weaknesses above. The reviewer indicated that a sufficiently refined version could merit a more positive assessment."
        ],
        "scores": [
          {
            "label": "Overall",
            "value": 3,
            "scale": "1–6 (1 strong reject; 6 strong accept)",
            "asOf": "2026-07-24",
            "note": "Borderline reject"
          },
          {
            "label": "Quality",
            "value": 2,
            "asOf": "2026-07-24",
            "scale": "1–4 (1 poor; 2 fair; 3 good; 4 excellent)"
          },
          {
            "label": "Clarity",
            "value": 1,
            "asOf": "2026-07-24",
            "scale": "1–4 (1 poor; 2 fair; 3 good; 4 excellent)"
          },
          {
            "label": "Significance",
            "value": 4,
            "asOf": "2026-07-24",
            "scale": "1–4 (1 poor; 2 fair; 3 good; 4 excellent)"
          },
          {
            "label": "Originality",
            "value": 4,
            "asOf": "2026-07-24",
            "scale": "1–4 (1 poor; 2 fair; 3 good; 4 excellent)"
          },
          {
            "label": "Confidence",
            "value": 3,
            "asOf": "2026-07-24",
            "note": "3 denotes fairly confident under the official rubric.",
            "scale": "1–5 (1 educated guess; 5 certain)"
          }
        ]
      },
      {
        "label": "Reviewer 2",
        "snapshotOn": "2026-07-24",
        "kind": "review",
        "strengths": [
          "The distinction between training reward, held-out accuracy and diagnostics addresses misleading online reward curves and improves reporting practice."
        ],
        "concerns": [
          "The writing is difficult to follow and the abstract is too long and detailed.",
          "The consequences of omitting clipping and reference KL are not adequately justified.",
          "A mean across ten heterogeneous models has little clear scientific interpretation.",
          "The capacity-ceiling discussion repeats content.",
          "The Table 2 ledger-versus-summary discrepancy raises concerns about other configurations and the completeness of the audit.",
          "Section 4.4 lacks concrete ZVF/GU results across runs and configurations.",
          "The source and consequences of high-reward, high-ZVF behavior are insufficiently analyzed."
        ],
        "requests": [
          "Explain why clipping, reference KL and completion-only masking were omitted and how adding them might change the conclusions.",
          "State whether every ledger and summary row was checked or only the discrepant PPO row.",
          "Provide per-run ZVF and GU alongside reward levels and held-out accuracy.",
          "Analyze the sources and potential consequences of apparent saturation.",
          "Explain what the heterogeneous-model average is intended to show, or revise its use."
        ],
        "scores": [
          {
            "label": "Overall",
            "value": 3,
            "scale": "1–6 (1 strong reject; 6 strong accept)",
            "asOf": "2026-07-24",
            "note": "Borderline reject"
          },
          {
            "label": "Quality",
            "value": 3,
            "asOf": "2026-07-24",
            "scale": "1–4 (1 poor; 2 fair; 3 good; 4 excellent)"
          },
          {
            "label": "Clarity",
            "value": 1,
            "asOf": "2026-07-24",
            "scale": "1–4 (1 poor; 2 fair; 3 good; 4 excellent)"
          },
          {
            "label": "Significance",
            "value": 2,
            "asOf": "2026-07-24",
            "scale": "1–4 (1 poor; 2 fair; 3 good; 4 excellent)"
          },
          {
            "label": "Originality",
            "value": 3,
            "asOf": "2026-07-24",
            "scale": "1–4 (1 poor; 2 fair; 3 good; 4 excellent)"
          },
          {
            "label": "Confidence",
            "value": 3,
            "asOf": "2026-07-24",
            "note": "3 denotes fairly confident under the official rubric.",
            "scale": "1–5 (1 educated guess; 5 certain)"
          }
        ]
      },
      {
        "label": "Reviewer 3",
        "snapshotOn": "2026-07-24",
        "kind": "review",
        "strengths": [
          "The study addresses a timely area and covers modern language models of varied sizes.",
          "The range of evaluation domains could support broad diagnostic testing.",
          "The proposed monitoring measures are inexpensive and easy to apply."
        ],
        "concerns": [
          "The method is barely introduced before the results; the abstract carries equations, hyperparameters and methodological detail that belong elsewhere.",
          "The Introduction does not adequately motivate the problem or provide intuition for readers less familiar with RL for language models.",
          "Related work is brief and may not reflect the current field.",
          "The paper does not adequately explain what the diagnostics mean, why they were chosen or when they apply.",
          "The use-inspired contribution lacks a concrete real-world application.",
          "The prose repeatedly explains what the work is not rather than positively stating its contribution.",
          "Several evaluations use one seed, and analyses are not run across all listed corpora and tasks without a clear selection rationale.",
          "Table 2 extends beyond the page, and filenames appear in the main paper.",
          "Diagnostic and pipeline limitations, including the LoRA configuration's applicability, are insufficiently examined.",
          "The negative assessment was attributed to the missing real-world application for the selected contribution type, very low clarity and unexplained evaluation choices."
        ],
        "requests": [
          "Provide a motivation-led introduction, method walkthrough and fuller related-work discussion before the evidence and results.",
          "Identify the real-world use case and justify choosing a non-canonical implementation over a faithful GRPO implementation.",
          "Explain which corpus, task and algorithm supports each analysis and why, and address single-seed evidence.",
          "Correct the layout and replace filenames with substantive results; any necessary artifact references should be placed appropriately.",
          "Undertake a substantial revision rather than relying on minor rebuttal-stage edits."
        ],
        "scores": [
          {
            "label": "Overall",
            "value": 1,
            "scale": "1–6 (1 strong reject; 6 strong accept)",
            "asOf": "2026-07-24",
            "note": "Strong reject"
          },
          {
            "label": "Quality",
            "value": 1,
            "asOf": "2026-07-24",
            "scale": "1–4 (1 poor; 2 fair; 3 good; 4 excellent)"
          },
          {
            "label": "Clarity",
            "value": 1,
            "asOf": "2026-07-24",
            "scale": "1–4 (1 poor; 2 fair; 3 good; 4 excellent)"
          },
          {
            "label": "Significance",
            "value": 2,
            "asOf": "2026-07-24",
            "scale": "1–4 (1 poor; 2 fair; 3 good; 4 excellent)"
          },
          {
            "label": "Originality",
            "value": 3,
            "asOf": "2026-07-24",
            "scale": "1–4 (1 poor; 2 fair; 3 good; 4 excellent)"
          },
          {
            "label": "Confidence",
            "value": 3,
            "asOf": "2026-07-24",
            "note": "3 denotes fairly confident under the official rubric.",
            "scale": "1–5 (1 educated guess; 5 certain)"
          }
        ]
      },
      {
        "label": "Reviewer 3 follow-up",
        "date": "2026-07-30",
        "kind": "reviewer_followup",
        "strengths": [],
        "concerns": [
          "The response partly addressed the intended use, but structural concerns remained unresolved because the proposed substantial rewrite could not be assessed during discussion.",
          "Moving filenames into an appendix would not solve the deeper issue that actual results must be presented.",
          "The described practitioner framework still did not demonstrate a concrete application arising from needs outside the research community, as the reviewer understood the use-inspired requirement.",
          "The reviewer still lacked an explanation for single-seed evaluations and missing corpus/task combinations, and interpreted the matched five-seed comparison as withdrawn.",
          "The reviewer kept the assessment unchanged."
        ],
        "requests": []
      },
      {
        "label": "Reviewer 1 follow-up",
        "date": "2026-08-02",
        "kind": "reviewer_followup",
        "strengths": [],
        "concerns": [
          "The reviewer said the authors' commitments addressed most concerns, but the current version still appeared rushed and unfinished.",
          "Uncertainty about the final quality prevented a positive assessment at that stage; further refinement before a later conference submission was encouraged."
        ],
        "requests": []
      }
    ],
    "authorResponses": [
      {
        "label": "Author discussion comment",
        "date": "2026-07-30",
        "points": [
          "The authors accepted clarity and structure as a central problem and proposed a motivation-led Introduction with a small worked example, followed by Background, a new standalone Diagnostics section, an explicit corpus/task/claim matrix, claim-organized Results and affirmative Limitations.",
          "They proposed removing equations and hyperparameters from the abstract, fixing overflowing tables and charts and moving internal filenames to the appendix.",
          "They said two positive cases did not justify a validated early-step rule and committed to recasting it as a descriptive per-run observation without validated-triage language.",
          "They disclosed that the reported five-seed estimator comparison used batch size 4 for one seed and batch size 8 for the other four. Arms within each seed shared a batch size, but the pooled result combined configurations.",
          "They reported a historical paired difference of +0.5 percentage points, a 95% interval of roughly −0.9 to +1.9, p about 0.37 and sign changes across seeds. They emphasized that a near-ceiling null result does not establish equivalence or a capability gain.",
          "They clarified that the estimator toggle covered importance weighting and completion masking, not reference KL.",
          "They asked whether the planned restructuring matched the reviewers' requests. They explicitly described these as camera-ready commitments because the submission PDF was frozen, not implemented revisions."
        ]
      },
      {
        "label": "Author reply addressing Reviewer 1",
        "date": "2026-08-02",
        "points": [
          "The authors corrected their earlier Qwen3-8B GSM8K ZVF figure from approximately 8.5% to a reported 6.7% mean over 30 logged steps; they said the earlier value came from prose without a supporting artifact. Reported first-five and last-ten values were zero.",
          "They said they had demoted the early-step rule, not withdrawn the five-seed matched estimator control, and reiterated the between-seed batch-size difference.",
          "They reported historical held-out means of 92.6% versus 92.1%, with a 92.4% base, using 200 items per arm and seed. Reported paired differences were +0.5, −1.0, 0.0, +1.0 and +2.0 percentage points; mean +0.50, 95% interval −0.89 to +1.89, t(4) = 1.00 and p = 0.374.",
          "They explicitly said that this did not repair the weak capability evidence: absence of a detected difference is not equivalence, and neither arm had an established capability gain.",
          "They again limited the toggle to importance weighting and completion masking, excluding reference KL.",
          "They asked whether the proposed motivation-first structure, diagnostics section, evidence matrix and claim-organized results met the request for refinement, while explicitly retaining the status of a future commitment."
        ]
      },
      {
        "label": "Author reply addressing Reviewer 2",
        "date": "2026-08-02",
        "points": [
          "The authors repeated the ZVF correction from approximately 8.5% to a reported 6.7% and acknowledged a recurring failure mode in which prose numbers were reused without recomputation from logs.",
          "They stated explicitly that only the Qwen3-8B PPO row had been audited; the full ledger-summary sweep remained unfinished.",
          "They supplied a historical per-seed group-size sweep for Qwen2.5-0.5B on arithmetic-verifiable training, 40 steps, with seeds 42, 123 and 456. Each triple below follows that seed order and reports last-ten reward / mean ZVF / early ZVF / held-out accuracy.",
          "G = 2: reward 0.978, 0.975, 0.978; mean ZVF 0.827, 0.852, 0.836; early ZVF 0.613, 0.537, 0.575; held-out 0.990, 0.975, 0.980.",
          "G = 4: reward 0.969, 0.973, 0.972; mean ZVF 0.758, 0.781, 0.752; early ZVF 0.287, 0.287, 0.188; held-out 0.990, 0.990, 0.985.",
          "G = 8: reward 0.973, 0.980, 0.966; mean ZVF 0.702, 0.692, 0.678; early ZVF 0.138, 0.087, 0.062; held-out 0.985, 0.995, 0.990.",
          "G = 16: reward 0.989, 0.987, 0.975; mean ZVF 0.622, 0.634, 0.637; early ZVF 0.000, 0.037, 0.037; held-out 0.970, 0.975, 0.990.",
          "Across the reported 480 logged steps, the authors reported ZVF spanning zero to one, with Spearman correlations −0.74 with gradient norm and −0.53 with advantage variance. They interpreted decreasing mean ZVF with group size as consistent with the mixed-group mechanism.",
          "They reiterated their disagreement with the claim that the five-seed comparison was withdrawn, reported the same narrow near-ceiling null result, and disclaimed equivalence, capability improvement and reference-KL coverage."
        ]
      }
    ],
    "authorResponseReceipts": [
      {
        "label": "Three original rebuttals",
        "date": "2026-07-24",
        "note": "Three official notifications confirm that reviewer-specific rebuttals were posted. Their actual posted text is not included in those receipts. A separately archived bundle contains draft responses, but the drafts have not been verified as identical to the submitted rebuttals and are not substituted for them."
      },
      {
        "label": "Confidential author-to-chair comment",
        "date": "2026-07-25",
        "note": "Posting receipt is available; the comment text is not included and is not reproduced."
      }
    ],
    "notes": [
      "Current evidence caveat, checked 18 October 2026: the saved provenance audit dated 28 July 2026 found that three of the five claimed seed pairs were backfilled summaries without unique upstream-run or treatment links. Two live seed pairs were partly corroborated, and the 92.6% versus 92.1% arithmetic could be reproduced, but the audit rejected this as admissible five-seed matched transfer evidence. A revised response draft withdrew that transfer claim; posting of that revision is not verified. The August 2 comments below are genuine historical author statements and do not override that evidence limitation.",
      "Faithful paraphrases of source feedback, not verbatim review publication. Historical author statements are attributed and do not establish that a change was implemented or that an experimental claim is currently verified.",
      "This is the historical review and discussion record, not an assessment of the current results. Author-reported numbers and camera-ready commitments below are attributed statements from that discussion, not newly verified measurements or completed fixes.",
      "The final paper-specific meta-review or decision rationale is not present in the checked sources.",
      "Actual text of the three posted July 24 rebuttals remains unverified; only their posting receipts and separate drafts are available.",
      "A confidential author-to-chair comment has a receipt but its content was not included and is not reproduced.",
      "Latest portal state and any subsequent review edits remain unverified.",
      "No implementation or resolution status is inferred from author commitments.",
      "Reviewer 1: Limitations were acknowledged.",
      "Reviewer 2: Limitations were acknowledged.",
      "Reviewer 3: The reviewer considered the impact statement acceptable but the treatment of limitations insufficient.",
      "Reviewer 3 follow-up: The authors later disputed the interpretation that the five-seed comparison had been withdrawn. Both positions are preserved below; this summary does not resolve that historical disagreement.",
      "All three original reviewers recorded no or very minor ethics concerns. Original review posting dates are unavailable; 24 July is the archive retrieval date."
    ],
    "authorResponseWarning": {
      "label": "Historical author statements; five-seed transfer claim unsupported by the saved provenance audit",
      "checkedAt": "2026-10-18",
      "sourceDate": "2026-07-28",
      "sourceBasis": "Saved read-only Tinker/W&B provenance audit and unposted revised-response record",
      "text": "Current evidence caveat, checked 18 October 2026: the saved provenance audit dated 28 July 2026 found that three of the five claimed seed pairs were backfilled summaries without unique upstream-run or treatment links. Two live seed pairs were partly corroborated, and the 92.6% versus 92.1% arithmetic could be reproduced, but the audit rejected this as admissible five-seed matched transfer evidence. A revised response draft withdrew that transfer claim; posting of that revision is not verified. The August 2 comments below are genuine historical author statements and do not override that evidence limitation."
    }
  },
  "fllmpt-neurips-2026-3": {
    "evidence": "Original decision notification and thread, title/venue searches, and organizer-address correspondence searches in connected email, including Spam and Trash; checked 18 October 2026.",
    "coverage": {
      "reviews": "not_found",
      "decision": "not_found",
      "responses": "not_found"
    },
    "coverageNotes": {
      "reviews": "No individual review text was retrieved from the checked email. Portal-only reviews remain unverified.",
      "decision": "The decision notification contains no submission-specific rationale.",
      "responses": "No posted author-response text was found in the checked email. This does not establish that no portal response exists."
    },
    "decisionReasons": [],
    "reviews": [],
    "authorResponses": [],
    "authorResponseReceipts": [],
    "notes": [
      "The original notification confirms rejection but provides no rationale or review text. No posted author response was found in connected email. Portal-only material has not been retrieved."
    ]
  },
  "fllmpt-neurips-2026-4": {
    "evidence": "Original decision notification and thread, title/venue searches, and organizer-address correspondence searches in connected email, including Spam and Trash; checked 18 October 2026.",
    "coverage": {
      "reviews": "not_found",
      "decision": "not_found",
      "responses": "not_found"
    },
    "coverageNotes": {
      "reviews": "No individual review text was retrieved from the checked email. Portal-only reviews remain unverified.",
      "decision": "The decision notification contains no submission-specific rationale.",
      "responses": "No posted author-response text was found in the checked email. This does not establish that no portal response exists."
    },
    "decisionReasons": [],
    "reviews": [],
    "authorResponses": [],
    "authorResponseReceipts": [],
    "notes": [
      "The original notification confirms rejection but provides no rationale or review text. No posted author response was found in connected email. Portal-only material has not been retrieved."
    ]
  },
  "fllmpt-neurips-2026-5": {
    "evidence": "Original decision notification and thread, title/venue searches, and organizer-address correspondence searches in connected email, including Spam and Trash; checked 18 October 2026.",
    "coverage": {
      "reviews": "not_found",
      "decision": "not_found",
      "responses": "not_found"
    },
    "coverageNotes": {
      "reviews": "No individual review text was retrieved from the checked email. Portal-only reviews remain unverified.",
      "decision": "The decision notification contains no submission-specific rationale.",
      "responses": "No posted author-response text was found in the checked email. This does not establish that no portal response exists."
    },
    "decisionReasons": [],
    "reviews": [],
    "authorResponses": [],
    "authorResponseReceipts": [],
    "notes": [
      "The original notification confirms rejection but provides no rationale or review text. No posted author response was found in connected email. Portal-only material has not been retrieved."
    ]
  },
  "fllmpt-neurips-2026-6": {
    "evidence": "Original decision notification and thread, title/venue searches, and organizer-address correspondence searches in connected email, including Spam and Trash; checked 18 October 2026.",
    "coverage": {
      "reviews": "not_found",
      "decision": "not_found",
      "responses": "not_found"
    },
    "coverageNotes": {
      "reviews": "No individual review text was retrieved from the checked email. Portal-only reviews remain unverified.",
      "decision": "The decision notification contains no submission-specific rationale.",
      "responses": "No posted author-response text was found in the checked email. This does not establish that no portal response exists."
    },
    "decisionReasons": [],
    "reviews": [],
    "authorResponses": [],
    "authorResponseReceipts": [],
    "notes": [
      "The original notification confirms rejection but provides no rationale or review text. No posted author response was found in connected email. Portal-only material has not been retrieved."
    ]
  },
  "fllmpt-neurips-2026-7": {
    "evidence": "Original decision notification and thread, title/venue searches, and organizer-address correspondence searches in connected email, including Spam and Trash; checked 18 October 2026.",
    "coverage": {
      "reviews": "not_found",
      "decision": "not_found",
      "responses": "not_found"
    },
    "coverageNotes": {
      "reviews": "No individual review text was retrieved from the checked email. Portal-only reviews remain unverified.",
      "decision": "The decision notification contains no submission-specific rationale.",
      "responses": "No posted author-response text was found in the checked email. This does not establish that no portal response exists."
    },
    "decisionReasons": [],
    "reviews": [],
    "authorResponses": [],
    "authorResponseReceipts": [],
    "notes": [
      "The original notification confirms rejection but provides no rationale or review text. No posted author response was found in connected email. Portal-only material has not been retrieved."
    ]
  },
  "fllmpt-neurips-2026-8": {
    "evidence": "Original decision notification and thread, title/venue searches, and organizer-address correspondence searches in connected email, including Spam and Trash; checked 18 October 2026.",
    "coverage": {
      "reviews": "not_found",
      "decision": "not_found",
      "responses": "not_found"
    },
    "coverageNotes": {
      "reviews": "No individual review text was retrieved from the checked email. Portal-only reviews remain unverified.",
      "decision": "The decision notification contains no submission-specific rationale.",
      "responses": "No posted author-response text was found in the checked email. This does not establish that no portal response exists."
    },
    "decisionReasons": [],
    "reviews": [],
    "authorResponses": [],
    "authorResponseReceipts": [],
    "notes": [
      "The original notification confirms rejection but provides no rationale or review text. No posted author response was found in connected email. Portal-only material has not been retrieved."
    ]
  },
  "fllmpt-neurips-2026-9": {
    "evidence": "Original decision notification and thread, title/venue searches, and organizer-address correspondence searches in connected email, including Spam and Trash; checked 18 October 2026.",
    "coverage": {
      "reviews": "not_found",
      "decision": "not_found",
      "responses": "not_found"
    },
    "coverageNotes": {
      "reviews": "No individual review text was retrieved from the checked email. Portal-only reviews remain unverified.",
      "decision": "The decision notification contains no submission-specific rationale.",
      "responses": "No posted author-response text was found in the checked email. This does not establish that no portal response exists."
    },
    "decisionReasons": [],
    "reviews": [],
    "authorResponses": [],
    "authorResponseReceipts": [],
    "notes": [
      "The original notification confirms rejection but provides no rationale or review text. No posted author response was found in connected email. Portal-only material has not been retrieved."
    ]
  },
  "fllmpt-neurips-2026-10": {
    "evidence": "Original decision notification and thread, title/venue searches, and organizer-address correspondence searches in connected email, including Spam and Trash; checked 18 October 2026.",
    "coverage": {
      "reviews": "not_found",
      "decision": "not_found",
      "responses": "not_found"
    },
    "coverageNotes": {
      "reviews": "No individual review text was retrieved from the checked email. Portal-only reviews remain unverified.",
      "decision": "The decision notification contains no submission-specific rationale.",
      "responses": "No posted author-response text was found in the checked email. This does not establish that no portal response exists."
    },
    "decisionReasons": [],
    "reviews": [],
    "authorResponses": [],
    "authorResponseReceipts": [],
    "notes": [
      "The original notification confirms rejection but provides no rationale or review text. No posted author response was found in connected email. Portal-only material has not been retrieved."
    ]
  },
  "fllmpt-neurips-2026-11": {
    "evidence": "Original decision notification and thread, title/venue searches, and organizer-address correspondence searches in connected email, including Spam and Trash; checked 18 October 2026.",
    "coverage": {
      "reviews": "not_found",
      "decision": "not_found",
      "responses": "not_found"
    },
    "coverageNotes": {
      "reviews": "No individual review text was retrieved from the checked email. Portal-only reviews remain unverified.",
      "decision": "The decision notification contains no submission-specific rationale.",
      "responses": "No posted author-response text was found in the checked email. This does not establish that no portal response exists."
    },
    "decisionReasons": [],
    "reviews": [],
    "authorResponses": [],
    "authorResponseReceipts": [],
    "notes": [
      "The original notification confirms rejection but provides no rationale or review text. No posted author response was found in connected email. Portal-only material has not been retrieved."
    ]
  },
  "fllmpt-neurips-2026-12": {
    "evidence": "Original decision notification and thread, title/venue searches, and organizer-address correspondence searches in connected email, including Spam and Trash; checked 18 October 2026.",
    "coverage": {
      "reviews": "not_found",
      "decision": "not_found",
      "responses": "not_found"
    },
    "coverageNotes": {
      "reviews": "No individual review text was retrieved from the checked email. Portal-only reviews remain unverified.",
      "decision": "The decision notification contains no submission-specific rationale.",
      "responses": "No posted author-response text was found in the checked email. This does not establish that no portal response exists."
    },
    "decisionReasons": [],
    "reviews": [],
    "authorResponses": [],
    "authorResponseReceipts": [],
    "notes": [
      "The original notification confirms rejection but provides no rationale or review text. No posted author response was found in connected email. Portal-only material has not been retrieved."
    ]
  },
  "fllmpt-neurips-2026-13": {
    "evidence": "Original decision notification and thread, title/venue searches, and organizer-address correspondence searches in connected email, including Spam and Trash; checked 18 October 2026.",
    "coverage": {
      "reviews": "not_found",
      "decision": "not_found",
      "responses": "not_found"
    },
    "coverageNotes": {
      "reviews": "No individual review text was retrieved from the checked email. Portal-only reviews remain unverified.",
      "decision": "The decision notification contains no submission-specific rationale.",
      "responses": "No posted author-response text was found in the checked email. This does not establish that no portal response exists."
    },
    "decisionReasons": [],
    "reviews": [],
    "authorResponses": [],
    "authorResponseReceipts": [],
    "notes": [
      "The original notification confirms rejection but provides no rationale or review text. No posted author response was found in connected email. Portal-only material has not been retrieved."
    ]
  },
  "judge-2026-1": {
    "evidence": "Original decision notification and thread, title/venue searches, and organizer-address correspondence searches in connected email, including Spam and Trash; checked 18 October 2026.",
    "coverage": {
      "reviews": "not_found",
      "decision": "not_found",
      "responses": "not_found"
    },
    "coverageNotes": {
      "reviews": "No individual review text was retrieved from the checked email. Portal-only reviews remain unverified.",
      "decision": "The decision notification contains no submission-specific rationale.",
      "responses": "No posted author-response text was found in the checked email. This does not establish that no portal response exists."
    },
    "decisionReasons": [],
    "reviews": [],
    "authorResponses": [],
    "authorResponseReceipts": [],
    "notes": [
      "The original notification confirms a program-chair desk rejection but provides no reason. No scientific peer-review text or posted author response was found in connected email. Portal-only material has not been retrieved. Revision receipts were found before the desk rejection; these are submission updates, not responses to reviews and do not establish that reviewer requests were implemented."
    ]
  },
  "judge-2026-2": {
    "evidence": "Original decision notification and thread, title/venue searches, and organizer-address correspondence searches in connected email, including Spam and Trash; checked 18 October 2026.",
    "coverage": {
      "reviews": "not_found",
      "decision": "not_found",
      "responses": "not_found"
    },
    "coverageNotes": {
      "reviews": "No individual review text was retrieved from the checked email. Portal-only reviews remain unverified.",
      "decision": "The decision notification contains no submission-specific rationale.",
      "responses": "No posted author-response text was found in the checked email. This does not establish that no portal response exists."
    },
    "decisionReasons": [],
    "reviews": [],
    "authorResponses": [],
    "authorResponseReceipts": [],
    "notes": [
      "The original notification confirms a program-chair desk rejection but provides no reason. No scientific peer-review text or posted author response was found in connected email. Portal-only material has not been retrieved. Revision receipts were found before the desk rejection; these are submission updates, not responses to reviews and do not establish that reviewer requests were implemented."
    ]
  },
  "judge-2026-3": {
    "evidence": "Original decision notification and thread, title/venue searches, and organizer-address correspondence searches in connected email, including Spam and Trash; checked 18 October 2026.",
    "coverage": {
      "reviews": "not_found",
      "decision": "not_found",
      "responses": "not_found"
    },
    "coverageNotes": {
      "reviews": "No individual review text was retrieved from the checked email. Portal-only reviews remain unverified.",
      "decision": "The decision notification contains no submission-specific rationale.",
      "responses": "No posted author-response text was found in the checked email. This does not establish that no portal response exists."
    },
    "decisionReasons": [],
    "reviews": [],
    "authorResponses": [],
    "authorResponseReceipts": [],
    "notes": [
      "The original notification confirms a program-chair desk rejection but provides no reason. No scientific peer-review text or posted author response was found in connected email. Portal-only material has not been retrieved. Revision receipts were found before the desk rejection; these are submission updates, not responses to reviews and do not establish that reviewer requests were implemented."
    ]
  },
  "neurips-2026-automlr-workshop-2": {
    "evidence": "Original decision notification and thread, title/venue searches, and organizer-address correspondence searches in connected email, including Spam and Trash; checked 18 October 2026.",
    "coverage": {
      "reviews": "not_found",
      "decision": "not_found",
      "responses": "not_found"
    },
    "coverageNotes": {
      "reviews": "No individual review text was retrieved from the checked email. Portal-only reviews remain unverified.",
      "decision": "The decision notification contains no submission-specific rationale.",
      "responses": "No posted author-response text was found in the checked email. This does not establish that no portal response exists."
    },
    "decisionReasons": [],
    "reviews": [],
    "authorResponses": [],
    "authorResponseReceipts": [],
    "notes": [
      "The original notification confirms a program-chair desk rejection but provides no reason. No scientific peer-review text or posted author response was found in connected email. Portal-only material has not been retrieved."
    ]
  },
  "neurips-2026-workshop-math-ai-3": {
    "evidence": "Original decision notification and thread, title/venue searches, and organizer-address correspondence searches in connected email, including Spam and Trash; checked 18 October 2026.",
    "coverage": {
      "reviews": "not_found",
      "decision": "not_found",
      "responses": "not_found"
    },
    "coverageNotes": {
      "reviews": "No individual review text was retrieved from the checked email. Portal-only reviews remain unverified.",
      "decision": "The decision notification contains no submission-specific rationale.",
      "responses": "No posted author-response text was found in the checked email. This does not establish that no portal response exists."
    },
    "decisionReasons": [],
    "reviews": [],
    "authorResponses": [],
    "authorResponseReceipts": [],
    "notes": [
      "The original notification confirms a program-chair desk rejection but provides no reason. No scientific peer-review text or posted author response was found in connected email. Portal-only material has not been retrieved."
    ]
  },
  "neurips-2026-workshop-math-ai-4": {
    "evidence": "Original decision notification and thread, title/venue searches, and organizer-address correspondence searches in connected email, including Spam and Trash; checked 18 October 2026.",
    "coverage": {
      "reviews": "not_found",
      "decision": "not_found",
      "responses": "not_found"
    },
    "coverageNotes": {
      "reviews": "No individual review text was retrieved from the checked email. Portal-only reviews remain unverified.",
      "decision": "The decision notification contains no submission-specific rationale.",
      "responses": "No posted author-response text was found in the checked email. This does not establish that no portal response exists."
    },
    "decisionReasons": [],
    "reviews": [],
    "authorResponses": [],
    "authorResponseReceipts": [],
    "notes": [
      "The original notification confirms a program-chair desk rejection but provides no reason. No scientific peer-review text or posted author response was found in connected email. Portal-only material has not been retrieved."
    ]
  },
  "neurips-2026-workshop-math-ai-5": {
    "evidence": "Original decision notification and thread, title/venue searches, and organizer-address correspondence searches in connected email, including Spam and Trash; checked 18 October 2026.",
    "coverage": {
      "reviews": "not_found",
      "decision": "not_found",
      "responses": "not_found"
    },
    "coverageNotes": {
      "reviews": "No individual review text was retrieved from the checked email. Portal-only reviews remain unverified.",
      "decision": "The decision notification contains no submission-specific rationale.",
      "responses": "No posted author-response text was found in the checked email. This does not establish that no portal response exists."
    },
    "decisionReasons": [],
    "reviews": [],
    "authorResponses": [],
    "authorResponseReceipts": [],
    "notes": [
      "The original notification confirms a program-chair desk rejection but provides no reason. No scientific peer-review text or posted author response was found in connected email. Portal-only material has not been retrieved."
    ]
  },
  "neurips-2026-workshop-math-ai-6": {
    "evidence": "Original decision notification and thread, title/venue searches, and organizer-address correspondence searches in connected email, including Spam and Trash; checked 18 October 2026.",
    "coverage": {
      "reviews": "not_found",
      "decision": "not_found",
      "responses": "not_found"
    },
    "coverageNotes": {
      "reviews": "No individual review text was retrieved from the checked email. Portal-only reviews remain unverified.",
      "decision": "The decision notification contains no submission-specific rationale.",
      "responses": "No posted author-response text was found in the checked email. This does not establish that no portal response exists."
    },
    "decisionReasons": [],
    "reviews": [],
    "authorResponses": [],
    "authorResponseReceipts": [],
    "notes": [
      "The original notification confirms a program-chair desk rejection but provides no reason. No scientific peer-review text or posted author response was found in connected email. Portal-only material has not been retrieved."
    ]
  },
  "pre-to-post-neurips-2026-1": {
    "evidence": "Original decision notification and thread, title/venue searches, and organizer-address correspondence searches in connected email, including Spam and Trash; checked 18 October 2026.",
    "coverage": {
      "reviews": "not_found",
      "decision": "not_found",
      "responses": "not_found"
    },
    "coverageNotes": {
      "reviews": "No individual review text was retrieved from the checked email. Portal-only reviews remain unverified.",
      "decision": "The decision notification contains no submission-specific rationale.",
      "responses": "No posted author-response text was found in the checked email. This does not establish that no portal response exists."
    },
    "decisionReasons": [],
    "reviews": [],
    "authorResponses": [],
    "authorResponseReceipts": [],
    "notes": [
      "The original notification confirms a program-chair desk rejection but provides no reason. No scientific peer-review text or posted author response was found in connected email. Portal-only material has not been retrieved."
    ]
  },
  "pre-to-post-neurips-2026-2": {
    "evidence": "Original decision notification and thread, title/venue searches, and organizer-address correspondence searches in connected email, including Spam and Trash; checked 18 October 2026.",
    "coverage": {
      "reviews": "not_found",
      "decision": "not_found",
      "responses": "not_found"
    },
    "coverageNotes": {
      "reviews": "No individual review text was retrieved from the checked email. Portal-only reviews remain unverified.",
      "decision": "The decision notification contains no submission-specific rationale.",
      "responses": "No posted author-response text was found in the checked email. This does not establish that no portal response exists."
    },
    "decisionReasons": [],
    "reviews": [],
    "authorResponses": [],
    "authorResponseReceipts": [],
    "notes": [
      "The original notification confirms a program-chair desk rejection but provides no reason. No scientific peer-review text or posted author response was found in connected email. Portal-only material has not been retrieved."
    ]
  },
  "pre-to-post-neurips-2026-3": {
    "evidence": "Original decision notification and thread, title/venue searches, and organizer-address correspondence searches in connected email, including Spam and Trash; checked 18 October 2026.",
    "coverage": {
      "reviews": "not_found",
      "decision": "not_found",
      "responses": "not_found"
    },
    "coverageNotes": {
      "reviews": "No individual review text was retrieved from the checked email. Portal-only reviews remain unverified.",
      "decision": "The decision notification contains no submission-specific rationale.",
      "responses": "No posted author-response text was found in the checked email. This does not establish that no portal response exists."
    },
    "decisionReasons": [],
    "reviews": [],
    "authorResponses": [],
    "authorResponseReceipts": [],
    "notes": [
      "The original notification confirms a program-chair desk rejection but provides no reason. No scientific peer-review text or posted author response was found in connected email. Portal-only material has not been retrieved."
    ]
  },
  "pre-to-post-neurips-2026-4": {
    "evidence": "Original decision notification and thread, title/venue searches, and organizer-address correspondence searches in connected email, including Spam and Trash; checked 18 October 2026.",
    "coverage": {
      "reviews": "not_found",
      "decision": "not_found",
      "responses": "not_found"
    },
    "coverageNotes": {
      "reviews": "No individual review text was retrieved from the checked email. Portal-only reviews remain unverified.",
      "decision": "The decision notification contains no submission-specific rationale.",
      "responses": "No posted author-response text was found in the checked email. This does not establish that no portal response exists."
    },
    "decisionReasons": [],
    "reviews": [],
    "authorResponses": [],
    "authorResponseReceipts": [],
    "notes": [
      "The original notification confirms a program-chair desk rejection but provides no reason. No scientific peer-review text or posted author response was found in connected email. Portal-only material has not been retrieved."
    ]
  },
  "pre-to-post-neurips-2026-5": {
    "evidence": "Original decision notification and thread, title/venue searches, and organizer-address correspondence searches in connected email, including Spam and Trash; checked 18 October 2026.",
    "coverage": {
      "reviews": "not_found",
      "decision": "not_found",
      "responses": "not_found"
    },
    "coverageNotes": {
      "reviews": "No individual review text was retrieved from the checked email. Portal-only reviews remain unverified.",
      "decision": "The decision notification contains no submission-specific rationale.",
      "responses": "No posted author-response text was found in the checked email. This does not establish that no portal response exists."
    },
    "decisionReasons": [],
    "reviews": [],
    "authorResponses": [],
    "authorResponseReceipts": [],
    "notes": [
      "The original notification confirms a program-chair desk rejection but provides no reason. No scientific peer-review text or posted author response was found in connected email. Portal-only material has not been retrieved."
    ]
  },
  "pre-to-post-neurips-2026-6": {
    "evidence": "Original decision notification and thread, title/venue searches, and organizer-address correspondence searches in connected email, including Spam and Trash; checked 18 October 2026.",
    "coverage": {
      "reviews": "not_found",
      "decision": "not_found",
      "responses": "not_found"
    },
    "coverageNotes": {
      "reviews": "No individual review text was retrieved from the checked email. Portal-only reviews remain unverified.",
      "decision": "The decision notification contains no submission-specific rationale.",
      "responses": "No posted author-response text was found in the checked email. This does not establish that no portal response exists."
    },
    "decisionReasons": [],
    "reviews": [],
    "authorResponses": [],
    "authorResponseReceipts": [],
    "notes": [
      "The original notification confirms a program-chair desk rejection but provides no reason. No scientific peer-review text or posted author response was found in connected email. Portal-only material has not been retrieved."
    ]
  },
  "pre-to-post-neurips-2026-7": {
    "evidence": "Original decision notification and thread, title/venue searches, and organizer-address correspondence searches in connected email, including Spam and Trash; checked 18 October 2026.",
    "coverage": {
      "reviews": "not_found",
      "decision": "not_found",
      "responses": "not_found"
    },
    "coverageNotes": {
      "reviews": "No individual review text was retrieved from the checked email. Portal-only reviews remain unverified.",
      "decision": "The decision notification contains no submission-specific rationale.",
      "responses": "No posted author-response text was found in the checked email. This does not establish that no portal response exists."
    },
    "decisionReasons": [],
    "reviews": [],
    "authorResponses": [],
    "authorResponseReceipts": [],
    "notes": [
      "The original notification confirms a program-chair desk rejection but provides no reason. No scientific peer-review text or posted author response was found in connected email. Portal-only material has not been retrieved."
    ]
  },
  "pre-to-post-neurips-2026-8": {
    "evidence": "Original decision notification and thread, title/venue searches, and organizer-address correspondence searches in connected email, including Spam and Trash; checked 18 October 2026.",
    "coverage": {
      "reviews": "not_found",
      "decision": "not_found",
      "responses": "not_found"
    },
    "coverageNotes": {
      "reviews": "No individual review text was retrieved from the checked email. Portal-only reviews remain unverified.",
      "decision": "The decision notification contains no submission-specific rationale.",
      "responses": "No posted author-response text was found in the checked email. This does not establish that no portal response exists."
    },
    "decisionReasons": [],
    "reviews": [],
    "authorResponses": [],
    "authorResponseReceipts": [],
    "notes": [
      "The original notification confirms a program-chair desk rejection but provides no reason. No scientific peer-review text or posted author response was found in connected email. Portal-only material has not been retrieved."
    ]
  },
  "pre-to-post-neurips-2026-9": {
    "evidence": "Original decision notification and thread, title/venue searches, and organizer-address correspondence searches in connected email, including Spam and Trash; checked 18 October 2026.",
    "coverage": {
      "reviews": "not_found",
      "decision": "not_found",
      "responses": "not_found"
    },
    "coverageNotes": {
      "reviews": "No individual review text was retrieved from the checked email. Portal-only reviews remain unverified.",
      "decision": "The decision notification contains no submission-specific rationale.",
      "responses": "No posted author-response text was found in the checked email. This does not establish that no portal response exists."
    },
    "decisionReasons": [],
    "reviews": [],
    "authorResponses": [],
    "authorResponseReceipts": [],
    "notes": [
      "The original notification confirms a program-chair desk rejection but provides no reason. No scientific peer-review text or posted author response was found in connected email. Portal-only material has not been retrieved."
    ]
  },
  "pre-to-post-neurips-2026-10": {
    "evidence": "Original decision notification and thread, title/venue searches, and organizer-address correspondence searches in connected email, including Spam and Trash; checked 18 October 2026.",
    "coverage": {
      "reviews": "not_found",
      "decision": "not_found",
      "responses": "not_found"
    },
    "coverageNotes": {
      "reviews": "No individual review text was retrieved from the checked email. Portal-only reviews remain unverified.",
      "decision": "The decision notification contains no submission-specific rationale.",
      "responses": "No posted author-response text was found in the checked email. This does not establish that no portal response exists."
    },
    "decisionReasons": [],
    "reviews": [],
    "authorResponses": [],
    "authorResponseReceipts": [],
    "notes": [
      "The original notification confirms a program-chair desk rejection but provides no reason. No scientific peer-review text or posted author response was found in connected email. Portal-only material has not been retrieved."
    ]
  },
  "pre-to-post-neurips-2026-11": {
    "evidence": "Original decision notification and thread, title/venue searches, and organizer-address correspondence searches in connected email, including Spam and Trash; checked 18 October 2026.",
    "coverage": {
      "reviews": "not_found",
      "decision": "not_found",
      "responses": "not_found"
    },
    "coverageNotes": {
      "reviews": "No individual review text was retrieved from the checked email. Portal-only reviews remain unverified.",
      "decision": "The decision notification contains no submission-specific rationale.",
      "responses": "No posted author-response text was found in the checked email. This does not establish that no portal response exists."
    },
    "decisionReasons": [],
    "reviews": [],
    "authorResponses": [],
    "authorResponseReceipts": [],
    "notes": [
      "The original notification confirms a program-chair desk rejection but provides no reason. No scientific peer-review text or posted author response was found in connected email. Portal-only material has not been retrieved."
    ]
  },
  "neurips-2026-proposal-488": {
    "evidence": "Original proposal decision notification and thread, exact-title and portal-reference email searches from June 2026 onward, and organizer-address correspondence searches, including Spam and Trash; checked 18 October 2026.",
    "coverage": {
      "reviews": "not_found",
      "decision": "partial",
      "responses": "not_found"
    },
    "coverageNotes": {
      "reviews": "No individual review text was retrieved from the checked email. Portal-only reviews remain unverified.",
      "decision": "Only general selection context is available; it does not establish a specific reason for this submission.",
      "responses": "No posted author-response text was found in the checked email. This does not establish that no portal response exists."
    },
    "decisionReasons": [
      {
        "label": "General organizer selection context (not a submission-specific rationale)",
        "points": [
          "The rejection notice gives program-wide selection context: limited workshop capacity, overlap among proposed themes, a balanced program, diversity of topics and speakers, and novelty relative to prior workshops.",
          "The notice says most proposals received at least two reviews and an area-chair metareview. These are general process statements, not verified reviewer counts for this proposal.",
          "The notice directs the author to a proposal-specific metareview in the submission portal, but the email does not contain it."
        ],
        "date": "2026-07-12"
      }
    ],
    "reviews": [],
    "authorResponses": [],
    "authorResponseReceipts": [],
    "notes": [
      "Rejection is verified. The available email contains general selection context, not a proposal-specific rationale. The referenced metareview and individual reviews were not present in connected email. No posted rebuttal or author response was found; speaker-invitation correspondence and submission-update receipts are not review responses."
    ]
  },
  "ai4metascience-2026-6": {
    "evidence": "Original submission receipt and decision-availability notification; exact-title and venue email searches, including Spam and Trash; checked 18 October 2026.",
    "coverage": {
      "reviews": "not_found",
      "decision": "partial",
      "responses": "not_found"
    },
    "coverageNotes": {
      "reviews": "No individual review text was retrieved from the checked email. Portal-only reviews remain unverified.",
      "decision": "Only general selection context is available; it does not establish a specific reason for this submission.",
      "responses": "No posted author-response text was found in the checked email. This does not establish that no portal response exists."
    },
    "decisionReasons": [
      {
        "label": "General organizer selection context (not a submission-specific rationale)",
        "points": [
          "The decision-availability notice describes a general holistic assessment using reviews, reviewer confidence, workshop relevance, contribution clarity, and the acceptance threshold.",
          "The notice cautions that favorable reviews did not guarantee acceptance and that some position papers were judged insufficiently developed in argument, evidence, positioning, or contribution. It does not say whether that assessment applies to this submission."
        ],
        "date": "2026-09-30"
      }
    ],
    "reviews": [],
    "authorResponses": [],
    "authorResponseReceipts": [],
    "notes": [
      "A submission receipt and decision-availability notice are verified. The notice does not state acceptance or rejection and contains no paper-specific rationale, individual reviews, or author response. Portal-only material has not been retrieved."
    ]
  }
};

export const COVERAGE_LABELS = {
  available: "Available in checked sources",
  partial: "Partial coverage",
  not_found: "Not found in checked sources",
  unavailable: "Source unavailable",
  not_verified: "Not yet verified",
};

const unverified = {
  evidence: "Feedback sources have not yet been verified for this submission.",
  coverage: { reviews: "not_verified", decision: "not_verified", responses: "not_verified" },
  coverageNotes: {},
  decisionReasons: [], reviews: [], authorResponses: [], authorResponseReceipts: [], notes: [],
};

export function feedbackFor(id, data = SUBMISSION_FEEDBACK) {
  return Object.hasOwn(data, id) ? data[id] : unverified;
}

export function feedbackCounts(records, data = SUBMISSION_FEEDBACK) {
  const feedback = records.map(({ id }) => feedbackFor(id, data));
  return {
    total: records.length,
    withReviews: feedback.filter(f => f.reviews.length > 0).length,
    withDecisionReasons: feedback.filter(f => f.decisionReasons.length > 0).length,
    withResponses: feedback.filter(f => f.authorResponses.length > 0).length,
    withoutReviews: feedback.filter(f => f.reviews.length === 0).length,
  };
}

export function feedbackSummary(feedback) {
  const missing = (key, label) => `${label} ${feedback.coverage[key] === "not_verified" ? "not yet verified" : "not retrieved"}`;
  const partial = key => feedback.coverage[key] === "partial" ? " (partial coverage)" : "";
  const hasFollowups = feedback.reviews.some(entry => entry.kind && entry.kind !== "review");
  return [
    feedback.reviews.length ? `${feedback.reviews.length} ${hasFollowups ? "reviewer feedback entries" : `review ${feedback.reviews.length === 1 ? "report" : "reports"}`}${partial("reviews")}` : missing("reviews", "Review text"),
    feedback.decisionReasons.length ? `Committee / decision feedback available${partial("decision")}` : missing("decision", "Decision reasons"),
    feedback.authorResponses.length ? `${feedback.authorResponses.length} author ${feedback.authorResponses.length === 1 ? "response" : "responses"}${partial("responses")}` : missing("responses", "Author response"),
  ].join(" · ");
}

function searchableFeedback(feedback) {
  return [feedback.evidence, ...feedback.notes, ...Object.values(feedback.coverageNotes),
    ...Object.values(feedback.authorResponseWarning || {}),
    ...feedback.decisionReasons.flatMap(entry => [entry.label, ...entry.points]),
    ...feedback.reviews.flatMap(entry => [entry.label, ...entry.strengths, ...entry.concerns, ...entry.requests,
      ...(entry.scores || []).flatMap(score => [score.label, score.value, score.scale, score.note])]),
    ...feedback.authorResponses.flatMap(entry => [entry.label, ...entry.points]),
    ...(feedback.authorResponseReceipts || []).flatMap(entry => [entry.label, entry.note]),
  ].join(" ");
}

export function filterSubmissions(records, { query = "", coverage = "all" } = {}, data = SUBMISSION_FEEDBACK) {
  const terms = query.trim().toLocaleLowerCase("en").split(/\s+/).filter(Boolean);
  return records.filter(record => {
    const feedback = feedbackFor(record.id, data);
    if (coverage === "with_reviews" && !feedback.reviews.length) return false;
    if (coverage === "with_reasons" && !feedback.decisionReasons.length) return false;
    if (coverage === "with_responses" && !feedback.authorResponses.length) return false;
    if (coverage === "without_reviews" && feedback.reviews.length) return false;
    const text = [record.venue, record.kind, record.number, record.title, record.decision, searchableFeedback(feedback)].join(" ").toLocaleLowerCase("en");
    return terms.every(term => text.includes(term));
  });
}
