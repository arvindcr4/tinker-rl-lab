// Curated administrative metadata, verified from organizer correspondence.
// Keep this separate from data.json and the experimental analysis. Never add
// email bodies, addresses, message IDs, reviewer details, or private portal URLs.
// This is a dated snapshot; acceptance does not establish final publication.
export const SUBMISSIONS_VERIFIED_ON = "2026-10-18";

export const PRIMARY_SUBMISSIONS = [
  {
    id: "icicc-2027-243",
    venue: "ICICC 2027",
    kind: "Conference paper",
    number: "243",
    title: "Receipt-Bound Held-Out Evaluation of a GRPO-Trained Language Model Across Fourteen Agentic and Reasoning Suites",
    submittedOn: "2026-09-24",
    decision: "Accepted",
    decisionOn: "2026-10-11",
    evidence: "Organizer acceptance notification",
    note: "Accepted by organizer notification. Camera-ready submission, registration and final publication are unverified.",
    deadline: { label: "Organizer-stated registration deadline", date: "2026-10-30", caveat: "Timezone unspecified; completion unverified" },
  },
  {
    id: "neurips-2026-main-36320",
    venue: "NeurIPS 2026",
    kind: "Main-track paper",
    number: "36320",
    title: "Reward Contrast, Not Algorithm Labels: A Diagnostic Audit of Critic-Free Group-Relative RL for LLMs",
    submittedOn: "2026-05-05",
    decision: "Rejected",
    decisionOn: "2026-09-24",
    evidence: "Main-track decision notification",
    note: "The main-track decision is final in the verified correspondence. Public availability and the selected public-release option are unverified.",
  },
];

export const WORKSHOP_PROPOSAL = {
  id: "neurips-2026-proposal-488",
  venue: "NeurIPS 2026",
  kind: "Workshop organization proposal",
  number: "488",
  title: "Diagnosing RL Post-Training: Measurement, Reproducibility, and Failure Modes in RL Fine-Tuning of LLMs",
  submittedOn: "2026-06-07",
  decision: "Rejected",
  decisionOn: "2026-07-12",
  evidence: "Workshop proposal decision notification",
  note: "This decision concerns a proposal to organize a workshop, not a research-paper submission.",
};

export const WORKSHOP_PAPERS = [
  {
    "id": "fllmpt-neurips-2026-3",
    "venue": "FLLMPT NeurIPS 2026",
    "kind": "Workshop paper",
    "number": "3",
    "title": "Does Reinforcement Finetuning Actually Pay Off? A Systematic Evidence Map of LLM RFT Claims at ICLR/ICML/ACL 2026",
    "submittedOn": "2026-08-02",
    "decision": "Rejected",
    "decisionOn": "2026-09-30"
  },
  {
    "id": "fllmpt-neurips-2026-4",
    "venue": "FLLMPT NeurIPS 2026",
    "kind": "Workshop paper",
    "number": "4",
    "title": "Group Relative Policy Optimization Under the Microscope: When GRPO Beats PPO for LLM Reasoning",
    "submittedOn": "2026-08-02",
    "decision": "Rejected",
    "decisionOn": "2026-09-30"
  },
  {
    "id": "fllmpt-neurips-2026-5",
    "venue": "FLLMPT NeurIPS 2026",
    "kind": "Workshop paper",
    "number": "5",
    "title": "Verifiable Rewards vs Preferences: A Head-to-Head Agenda for Reasoning Post-Training",
    "submittedOn": "2026-08-02",
    "decision": "Rejected",
    "decisionOn": "2026-09-30"
  },
  {
    "id": "fllmpt-neurips-2026-6",
    "venue": "FLLMPT NeurIPS 2026",
    "kind": "Workshop paper",
    "number": "6",
    "title": "DPO After the Hype: Robustness, Geometry, and Failure Modes of Direct Preference Optimization",
    "submittedOn": "2026-08-02",
    "decision": "Rejected",
    "decisionOn": "2026-09-30"
  },
  {
    "id": "fllmpt-neurips-2026-7",
    "venue": "FLLMPT NeurIPS 2026",
    "kind": "Workshop paper",
    "number": "7",
    "title": "Process Beats Outcome? Credit Assignment with Step-Level Rewards in LLM Reasoning",
    "submittedOn": "2026-08-02",
    "decision": "Rejected",
    "decisionOn": "2026-09-30"
  },
  {
    "id": "fllmpt-neurips-2026-8",
    "venue": "FLLMPT NeurIPS 2026",
    "kind": "Workshop paper",
    "number": "8",
    "title": "SFT Is Not Solved: A Critical Review of Instruction-Tuning Recipes and Matched-Compute Baselines",
    "submittedOn": "2026-08-02",
    "decision": "Rejected",
    "decisionOn": "2026-09-30"
  },
  {
    "id": "fllmpt-neurips-2026-9",
    "venue": "FLLMPT NeurIPS 2026",
    "kind": "Workshop paper",
    "number": "9",
    "title": "Reward Models Are the Bottleneck: Ranking Quality, Overoptimization, and Downstream Policy Impact",
    "submittedOn": "2026-08-02",
    "decision": "Rejected",
    "decisionOn": "2026-09-30"
  },
  {
    "id": "fllmpt-neurips-2026-10",
    "venue": "FLLMPT NeurIPS 2026",
    "kind": "Workshop paper",
    "number": "10",
    "title": "RLHF Revisited: Isolating Human Feedback Signal from Optimization Theater",
    "submittedOn": "2026-08-02",
    "decision": "Rejected",
    "decisionOn": "2026-09-30"
  },
  {
    "id": "fllmpt-neurips-2026-11",
    "venue": "FLLMPT NeurIPS 2026",
    "kind": "Workshop paper",
    "number": "11",
    "title": "Beyond Vanilla DPO: A Comparative Field Guide to Preference Objectives for LLM Alignment",
    "submittedOn": "2026-08-02",
    "decision": "Rejected",
    "decisionOn": "2026-09-30"
  },
  {
    "id": "fllmpt-neurips-2026-12",
    "venue": "FLLMPT NeurIPS 2026",
    "kind": "Workshop paper",
    "number": "12",
    "title": "Beyond Named Algorithms: What Else Is Moving the Needle in LLM Post-Training?",
    "submittedOn": "2026-08-02",
    "decision": "Rejected",
    "decisionOn": "2026-09-30"
  },
  {
    "id": "fllmpt-neurips-2026-13",
    "venue": "FLLMPT NeurIPS 2026",
    "kind": "Workshop paper",
    "number": "13",
    "title": "Ten Causal Hypotheses from Indexing 1526 LLM Post-Training Papers",
    "submittedOn": "2026-08-02",
    "decision": "Rejected",
    "decisionOn": "2026-09-30"
  },
  {
    "id": "judge-2026-1",
    "venue": "JUDGe 2026",
    "kind": "Workshop paper",
    "number": "1",
    "title": "Reward Models Are the Bottleneck: Ranking Quality, Overoptimization, and Downstream Policy Impact",
    "submittedOn": "2026-08-02",
    "decision": "Desk-rejected",
    "decisionOn": "2026-09-08"
  },
  {
    "id": "judge-2026-2",
    "venue": "JUDGe 2026",
    "kind": "Workshop paper",
    "number": "2",
    "title": "Beyond Vanilla DPO: A Comparative Field Guide to Preference Objectives for LLM Alignment",
    "submittedOn": "2026-08-02",
    "decision": "Desk-rejected",
    "decisionOn": "2026-09-08"
  },
  {
    "id": "judge-2026-3",
    "venue": "JUDGe 2026",
    "kind": "Workshop paper",
    "number": "3",
    "title": "DPO After the Hype: Robustness, Geometry, and Failure Modes of Direct Preference Optimization",
    "submittedOn": "2026-08-02",
    "decision": "Desk-rejected",
    "decisionOn": "2026-09-08"
  },
  {
    "id": "neurips-2026-automlr-workshop-2",
    "venue": "NeurIPS 2026 AutoMLR Workshop",
    "kind": "Workshop paper",
    "number": "2",
    "title": "Ten Causal Hypotheses from Indexing 1526 LLM Post-Training Papers",
    "submittedOn": "2026-08-02",
    "decision": "Desk-rejected",
    "decisionOn": "2026-09-06"
  },
  {
    "id": "neurips-2026-workshop-math-ai-3",
    "venue": "NeurIPS 2026 Workshop MATH-AI",
    "kind": "Workshop paper",
    "number": "3",
    "title": "Group Relative Policy Optimization Under the Microscope: When GRPO Beats PPO for LLM Reasoning",
    "submittedOn": "2026-08-02",
    "decision": "Desk-rejected",
    "decisionOn": "2026-09-07"
  },
  {
    "id": "neurips-2026-workshop-math-ai-4",
    "venue": "NeurIPS 2026 Workshop MATH-AI",
    "kind": "Workshop paper",
    "number": "4",
    "title": "Verifiable Rewards vs Preferences: A Head-to-Head Agenda for Reasoning Post-Training",
    "submittedOn": "2026-08-02",
    "decision": "Desk-rejected",
    "decisionOn": "2026-09-07"
  },
  {
    "id": "neurips-2026-workshop-math-ai-5",
    "venue": "NeurIPS 2026 Workshop MATH-AI",
    "kind": "Workshop paper",
    "number": "5",
    "title": "Process Beats Outcome? Credit Assignment with Step-Level Rewards in LLM Reasoning",
    "submittedOn": "2026-08-02",
    "decision": "Desk-rejected",
    "decisionOn": "2026-09-07"
  },
  {
    "id": "neurips-2026-workshop-math-ai-6",
    "venue": "NeurIPS 2026 Workshop MATH-AI",
    "kind": "Workshop paper",
    "number": "6",
    "title": "Does Reinforcement Finetuning Actually Pay Off? A Systematic Evidence Map of LLM RFT Claims at ICLR/ICML/ACL 2026",
    "submittedOn": "2026-08-02",
    "decision": "Desk-rejected",
    "decisionOn": "2026-09-07"
  },
  {
    "id": "pre-to-post-neurips-2026-1",
    "venue": "Pre-to-Post NeurIPS 2026",
    "kind": "Workshop paper",
    "number": "1",
    "title": "Does Reinforcement Finetuning Actually Pay Off? A Systematic Evidence Map of LLM RFT Claims at ICLR/ICML/ACL 2026",
    "submittedOn": "2026-08-02",
    "decision": "Desk-rejected",
    "decisionOn": "2026-09-07"
  },
  {
    "id": "pre-to-post-neurips-2026-2",
    "venue": "Pre-to-Post NeurIPS 2026",
    "kind": "Workshop paper",
    "number": "2",
    "title": "Group Relative Policy Optimization Under the Microscope: When GRPO Beats PPO for LLM Reasoning",
    "submittedOn": "2026-08-02",
    "decision": "Desk-rejected",
    "decisionOn": "2026-09-07"
  },
  {
    "id": "pre-to-post-neurips-2026-3",
    "venue": "Pre-to-Post NeurIPS 2026",
    "kind": "Workshop paper",
    "number": "3",
    "title": "SFT Is Not Solved: A Critical Review of Instruction-Tuning Recipes and Matched-Compute Baselines",
    "submittedOn": "2026-08-02",
    "decision": "Desk-rejected",
    "decisionOn": "2026-09-07"
  },
  {
    "id": "pre-to-post-neurips-2026-4",
    "venue": "Pre-to-Post NeurIPS 2026",
    "kind": "Workshop paper",
    "number": "4",
    "title": "Verifiable Rewards vs Preferences: A Head-to-Head Agenda for Reasoning Post-Training",
    "submittedOn": "2026-08-02",
    "decision": "Desk-rejected",
    "decisionOn": "2026-09-07"
  },
  {
    "id": "pre-to-post-neurips-2026-5",
    "venue": "Pre-to-Post NeurIPS 2026",
    "kind": "Workshop paper",
    "number": "5",
    "title": "DPO After the Hype: Robustness, Geometry, and Failure Modes of Direct Preference Optimization",
    "submittedOn": "2026-08-02",
    "decision": "Desk-rejected",
    "decisionOn": "2026-09-07"
  },
  {
    "id": "pre-to-post-neurips-2026-6",
    "venue": "Pre-to-Post NeurIPS 2026",
    "kind": "Workshop paper",
    "number": "6",
    "title": "Reward Models Are the Bottleneck: Ranking Quality, Overoptimization, and Downstream Policy Impact",
    "submittedOn": "2026-08-02",
    "decision": "Desk-rejected",
    "decisionOn": "2026-09-07"
  },
  {
    "id": "pre-to-post-neurips-2026-7",
    "venue": "Pre-to-Post NeurIPS 2026",
    "kind": "Workshop paper",
    "number": "7",
    "title": "RLHF Revisited: Isolating Human Feedback Signal from Optimization Theater",
    "submittedOn": "2026-08-02",
    "decision": "Desk-rejected",
    "decisionOn": "2026-09-07"
  },
  {
    "id": "pre-to-post-neurips-2026-8",
    "venue": "Pre-to-Post NeurIPS 2026",
    "kind": "Workshop paper",
    "number": "8",
    "title": "Beyond Vanilla DPO: A Comparative Field Guide to Preference Objectives for LLM Alignment",
    "submittedOn": "2026-08-02",
    "decision": "Desk-rejected",
    "decisionOn": "2026-09-07"
  },
  {
    "id": "pre-to-post-neurips-2026-9",
    "venue": "Pre-to-Post NeurIPS 2026",
    "kind": "Workshop paper",
    "number": "9",
    "title": "Process Beats Outcome? Credit Assignment with Step-Level Rewards in LLM Reasoning",
    "submittedOn": "2026-08-02",
    "decision": "Desk-rejected",
    "decisionOn": "2026-09-07"
  },
  {
    "id": "pre-to-post-neurips-2026-10",
    "venue": "Pre-to-Post NeurIPS 2026",
    "kind": "Workshop paper",
    "number": "10",
    "title": "Beyond Named Algorithms: What Else Is Moving the Needle in LLM Post-Training?",
    "submittedOn": "2026-08-02",
    "decision": "Desk-rejected",
    "decisionOn": "2026-09-07"
  },
  {
    "id": "pre-to-post-neurips-2026-11",
    "venue": "Pre-to-Post NeurIPS 2026",
    "kind": "Workshop paper",
    "number": "11",
    "title": "Ten Causal Hypotheses from Indexing 1526 LLM Post-Training Papers",
    "submittedOn": "2026-08-02",
    "decision": "Desk-rejected",
    "decisionOn": "2026-09-07"
  },
  {
    id: "ai4metascience-2026-6",
    venue: "AI4MetaScience",
    kind: "Workshop paper",
    number: "6",
    title: "Ten Causal Hypotheses from Indexing 1526 LLM Post-Training Papers",
    submittedOn: "2026-08-02",
    decision: "Decision available; outcome unverified",
    decisionOn: "2026-09-30",
    decisionDateLabel: "Decision-availability notice",
    note: "The notification says decisions and reviews are available, but does not contain an acceptance or rejection outcome. The outcome remains unverified.",
  }
];

export function workshopGroups(records = WORKSHOP_PAPERS) {
  const groups = new Map();
  for (const record of records) {
    if (!groups.has(record.venue)) groups.set(record.venue, []);
    groups.get(record.venue).push(record);
  }
  return [...groups].map(([venue, submissions]) => ({ venue, submissions }))
    .sort((a, b) => b.submissions.length - a.submissions.length || a.venue.localeCompare(b.venue));
}

export function decisionSummary(records) {
  const counts = new Map();
  for (const { decision } of records) counts.set(decision, (counts.get(decision) || 0) + 1);
  return [...counts].map(([decision, count]) => `${count} ${decision.toLowerCase()}`).join(" · ");
}


// Dates are recorded in UTC; avoid shifting a notification to the previous day
// when the offline page is opened in a different local timezone.
export function submissionDate(value) {
  return new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" })
    .format(new Date(`${value}T00:00:00Z`));
}
