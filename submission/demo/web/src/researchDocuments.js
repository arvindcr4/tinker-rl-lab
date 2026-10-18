// Curated public-document index, separate from experimental data and venue decisions.
// Links are pinned to the reviewed public tree, never to local drafts or moving HEAD.
export const DOCUMENTS_VERIFIED_ON = "2026-10-18";
export const PUBLIC_DOCUMENT_COMMIT = "eb9eb4927ddf1c5b24017f1fd913c96744bb6dd4";
const repository = "https://github.com/arvindcr4/tinker-rl-lab";
const sem3 = "platform_hybrid/sem 3 work";
const sem4 = "platform_hybrid/sem 4 work";
const paper = "platform_hybrid/paper";
const review = "reports/public_revision_2026-10-11";

const document = (path, label, format, kind = "blob") => ({
  path, label, format, kind,
  href: `${repository}/${kind}/${PUBLIC_DOCUMENT_COMMIT}/${path.split("/").map(encodeURIComponent).join("/")}`,
});

export const DOCUMENTS = {
  history: document("PROJECT_HISTORY.md", "Academic history", "Markdown"),
  currentStatus: document("CURRENT_PROJECT_STATUS.md", "Current project status", "Markdown"),
  thesis: document(`${review}/thesis/Thesis_Report_ArvindCR_2026-10-11_ESA_CORRECTED_REVIEW.pdf`, "Read the review thesis", "PDF · 317 pages"),
  thesisSources: document(`${review}/thesis/README.md`, "Editable sources & build guide", "Markdown"),
  thesisReview: document(`${review}/README.md`, "Export and review status", "Markdown"),
  completion: document(`${review}/COMPLETION_ASSESSMENT.md`, "Evidence and approval limits", "Markdown"),
  errata: document("reports/runtime_followup_2026-10-11/MANUSCRIPT_ERRATA.md", "Runtime and manuscript errata", "Markdown"),
  capstone: document(`${sem3}/deliverables/capstone-final-report.docx`, "Open the capstone report", "DOCX"),
  capstoneSource: document(`${sem3}/source/capstone_final_report.tex`, "Inspect the frozen report source", "LaTeX"),
  groupCitation: document(`${sem3}/CITATION.cff`, "Group authorship record", "CFF"),
  groupContributions: document(`${sem3}/CONTRIBUTIONS.md`, "Recorded contributions", "Markdown"),
  groupProvenance: document(`${sem3}/PROVENANCE.md`, "Archive provenance", "Markdown"),
  groupErratum: document(`${sem3}/REVIEW43_ERRATUM.md`, "Read the frozen-report erratum", "Markdown"),
  preparedPackage: document(`${sem3}/submissions/neurips-main-track/SUBMISSION_MANIFEST.md`, "Historical package manifest", "Markdown"),
  continuationMap: document(`${sem4}/EXPERIMENTS.md`, "Paper-to-evidence map", "Markdown"),
  continuationProvenance: document(`${sem4}/PROVENANCE.md`, "Solo-continuation provenance", "Markdown"),
  roster: document(`${paper}/PAPERS_README.md`, "Full current manuscript roster", "Markdown"),
  fraudArchive: document(`${paper}/archive/absorbed/P08_fraud/paper_P8_fraud.tex`, "Archived fraud-study source", "LaTeX"),
};

// These are topic labels, not verbatim historical titles or accepted-paper claims.
// The roster supplies the revised scope; the semester map supplies the source paths.
export const CONTINUATION_PAPERS = [
  { id: "P1", title: "Scaling-law limits", focus: "Cross-library and cross-scale behavior, with identifiability limits made explicit.", boundary: "An audit, not an established positive scaling law.", source: document(`${paper}/paper_P1_scaling.tex`, "P1 manuscript source", "LaTeX") },
  { id: "P2", title: "Zero-variance fraction", focus: "A descriptive diagnostic for missing reward contrast within a completion group.", boundary: "Diagnostic accounting does not by itself establish a training gain.", source: document(`${paper}/paper_P2_zvf.tex`, "P2 manuscript source", "LaTeX") },
  { id: "P3", title: "Group size and budget", focus: "How completion-group size interacts with budget and reward regime.", boundary: "Measured bounds and reconstructed hypotheses remain separate; no universal optimum.", source: document(`${paper}/paper_P3_group_size.tex`, "P3 manuscript source", "LaTeX") },
  { id: "P4", title: "Length bias and generalization", focus: "A bounded null test for length effects under a 200-token response cap.", boundary: "The result is scoped to the tested cap and setup.", source: document(`${paper}/paper_P4_length_bias.tex`, "P4 manuscript source", "LaTeX") },
  { id: "P5", title: "Report the stack", focus: "An eight-item reporting standard for stack-conditioned RL results.", boundary: "Reporting completeness is separate from experimental validity.", source: document(`${paper}/paper_P5_minreport.tex`, "P5 manuscript source", "LaTeX") },
  { id: "P6", title: "GRPO stack registry", focus: "A machine-readable seven-field run-start registry and explicit variant differences.", boundary: "Held-out pass@k belongs in the evaluation report, beyond the run-start fields.", source: document(`${paper}/paper_P6_registry.tex`, "P6 manuscript source", "LaTeX") },
  { id: "P7", title: "From diagnostic to controller?", focus: "A retrospective controller audit and a prospective intervention test plan.", boundary: "Replay and simulations do not establish closed-loop learning benefits.", source: document(`${paper}/paper_P7_zvf_controller.tex`, "P7 manuscript source", "LaTeX") },
];

export const GROUP_MEMBERS = [
  { name: "Arvind C R", contribution: "Repository integration, Tinker/SkyRL experiments, methodology and editing" },
  { name: "Sandhya Jeyaraj", contribution: "Tool-call scaling study, methodology and visualization" },
  { name: "Madhu Kumara L", contribution: "Code-generation GRPO work, methodology and visualization" },
  { name: "Mohammad Rafi", contribution: "Formal analysis, literature review and original report drafting" },
  { name: "Dhruva N Murthy", contribution: "Baseline, SFT and GRPO tool-use pipeline; review and editing" },
  { name: "Arumugam Chetty K", contribution: "Aerospace-domain DPO pipeline and formal analysis" },
];

export const GROUP_GUIDES = ["Anwesh Reddy Paduri", "Narayana Darapaneni"];
export const SOLO_GUIDE = "Ramesh Prakash Guledgudd";
export const SEMESTER_BOUNDARY = { tag: "capstone-final-2026-04-25", commit: "21a99ef766543a1ca86bfbad188445d3596ea73c" };
