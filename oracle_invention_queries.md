# Oracle Invention Queries — ZVF Program 4 Papers × 10 Prompts

> Uses `npx -y @steipete/oracle` to bundle each paper and ask a frontier model to invent something new.
> All four PDFs are < 1 MB, so they attach directly. For the two `.tex` papers I also include the source so the AI sees the TODO/gap markers and bib stubs.

---

## Paper 1 — MIN-REPORT-RL Position Paper (`zvf-program/position/min_report_rl.pdf`)

Theme: a 7-item minimum-reportable-stack standard + controlled reproducibility audit for the GRPO family.

```bash
# 1.1 — Invent an auto-reporting telemetry plugin for TRL
npx -y @steipete/oracle -p \
"Read the attached MIN-REPORT-RL position paper. Invent a TRL plugin that automatically emits the 7-item minimum-reportable-stack block as JSON at the start of every GRPO run and logs per-step ZVF/GU. Specify the hooks, config flags, and output schema." \
--file zvf-program/position/min_report_rl.pdf --file zvf-program/position/min_report_rl.tex

# 1.2 — Invent a stack-diff comparison tool
npx -y @steipete/oracle -p \
"Read the attached paper. Invent a CLI tool that takes two GRPO run manifests and reports which stack levers differ and whether each difference is large enough to flip a comparison. Define the diff taxonomy and output format." \
--file zvf-program/position/min_report_rl.pdf --file zvf-program/position/min_report_rl.tex

# 1.3 — Invent a MIN-REPORT-RL compliance checker
npx -y @steipete/oracle -p \
"Read the attached paper. Invent an automated compliance checker that scans a GRPO paper or repo and scores it against the 7-item MIN-REPORT-RL checklist. Provide the scoring rubric, heuristics for each item, and example outputs." \
--file zvf-program/position/min_report_rl.pdf --file zvf-program/position/min_report_rl.tex --file zvf-program/position/CHECKLIST.md

# 1.4 — Invent a reproducibility audit automation
npx -y @steipete/oracle -p \
"Read the attached paper. Invent an automated pipeline that pre-registers and runs the controlled single-stack DAPO/GSPO/Dr.GRPO/MAD-GRPO audit described in the paper. Specify inputs, arm definitions, survival verdict logic, and the output tables." \
--file zvf-program/position/min_report_rl.pdf --file zvf-program/position/min_report_rl.tex

# 1.5 — Invent a citation-verification bot for GRPO papers
npx -y @steipete/oracle -p \
"Read the attached paper. Invent a bot that scans GRPO-family papers, extracts every claim tied to a specific stack lever, and checks whether that lever is actually reported in the cited paper. Explain the claim-extraction model and the report format." \
--file zvf-program/position/min_report_rl.pdf --file zvf-program/position/min_report_rl.tex

# 1.6 — Invent a stack-effect attribution scorecard
npx -y @steipete/oracle -p \
"Read the attached paper. Invent a scorecard that attributes an observed head-to-head delta between two GRPO variants to stack differences vs. algorithmic differences. Define the attribution method, required measurements, and confidence language." \
--file zvf-program/position/min_report_rl.pdf --file zvf-program/position/min_report_rl.tex

# 1.7 — Invent a community-maintained GRPO variant registry
npx -y @steipete/oracle -p \
"Read the attached paper. Invent a public registry format for GRPO-family variants where each entry records the defining delta plus the full MIN-REPORT-RL block. Provide the JSON schema, an example entry for DAPO, and a query API." \
--file zvf-program/position/min_report_rl.pdf --file zvf-program/position/min_report_rl.tex

# 1.8 — Invent a reproducibility threat-modeling framework
npx -y @steipete/oracle -p \
"Read the attached paper. Invent a threat-modeling framework for reproducibility failures in RL-for-LLM papers. List threat categories, mitigations, and how they map to the 7 checklist items." \
--file zvf-program/position/min_report_rl.pdf --file zvf-program/position/min_report_rl.tex

# 1.9 — Invent an automated single-stack re-implementation harness
npx -y @steipete/oracle -p \
"Read the attached paper. Invent a harness that re-implements a GRPO variant as a minimal config override on a shared trainer. Define the hook interface, baseline constraints, and how it enforces identical stack across arms." \
--file zvf-program/position/min_report_rl.pdf --file zvf-program/position/min_report_rl.tex

# 1.10 — Invent a venue review checklist integration
npx -y @steipete/oracle -p \
"Read the attached paper. Invent a way to integrate the MIN-REPORT-RL checklist into OpenReview / HotCRP so reviewers see a structured stack-report form and an auto-populated compliance summary." \
--file zvf-program/position/min_report_rl.pdf --file zvf-program/position/min_report_rl.tex
```

---

## Paper 2 — ZVF Theory Paper (`zvf-program/theory/zvf_theory.pdf`)

Theme: T1 ZVF confidence interval, T2 wasted-compute lower bound, T3 optimal group size G*.

```bash
# 2.1 — Invent an online ZVF confidence-interval calculator
npx -y @steipete/oracle -p \
"Read the attached ZVF theory paper. Invent a drop-in Python module that computes the per-step ZVF and its exact 95%% confidence interval from a batch of group rewards, handling boundaries with Wilson intervals. Provide the API and a usage example." \
--file zvf-program/theory/zvf_theory.pdf --file zvf-program/theory/zvf_theory.tex --file zvf-program/theory/THEORY_NOTES.md

# 2.2 — Invent a wasted-compute early-stopping rule
npx -y @steipete/oracle -p \
"Read the attached paper. Invent an early-stopping rule based on the T2 wasted-compute lower bound. Specify the trigger condition, the confidence-level choice, and a conservative variant that avoids false stops." \
--file zvf-program/theory/zvf_theory.pdf --file zvf-program/theory/zvf_theory.tex

# 2.3 — Invent an adaptive group-size controller based on G*
npx -y @steipete/oracle -p \
"Read the attached paper. Invent a training-time controller that sets group size G to the T3 optimal G* each step, using a running Beta fit of per-prompt success rates. Give the update equations, stability safeguards, and pseudocode." \
--file zvf-program/theory/zvf_theory.pdf --file zvf-program/theory/zvf_theory.tex

# 2.4 — Invent a Beta-prior difficulty estimator
npx -y @steipete/oracle -p \
"Read the attached paper. Invent an online estimator for the reward-density prior phi as a Beta distribution from streaming per-prompt success rates. Derive the MLE or method-of-moments update and state its convergence behavior." \
--file zvf-program/theory/zvf_theory.pdf --file zvf-program/theory/zvf_theory.tex

# 2.5 — Invent a closed-loop stability analysis for the G* controller
npx -y @steipete/oracle -p \
"Read the attached paper. Invent a stability analysis for the closed loop G_{t+1} = G*(phi_hat_t). State conditions under which the loop converges, oscillates, or diverges, and propose a damped update that guarantees bounded G." \
--file zvf-program/theory/zvf_theory.pdf --file zvf-program/theory/zvf_theory.tex

# 2.6 — Invent a ZVF calibration under curriculum sampling
npx -y @steipete/oracle -p \
"Read the attached paper. Invent a corrected ZVF confidence interval and estimator for curriculum or replay sampling, where across-group i.i.d. is violated. Provide the bias correction and a block-bootstrap alternative." \
--file zvf-program/theory/zvf_theory.pdf --file zvf-program/theory/zvf_theory.tex

# 2.7 — Invent a non-binary reward ZVF generalization
npx -y @steipete/oracle -p \
"Read the attached paper. Invent a generalization of ZVF/GU to continuous or dense rewards, where exact zero variance is rare. Define a tolerance-based informativeness kernel and show it reduces to binary ZVF at the limit." \
--file zvf-program/theory/zvf_theory.pdf --file zvf-program/theory/zvf_theory.tex

# 2.8 — Invent a multi-prompt difficulty prior inference
npx -y @steipete/oracle -p \
"Read the attached paper. Invent a hierarchical model that infers per-prompt success probabilities and a population prior simultaneously, so G* can be personalized per prompt. Give the inference algorithm and complexity." \
--file zvf-program/theory/zvf_theory.pdf --file zvf-program/theory/zvf_theory.tex

# 2.9 — Invent a ZVF-powered learning-rate scheduler
npx -y @steipete/oracle -p \
"Read the attached paper. Invent a learning-rate scheduler that uses ZVF and its CI to scale the step size: small or uncertain updates when signal is scarce, larger updates when contrast is abundant. Provide the rule and intuition." \
--file zvf-program/theory/zvf_theory.pdf --file zvf-program/theory/zvf_theory.tex

# 2.10 — Invent a theoretical ZVF test harness
npx -y @steipete/oracle -p \
"Read the attached paper. Invent a test harness that empirically validates the T1 CI coverage, the T2 compute floor, and the T3 G* optimum on synthetic binary-reward data. Specify experiments, metrics, and expected pass criteria." \
--file zvf-program/theory/zvf_theory.pdf --file zvf-program/theory/zvf_theory.tex
```

---

## Paper 3 — ZVF Program Progress Deck (`zvf-program/ZVF_Program_Progress_2026-06-14.pdf`)

Theme: four-pillar program status, scorecard, and next moves for the ZVF Program.

```bash
# 3.1 — Invent a four-pillar program management dashboard
npx -y @steipete/oracle -p \
"Read the attached progress deck. Invent a live dashboard that tracks the four ZVF Program pillars (Method, Theory, Open Source, Position) with artifact status, blockers, and next deadlines. List the data sources and UI panels." \
--file zvf-program/ZVF_Program_Progress_2026-06-14.pdf

# 3.2 — Invent an artifact readiness scorecard
npx -y @steipete/oracle -p \
"Read the attached deck. Invent a scorecard that rates each pillar's artifact as SHIPPABLE, LAUNCH-READY, DRAFT, or BLOCKED using objective signals from tests, compilation, docs, and TODO counts. Define the scoring rules." \
--file zvf-program/ZVF_Program_Progress_2026-06-14.pdf

# 3.3 — Invent a conference-target milestone tracker
npx -y @steipete/oracle -p \
"Read the attached deck. Invent a milestone tracker that maps each pillar to its target venue (ICLR, AISTATS/COLT, NeurIPS D&B, NeurIPS Position) and tracks submission-readiness deadlines, required experiments, and missing artifacts." \
--file zvf-program/ZVF_Program_Progress_2026-06-14.pdf

# 3.4 — Invent a cross-pillar dependency visualizer
npx -y @steipete/oracle -p \
"Read the attached deck. Invent a tool that visualizes dependencies across the four pillars and warns when a delay in one pillar propagates to others. Describe the DAG, node attributes, and critical-path highlighting." \
--file zvf-program/ZVF_Program_Progress_2026-06-14.pdf

# 3.5 — Invent a submission-readiness risk analyzer
npx -y @steipete/oracle -p \
"Read the attached deck. Invent a risk analyzer that scans each pillar and outputs a ranked list of risks to hitting the target venue deadline, with mitigation suggestions." \
--file zvf-program/ZVF_Program_Progress_2026-06-14.pdf

# 3.6 — Invent an automated slide-deck updater
npx -y @steipete/oracle -p \
"Read the attached deck. Invent a pipeline that regenerates the progress deck from live repo signals (test status, TODO counts, run manifests, compile logs). Specify the template engine and data bindings." \
--file zvf-program/ZVF_Program_Progress_2026-06-14.pdf

# 3.7 — Invent a Git-based research program status reporter
npx -y @steipete/oracle -p \
"Read the attached deck. Invent a CLI that scans a research repo and produces a program-status report: active branches, recent commits per pillar, open TODOs, and artifact build status." \
--file zvf-program/ZVF_Program_Progress_2026-06-14.pdf

# 3.8 — Invent a pillar health indicator suite
npx -y @steipete/oracle -p \
"Read the attached deck. Invent a suite of quantitative health indicators for each pillar (code coverage, doc completeness, proof gaps, citation stubs, experiment count). Provide formulas and aggregation weights." \
--file zvf-program/ZVF_Program_Progress_2026-06-14.pdf

# 3.9 — Invent a reviewer-objection prep tool
npx -y @steipete/oracle -p \
"Read the attached deck. Invent a tool that, for each pillar, generates the most likely reviewer objections and pre-drafts responses backed by evidence in the repo." \
--file zvf-program/ZVF_Program_Progress_2026-06-14.pdf

# 3.10 — Invent an open-science timeline planner
npx -y @steipete/oracle -p \
"Read the attached deck. Invent a timeline planner that schedules releases of code, data, checkpoints, and papers so each aligns with venue deadlines and community expectations. Include buffer recommendations." \
--file zvf-program/ZVF_Program_Progress_2026-06-14.pdf
```

---

## Paper 4 — ZVF Program Lightning Deck (`zvf-program/ZVF_Program_Progress_2026-06-14_lightning.pdf`)

Theme: condensed program status, shippable artifacts, and immediate next steps.

```bash
# 4.1 — Invent a one-page program health generator
npx -y @steipete/oracle -p \
"Read the attached lightning deck. Invent a script that generates a one-page program health summary from repo signals: 4 pillars, top artifact, status, and one blocking item each. Specify the input sources and output format." \
--file zvf-program/ZVF_Program_Progress_2026-06-14_lightning.pdf

# 4.2 — Invent an executive summary auto-generator
npx -y @steipete/oracle -p \
"Read the attached deck. Invent a generator that writes a 3-sentence executive summary of the ZVF Program from live repo data, in a tone suitable for an advisor or program manager." \
--file zvf-program/ZVF_Program_Progress_2026-06-14_lightning.pdf

# 4.3 — Invent a lightning-talk slide generator
npx -y @steipete/oracle -p \
"Read the attached deck. Invent a tool that auto-generates a 5-slide lightning talk from the current program state, with one slide per pillar and a status callout. Describe the template and content rules." \
--file zvf-program/ZVF_Program_Progress_2026-06-14_lightning.pdf

# 4.4 — Invent a quick status badge system
npx -y @steipete/oracle -p \
"Read the attached deck. Invent a badge system for README files that shows each pillar's status (SHIPPABLE, LAUNCH-READY, DRAFT, TODO) and updates from CI. Specify the badge endpoints and JSON schema." \
--file zvf-program/ZVF_Program_Progress_2026-06-14_lightning.pdf

# 4.5 — Invent a progress snapshot CLI
npx -y @steipete/oracle -p \
"Read the attached deck. Invent a CLI command `zvf-status` that prints the latest pillar snapshot, artifact paths, test counts, and next action. Define flags and output modes." \
--file zvf-program/ZVF_Program_Progress_2026-06-14_lightning.pdf

# 4.6 — Invent a minimal progress report formatter
npx -y @steipete/oracle -p \
"Read the attached deck. Invent a formatter that turns the program snapshot into Markdown, Slack, or email updates with the right level of detail for each audience." \
--file zvf-program/ZVF_Program_Progress_2026-06-14_lightning.pdf

# 4.7 — Invent a stakeholder update automator
npx -y @steipete/oracle -p \
"Read the attached deck. Invent an automator that sends scheduled stakeholder updates when a pillar changes status, including only the changed items and the implied next steps." \
--file zvf-program/ZVF_Program_Progress_2026-06-14_lightning.pdf

# 4.8 — Invent a key-metrics one-liner extractor
npx -y @steipete/oracle -p \
"Read the attached deck. Invent a tool that extracts the single most important metric per pillar (e.g., tests passing, theorems remaining with gaps, TODO citations, audit cells filled) and formats them as a one-liner." \
--file zvf-program/ZVF_Program_Progress_2026-06-14_lightning.pdf

# 4.9 — Invent a shippable artifact highlight reel
npx -y @steipete/oracle -p \
"Read the attached deck. Invent a highlight-reel generator that lists all currently shippable artifacts, their locations, and the one-line evidence that makes each shippable." \
--file zvf-program/ZVF_Program_Progress_2026-06-14_lightning.pdf

# 4.10 — Invent a 30-second pitch generator
npx -y @steipete/oracle -p \
"Read the attached deck. Invent a generator that produces a 30-second spoken pitch for the ZVF Program from the latest repo state, suitable for a hallway conversation or social post." \
--file zvf-program/ZVF_Program_Progress_2026-06-14_lightning.pdf
```

---

## Quick usage notes

- Run from the repo root so relative `--file` paths resolve.
- Oracle will auto-pick API mode if `OPENAI_API_KEY` is set; otherwise it falls back to browser automation.
- To preview what will be sent without spending tokens, append `--dry-run summary`.
- If you prefer a single model, add e.g. `--models gpt-5.5-pro` or `--models claude-sonnet-4.5`.
