import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT_DIR = "/Users/arvind/Developer/tinker-rl-lab/.codex-pptx/neurips-papers-e1e14-eli12/rendered";
const FINAL_PPTX = "/Users/arvind/Developer/tinker-rl-lab/outputs/NeurIPS_Review_12_Ideas_E1_E14_ELI12_Dark_2026-08-09.pptx";

const W = 1280;
const H = 720;
const C = {
  ink: "#F4F7FB",
  muted: "#AEB8C8",
  light: "#171D27",
  line: "#2A3445",
  blue: "#7EA5FF",
  blueLight: "#172541",
  green: "#49D6AE",
  greenLight: "#112A26",
  amber: "#FFB454",
  amberLight: "#302415",
  red: "#FF7F86",
  redLight: "#321C22",
  white: "#0B0F16",
  onAccent: "#071018",
};

const SRC = {
  review: "/Users/arvind/.codex/attachments/191be32b-d5ed-4464-a778-7ed4c495f73f/pasted-text.txt",
  roster: "/Users/arvind/Developer/tinker-rl-lab/platform_hybrid/paper/PAPERS_README.md",
  corrections: "/Users/arvind/Developer/tinker-rl-lab/platform_hybrid/paper/REVIEWER_36320_CORRECTION_MANIFEST.md",
  rebuttal: "/Users/arvind/Developer/tinker-rl-lab/zvf-program/flagship/paper/NEURIPS_2026_OPENREVIEW_REBUTTAL_FINAL.md",
  breakthrough: "/Users/arvind/Developer/tinker-rl-lab/BREAKTHROUGH_CHASE_18_ARTIFACTS.md",
  contract: "/Users/arvind/Developer/tinker-rl-lab/zvf-program/flagship/PAVLOVS_LIST_TASK_CONTRACT.md",
  citationPlan: "/Users/arvind/Developer/tinker-rl-lab/zvf-program/flagship/PAVLOV_PAPER_CITATION_PLAN_2026-08-09.md",
  sprint: "/Users/arvind/Developer/tinker-rl-lab/outputs/PAVLOV_E1_E14_LOCAL_SPRINT_2026-08-09.md",
  train: "/Users/arvind/Developer/tinker-rl-lab/checkpoints/grpo/pavlov_portfolio_api_swegym_qwen36_20260809_seed809.json",
  baseEval: "/Users/arvind/Developer/tinker-rl-lab/outputs/pavlov_portfolio_eval/base_reasoning_stripped_seed1810.json",
  trainedEval: "/Users/arvind/Developer/tinker-rl-lab/outputs/pavlov_portfolio_eval/trained_step40_seed1810.json",
  e11: "/Users/arvind/Developer/tinker-rl-lab/outputs/e11_verilog_eval/e11_trained_step40_receipt.json",
};

function addBox(slide, { left, top, width, height, fill = C.light, line = C.line, radius = "rounded-xl", name }) {
  return slide.shapes.add({
    geometry: "roundRect",
    name,
    position: { left, top, width, height },
    fill,
    line: { style: "solid", fill: line, width: 1 },
    borderRadius: radius,
  });
}

function addText(slide, text, { left, top, width, height, size = 20, color = C.ink, bold = false, align = "left", valign = "top", name, italic = false }) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name,
    position: { left, top, width, height },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: "Helvetica Neue",
    fontSize: size,
    color,
    bold,
    italic,
    alignment: align,
    verticalAlignment: valign,
  };
  return shape;
}

function addHeader(slide, title, kicker, page) {
  slide.background.fill = C.white;
  addText(slide, kicker.toUpperCase(), { left: 42, top: 34, width: 620, height: 28, size: 15, color: C.blue, bold: true, name: `kicker-${page}` });
  addText(slide, title, { left: 42, top: 70, width: 1170, height: 86, size: 39, bold: true, name: `title-${page}` });
  addText(slide, String(page).padStart(2, "0"), { left: 1184, top: 664, width: 54, height: 22, size: 13, color: C.muted, align: "right", name: `page-${page}` });
}

function addNotes(slide, sources, talkTrack) {
  const block = [
    talkTrack,
    "",
    "[Sources]",
    ...sources.map((s) => `- ${s}`),
    "[/Sources]",
  ].join("\n");
  slide.speakerNotes.textFrame.setText(block);
  slide.speakerNotes.setVisible(true);
}

function addStatusPill(slide, text, left, top, width, fill, color) {
  addBox(slide, { left, top, width, height: 34, fill, line: fill, radius: "rounded-full", name: `pill-${text}-${left}` });
  addText(slide, text, { left: left + 10, top: top + 6, width: width - 20, height: 22, size: 14, color, bold: true, align: "center" });
}

function addPaperRow(slide, y, id, title, correction, tone = "blue") {
  const fills = tone === "green" ? [C.greenLight, C.green] : tone === "amber" ? [C.amberLight, C.amber] : [C.blueLight, C.blue];
  addBox(slide, { left: 42, top: y, width: 1196, height: 70, fill: C.white, line: C.line, name: `paper-${id}` });
  addBox(slide, { left: 55, top: y + 14, width: 66, height: 42, fill: fills[0], line: fills[0], name: `paper-id-${id}` });
  addText(slide, id, { left: 60, top: y + 22, width: 56, height: 24, size: id.length > 2 ? 16 : 18, color: fills[1], bold: true, align: "center" });
  addText(slide, title, { left: 140, top: y + 11, width: 322, height: 48, size: 18, bold: true, valign: "middle" });
  addText(slide, correction, { left: 486, top: y + 11, width: 724, height: 48, size: 17, color: C.muted, valign: "middle" });
}

function addLaneRow(slide, y, id, suite, domain, status) {
  const statusColor = status === "SCORED" ? C.green : status === "PARTIAL" ? C.amber : C.red;
  const statusFill = status === "SCORED" ? C.greenLight : status === "PARTIAL" ? C.amberLight : C.redLight;
  addBox(slide, { left: 42, top: y, width: 1196, height: 61, fill: C.white, line: C.line, name: `lane-${id}` });
  addText(slide, id, { left: 56, top: y + 17, width: 66, height: 26, size: 18, color: C.blue, bold: true });
  addText(slide, suite, { left: 132, top: y + 12, width: 317, height: 37, size: 18, bold: true, valign: "middle" });
  addText(slide, domain, { left: 470, top: y + 12, width: 475, height: 37, size: 16, color: C.muted, valign: "middle" });
  addStatusPill(slide, status, 1028, y + 14, 174, statusFill, statusColor);
}

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  await fs.mkdir(OUT_DIR, { recursive: true });
  await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });

  const deck = Presentation.create({ slideSize: { width: W, height: H } });

  // 1 — minimal title slide, based on Codex Grid slide 01.
  {
    const s = deck.slides.add();
    s.background.fill = C.white;
    addText(s, "RESEARCH UPDATE · AUGUST 2026", { left: 42, top: 42, width: 620, height: 32, size: 16, color: C.blue, bold: true });
    addText(s, "How the NeurIPS review\nreshaped my research", { left: 42, top: 176, width: 1080, height: 250, size: 64, bold: true, valign: "bottom" });
    addText(s, "12 corrected research ideas · 2 serious paper directions · 14 new evaluation suites", { left: 42, top: 500, width: 1080, height: 74, size: 26, color: C.muted });
    addText(s, "Plain-language version", { left: 42, top: 625, width: 360, height: 30, size: 16, color: C.green, bold: true });
    addNotes(s, [SRC.review, SRC.roster, SRC.breakthrough, SRC.contract], "Open with the simple message: the review forced the project to become smaller in claims and larger in evidence.");
  }

  // 2 — what reviewers said.
  {
    const s = deck.slides.add();
    addHeader(s, "The reviewers liked the idea—but could not trust the story", "What NeurIPS #36320 said", 2);
    addBox(s, { left: 42, top: 182, width: 565, height: 430, fill: C.greenLight, line: C.greenLight, name: "review-strengths" });
    addText(s, "What they valued", { left: 72, top: 214, width: 480, height: 42, size: 27, color: C.green, bold: true });
    addText(s, "• Important problem: RL training curves can mislead\n\n• Cheap diagnostics: ZVF and GU are easy to monitor\n\n• Good research instinct: separate reward, capability, and algorithm labels", { left: 72, top: 278, width: 485, height: 280, size: 22, color: C.ink });
    addBox(s, { left: 635, top: 182, width: 603, height: 430, fill: C.redLight, line: C.redLight, name: "review-problems" });
    addText(s, "What broke confidence", { left: 665, top: 214, width: 520, height: 42, size: 27, color: C.red, bold: true });
    addText(s, "• The paper was hard to follow\n\n• The runner was not standard GRPO\n\n• Different models and tasks were mixed together\n\n• Some rows had weak or conflicting provenance\n\n• “Use-inspired” was claimed without a real user outcome", { left: 665, top: 278, width: 520, height: 304, size: 20, color: C.ink });
    addNotes(s, [SRC.review], "Explain this like a school project: the idea was interesting, but the report mixed several experiments and did not show exactly which evidence proved each claim.");
  }

  // 3 — correction process timeline, preserving Codex Grid slide 17 hierarchy.
  {
    const s = deck.slides.add();
    addHeader(s, "I treated the review like a bug report", "Correction strategy", 3);
    const xs = [80, 370, 660, 950];
    const labels = ["1. Admit", "2. Audit", "3. Narrow", "4. Rebuild"];
    const bodies = [
      "Say clearly which claims were unsupported.",
      "Trace every number to its run, seed, model, and evaluator.",
      "Keep only claims the evidence can actually carry.",
      "Make future claims pass stronger, multi-domain tests.",
    ];
    s.shapes.add({ geometry: "straightConnector1", position: { left: 100, top: 321, width: 1015, height: 1 }, fill: "none", line: { style: "solid", fill: C.ink, width: 2 } });
    xs.forEach((x, i) => {
      s.shapes.add({ geometry: "ellipse", position: { left: x + 82, top: 310, width: 24, height: 24 }, fill: i < 3 ? C.blue : C.green, line: { style: "solid", fill: i < 3 ? C.blue : C.green, width: 1 } });
      addText(s, labels[i], { left: x, top: 244, width: 190, height: 40, size: 21, color: i < 3 ? C.blue : C.green, bold: true, align: "center" });
      addText(s, bodies[i], { left: x - 8, top: 370, width: 210, height: 135, size: 19, color: C.muted, align: "center" });
    });
    addBox(s, { left: 210, top: 548, width: 860, height: 70, fill: C.light, line: C.light, name: "big-correction" });
    addText(s, "Biggest correction: a clean ‘I do not know yet’ is stronger than a shaky result.", { left: 236, top: 568, width: 808, height: 32, size: 22, bold: true, align: "center" });
    addNotes(s, [SRC.rebuttal, SRC.corrections], "Walk through the four steps. Stress that withdrawing weak claims was a research improvement, not a defeat.");
  }

  // 4 — P1 to P6.
  {
    const s = deck.slides.add();
    addHeader(s, "What changed in research ideas P1–P6", "12-idea correction pass · part 1", 4);
    addPaperRow(s, 166, "P1", "Scaling", "Stopped pooling selected checkpoints. Now a limits and identifiability audit.");
    addPaperRow(s, 242, "P2", "ZVF diagnostic", "ZVF is a useful description—not the whole gradient and not a universal predictor.");
    addPaperRow(s, 318, "P3", "Group size", "Reports budget-specific measurements; no one-size-fits-all group recommendation.");
    addPaperRow(s, 394, "P4", "Length bias", "Claims only a bounded null under a 200-token cap.");
    addPaperRow(s, 470, "P5", "MIN-REPORT-RL", "Requires provenance, estimands, missing cells, and held-out pass@k reporting.", "green");
    addPaperRow(s, 546, "P6", "Run registry", "Treats conflicting records as quarantined evidence, not as values to average away.", "green");
    addNotes(s, [SRC.roster, SRC.corrections], "Use one sentence per research idea. Each manuscript direction now has a smaller and more precise job.");
  }

  // 5 — P7 to P12.
  {
    const s = deck.slides.add();
    addHeader(s, "What changed in research ideas P7–P12", "12-idea correction pass · part 2", 5);
    addPaperRow(s, 166, "P7", "ZVF controller", "Now a retrospective audit and test plan; no claimed controller benefit.");
    addPaperRow(s, 242, "P8", "Workshop note", "Kept as a case-specific artifact note; no rankings or broad benchmark claim.");
    addPaperRow(s, 318, "P9", "Dataset & benchmark", "Uses evidence tiers and quarantines uncertain provenance.", "green");
    addPaperRow(s, 394, "P10", "ZVF theory", "Limits proofs to centered reward contrast; does not claim total-gradient behavior.");
    addPaperRow(s, 470, "P11", "Survival audit", "Reports the 40-unit single-stack result as bounded; conclusions remain inconclusive.", "green");
    addPaperRow(s, 546, "P12", "Signal starvation", "Keeps PPO/SAO routing as a proposal until prospective tests succeed.", "amber");
    addNotes(s, [SRC.roster, SRC.corrections], "Make clear that all 12 ideas remain useful, but only two are serious paper directions.");
  }

  // 6 — the two serious directions.
  {
    const s = deck.slides.add();
    addHeader(s, "Two paper directions now deserve serious submission work", "The audit’s survivors", 6);
    addBox(s, { left: 42, top: 180, width: 570, height: 434, fill: C.greenLight, line: C.green, name: "paper-a" });
    addStatusPill(s, "DEFENSIBLE NOW", 74, 208, 190, C.green, C.onAccent);
    addText(s, "Paper A\nVerify the treatment", { left: 74, top: 264, width: 490, height: 104, size: 32, bold: true });
    addText(s, "Simple idea: an algorithm name is only a label. Prove what code actually ran, then bind every result to the exact stack and evidence.", { left: 74, top: 390, width: 492, height: 120, size: 21, color: C.muted });
    addText(s, "Best fit: artifact / methodology / reproducibility paper", { left: 74, top: 540, width: 492, height: 42, size: 18, color: C.green, bold: true });

    addBox(s, { left: 640, top: 180, width: 598, height: 434, fill: C.amberLight, line: C.amber, name: "paper-b" });
    addStatusPill(s, "HIGH UPSIDE · GATED", 672, 208, 225, C.amber, C.onAccent);
    addText(s, "Paper B\nTRIAGE-RL", { left: 672, top: 264, width: 500, height: 104, size: 32, bold: true });
    addText(s, "Simple idea: not all weak learning signals mean the same thing. Solved, failed, clipped, and unsafe cases need different actions.", { left: 672, top: 390, width: 510, height: 120, size: 21, color: C.muted });
    addText(s, "Becomes flagship-worthy only if the matched-budget experiment wins", { left: 672, top: 540, width: 510, height: 42, size: 18, color: C.amber, bold: true });
    addNotes(s, [SRC.breakthrough], "Use the phrase ‘worthy of serious submission work,’ not ‘accepted’ or ‘finished.’ Paper A is evidence-backed now. Paper B is the high-upside bet but still has a kill test.");
  }

  // 7 — why E1-E14.
  {
    const s = deck.slides.add();
    addHeader(s, "Why GSM8K alone could never prove usefulness", "The expansion decision", 7);
    addText(s, "A model that passes one math worksheet may still fail at real work.", { left: 42, top: 176, width: 1120, height: 60, size: 31, bold: true });
    const cols = [42, 453, 864];
    const heads = ["Old test", "Missing abilities", "New rule"];
    const bodies = [
      "Mostly short math answers.\n\nUseful as calibration—not as the main claim.",
      "Use tools, edit files, browse, handle finance, write code, build artifacts, stay safe, work for many steps.",
      "Train across domains.\n\nEvaluate on unseen task families.\n\nNever hide a weak domain inside one average.",
    ];
    cols.forEach((x, i) => {
      addBox(s, { left: x, top: 278, width: 374, height: 290, fill: i === 2 ? C.blueLight : C.light, line: i === 2 ? C.blue : C.line, name: `why-${i}` });
      addText(s, heads[i], { left: x + 24, top: 310, width: 326, height: 38, size: 25, color: i === 2 ? C.blue : C.ink, bold: true });
      addText(s, bodies[i], { left: x + 24, top: 370, width: 326, height: 160, size: 21, color: C.muted });
    });
    addText(s, "E1–E14 turns ‘useful’ from a slogan into fourteen separate report cards.", { left: 184, top: 594, width: 912, height: 42, size: 24, color: C.green, bold: true, align: "center" });
    addNotes(s, [SRC.contract], "Use the school analogy: GSM8K is one worksheet. E1–E14 is a report card covering different subjects and real tasks.");
  }

  // 8 — E1 to E7.
  {
    const s = deck.slides.add();
    addHeader(s, "E1–E7: code, enterprise, browsers, and security", "The new usefulness test · part 1", 8);
    addLaneRow(s, 160, "E1", "SWE-bench Pro", "Hard repository repair", "BLOCKED");
    addLaneRow(s, 226, "E2", "Frontier SWE", "Frontier code and ML work", "BLOCKED");
    addLaneRow(s, 292, "E3", "SDAB", "Production systems and enterprise", "BLOCKED");
    addLaneRow(s, 358, "E4", "BankerToolBench", "Finance tools and state changes", "BLOCKED");
    addLaneRow(s, 424, "E5", "APEX Agents", "Long professional workflows", "PARTIAL");
    addLaneRow(s, 490, "E6", "WebBench", "Browser and computer use", "BLOCKED");
    addLaneRow(s, 556, "E7", "BinaryAudit", "Security and binary analysis", "PARTIAL");
    addNotes(s, [SRC.citationPlan, SRC.sprint], "Blocked means the exact benchmark score is not available yet. Partial means plumbing or a verifier ran on a controlled fixture, not that the model passed the benchmark.");
  }

  // 9 — E8 to E14.
  {
    const s = deck.slides.add();
    addHeader(s, "E8–E14: science, ML, safety, chips, design, games, math", "The new usefulness test · part 2", 9);
    addLaneRow(s, 160, "E8", "LifeSciBench", "Life science workflows", "BLOCKED");
    addLaneRow(s, 226, "E9", "MLE-bench", "Build and evaluate ML systems", "PARTIAL");
    addLaneRow(s, 292, "E10", "AgentHarm", "Safety under tool use", "PARTIAL");
    addLaneRow(s, 358, "E11", "VerilogEval", "Chip-design code", "SCORED");
    addLaneRow(s, 424, "E12", "AppBench", "Build usable visual apps", "BLOCKED");
    addLaneRow(s, 490, "E13", "OpenReward Games", "Planning in game worlds", "BLOCKED");
    addLaneRow(s, 556, "E14", "FrontierMath", "Hard private mathematics", "BLOCKED");
    addNotes(s, [SRC.citationPlan, SRC.sprint, SRC.e11], "E11 is now the one scored lane: a four-task smoke, not the full 312-task benchmark. The older sprint index predates that trained-model receipt, so the live E11 receipt takes precedence.");
  }

  // 10 — current results.
  {
    const s = deck.slides.add();
    addHeader(s, "The expansion has one small win—and thirteen open gates", "What has actually run", 10);
    s.charts.add("bar", {
      position: { left: 42, top: 194, width: 610, height: 390 },
      categories: ["Starting model", "Trained model"],
      series: [{ name: "Mean reward", values: [0.396, 0.599], valuesFormatCode: "0.000", fill: C.blue }],
      barOptions: { direction: "column", grouping: "clustered", gapWidth: 65 },
      hasLegend: false,
      chartFill: C.white,
      plotAreaFill: C.white,
      xAxis: { textStyle: { fill: C.muted, fontSize: 17 }, line: { style: "solid", fill: C.line, width: 1 } },
      yAxis: { min: 0, max: 0.7, majorUnit: 0.1, numberFormatCode: "0.0", textStyle: { fill: C.muted, fontSize: 14 }, majorGridlines: { style: "solid", fill: C.line, width: 1 } },
      dataLabels: { showValue: true, position: "outEnd", textStyle: { fill: C.ink, fontSize: 17, bold: true } },
    });
    addBox(s, { left: 690, top: 194, width: 548, height: 178, fill: C.greenLight, line: C.green, name: "portfolio-win" });
    addText(s, "+51%", { left: 722, top: 220, width: 180, height: 64, size: 42, color: C.green, bold: true });
    addText(s, "mean reward on the same 32 unseen API and code examples", { left: 722, top: 292, width: 470, height: 54, size: 20, color: C.muted });
    addBox(s, { left: 690, top: 396, width: 548, height: 188, fill: C.blueLight, line: C.blue, name: "e11-win" });
    addText(s, "4 / 4", { left: 722, top: 422, width: 180, height: 64, size: 42, color: C.blue, bold: true });
    addText(s, "tiny E11 trained-model HDL smoke passed\nFull VerilogEval has 312 prompts, so this is not the full score.", { left: 722, top: 492, width: 470, height: 76, size: 19, color: C.muted });
    addText(s, "Evidence boundary: 1 of 14 lanes scored today; 13 still need exact model-level results.", { left: 100, top: 621, width: 1080, height: 32, size: 22, color: C.amber, bold: true, align: "center" });
    addNotes(s, [SRC.train, SRC.baseEval, SRC.trainedEval, SRC.e11, "https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-pavlov/runs/fcsr357r", "https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-pavlov/runs/u9v7kh9w", "https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-pavlov/runs/d2mbnyyz"], "Explain the bar chart carefully: it is a matched local portfolio check, not an official SWE-bench Pro score. The E11 result is only four tasks.");
  }

  // 11 — close.
  {
    const s = deck.slides.add();
    s.background.fill = C.white;
    addText(s, "THE NEW RESEARCH PROGRAM", { left: 42, top: 42, width: 540, height: 30, size: 16, color: C.blue, bold: true });
    addText(s, "Less claiming.\nMore proving.", { left: 42, top: 168, width: 720, height: 220, size: 68, bold: true, valign: "bottom" });
    addText(s, "12 research ideas now say exactly what their evidence supports.\n2 paper directions carry the real novelty.\n14 suites test whether the work matters beyond one math benchmark.", { left: 42, top: 452, width: 830, height: 138, size: 24, color: C.muted });
    addBox(s, { left: 920, top: 170, width: 318, height: 410, fill: C.light, line: C.line, name: "next-proof" });
    addText(s, "What earns the next claim", { left: 950, top: 205, width: 258, height: 70, size: 25, color: C.ink, bold: true });
    addText(s, "Exact data\n\nUnseen split\n\nNative verifier\n\nW&B run\n\nHF checkpoint\n\nMatched baseline", { left: 950, top: 300, width: 250, height: 240, size: 20, color: C.ink });
    addText(s, "The goal is not fourteen green boxes today. The goal is fourteen results we can defend tomorrow.", { left: 42, top: 628, width: 1130, height: 38, size: 21, color: C.green, bold: true });
    addNotes(s, [SRC.corrections, SRC.breakthrough, SRC.contract, SRC.e11], "Close on research maturity: the project is now designed to fail honestly and to earn claims one receipt at a time.");
  }

  for (const [index, slide] of deck.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(path.join(OUT_DIR, `${stem}.png`), await deck.export({ slide, format: "png", scale: 1 }));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(OUT_DIR, `${stem}.layout.json`), await layout.text());
  }
  await writeBlob(path.join(OUT_DIR, "deck-montage.webp"), await deck.export({ format: "webp", montage: true, scale: 1 }));
  const pptx = await PresentationFile.exportPptx(deck);
  await pptx.save(FINAL_PPTX);
  console.log(JSON.stringify({ final: FINAL_PPTX, slides: deck.slides.items.length, renderDir: OUT_DIR }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
