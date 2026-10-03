const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.defineLayout({ name: "WIDE", width: 13.333, height: 7.5 });
pres.layout = "WIDE";
pres.title = "Tinker RL Lab — Phase 2 defense";
pres.author = "Arvind C R";
pres.subject = "UE20CS972 final defense, aligned to the 3 October 2026 public thesis";

const C = {
  ink: "1C1917",
  body: "3F3A36",
  muted: "6F675F",
  paper: "F7F4EF",
  white: "FFFFFF",
  maroon: "6E2C3A",
  forest: "1F4D3A",
  gold: "8C734A",
  brick: "8C3A2F",
  card: "FFFCF8",
  line: "E6DFD4",
  cream: "F3E6D4",
};
const F = { title: "Georgia", body: "Calibri" };
const W = 13.333;
const N = 12;

function footer(slide, n) {
  slide.addText(String(n) + "  /  " + N, {
    x: 0.5, y: 7.12, w: 1.4, h: 0.24,
    fontFace: F.body, fontSize: 11, color: C.muted, margin: 0,
  });
  slide.addText("UE20CS972  ·  Phase 2 defense  ·  3 October 2026", {
    x: 4.2, y: 7.12, w: 8.6, h: 0.24,
    fontFace: F.body, fontSize: 11, color: C.muted, align: "right", margin: 0,
  });
}

function paper(slide) {
  slide.background = { color: C.paper };
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.12, h: 7.5, fill: { color: C.maroon },
  });
}

function h1(slide, text) {
  slide.addText(text, {
    x: 0.48, y: 0.28, w: 12.3, h: 0.52,
    fontFace: F.title, fontSize: 28, color: C.ink, margin: 0,
  });
}

// 1 — title
{
  const s = pres.addSlide();
  s.background = { color: C.maroon };
  s.addText("PES UNIVERSITY", {
    x: 0.7, y: 0.42, w: 8, h: 0.28,
    fontFace: F.body, fontSize: 13, color: "E7C9C4", margin: 0, charSpacing: 1.4,
  });
  s.addText("Tinker RL Lab", {
    x: 0.7, y: 1.7, w: 11.5, h: 0.85,
    fontFace: F.title, fontSize: 48, color: C.white, margin: 0,
  });
  s.addText("Post-training of large language models\nunder a fixed stack", {
    x: 0.7, y: 2.65, w: 10.5, h: 1.15,
    fontFace: F.title, fontSize: 26, color: "F3E6D4", margin: 0,
  });
  s.addText("Arvind C R    ·    SRN PES2PGE24DS140\nM.Tech Data Science and Machine Learning\nGuide: Ramesh Prakash Guledgudd\nUE20CS972  ·  Project Phase 2", {
    x: 0.7, y: 4.55, w: 8, h: 1.45,
    fontFace: F.body, fontSize: 16, color: C.white, margin: 0,
  });
  s.addNotes("Open by naming the degree and the rule, not the infrastructure. Say: this is a measurement thesis. The algorithmic levers I could test are mostly noise-limited. I will show what survived, and I will say what I am not claiming. Public revision dated 3 October 2026. Do not use the September review deck.");
}

// 2 — rule
{
  const s = pres.addSlide();
  paper(s);
  h1(s, "Hold the stack fixed. Change one factor.");
  s.addText("If the stack also moves, the number is not an algorithm result. That rule is applied twice.", {
    x: 0.48, y: 0.95, w: 12.2, h: 0.4,
    fontFace: F.body, fontSize: 16, color: C.body, margin: 0,
  });
  const cards = [
    ["P1–P8", "What does GRPO do?", "Libraries, scales, group sizes, length, and a reporting standard. Only Tinker and TRL produced completed runs, and they used different base checkpoints."],
    ["E1–E14", "What does one actor score?", "One Tinker-trained checkpoint, 40 steps, fourteen native suites. Scores describe that checkpoint. They are not a post-training delta."],
  ];
  cards.forEach((c, i) => {
    const x = 0.48 + i * 6.35;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.65, w: 6.05, h: 3.7, fill: { color: C.card },
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.65, w: 0.1, h: 3.7, fill: { color: i === 0 ? C.forest : C.gold },
    });
    s.addText(c[0], {
      x: x + 0.35, y: 1.9, w: 5.4, h: 0.4,
      fontFace: F.body, fontSize: 14, color: C.gold, margin: 0,
    });
    s.addText(c[1], {
      x: x + 0.35, y: 2.3, w: 5.4, h: 0.9,
      fontFace: F.title, fontSize: 26, color: C.ink, margin: 0, valign: "top",
    });
    s.addText(c[2], {
      x: x + 0.35, y: 3.35, w: 5.4, h: 1.75,
      fontFace: F.body, fontSize: 16, color: C.body, margin: 0, valign: "top",
    });
  });
  footer(s, 2);
  s.addNotes("Source: thesis §9.1 and §9.3. If asked whether this is a finished seven-library benchmark, the answer is no. Comparative framework claims stop at the completed runs.");
}

// 3 — phenomenon
{
  const s = pres.addSlide();
  paper(s);
  h1(s, "Solved prompts spend compute and teach nothing");
  s.addText("Eight samples, all correct. Group mean is 1, group standard deviation is 0, every advantage is 0. All-wrong groups do the same. ZVF counts how often that happens. It does not say whether the group was mastered or failed.", {
    x: 0.48, y: 1.0, w: 7.5, h: 2.15,
    fontFace: F.body, fontSize: 16, color: C.body, margin: 0,
  });
  const stats = [
    ["0.72–0.77", "Phase-1 main configuration"],
    ["0.16–0.84", "Range across measured runs"],
    ["ρ = 0.27", "vs held-out outcome, n = 23"],
    ["CI crosses 0", "[−0.37, 0.88]"],
  ];
  stats.forEach((row, i) => {
    const y = 1.0 + i * 1.35;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 8.3, y, w: 4.5, h: 1.2, fill: { color: C.card },
    });
    s.addText(row[0], {
      x: 8.5, y: y + 0.12, w: 4.15, h: 0.55,
      fontFace: F.title, fontSize: 24, color: C.maroon, margin: 0,
    });
    s.addText(row[1], {
      x: 8.5, y: y + 0.68, w: 4.15, h: 0.35,
      fontFace: F.body, fontSize: 13, color: C.muted, margin: 0,
    });
  });
  s.addText("Three stored GSM8K tensors (Qwen3-8B, sampling only) reproduce 0.130, 0.190 and 0.155, pooled 0.1583. After mean reward is regressed out, r = +0.21, n = 17, and that interval crosses zero.", {
    x: 0.48, y: 5.55, w: 7.5, h: 1.2,
    fontFace: F.body, fontSize: 15, color: C.body, margin: 0,
  });
  footer(s, 3);
  s.addNotes("RQ1. Chapter 6 §6.3.4. The three tensors are sampling-only, G = 8, 200 problems per seed, not a training delta. Pooled ZVF is 0.1583. The residual after mean reward is r = +0.21, 95% CI [−0.31, +0.62], n = 17. ZVF is a diagnostic of advantage-signal starvation, not a predictor. High ZVF is also what mastery looks like.");
}

// 4 — four answers
{
  const s = pres.addSlide();
  paper(s);
  h1(s, "Four questions, at the strength of the evidence");
  const items = [
    ["RQ1  ZVF", "Recomputable. Association with held-out accuracy is not distinguishable from zero."],
    ["RQ2  Labels", "Unanswered at this budget. Ceiling nulls are not equivalence. One unsaturated gap is suggestive and baseline-specific."],
    ["RQ3  Manifests", "Yes for the one pair tested. stackdiff flags the open-versus-closed DAPO pair from manifests alone. Whether that flag matters for outcomes was not tested beyond that pair."],
    ["RQ4  Governance", "Receipt-bound scores and a terminal state for every lane. No post-training delta."],
  ];
  items.forEach((item, i) => {
    const x = 0.48 + (i % 2) * 6.35;
    const y = 1.15 + Math.floor(i / 2) * 2.75;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 6.05, h: 2.5, fill: { color: C.card },
    });
    s.addText(item[0], {
      x: x + 0.28, y: y + 0.28, w: 5.5, h: 0.45,
      fontFace: F.title, fontSize: 20, color: C.ink, margin: 0,
    });
    s.addText(item[1], {
      x: x + 0.28, y: y + 0.9, w: 5.5, h: 1.25,
      fontFace: F.body, fontSize: 16, color: C.body, margin: 0,
    });
  });
  footer(s, 4);
  s.addNotes("These four sentences are the abstract of the defense. If time is cut, stay on this slide and the next two. Source: §9.3. RQ3 is one manifest pair only. Do not say stackdiff predicts a score change.");
}

// 5 — suggestive gap
{
  const s = pres.addSlide();
  paper(s);
  h1(s, "One gap is suggestive. It is not a PPO result.");
  s.addText("+5.0 pp", {
    x: 0.48, y: 1.2, w: 5.5, h: 1.15,
    fontFace: F.title, fontSize: 60, color: C.maroon, margin: 0,
  });
  s.addText("GRPO, group size 8, minus a minimal learned-critic PPO.\nQwen2.5-1.5B, GSM8K chain-of-thought, five paired seeds.", {
    x: 0.48, y: 2.5, w: 6.2, h: 0.9,
    fontFace: F.body, fontSize: 16, color: C.body, margin: 0,
  });
  const pills = [
    ["Paired t", "p = 0.016"],
    ["Exact sign-flip", "p = 0.0625"],
    ["G8 minus G2", "−0.5 pp, inconclusive"],
  ];
  pills.forEach((p, i) => {
    const y = 3.65 + i * 0.95;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.48, y, w: 6.0, h: 0.8, fill: { color: C.card },
    });
    s.addText(p[0], {
      x: 0.7, y: y + 0.18, w: 2.6, h: 0.45,
      fontFace: F.body, fontSize: 15, color: C.muted, margin: 0,
    });
    s.addText(p[1], {
      x: 3.2, y: y + 0.16, w: 3.0, h: 0.48,
      fontFace: F.title, fontSize: 18, color: C.ink, margin: 0,
    });
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 7.15, y: 1.2, w: 5.65, h: 5.3, fill: { color: C.card },
  });
  s.addText("Say this before the number", {
    x: 7.4, y: 1.45, w: 5.2, h: 0.35,
    fontFace: F.body, fontSize: 13, color: C.gold, margin: 0,
  });
  s.addText("The critic was untuned. Prompt exposure differed: 16 prompts × 8 completions for GRPO, against 128 prompts for PPO. The 200-token cap can still bind.\n\nOn the saturated 0.5B addition task both methods sit at 0.99. Paired difference −0.002, p = 0.374. A null at the ceiling cannot separate no effect from no headroom.\n\nG = 8 versus G = 2 on the unsaturated rerun is −0.005, interval crossing both zero and the ±0.02 equivalence margin.", {
    x: 7.4, y: 1.95, w: 5.15, h: 4.2,
    fontFace: F.body, fontSize: 15, color: C.body, margin: 0, valign: "top",
  });
  footer(s, 5);
  s.addNotes("§6.5.5 and §9.3. Interval on the +5.0 pp gap is [+1.5, +8.5] percentage points. Do not say GRPO beats PPO. Do not say the methods are equivalent. Holm family p = 0.25 belongs to audit H7, not to this gap. Both contrasts can show a sign-flip p of 0.0625 because five pairs agreed.");
}

// 6 — what failed
{
  const s = pres.addSlide();
  paper(s);
  h1(s, "What the evidence does not support");
  const gone = [
    ["No scaling law", "Flat fit over about 2.4 orders of magnitude. Scale is collinear with the recipe. Single-seed anchors."],
    ["No length-bias finding", "In the controlled comparison, completion length drifts shorter. GRPO and Dr. GRPO are indistinguishable on held-out gain."],
    ["No group-size optimum", "G = 2 through 16 are not separable on the three-seed sweep. G = 16 versus G = 2 fails equivalence at ±0.02 (TOST p = 0.056)."],
    ["No decisive audit headline", "Seven audited headlines: none decisive, one suggestive (H7, iter-136 efficiency). Holm on that family of eight moves the p to 0.25. That is not the +5.0 pp gap."],
  ];
  gone.forEach((g, i) => {
    const x = 0.48 + (i % 2) * 6.35;
    const y = 1.15 + Math.floor(i / 2) * 2.75;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 6.05, h: 2.5, fill: { color: C.card },
    });
    s.addText(g[0], {
      x: x + 0.28, y: y + 0.25, w: 5.5, h: 0.5,
      fontFace: F.title, fontSize: 20, color: C.brick, margin: 0,
    });
    s.addText(g[1], {
      x: x + 0.28, y: y + 0.9, w: 5.5, h: 1.3,
      fontFace: F.body, fontSize: 15, color: C.body, margin: 0,
    });
  });
  footer(s, 6);
  s.addNotes("H7 is the iter-136 late-training efficiency contrast. Its uncorrected sign-flip is 0.0625, and Holm across the family of eight raises that p to 0.25. The +5.0 pp GSM8K gap on the previous slide is a different contrast with the same sign-flip p. Do not merge them. If an examiner quotes an older slide that says 99.9% Tinker versus 73.4% TRL, stop them. Those are training rewards on different model sizes. The same-model pair still mixes Base and Instruct checkpoints. §9.2.");
}

// 7 — campaign
{
  const s = pres.addSlide();
  paper(s);
  h1(s, "Campaign scores are checkpoint scores");
  s.addTable(
    [
      [
        { text: "Scope", options: { fill: { color: C.maroon }, color: C.white, bold: true } },
        { text: "Score", options: { fill: { color: C.maroon }, color: C.white, bold: true } },
        { text: "Read it as", options: { fill: { color: C.maroon }, color: C.white, bold: true } },
      ],
      ["VerilogEval, full coverage", "129 / 312 = 41.35%", "150 of 183 failures are extraction failures, most likely from the 1,024-token budget"],
      ["Omni-MATH, all dispositions", "2,271 / 4,428 = 51.29%", "51.31% is the 4,426 judged rows only"],
      ["SWE-bench Pro", "2 / 731 = 0.274%", "593 of 713 patches corrupt for git apply"],
      ["WebArena", "90 / 812 = 0.111", "Lower bound. 108 judge tasks ungraded"],
      ["AgentDojo utility", "88 / 97 = 0.907", "Benign utility only. Not an attack result"],
    ].map((row, idx) => {
      if (idx === 0) return row;
      const bg = idx % 2 === 0 ? "F3EDE4" : C.white;
      return row.map((cell) => ({
        text: cell,
        options: { fill: { color: bg }, color: C.ink },
      }));
    }),
    {
      x: 0.48, y: 1.1, w: 12.35, h: 3.7,
      colW: [3.7, 3.5, 5.15],
      border: [{ pt: 0, color: C.paper }, { pt: 0, color: C.paper }, { pt: 0, color: C.paper }, { pt: 0, color: C.paper }],
      fontFace: F.body,
      fontSize: 13,
      color: C.ink,
      valign: "middle",
      align: "left",
    }
  );
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.48, y: 5.05, w: 12.35, h: 1.75, fill: { color: "F3E6D4" },
  });
  s.addText("Do not average these rows. Denominators count different things. The paired base-versus-trained comparison, 5 to 151 items per lane, is inconclusive in every lane. Original contracts and replacement scopes stay in separate columns.", {
    x: 0.72, y: 5.28, w: 11.9, h: 1.3,
    fontFace: F.body, fontSize: 16, color: C.ink, margin: 0,
  });
  footer(s, 7);
  s.addNotes("These are harness scores. VerilogEval: the 150 extraction failures come from the token cap with thinking on, not format; same adapter 33/50 thinking off at 4,096 tokens vs 23/50 on a 50-item subset, engine and temperature also changed. Omni-MATH: capped answers 34.5% correct, uncapped 85.9%, confounded by difficulty. SWE-bench Pro: 593 of 713 patches failed git apply as corrupt hunks, so tests ran on the unpatched base; Wilson interval 0.08% to 0.99%. Table C.3 is the status table if they want every lane. Six original contracts are CLOSED_EXTERNAL: SDAB, BinaryAudit payload, LifeSciBench package, AgentHarm, AppBench, FrontierMath. Say the reopen condition, not a pending label.");
}

// 8 — C1
{
  const s = pres.addSlide();
  paper(s);
  h1(s, "C1 draws a boundary. It does not repair the score.");
  const nums = [
    ["11 / 64", "Frozen endpoint", "17.19%. Wilson interval 9.88–28.21%. Parser v2. Not changed after review."],
    ["3 / 64", "Numerical witnesses", "Assistant-assisted. Separate count. Not added back into 11."],
    ["1 case", "Wrong reference", "Eight correct opening answers, then a match to an incorrect checkpoint reference."],
  ];
  nums.forEach((n, i) => {
    const x = 0.48 + i * 4.2;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.35, w: 4.0, h: 3.55, fill: { color: C.card },
    });
    s.addText(n[0], {
      x: x + 0.25, y: 1.55, w: 3.5, h: 0.7,
      fontFace: F.title, fontSize: 32, color: C.maroon, margin: 0, valign: "top",
    });
    s.addText(n[1], {
      x: x + 0.25, y: 2.3, w: 3.5, h: 0.45,
      fontFace: F.title, fontSize: 18, color: C.ink, margin: 0, valign: "top",
    });
    s.addText(n[2], {
      x: x + 0.25, y: 2.9, w: 3.5, h: 1.7,
      fontFace: F.body, fontSize: 15, color: C.body, margin: 0, valign: "top",
    });
  });
  footer(s, 8);
  s.addNotes("The cohort is not a random benchmark sample. Assistant review is not independent human validation. The witnesses reject certainty from eight equal wrong numbers on this checkpoint. They are not a gradient, a training gain, or a controller result. Santana's first successful number came from an incorrect argument.");
}

// 9 — contribution
{
  const s = pres.addSlide();
  paper(s);
  h1(s, "What I will defend as the contribution");
  const bits = [
    ["01", "A diagnostic", "ZVF and pairwise contrast density, recomputable from stored tensors, with the association stated at the width of its interval."],
    ["02", "A minimum report", "Eight items: seven manifest fields and one evaluation item. A label without its stack is not an experimental specification."],
    ["03", "A ledger with gates", "Every campaign lane has a terminal state and a reopen condition. Missing, simulated, and measured results stay distinct."],
  ];
  bits.forEach((b, i) => {
    const y = 1.15 + i * 1.85;
    s.addText(b[0], {
      x: 0.48, y: y, w: 1.2, h: 0.7,
      fontFace: F.title, fontSize: 28, color: C.gold, margin: 0,
    });
    s.addText(b[1], {
      x: 1.9, y: y, w: 10.6, h: 0.48,
      fontFace: F.title, fontSize: 22, color: C.ink, margin: 0,
    });
    s.addText(b[2], {
      x: 1.9, y: y + 0.55, w: 10.6, h: 0.9,
      fontFace: F.body, fontSize: 16, color: C.body, margin: 0,
    });
  });
  footer(s, 9);
  s.addNotes("The novelty sentence, if asked: a targeted search through July 2026 did not find an equivalent GRPO-specific machine-readable run datasheet with rollout provenance. That is an evidence-bounded claim, not a proof that none exists. §9.3.");
}

// 10 — examiner
{
  const s = pres.addSlide();
  paper(s);
  h1(s, "The four pushes I expect");
  const pushes = [
    ["Seed count", "No family reaches Henderson’s ten seeds. Single-seed Tinker figures are descriptive. The open-source contrasts use three to eight."],
    ["Closed trainer", "Tinker’s server-side loss, normalisation and hardware are not visible. Those results are the platform’s GRPO, not an abstract algorithm."],
    ["Lost sources", "Training raw logs and the E4 raw job receipt are unresolved. A sealed request without its execution source is not a rerun."],
    ["Power", "Inconclusive is the result at this n. I will not upgrade it to “no effect” in the room."],
  ];
  pushes.forEach((p, i) => {
    const y = 1.1 + i * 1.4;
    s.addShape(pres.shapes.OVAL, {
      x: 0.55, y: y + 0.15, w: 0.42, h: 0.42, fill: { color: C.maroon },
    });
    s.addText(String(i + 1), {
      x: 0.55, y: y + 0.2, w: 0.42, h: 0.34,
      fontFace: F.body, fontSize: 14, color: C.white, align: "center", margin: 0,
    });
    s.addText(p[0], {
      x: 1.25, y: y, w: 11.3, h: 0.4,
      fontFace: F.title, fontSize: 18, color: C.ink, margin: 0,
    });
    s.addText(p[1], {
      x: 1.25, y: y + 0.42, w: 11.3, h: 0.7,
      fontFace: F.body, fontSize: 15, color: C.body, margin: 0,
    });
  });
  footer(s, 10);
  s.addNotes("Threat table 9.1 if they want the full list. Horizons are 30–50 steps for the Tinker runs: no asymptotic claim. Environmental accounting is uncertain because Tinker exposes no hardware telemetry.");
}

// 11 — demo
{
  const s = pres.addSlide();
  paper(s);
  h1(s, "Ninety seconds, then stop");
  const steps = [
    ["1", "Scope", "Mechanism and artifact integrity. This laptop is not retraining the model."],
    ["2", "Group signal", "Mixed rewards get signed advantages. Equal rewards, right or wrong, get zero."],
    ["3", "The hash", "The page recomputes the recorded artifact and shows the SHA-256 of the reviewed bytes."],
    ["4", "The boundary", "A pass does not prove generalisation or a gain over the base model."],
  ];
  steps.forEach((st, i) => {
    const y = 1.15 + i * 1.2;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.48, y, w: 8.15, h: 1.05, fill: { color: C.card },
    });
    s.addText(st[0], {
      x: 0.68, y: y + 0.25, w: 0.5, h: 0.5,
      fontFace: F.title, fontSize: 22, color: C.maroon, margin: 0,
    });
    s.addText(st[1], {
      x: 1.3, y: y + 0.1, w: 7.0, h: 0.35,
      fontFace: F.title, fontSize: 16, color: C.ink, margin: 0,
    });
    s.addText(st[2], {
      x: 1.3, y: y + 0.48, w: 7.0, h: 0.42,
      fontFace: F.body, fontSize: 14, color: C.body, margin: 0,
    });
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 8.9, y: 1.15, w: 3.95, h: 4.8, fill: { color: C.ink },
  });
  s.addText("BEFORE THE ROOM", {
    x: 9.1, y: 1.4, w: 3.55, h: 0.3,
    fontFace: F.body, fontSize: 12, color: C.gold, margin: 0,
  });
  s.addText("demo.sh --self-test\ndemo.sh\n\nOpen the HTML file directly if the port is busy.\n\nIf the SHA fails, show the failure. Do not skip it.", {
    x: 9.1, y: 1.9, w: 3.55, h: 3.6,
    fontFace: F.body, fontSize: 14, color: C.white, margin: 0,
  });
  footer(s, 11);
  s.addNotes("Commands live in submission/demo/DEFENSE_RUNBOOK.md. Fallback folder: submission/demo/defense_fallback. The 68.75% figure on the dashboard is the mean of 80 recorded binary rewards in one artifact. It is not a benchmark accuracy.");
}

// 12 — close
{
  const s = pres.addSlide();
  s.background = { color: C.ink };
  s.addText("The measurements are informative.\nThe algorithmic levers, at this budget,\nare noise-limited.", {
    x: 0.7, y: 1.45, w: 11.8, h: 2.3,
    fontFace: F.title, fontSize: 32, color: C.white, margin: 0,
  });
  s.addText("A label without its stack is not a result I will defend.", {
    x: 0.7, y: 4.05, w: 11, h: 0.5,
    fontFace: F.body, fontSize: 18, color: "E7C9C4", margin: 0,
  });
  s.addText("Questions", {
    x: 0.7, y: 5.5, w: 4, h: 0.5,
    fontFace: F.title, fontSize: 22, color: C.white, margin: 0,
  });
  s.addText("Arvind C R  ·  Ramesh Prakash Guledgudd", {
    x: 6.5, y: 5.55, w: 6.1, h: 0.4,
    fontFace: F.body, fontSize: 14, color: "C8C2BA", align: "right", margin: 0,
  });
  s.addNotes("Stop. Do not add a new claim in the last thirty seconds. If the first question is hostile, answer the number, then the limit, then stop.");
}

pres.writeFile({ fileName: "reports/final_defense_2026-10-03/TinkerRL_Phase2_Defense_2026-10-03.pptx" })
  .then(() => console.log("wrote deck"))
  .catch((err) => {
    console.error(err);
    process.exit(1);
  });
