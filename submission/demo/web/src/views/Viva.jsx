import React from "react";
import { CONTRASTS, DATA, SS, nSeeds, zvfStats } from "../data.js";
import { pct, pfmt, sgn } from "../stats.js";

// The hardest questions two independent reviewers raised on the thesis, answered from the
// thesis text, each linked to the view that shows the evidence. Numbers are computed, not typed.
export default function Viva() {
  const s = CONTRASTS.find((c) => c.a === "grpo_g8" && c.b === "ppo").stats;
  const z = zvfStats("grpo_g8", 8), ck = DATA.grader_check;
  const QA = [
    {
      q: "What is your individual contribution, given the Semester-3 group work and the AI agents?",
      a: <>The group built the multi-framework infrastructure and baseline integrations, frozen at tag <span className="mono">capstone-final-2026-04-25</span>. The research direction, the experimental expansion, the statistical analyses, the P1–P8 reports, the registry, the adaptive group-size controller, the fraud study and the E1–E14 campaign are mine. AI agents wrote much of the code and drafts under my direction. Every reported number is tied to a stored run and can be re-checked, as the evidence audit and the re-graded answers show.</>,
      go: [["audit", "Evidence audit"], ["answers", "Real answers"]],
    },
    {
      q: "Table 6.2 is provisional. Why should we accept the GRPO vs PPO conclusion?",
      a: <>Over the {s.n} completed paired seeds, GRPO G = 8 minus PPO is {sgn(s.md)} with 95% CI [{sgn(s.lo)}, {sgn(s.hi)}], and TOST at ±0.02 gives p = {pfmt(s.tost)}. The conclusion is limited to "no detectable GRPO advantage on this configuration". PPO has {nSeeds("ppo")} of {SS.seeds_planned} seeds and the exposure-matched arm is pending, so the result is labelled provisional rather than final. The earlier +0.050 gap came from a 200-token cap and a cold critic; PPO ended below its own start.</>,
      go: [["samestack", "GRPO vs PPO"], ...(DATA.platform ? [["platform", "Platform calibration"]] : [])],
    },
    {
      q: "Is ZVF a contribution, or just the known binomial identity?",
      a: <>The identity p<sup>G</sup> + (1−p)<sup>G</sup> is elementary, and the U-shape follows from it. The contribution is measuring ZVF from stored per-step tensors as training telemetry: mean ZVF is {pct(z.mean)} for GRPO G = 8 here, and {pct(z.above, 0)} of steps sit above the equal-difficulty curve because prompts differ in difficulty. The historical correlation with outcomes is withdrawn; on the eight non-simulated cells it is ρ = −0.37 (p = 0.36).</>,
      go: [["zvf", "Zero-variance groups"], ["advantage", "Advantage lab"]],
    },
    {
      q: "The title says multi-framework. What cross-framework result do you claim?",
      a: <>None. Only Tinker and TRL produced completed runs, on different base checkpoints, so no framework ranking is claimed. That is the reason for the registry: stacks sharing the label "grpo" differ in outcome-critical fields, which <span className="mono">stackdiff</span> makes visible.</>,
      go: [["stackdiff", "Stack diff"]],
    },
    {
      q: "Does the E1–E14 campaign show that GRPO post-training helped?",
      a: <>No. The actor was trained for 40 steps, and the paired trained-versus-base comparison (Table 8.B) is inconclusive in every lane. The campaign contributes evaluation governance: denominators, receipts and terminal states that are never pooled, not a post-training gain.</>,
      go: [],
    },
    {
      q: "How do we know the grader that produced these accuracies is right?",
      a: <>An independent balanced-brace parser re-grades every stored completion. In this page a port of it reproduces the stored score on {ck ? `${ck.completions.toLocaleString("en")} answers from ${ck.runs} runs with ${ck.disagreements} disagreements` : "the stored answers"}, and you can test it on any input.</>,
      go: [["answers", "Real answers"]],
    },
  ];
  return (
    <section className="view">
      <div className="panel">
        <h2>Hard questions, answered from the evidence</h2>
        <p className="small">These are the questions two independent reviewers expected an examiner to ask. Each answer follows the thesis text and links to the view that shows the evidence. The numbers in the answers are computed from the same data as the rest of the page.</p>
      </div>
      {QA.map(({ q, a, go }, i) => (
        <div key={i} className="panel qa">
          <div className="eyebrow">Question {i + 1}</div>
          <h3>{q}</h3>
          <p>{a}</p>
          {go.length > 0 && <div className="row">{go.map(([id, label]) => <a key={id} className="next" href={`#${id}`}>{label} →</a>)}</div>}
        </div>
      ))}
    </section>
  );
}
