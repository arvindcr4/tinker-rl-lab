import React from "react";
import { ARM_NAME, CONTRASTS, DATA, MODEL, SD, SS, nSeeds, zvfStats } from "../data.js";
import { pct, pfmt, sgn } from "../stats.js";
import { SubmissionOverview } from "./Submissions.jsx";

export default function Overview() {
  const c = CONTRASTS.find((c) => c.id === "g8-ppo"), s = c?.stats;
  const zs = zvfStats("grpo_g8", 8);
  const same = Object.values(SD.pairs).filter((p) => p.same_label);
  const au = DATA.audit.summary;
  const cards = [
    {
      k: "GRPO vs PPO-style, same stack", big: s ? sgn(s.md, 3) : "Pending", go: "samestack",
      pill: [c?.equivalent ? "good" : "warn", c?.equivalent ? "Adjusted equivalence established" : "No confirmatory verdict"],
      text: s ? `${s.n} validated paired seeds; ${MODEL.completion.selected}/${MODEL.completion.planned ?? "unknown"} planned runs selected, ${MODEL.completion.missing ?? "unknown"} missing; mean difference ${sgn(s.md, 3)}, 95% difference CI [${sgn(s.lo, 3)}, ${sgn(s.hi, 3)}], 90% equivalence CI [${sgn(s.eqLo, 3)}, ${sgn(s.eqHi, 3)}]. Adjusted combined TOST p = ${pfmt(s.holmTost)}. ${c.decision}.` :
        `Canonical analysis ${MODEL.analysis?.status || "unavailable"}. Historical means are descriptive; missing run identity, protocol or ordered-item hashes cannot be repaired by matching seed numbers.`,
    },
    {
      k: "Zero-variance groups", big: pct(zs.mean, 0), go: "zvf", pill: ["info", "Mean ZVF, GRPO G = 8"],
      text: `Share of prompts per step whose 8 answers all got the same reward, giving zero group-relative advantage, across ${zs.n} training steps. ${pct(zs.above, 0)} of steps sit above the equal-difficulty curve.`,
    },
    {
      k: "Same label, different stack", big: `${same.filter((p) => p.level === 5).length}/${same.length}`, go: "stackdiff", pill: ["bad", "R5 outcome-critical"],
      text: `Pairs of registry stacks that share a method label (for example "grpo") yet differ in a known outcome-critical field.`,
    },
    {
      k: "Thesis number audit", big: `${au.failed_checks} failed`, go: "audit",
      pill: [DATA.audit.status === "PASS" ? "good" : "warn", DATA.audit.status],
      text: `${au.passed_checks} of ${au.registered_token_occurrences} registered values pass, ${au.missing_artifact_values} have evidence outside the public checkout, ${au.unbound_values} unbound.`,
    },
  ];
  const st = SS.setup;
  return (
    <section className="view">
      <div>
        <h2>Explore the evidence</h2>
        <p className="small">Start with a finding, then inspect the analysis and its limits.</p>
      </div>
      <div className="overview-grid">
        {cards.map((c) => (
          <a key={c.k} className="panel finding" href={`#${c.go}`}>
            <h3 className="eyebrow">{c.k}</h3>
            <div className="big">{c.big}</div>
            <div><span className={`pill ${c.pill[0]}`}>{c.pill[1]}</span></div>
            <p className="small">{c.text}</p>
            <span className="card-link">Explore this finding <span aria-hidden="true">→</span></span>
          </a>
        ))}
      </div>
      <SubmissionOverview />
      <div className="panel">
        <h3>What this project built</h3>
        <p>A measurement system for RL post-training of language models: a same-stack GRPO-versus-PPO-style harness on GSM8K, zero-variance-fraction (ZVF) telemetry for group-relative advantages, a stack-provenance registry with a <span className="mono">stackdiff</span> tool, and a number audit that reports which registered thesis values are bound to a recomputation or source artifact, and which remain missing or unbound.</p>
      </div>
      <div className="panel">
        <h3>Same-stack harness (v2)</h3>
        <dl className="stat">
          <dt>Model</dt><dd>{st.model}</dd>
          <dt>Training</dt><dd>{st.n_steps} steps × {st.n_gen} generations, {st.k_epochs} epochs/step, lr {st.lr}</dd>
          <dt>Response cap</dt><dd>{st.max_new} tokens</dd>
          <dt>Held-out eval</dt><dd>{st.n_eval} GSM8K test items, greedy, strict boxed answer</dd>
          <dt>Historical/descriptive records loaded</dt><dd>{SS.runs.length} ({Object.keys(ARM_NAME).map((a) => `${a} ${nSeeds(a)}`).join(", ")})</dd>
        </dl>
      </div>
    </section>
  );
}
