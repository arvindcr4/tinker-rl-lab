import React from "react";
import { ARM_NAME, CONTRASTS, DATA, SD, SS, nSeeds, provisional, zvfStats } from "../data.js";
import { pct, pfmt, sgn } from "../stats.js";

export default function Overview() {
  const s = CONTRASTS.find((c) => c.a === "grpo_g8" && c.b === "ppo").stats;
  const zs = zvfStats("grpo_g8", 8);
  const same = Object.values(SD.pairs).filter((p) => p.same_label);
  const au = DATA.audit.summary;
  const cards = [
    {
      k: "GRPO vs PPO, same stack", big: sgn(s.md, 3), go: "samestack",
      pill: s.lo < 0 && s.tost < 0.05 ? ["good", "Equivalent within ±0.02"] : ["warn", "Not settled"],
      text: `GRPO G = 8 minus PPO in held-out GSM8K accuracy, ${s.n} paired seeds, 95% CI [${sgn(s.lo, 3)}, ${sgn(s.hi, 3)}]. TOST p = ${pfmt(s.tost)}.` +
        (provisional() ? ` Provisional: PPO has ${nSeeds("ppo")} of ${SS.seeds_planned} seeds and the exposure-matched arm has ${nSeeds("ppo_8x8")}.` : ""),
    },
    {
      k: "Zero-variance groups", big: pct(zs.mean, 0), go: "zvf", pill: ["info", "Mean ZVF, GRPO G = 8"],
      text: `Share of prompts per step whose 8 answers all got the same reward, so contributed no gradient, across ${zs.n} training steps. ${pct(zs.above, 0)} of steps sit above the equal-difficulty curve.`,
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
      <div className="grid3">
        {cards.map((c) => (
          <a key={c.k} className="panel finding" href={`#${c.go}`}>
            <div className="eyebrow">{c.k}</div>
            <div className="big">{c.big}</div>
            <div><span className={`pill ${c.pill[0]}`}>{c.pill[1]}</span></div>
            <p className="small">{c.text}</p>
          </a>
        ))}
      </div>
      <div className="panel">
        <h3>What this project built</h3>
        <p>A measurement system for RL post-training of language models: a same-stack GRPO-versus-PPO harness on GSM8K, zero-variance-fraction (ZVF) telemetry for group-relative advantages, a stack-provenance registry with a <span className="mono">stackdiff</span> tool, and a number audit that binds every printed thesis value to a recomputation or a source artifact.</p>
      </div>
      <div className="panel">
        <h3>Same-stack harness (v2)</h3>
        <dl className="stat">
          <dt>Model</dt><dd>{st.model}</dd>
          <dt>Training</dt><dd>{st.n_steps} steps × {st.n_gen} generations, {st.k_epochs} epochs/step, lr {st.lr}</dd>
          <dt>Response cap</dt><dd>{st.max_new} tokens</dd>
          <dt>Held-out eval</dt><dd>{st.n_eval} GSM8K test items, greedy, strict boxed answer</dd>
          <dt>Runs loaded</dt><dd>{SS.runs.length} ({Object.keys(ARM_NAME).map((a) => `${a} ${nSeeds(a)}`).join(", ")})</dd>
        </dl>
      </div>
    </section>
  );
}
