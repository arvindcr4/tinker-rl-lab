import React, { useState } from "react";
import { pct, zvfCurve } from "../stats.js";

const PRESETS = {
  "All correct": (G) => Array(G).fill(1),
  "One correct": (G) => Array.from({ length: G }, (_, i) => +(i === 1)),
  "Half correct": (G) => Array.from({ length: G }, (_, i) => +(i % 2 === 0)),
  "All wrong": (G) => Array(G).fill(0),
};
const SIZES = [2, 4, 8, 16];

function AdvantageChart({ grpo, dr }) {
  const W = 520, H = 260, L = 40, R = 10, T = 14, B = 26, lim = 3;
  const y = (v) => T + ((H - T - B) * (lim - v)) / (2 * lim);
  const slot = (W - L - R) / grpo.length, bw = slot * 0.34;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Advantage per answer">
      {[-3, -2, -1, 0, 1, 2, 3].map((v) => (
        <g key={v}>
          <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke="var(--line)" strokeWidth={v ? 1 : 1.5} />
          <text x={L - 6} y={y(v) + 4} textAnchor="end">{v}</text>
        </g>
      ))}
      {grpo.map((g, i) => {
        const x0 = L + i * slot + slot * 0.12;
        return (
          <g key={i}>
            {[[0, Math.max(-lim, Math.min(lim, g)), "var(--grpo)"], [1, dr[i], "var(--ppo)"]].map(([k, v, fill]) => (
              <rect key={k} x={x0 + k * (bw + 2)} y={y(Math.max(0, v))} width={bw} height={Math.max(Math.abs(y(v) - y(0)), 0.5)} fill={fill} rx={2} />
            ))}
            <text x={x0 + bw} y={H - 8} textAnchor="middle">#{i + 1}</text>
          </g>
        );
      })}
    </svg>
  );
}

export default function AdvantageLab() {
  const [rewards, setRewards] = useState([1, 0, 0, 1, 0, 0, 0, 0]);
  const G = rewards.length, mean = rewards.reduce((a, b) => a + b, 0) / G;
  const std = Math.sqrt(rewards.reduce((a, r) => a + (r - mean) ** 2, 0) / G);
  const grpo = rewards.map((r) => (std > 0 ? (r - mean) / (std + 1e-4) : 0));
  const dr = rewards.map((r) => r - mean);
  const flip = (i) => setRewards(rewards.map((r, j) => (j === i ? 1 - r : r)));
  const resize = (n) => setRewards(Array.from({ length: n }, (_, i) => rewards[i % G] ?? 0));

  return (
    <section className="view">
      <div className="panel">
        <h2>Where GRPO's learning signal comes from</h2>
        <p className="small">GRPO samples a group of answers to one prompt and scores each one. Each answer's advantage is its reward minus the group mean, divided by the group's standard deviation. Dr. GRPO drops the division. Tap an answer to flip it between correct (1) and wrong (0).</p>
      </div>
      <div className="grid2">
        <div className="panel">
          <div className="row between" style={{ marginBottom: 10 }}>
            <h3 style={{ margin: 0 }}>One prompt, a group of {G} sampled answers</h3>
            <div className="seg" aria-label="Group size">
              {SIZES.map((n) => <button key={n} type="button" aria-pressed={n === G} onClick={() => resize(n)}>G = {n}</button>)}
            </div>
          </div>
          <div className="coins" style={{ gridTemplateColumns: `repeat(${Math.min(G, 8)}, minmax(0, 64px))` }}>
            {rewards.map((r, i) => (
              <button key={i} type="button" className={`coin ${r ? "on" : ""}`} aria-pressed={!!r}
                aria-label={`Answer ${i + 1}: ${r ? "correct" : "wrong"}`} onClick={() => flip(i)}>{r}</button>
            ))}
          </div>
          <div className="row" style={{ marginTop: 14 }}>
            {Object.entries(PRESETS).map(([k, v]) => (
              <button key={k} type="button" onClick={() => setRewards(v(G))}>{k}</button>
            ))}
          </div>
          <dl className="stat" style={{ marginTop: 14 }}>
            <dt>Group mean reward</dt><dd>{mean.toFixed(3)}</dd>
            <dt>Group std</dt><dd>{std.toFixed(3)}</dd>
            <dt>Answers with nonzero advantage</dt><dd>{grpo.filter((a) => Math.abs(a) > 1e-9).length} of {G}</dd>
            <dt>Chance a fresh group is all-same</dt><dd>{pct(zvfCurve(mean, G))} at p = {mean.toFixed(2)}</dd>
          </dl>
        </div>
        <div className="panel">
          <h3>Per-answer advantage</h3>
          <div className="legend">
            <span><i style={{ background: "var(--grpo)" }} />GRPO (÷ std)</span>
            <span><i style={{ background: "var(--ppo)" }} />Dr. GRPO (mean-centred)</span>
          </div>
          <div className="chart-box"><AdvantageChart grpo={grpo} dr={dr} /></div>
          {std === 0 ? (
            <div className="callout">Every answer has the same reward, so every advantage is zero. This group is a zero-variance group: it costs {G} generations and teaches the model nothing.</div>
          ) : (
            <p className="small">Rare correct answers get large positive advantages under GRPO because the small group std scales them up. Dr. GRPO keeps them proportional to the reward gap.</p>
          )}
        </div>
      </div>
    </section>
  );
}
