import React, { useState } from "react";
import { ARM_NAME, CONTRASTS, DATA, SHORT, SS, byArm, nSeeds } from "../data.js";
import { pct, pfmt, sgn } from "../stats.js";

function DiffChart({ c }) {
  const s = c.stats, n = c.seeds.length;
  const W = 520, H = 300, L = 48, R = 12, T = 14, B = 34;
  const lim = Math.max(0.03, ...c.diffs.map(Math.abs), Math.abs(s.lo), Math.abs(s.hi)) * 1.15;
  const y = (v) => T + ((H - T - B) * (lim - v)) / (2 * lim);
  const x = (i) => L + ((W - L - R) * (i + 0.5)) / (n + 1.6);
  const step = lim > 0.06 ? 0.02 : 0.01, ticks = [];
  for (let v = -Math.floor(lim / step) * step; v <= lim + 1e-9; v += step) ticks.push(v);
  const xm = x(n + 0.3);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Paired differences per seed">
      <rect x={L} y={y(0.02)} width={W - L - R} height={y(-0.02) - y(0.02)} fill="var(--band)" />
      {ticks.map((v) => (
        <g key={v.toFixed(3)}>
          <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke="var(--line)" strokeWidth={Math.abs(v) < 1e-9 ? 1.6 : 1} />
          <text x={L - 6} y={y(v) + 4} textAnchor="end">{(v >= 0 ? "+" : "") + v.toFixed(2)}</text>
        </g>
      ))}
      {c.diffs.map((d, i) => (
        <g key={c.seeds[i]}>
          <circle cx={x(i)} cy={y(d)} r={5} fill={d >= 0 ? "var(--grpo)" : "var(--ppo)"} />
          <text x={x(i)} y={H - 12} textAnchor="middle">{c.seeds[i]}</text>
        </g>
      ))}
      <line x1={xm} x2={xm} y1={y(s.hi)} y2={y(s.lo)} stroke="var(--grpo)" strokeWidth={3} />
      <rect x={xm - 7} y={y(s.md) - 7} width={14} height={14} fill="var(--grpo)" transform={`rotate(45 ${xm} ${y(s.md)})`} />
      <text x={xm} y={H - 12} textAnchor="middle">mean</text>
    </svg>
  );
}

const V1 = DATA.samestack_v1;

// Per-arm pre (shared base) and post (per seed) accuracy, from v1 or v2 records.
function prePost(version) {
  if (version === "v1") {
    return Object.entries(V1.arms).map(([a, v]) => ({ arm: a, pre: v.pre, posts: Object.values(v.post_by_seed) }));
  }
  return Object.keys(ARM_NAME).filter(nSeeds).map((a) => {
    const rs = Object.values(byArm[a]);
    return { arm: a, pre: rs.reduce((t, r) => t + r.pre, 0) / rs.length, posts: rs.map((r) => r.post) };
  });
}

function PrePostChart({ rows }) {
  const W = 520, H = 280, L = 52, R = 16, T = 16, B = 40;
  const vals = rows.flatMap((r) => [r.pre, ...r.posts]);
  const pad = (Math.max(...vals) - Math.min(...vals)) * 0.15 || 0.01;
  const lo = Math.min(...vals) - pad, hi = Math.max(...vals) + pad;
  const y = (v) => T + ((H - T - B) * (hi - v)) / (hi - lo);
  const col = (W - L - R) / rows.length;
  const raw = (hi - lo) / 4, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((v) => v >= raw);
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) ticks.push(v);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Accuracy before and after training, per arm">
      {ticks.map((v) => (
        <g key={v.toFixed(4)}>
          <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke="var(--line)" />
          <text x={L - 6} y={y(v) + 4} textAnchor="end">{v.toFixed(step < 0.01 ? 3 : 2)}</text>
        </g>
      ))}
      {rows.map((r, i) => {
        const x0 = L + col * i + col * 0.25, x1 = L + col * i + col * 0.7;
        const mean = r.posts.reduce((a, b) => a + b, 0) / r.posts.length;
        const color = r.arm.startsWith("ppo") ? "var(--ppo)" : "var(--grpo)";
        const down = mean < r.pre;
        return (
          <g key={r.arm}>
            <line x1={x0} x2={x1} y1={y(r.pre)} y2={y(mean)} stroke={down ? "var(--bad)" : color} strokeWidth={2.5} />
            <circle cx={x0} cy={y(r.pre)} r={5} fill="var(--panel)" stroke="var(--muted)" strokeWidth={2} />
            {r.posts.map((p, j) => <circle key={j} cx={x1 + ((j % 5) - 2) * 4} cy={y(p)} r={3} fill={color} fillOpacity={0.4} />)}
            <rect x={x1 - 8} y={y(mean) - 2} width={16} height={4} fill={down ? "var(--bad)" : color} />
            <text x={(x0 + x1) / 2} y={H - 22} textAnchor="middle">{SHORT[r.arm]}</text>
            <text x={x0} y={H - 8} textAnchor="middle">pre</text>
            <text x={x1} y={H - 8} textAnchor="middle">post</text>
          </g>
        );
      })}
    </svg>
  );
}

function Correction() {
  const [version, setVersion] = useState("v2");
  const rows = prePost(version);
  const ppo = rows.find((r) => r.arm === "ppo");
  const ppoPost = ppo.posts.reduce((a, b) => a + b, 0) / ppo.posts.length;
  return (
    <div className="grid2">
      <div className="panel">
        <div className="row between">
          <h3 style={{ margin: 0 }}>Accuracy before and after training</h3>
          {V1 && (
            <div className="seg">
              {["v1", "v2"].map((v) => <button key={v} type="button" aria-pressed={v === version} onClick={() => setVersion(v)}>{v}</button>)}
            </div>
          )}
        </div>
        <div className="chart-box"><PrePostChart rows={rows} /></div>
      </div>
      <div className="panel finding">
        <h3>{version === "v1" ? "v1: 200-token cap, cold critic, 200 test items" : `v2: ${SS.setup.max_new}-token cap, warm critic, ${SS.setup.n_eval} test items`}</h3>
        {version === "v1" ? (
          <>
            <p className="small">GRPO G = 8 beat PPO by {sgn(V1.g8_ppo.mean, 3)} (paired t p = {pfmt(V1.g8_ppo.p_t)}), but PPO finished at {ppoPost.toFixed(3)}, below its own starting accuracy of {ppo.pre.toFixed(3)}. A trained arm ending worse than its start points to a broken baseline, not a better algorithm. After Holm correction the sign-flip p is {pfmt(V1.g8_ppo.holm_sf)}.</p>
            <div className="callout">Superseded. Answers were cut at 200 tokens and the PPO critic started cold, so this gap is not a finding of the thesis.</div>
          </>
        ) : (
          <>
            <p className="small">With the cap raised and the critic warm-started, every arm improves from a base of {ppo.pre.toFixed(3)}, and the arms finish within {(Math.max(...rows.map((r) => r.posts.reduce((a, b) => a + b, 0) / r.posts.length)) - Math.min(...rows.map((r) => r.posts.reduce((a, b) => a + b, 0) / r.posts.length))).toFixed(3)} of each other.</p>
            <div className="callout ok">Training works for both methods; GRPO and PPO end statistically equivalent within ±0.02 on this configuration.</div>
          </>
        )}
      </div>
    </div>
  );
}

export default function SameStack() {
  const [idx, setIdx] = useState(0);
  const c = CONTRASTS[idx], s = c.stats;
  const sig = s.lo > 0 || s.hi < 0, eq = s.tost < 0.05;
  const prov = c.b === "ppo" && nSeeds("ppo") < SS.seeds_planned
    ? ` PPO has ${nSeeds("ppo")} of ${SS.seeds_planned} planned seeds, so this result is provisional.` : "";
  return (
    <section className="view">
      <div className="panel">
        <h2>Does GRPO beat PPO on the same stack?</h2>
        <p className="small">The first version of this comparison showed GRPO +0.050 over PPO, but it capped answers at 200 tokens and started PPO with a cold critic, so PPO ended below its own starting accuracy. Version 2 raises the cap to {SS.setup.max_new} tokens, warm-starts the critic and evaluates on all {SS.setup.n_eval} GSM8K test items. Toggle v1 and v2 to see the correction, then pick a contrast below; the statistics are recomputed in this page from the per-seed run files.</p>
      </div>
      <Correction />
      <div className="row">
        {CONTRASTS.map((k, i) => (
          <button key={k.a + k.b} type="button" aria-pressed={i === idx} onClick={() => setIdx(i)}>{SHORT[k.a]} − {SHORT[k.b]}</button>
        ))}
      </div>
      <div className="grid2">
        <div className="panel">
          <h3>Paired difference per seed, held-out accuracy</h3>
          <div className="chart-box"><DiffChart c={c} /></div>
          <div className="legend">
            <span><i style={{ background: "var(--band)", border: "1px solid var(--good)" }} />equivalence margin ±0.02</span>
            <span><i style={{ background: "var(--grpo)" }} />95% t interval</span>
          </div>
        </div>
        <div className="panel finding">
          <h3>{ARM_NAME[c.a]} minus {ARM_NAME[c.b]}</h3>
          <dl className="stat">
            <dt>Paired seeds</dt><dd>{s.n}</dd>
            <dt>Mean difference</dt><dd>{sgn(s.md)}</dd>
            <dt>95% t interval</dt><dd>[{sgn(s.lo)}, {sgn(s.hi)}]</dd>
            <dt>Paired t p</dt><dd>{pfmt(s.p)}</dd>
            <dt>Exact sign-flip p</dt><dd>{pfmt(s.sf)}</dd>
            <dt>TOST p, margin ±0.02</dt><dd>{pfmt(s.tost)}</dd>
          </dl>
          <div className={`callout ${eq ? "ok" : ""}`}>
            {sig ? "The interval excludes zero" : "The interval includes zero"};{" "}
            {eq ? "the difference is statistically equivalent to zero within ±0.02 accuracy." : "equivalence within ±0.02 is not shown."}{prov}
          </div>
        </div>
      </div>
      <div className="panel">
        <h3>Arms</h3>
        <div className="table-box"><table>
          <thead><tr><th>Arm</th><th className="num">Seeds</th><th className="num">Pre acc</th><th className="num">Post acc</th><th className="num">SD across seeds</th><th className="num">Truncated</th></tr></thead>
          <tbody>
            {Object.keys(ARM_NAME).filter(nSeeds).map((a) => {
              const rs = Object.values(byArm[a]), m = (k) => rs.reduce((t, r) => t + r[k], 0) / rs.length;
              const mp = m("post"), sd = Math.sqrt(rs.reduce((t, r) => t + (r.post - mp) ** 2, 0) / Math.max(1, rs.length - 1));
              return (
                <tr key={a}><td>{ARM_NAME[a]}</td><td className="num">{rs.length}</td><td className="num">{m("pre").toFixed(4)}</td>
                  <td className="num">{mp.toFixed(4)}</td><td className="num">{sd.toFixed(4)}</td><td className="num">{pct(m("post_trunc"))}</td></tr>
              );
            })}
          </tbody>
        </table></div>
      </div>
    </section>
  );
}
