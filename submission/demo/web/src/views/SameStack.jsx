import React, { useState } from "react";
import { ARM_NAME, CONTRASTS, DATA, MODEL, SHORT, SS } from "../data.js";
import { armRows } from "../analysis.js";
import { pct, pfmt, sgn } from "../stats.js";

function DiffChart({ c }) {
  const s = c?.stats;
  if (!s) return <p className="small">Paired inference unavailable: at least two validated, compatible pairs are required. Matching seed numbers alone are insufficient.</p>;
  const points = c.diffs.filter(Number.isFinite), n = points.length;
  const W = 520, H = 300, L = 48, R = 12, T = 14, B = 34;
  const lim = Math.max(0.03, ...points.map(Math.abs), Math.abs(s.lo), Math.abs(s.hi)) * 1.15;
  const y = (v) => T + ((H - T - B) * (lim - v)) / (2 * lim);
  const x = (i) => L + ((W - L - R) * (i + 0.5)) / (n + 1.6);
  const step = lim > 0.06 ? 0.02 : 0.01, ticks = [];
  for (let v = -Math.floor(lim / step) * step; v <= lim + 1e-9; v += step) ticks.push(v);
  const xm = x(n + 0.3), margin = MODEL.margin;
  return <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Canonical paired differences, 95% difference and 90% equivalence intervals">
    {margin && <rect x={L} y={y(margin)} width={W - L - R} height={y(-margin) - y(margin)} fill="var(--band)" />}
    {ticks.map((v) => <g key={v.toFixed(3)}>
      <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke="var(--line)" strokeWidth={Math.abs(v) < 1e-9 ? 1.6 : 1} />
      <text x={L - 6} y={y(v) + 4} textAnchor="end">{(v >= 0 ? "+" : "") + v.toFixed(2)}</text>
    </g>)}
    {points.map((d, i) => <g key={i}>
      <circle cx={x(i)} cy={y(d)} r={5} fill={d >= 0 ? "var(--grpo)" : "var(--ppo)"} />
      <text x={x(i)} y={H - 12} textAnchor="middle">{c.seeds[i] ?? i + 1}</text>
    </g>)}
    <line x1={xm - 4} x2={xm - 4} y1={y(s.hi)} y2={y(s.lo)} stroke="var(--grpo)" strokeWidth={3} />
    <line x1={xm + 4} x2={xm + 4} y1={y(s.eqHi)} y2={y(s.eqLo)} stroke="var(--ppo)" strokeWidth={5} />
    <circle cx={xm} cy={y(s.md)} r={5} fill="var(--grpo)" />
    <text x={xm} y={H - 12} textAnchor="middle">mean</text>
  </svg>;
}

const V1 = DATA.samestack_v1;
function prePost(version) {
  if (version === "v1") return Object.entries(V1?.arms || {}).flatMap(([arm, v]) => {
    const posts = Object.values(v.post_by_seed || {}).filter(Number.isFinite);
    return posts.length && Number.isFinite(v.pre) ? [{ arm, pre: v.pre, posts }] : [];
  });
  return armRows(SS.runs || []);
}
function PrePostChart({ rows }) {
  if (!rows.length) return <p className="small">No descriptive pre/post records available.</p>;
  const W = 520, H = 280, L = 52, R = 16, T = 16, B = 40;
  const vals = rows.flatMap((r) => [r.pre, ...r.posts]);
  const pad = (Math.max(...vals) - Math.min(...vals)) * 0.15 || 0.01;
  const lo = Math.min(...vals) - pad, hi = Math.max(...vals) + pad;
  const y = (v) => T + ((H - T - B) * (hi - v)) / (hi - lo);
  const col = (W - L - R) / rows.length;
  const raw = (hi - lo) / 4, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((v) => v >= raw), ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) ticks.push(v);
  return <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Descriptive accuracy before and after training, each arm's own pre-score">
    {ticks.map((v) => <g key={v.toFixed(4)}><line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke="var(--line)" /><text x={L - 6} y={y(v) + 4} textAnchor="end">{v.toFixed(step < 0.01 ? 3 : 2)}</text></g>)}
    {rows.map((r, i) => {
      const x0 = L + col * i + col * 0.25, x1 = L + col * i + col * 0.7;
      const mean = r.posts.reduce((a, b) => a + b, 0) / r.posts.length;
      const color = r.arm.startsWith("ppo") ? "var(--ppo)" : "var(--grpo)";
      return <g key={r.arm}>
        <line x1={x0} x2={x1} y1={y(r.pre)} y2={y(mean)} stroke={mean < r.pre ? "var(--bad)" : color} strokeWidth={2.5} />
        <circle cx={x0} cy={y(r.pre)} r={5} fill="var(--panel)" stroke="var(--muted)" strokeWidth={2} />
        {r.posts.map((p, j) => <circle key={j} cx={x1 + ((j % 5) - 2) * 4} cy={y(p)} r={3} fill={color} fillOpacity={0.4} />)}
        <rect x={x1 - 8} y={y(mean) - 2} width={16} height={4} fill={color} />
        <text x={(x0 + x1) / 2} y={H - 22} textAnchor="middle">{SHORT[r.arm] || r.arm}</text>
        <text x={x0} y={H - 8} textAnchor="middle">pre</text><text x={x1} y={H - 8} textAnchor="middle">post</text>
      </g>;
    })}
  </svg>;
}
function Correction() {
  const [version, setVersion] = useState("v2"), rows = prePost(version);
  const c = CONTRASTS.find((c) => c.id === "g8-ppo");
  return <div className="grid2">
    <div className="panel"><div className="row between"><h3 style={{ margin: 0 }}>Descriptive pre/post sample means</h3>
      {V1 && <div className="seg">{["v1", "v2"].map((v) => <button key={v} type="button" aria-pressed={v === version} onClick={() => setVersion(v)}>{v}</button>)}</div>}
    </div><div className="chart-box"><PrePostChart rows={rows} /></div></div>
    <div className="panel finding"><h3>{version === "v1" ? "v1: superseded comparison" : "v2: descriptive records and validated analysis"}</h3>
      {version === "v1" ? <><p className="small">The 200-token cap, cold critic and 200-item evaluation did not adequately support the intended comparison. A negative observed gain alone does not diagnose a broken baseline. The version change altered several factors simultaneously, so it cannot identify which change caused an outcome difference.</p><div className="callout">Superseded. The earlier gap is not evidence for a general algorithm advantage.</div></> : <>
        <p className="small">Each arm is compared with its own mean pre-training score. These are observed sample means from stored records, including records with unverified provenance. They do not establish a training gain or equivalence.</p>
        <div className={`callout ${c?.equivalent ? "ok" : ""}`}>{c?.decision || "Canonical comparison pending"}</div>
      </>}
    </div>
  </div>;
}
export default function SameStack() {
  const [idx, setIdx] = useState(0), c = CONTRASTS[idx] || CONTRASTS[0], s = c?.stats;
  const rows = armRows(SS.runs || []);
  return <section className="view">
    <div className="panel"><h2>Does this GRPO configuration differ from the PPO-style variant?</h2>
      <p className="small">This comparison concerns the implemented prompt-end detached linear terminal-reward baseline (no GAE), with a warm-started critic and its documented configuration, not PPO algorithms generally. v2 changes the response cap, critic initialization and evaluation size together. Canonical inference is computed offline from validated run IDs, actual environment, shared effective-stack, protocol and ordered-item hashes; seed identity alone is insufficient.</p>
      <div className="callout">Analysis: {MODEL.analysis?.status || "unavailable"}. Selected {MODEL.completion.selected} of {MODEL.completion.planned ?? "unknown"} planned arm-seed runs; {MODEL.completion.missing ?? "unknown"} missing. Historical run count is not planned-cohort completion.
        {MODEL.reasons.length > 0 && <ul>{MODEL.reasons.map((r, i) => <li key={i}>{r}</li>)}</ul>}
      </div>
    </div>
    <div className="panel"><details><summary>Run identity and cohort evidence</summary>
      <p className="small">Selected IDs come only from the offline selector. Unknown historical metadata remains unknown. Excluded records: {MODEL.analysis ? (MODEL.analysis.excluded_runs?.length || 0) : "unknown"}. Complete artifact bundles are required for manifested inputs.</p>
      <div className="table-box"><table><thead><tr><th>Arm / seed</th><th>Run ID</th><th>Environment</th><th>Actual environment hash</th><th>Shared effective-stack hash</th><th>Protocol hash</th><th>Ordered-item hash</th><th>Completion bundle</th><th>Selected</th></tr></thead><tbody>
        {(SS.runs || []).map((r, i) => <tr key={r.source_path || i}><td>{r.arm} / {r.seed}</td><td className="mono">{r.run_id || "unknown"}</td><td>{r.environment || "unknown"}</td><td className="mono">{r.environment_hash || "unknown"}</td><td className="mono">{r.shared_stack_hash || "unknown"}</td><td className="mono">{r.protocol_hash || "unknown"}</td><td className="mono">{r.ordered_test_item_hash || "unknown"}</td><td>{r.completion_bundle_validated ? "validated" : r.completion_failure ? "excluded" : "unknown"}</td><td>{MODEL.selectionValid && MODEL.selectedRuns.includes(r) ? "yes" : "no"}</td></tr>)}
      </tbody></table></div>
    </details></div>
    <Correction />
    <div className="row">{CONTRASTS.map((k, i) => <button key={k.id} type="button" aria-pressed={i === idx} onClick={() => setIdx(i)}>{SHORT[k.a]} − {SHORT[k.b]}{k.exploratory ? " (exploratory)" : ""}</button>)}</div>
    {!c ? <div className="panel">No contrasts available.</div> : <div className="grid2">
      <div className="panel"><h3>Validated paired differences in held-out accuracy</h3><div className="chart-box"><DiffChart c={c} /></div>
        {s && <div className="legend"><span><i style={{ background: "var(--band)" }} />declared equivalence margin {MODEL.margin === null ? "unavailable" : `±${100 * MODEL.margin} pp`}</span><span><i style={{ background: "var(--grpo)" }} />95% difference CI</span><span><i style={{ background: "var(--ppo)" }} />90% equivalence CI</span></div>}
      </div>
      <div className="panel finding"><h3>{ARM_NAME[c.a]} minus {ARM_NAME[c.b]}</h3>
        {s ? <dl className="stat">
          <dt>Validated paired seeds</dt><dd>{s.n}</dd><dt>Mean difference</dt><dd>{sgn(s.md)}</dd>
          <dt>95% difference CI</dt><dd>[{sgn(s.lo)}, {sgn(s.hi)}]</dd><dt>90% equivalence CI</dt><dd>[{sgn(s.eqLo)}, {sgn(s.eqHi)}]</dd>
          <dt>Paired t p (raw / Holm)</dt><dd>{pfmt(s.p)} / {pfmt(s.holmP)}</dd>
          <dt>Exact sign-flip p (raw / Holm)</dt><dd>{pfmt(s.sf)} / {pfmt(s.holmSF)}</dd>
          <dt>Combined TOST p (raw / Holm)</dt><dd>{pfmt(s.tost)} / {pfmt(s.holmTost)}</dd>
          <dt>Directional test excluding a positive +5 pp effect</dt><dd>{pfmt(s.positive5pp)}</dd>
        </dl> : <p className="small">Inferential statistics unavailable for this contrast. Empty, one-pair and incompatible cohorts do not produce a verdict.</p>}
        <div className={`callout ${c.equivalent ? "ok" : ""}`}>{c.decision}</div>
        <p className="small">The primary Holm family is fixed at three comparisons: G8−PPO-style, G8−PPO-style 8×8, and G8−G2. Missing tests contribute p = 1. G2−PPO-style is exploratory. Adjusted combined TOST p must be below {MODEL.alpha} at a validated final analysis to support equivalence.</p>
        <p className="small">Margin justification: {MODEL.justified ? MODEL.plan.margin_justification : "not declared. A 2 percentage-point margin is a sensitivity choice, not an established practical-equivalence threshold."} The directional +5 pp test does not rule out effects of either sign.</p>
      </div>
    </div>}
    <div className="panel"><h3>Stored arm summaries (descriptive; provenance may be unverified)</h3>
      <div className="table-box"><table><thead><tr><th>Arm</th><th className="num">Records</th><th className="num">Own pre mean</th><th className="num">Post mean</th><th className="num">Observed gain</th></tr></thead><tbody>
        {rows.map((r) => <tr key={r.arm}><td>{ARM_NAME[r.arm] || r.arm}</td><td className="num">{r.n}</td><td className="num">{pct(r.pre, 2)}</td><td className="num">{pct(r.post, 2)}</td><td className="num">{sgn(100 * r.gain, 2)} pp</td></tr>)}
        {!rows.length && <tr><td colSpan={5}>No arm records available.</td></tr>}
      </tbody></table></div>
    </div>
  </section>;
}
