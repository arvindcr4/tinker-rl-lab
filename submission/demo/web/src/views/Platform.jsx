import React, { useState } from "react";
import { DATA, SHORT, SS } from "../data.js";
import { pfmt, sgn } from "../stats.js";

// Cross-platform calibration (platform_calibration.json): Modal A10G is the reference platform,
// Colab A100 ran the remaining seeds. Every section tolerates missing, empty or n < 2 data,
// because the JSON is regenerated while runs are still landing.
const P = DATA.platform;
const FAMILY = [["g8-ppo", "grpo_g8", "ppo"], ["g8-ppo_8x8", "grpo_g8", "ppo_8x8"], ["g8-g2", "grpo_g8", "grpo_g2"]];
const PLANNED = SS.seeds_planned;

const platName = (p) => (P && p === P.reference ? `Modal ${p}` : `Colab ${p}`);
const asList = (d) => (Array.isArray(d) ? d : Object.values(d || {}));

// Normalise a one_sample() result, or its n < 2 stub, to { n, diffs, mean?, lo?, hi?, tost? }.
function est(e) {
  if (!e) return { n: 0, diffs: [] };
  const diffs = "diff" in e ? [e.diff] : asList(e.diffs);
  if (e.n >= 2 && e.t_ci) return { n: e.n, diffs, mean: e.mean, lo: e.t_ci[0], hi: e.t_ci[1], tost: e.tost_p };
  return { n: e.n ?? diffs.length, diffs };
}

function IntervalChart({ rows, margin }) {
  const W = 520, L = 128, R = 16, T = 8, RH = 44, B = 26, H = T + RH * rows.length + B;
  const vals = rows.flatMap((r) => [...r.diffs, r.lo, r.hi]).filter(Number.isFinite);
  const lim = Math.max(margin * 1.5, ...vals.map(Math.abs)) * 1.12;
  const x = (v) => L + ((W - L - R) * (v + lim)) / (2 * lim);
  const step = lim > 0.06 ? 0.02 : 0.01, ticks = [];
  for (let v = -Math.floor(lim / step) * step; v <= lim + 1e-9; v += step) ticks.push(v);
  const cy = (i) => T + RH * i + RH / 2;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Differences with 95% intervals against the ±${margin} equivalence band`}>
      <rect x={x(-margin)} y={T} width={x(margin) - x(-margin)} height={RH * rows.length} fill="var(--band)" />
      {ticks.map((v) => (
        <g key={v.toFixed(3)}>
          <line x1={x(v)} x2={x(v)} y1={T} y2={T + RH * rows.length} stroke="var(--line)" strokeWidth={Math.abs(v) < 1e-9 ? 1.6 : 1} />
          <text x={x(v)} y={H - 8} textAnchor="middle">{(v > 1e-9 ? "+" : "") + v.toFixed(2)}</text>
        </g>
      ))}
      {rows.map((r, i) => (
        <g key={r.label}>
          <text x={L - 10} y={cy(i) - 2} textAnchor="end" style={{ fill: "var(--ink)" }}>{r.label}</text>
          <text x={L - 10} y={cy(i) + 12} textAnchor="end">{r.n ? `n = ${r.n}` : "no pairs yet"}</text>
          {r.diffs.map((d, j) => <circle key={j} cx={x(d)} cy={cy(i) + 9} r={3.5} fill="var(--muted)" fillOpacity={0.55} />)}
          {Number.isFinite(r.lo) && (
            <>
              <line x1={x(r.lo)} x2={x(r.hi)} y1={cy(i) - 4} y2={cy(i) - 4} stroke="var(--grpo)" strokeWidth={3} />
              <rect x={x(r.mean) - 6} y={cy(i) - 10} width={12} height={12} fill="var(--grpo)" transform={`rotate(45 ${x(r.mean)} ${cy(i) - 4})`} />
            </>
          )}
        </g>
      ))}
    </svg>
  );
}

function Estimates({ rows, margin }) {
  return (
    <dl className="stat">
      {rows.flatMap((r) => [
        <dt key={r.label + "t"}>{r.label}</dt>,
        <dd key={r.label + "d"}>
          {r.n >= 2
            ? <>{sgn(r.mean)} [{sgn(r.lo)}, {sgn(r.hi)}], TOST p {r.tost == null ? "n/a" : pfmt(r.tost)}</>
            : r.n === 1 ? <>{sgn(r.diffs[0])}, one seed, no interval</> : "pending"}
        </dd>,
      ])}
      <dt>Margin</dt><dd>±{margin}</dd>
    </dl>
  );
}

function Verdict({ rows, what, planned }) {
  const main = rows[0];
  if (main.n < 2) {
    return <div className="callout">{main.n ? `Only ${main.n} paired seed so far` : "No paired seeds yet"}, so {what} cannot be estimated.</div>;
  }
  const tostTxt = main.tost == null ? "n/a" : pfmt(main.tost);
  if (planned && main.n < planned) {
    // No equivalence verdict on a partial sample: report the running estimate only.
    return (
      <div className="callout">
        {`${main.label}: interim estimate of ${what} ${sgn(main.mean)} [${sgn(main.lo)}, ${sgn(main.hi)}], TOST p = ${tostTxt}, from ${main.n} of ${planned} planned seeds. No equivalence verdict until all planned seeds are in.`}
      </div>
    );
  }
  const eq = main.tost != null && main.tost < 0.05;
  return (
    <div className={`callout ${eq ? "ok" : ""}`}>
      {eq ? `${main.label}: ${what} is equivalent to zero within ±${P.margin} (TOST p = ${tostTxt}).`
        : `${main.label}: equivalence of ${what} within ±${P.margin} is not shown (TOST p = ${tostTxt}).`}
    </div>
  );
}

function Contrasts() {
  const [metric, setMetric] = useState("post");
  const C = P[`contrasts_${metric}`] || {};
  return (
    <div className="panel">
      <div className="row between">
        <h3 style={{ margin: 0 }}>Primary contrasts, same-platform pairs only</h3>
        <div className="seg">
          {[["post", "Post accuracy (primary)"], ["gain", "Gain over own base"]].map(([k, label]) => (
            <button key={k} type="button" aria-pressed={k === metric} onClick={() => setMetric(k)}>{label}</button>
          ))}
        </div>
      </div>
      <div className="table-box" style={{ marginTop: 10 }}><table>
        <thead><tr>
          <th>Contrast</th><th className="num">Pairs</th><th>Pairs by platform</th><th className="num">Mean</th><th className="num">95% t interval</th>
          <th className="num">Holm p (t)</th><th className="num">Holm p (sign-flip)</th><th className="num">Holm TOST p</th><th className="num">State-matched</th>
        </tr></thead>
        <tbody>
          {FAMILY.map(([name, a, b]) => {
            const c = C[name], label = `${SHORT[a]} − ${SHORT[b]}`;
            if (!c) return <tr key={name}><td>{label}</td><td className="num">0/{PLANNED}</td><td colSpan={7} className="small">pending: no seed has both arms on one platform yet</td></tr>;
            const ok = c.n >= 2 && c.t_ci, h = (k) => (c[k] == null ? "n/a" : pfmt(c[k]));
            const sm = c.modal_state_matched;
            return (
              <tr key={name}>
                <td>{label}</td>
                <td className="num">{c.n}/{PLANNED}</td>
                <td className="mono">{Object.entries(c.strata || {}).map(([p, s]) => `${p} ${s.n} (${sgn(s.mean)})`).join(" · ")}</td>
                <td className="num">{sgn(c.mean)}</td>
                <td className="num" style={{ whiteSpace: "nowrap" }}>{ok ? `[${sgn(c.t_ci[0])}, ${sgn(c.t_ci[1])}]` : "n/a"}</td>
                <td className="num">{h("holm_p_t")}</td><td className="num">{h("holm_p_signflip")}</td><td className="num">{h("holm_tost_p")}</td>
                <td className="num">{sm ? `${sm.n}: ${sm.mean == null ? "n/a" : sgn(sm.mean)}` : "n/a"}</td>
              </tr>
            );
          })}
        </tbody>
      </table></div>
      <p className="small" style={{ marginTop: 10 }}>Holm adjusts across the pre-specified family of three contrasts, counting only contrasts that have data so far. Post accuracy with the paired t is primary; the sign-flip, TOST and gain columns are sensitivity analyses. Within one platform both arms share the same base answers, so gain and post differ only where the two Modal base states mix. "State-matched" keeps only Modal pairs whose arms share a base state, and gives their number and mean difference.</p>
    </div>
  );
}

export default function Platform() {
  if (!P) {
    return (
      <section className="view"><div className="panel">
        <h2>Platform calibration</h2>
        <p className="small">platform_calibration.json has not been generated yet, so there is nothing to show.</p>
      </div></section>
    );
  }
  const plats = P.platforms || [];
  const fx = P.platform_effect?.grpo_g8;
  const fxRows = [["Gain (primary)", fx?.gain], ["Post accuracy", fx?.post]].map(([label, e]) => ({ label, ...est(e) }));
  const ix = P.interaction_g8_ppo;
  const ixRows = [["Post accuracy", ix?.post], ["Gain", ix?.gain]].map(([label, e]) => ({ label, ...est(e) }));
  const done = FAMILY.reduce((t, [k]) => t + Math.min(P.contrasts_post?.[k]?.n ?? 0, PLANNED), 0) + Math.min(fx?.seeds?.length ?? 0, PLANNED);
  const total = (FAMILY.length + 1) * PLANNED;
  const agree = P.base?.verdicts_agree_with_each_modal_state;
  const other = plats.find((p) => p !== P.reference);
  return (
    <section className="view">
      <div className="panel">
        <div className="row between">
          <h2 style={{ margin: 0 }}>Platform calibration</h2>
          <span className={`pill ${done < total ? "warn" : "good"}`}>{done < total ? `Interim · pending: ${total - done} of ${total} seed pairs` : `Complete · ${total} seed pairs`}</span>
        </div>
        <p className="small" style={{ marginTop: 8 }}>The v2 same-stack runs started on Modal A10G GPUs. When Modal was paused, the remaining runs moved to Colab A100. The model, code, seeds and data are identical, but greedy decoding is not: bf16 arithmetic is not bit-identical across GPU generations, and one different token changes the rest of an answer. This view measures how much that matters and shows the contrasts computed so that GPUs never mix within a pair. The seed-pair count covers the three primary contrasts plus the GRPO G8 re-runs on {other || "the second platform"}, each planned at {PLANNED} seeds.</p>
      </div>

      <div className="grid2">
        <div className="panel">
          <h3>Why: the base model scores differently per GPU</h3>
          <div className="table-box"><table>
            <thead><tr><th>Platform</th><th>Base state</th><th className="num">Base accuracy</th><th className="num">Runs</th></tr></thead>
            <tbody>
              {plats.flatMap((p) => Object.entries(P.base?.[p] || {}).sort((a, b) => b[1].runs - a[1].runs).map(([h, s]) => (
                <tr key={p + h}><td>{platName(p)}</td><td className="mono">{h}</td><td className="num">{s.acc.toFixed(4)}</td><td className="num">{s.runs}</td></tr>
              )))}
            </tbody>
          </table></div>
          <p className="small" style={{ marginTop: 10 }}>
            The base model is evaluated greedily, so every run on one platform should store the same base answers. Modal itself produced {Object.keys(P.base?.[P.reference] || {}).length} distinct base states.
            {agree?.length ? ` The ${other} base gives the same right/wrong verdict as Modal on ${agree.join(" and ")} of ${SS.setup.n_eval} items.` : ""}
          </p>
        </div>
        <div className="panel finding">
          <h3>Design: one same-platform pair per seed</h3>
          <p className="small">Each seed contributes one paired difference, and both arms in the pair come from the same platform: Modal A10G wherever both arms exist there, otherwise {other ? platName(other) : "the other platform"}. Because a pair never mixes GPUs, the platform's effect on accuracy cancels inside it.</p>
          <p className="small">Two things can still go wrong. If the platform shifts accuracy beyond the margin, results from the two GPUs are not interchangeable; the platform effect below tests that. If the platform shifts one arm more than the other (a platform × arm interaction), a contrast pooled across GPUs is biased; the interaction below tests that.</p>
        </div>
      </div>

      <div className="grid2">
        <div className="panel">
          <h3>Platform effect, {SHORT.grpo_g8}: {fx?.direction || `${other || "A100"} − ${P.reference}`}</h3>
          <div className="chart-box"><IntervalChart rows={fxRows} margin={P.margin} /></div>
          <div className="legend">
            <span><i style={{ background: "var(--band)", border: "1px solid var(--good)" }} />equivalence margin ±{P.margin}</span>
            <span><i style={{ background: "var(--grpo)" }} />mean and 95% t interval</span>
            <span><i style={{ background: "var(--muted)" }} />one seed</span>
          </div>
        </div>
        <div className="panel finding">
          <h3>Same arm, same seed, different GPU</h3>
          <Estimates rows={fxRows} margin={P.margin} />
          <Verdict rows={fxRows} what="the platform effect" planned={PLANNED} />
          <p className="small">Gain over each platform's own base is primary here, because the base model itself scores differently per GPU; the post-accuracy shift includes that base offset.</p>
        </div>
      </div>

      <div className="grid2">
        <div className="panel">
          <h3>Interaction: {ix?.direction || `(G8 − PPO) on ${other || "A100"} − (G8 − PPO) on ${P.reference}`}</h3>
          <div className="chart-box"><IntervalChart rows={ixRows} margin={P.margin} /></div>
        </div>
        <div className="panel finding">
          <h3>Does the GPU change the GRPO − PPO gap?</h3>
          <Estimates rows={ixRows} margin={P.margin} />
          {ixRows[0].n < 2
            ? <div className="callout">{ixRows[0].n ? `Only ${ixRows[0].n} seed has` : "No seed yet has"} both GRPO G8 and PPO on both platforms, so the interaction cannot be estimated. Until it can, contrasts that mix platforms across seeds rest on the assumption that the GPU shifts both arms equally.</div>
            : <Verdict rows={ixRows} what="the interaction" />}
        </div>
      </div>

      <Contrasts />
    </section>
  );
}
