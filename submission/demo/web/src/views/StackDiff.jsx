import React, { useState } from "react";
import { SD } from "../data.js";
import { pct } from "../stats.js";

const PRESETS = [
  ["Tinker vs OpenRLHF, both 'grpo'", "tinker_grpo_qwen3-8b_gsm8k", "openrlhf_grpo_qwen3-8b_gsm8k"],
  ["Two Tinker 'grpo' runs", "tinker_grpo_qwen3-8b_gsm8k", "tinker_grpo_qwen3.5-4b_gsm8k"],
  ["GRPO vs DAPO, same stack", "tinker_grpo_qwen3.5-4b_gsm8k", "tinker_dapo_qwen3.5-4b_gsm8k"],
].filter(([, a, b]) => [a, b].every((id) => SD.entries.some((e) => e.id === id)));
const LEVEL_COLOR = ["var(--muted)", "var(--muted)", "var(--grpo)", "var(--ppo)", "var(--warn)", "var(--bad)"];
const show = (v) => (typeof v === "object" ? JSON.stringify(v) : String(v));

function Distribution() {
  const pairs = Object.values(SD.pairs);
  return (
    <>
      {[["All pairs", pairs], ["Same claimed label", pairs.filter((p) => p.same_label)]].map(([t, ps]) => {
        const cnt = [0, 0, 0, 0, 0, 0];
        ps.forEach((p) => cnt[p.level]++);
        return (
          <div key={t} style={{ marginBottom: 12 }}>
            <div className="row between">
              <span className="small"><b>{t}</b> ({ps.length})</span>
              <span className="small mono">{cnt.map((c, i) => (c ? `R${i} ${c}` : "")).filter(Boolean).join(" · ")}</span>
            </div>
            <div className="bar tall">
              {cnt.map((c, i) => c ? <span key={i} title={`R${i}: ${c}`} style={{ width: `${(100 * c) / ps.length}%`, background: LEVEL_COLOR[i] }} /> : null)}
            </div>
          </div>
        );
      })}
      <p className="small">{pairs.filter((p) => p.underspecified).length} of {pairs.length} pairs are underspecified: an outcome-critical field is unknown on at least one side.</p>
    </>
  );
}

export default function StackDiff() {
  const [a, setA] = useState(PRESETS[0]?.[1] ?? SD.entries[0].id);
  const [b, setB] = useState(PRESETS[0]?.[2] ?? SD.entries[1].id);
  const ea = SD.entries.find((e) => e.id === a), eb = SD.entries.find((e) => e.id === b);
  const flip = a > b, r = a === b ? null : SD.pairs[flip ? `${b}|${a}` : `${a}|${b}`];

  return (
    <section className="view">
      <div className="panel">
        <h2>A method label does not pin down a stack</h2>
        <p className="small">The registry records {SD.entries.length} training stacks field by field. <span className="mono">stackdiff</span> compares two stacks and reports the most severe known difference, from R0 (none) to R5 (checkpoint, reward, prompt template, dataset, eval protocol or execution mode). Fields unknown on either side never raise the level; they lower coverage instead.</p>
      </div>
      <div className="panel">
        <div className="grid2">
          {[["Stack A", a, setA], ["Stack B", b, setB]].map(([label, val, set]) => (
            <label key={label} className="small">{label}
              <select value={val} onChange={(e) => set(e.target.value)}>
                {SD.entries.map((e) => <option key={e.id} value={e.id}>{e.id}</option>)}
              </select>
            </label>
          ))}
        </div>
        <div className="row" style={{ marginTop: 12 }}>
          {PRESETS.map(([t, x, y]) => (
            <button key={t} type="button" aria-pressed={a === x && b === y} onClick={() => { setA(x); setB(y); }}>{t}</button>
          ))}
        </div>
      </div>
      {!r ? <div className="panel"><p className="small">Pick two different stacks.</p></div> : (
        <div className="grid2">
          <div className="panel finding">
            <div className="eyebrow">Verdict</div>
            <div className="row"><span className={`lvl l${r.level}`}>R{r.level}</span><b>{SD.risk[r.level].replace(/^R\d /, "")}</b></div>
            <p className="small">
              {r.same_label ? <>Both stacks are labelled <b className="mono">{ea.label}</b>. </> : <>Labels: <span className="mono">{ea.label}</span> vs <span className="mono">{eb.label}</span>. </>}
              {ea.framework} vs {eb.framework}.
            </p>
            <div className="small">Coverage: fields known on both sides <b className="mono">{pct(r.coverage, 0)}</b></div>
            <div className="bar"><span style={{ width: `${100 * r.coverage}%`, background: "var(--grpo)" }} /></div>
            {r.underspecified && (
              <div className="callout">Underspecified: {r.unknown_n} fields are unknown on at least one side
                {r.unknown_r5.length ? <>, including outcome-critical <span className="mono">{r.unknown_r5.join(", ")}</span></> : null}. More differences may be hidden.</div>
            )}
            {r.same_label && r.level >= 4 && (
              <div className="callout">Label-flip risk: the shared label does not pin the stack, so outcome differences cannot be attributed to the method.</div>
            )}
          </div>
          <div className="panel">
            <h3>Known differences</h3>
            <div className="table-box"><table>
              {r.differs.length ? (
                <>
                  <thead><tr><th>Level</th><th>Field</th><th>A</th><th>B</th></tr></thead>
                  <tbody>
                    {r.differs.map(([fd, sv, x, y]) => {
                      const [va, vb] = flip ? [y, x] : [x, y];
                      return <tr key={fd}><td><span className={`lvl l${sv}`}>R{sv}</span></td><td className="mono">{fd}</td><td>{show(va)}</td><td>{show(vb)}</td></tr>;
                    })}
                  </tbody>
                </>
              ) : <tbody><tr><td>No known differences.</td></tr></tbody>}
            </table></div>
          </div>
        </div>
      )}
      <div className="panel">
        <h3>All {Object.keys(SD.pairs).length} stack pairs</h3>
        <Distribution />
      </div>
    </section>
  );
}
