import React from "react";
import { DATA } from "../data.js";

const PARTS = [
  ["PASS", "var(--good)", "pass"],
  ["CONTEXT", "var(--muted)", "context only"],
  ["ARTIFACT_MISSING", "var(--warn)", "evidence not in public checkout"],
  ["FAIL", "var(--bad)", "failed"],
  ["UNBOUND", "var(--bad)", "unbound"],
];
const METHOD = {
  R: "Recomputed from raw records", S: "Statistics export", T: "Transcribed from source",
  C: "Matched against code", A: "Arithmetic on printed counts", P: "Bound to cited prose source",
};

export default function Audit() {
  const a = DATA.audit, s = a.summary, tc = a.token_counts, tot = s.registered_token_occurrences;
  const m = s.passed_by_method, max = Math.max(...Object.values(m));
  return (
    <section className="view">
      <div className="panel">
        <h2>Every printed number has a receipt</h2>
        <p className="small">The thesis number audit registers each printed value and checks it by a named method: recomputed from raw records (R), from a statistics export (S), transcribed from a source (T), matched against code (C), arithmetic on printed counts (A) or bound to a cited prose source (P). Values whose evidence is not in the public checkout are reported as unavailable, never as passing.</p>
      </div>
      <div className="grid2">
        <div className="panel finding">
          <div className="eyebrow">Public checkout audit</div>
          <div className="big">{s.passed_checks} / {tot}</div>
          <div className="row">
            <span className={`pill ${a.status === "PASS" ? "good" : "warn"}`}>{a.status}</span>
            <span className={`pill ${s.failed_checks ? "bad" : "good"}`}>{s.failed_checks} failed</span>
          </div>
          <div className="bar tall">
            {PARTS.map(([k, c]) => tc[k] ? <span key={k} style={{ width: `${(100 * tc[k]) / tot}%`, background: c }} /> : null)}
          </div>
          <div className="legend">
            {PARTS.filter(([k]) => tc[k]).map(([k, c, t]) => <span key={k}><i style={{ background: c }} />{tc[k]} {t}</span>)}
          </div>
          <p className="small">INCOMPLETE means some evidence is withheld from the public checkout, not that a value disagreed. Source: <span className="mono">{a.path}</span>.</p>
        </div>
        <div className="panel">
          <h3>Passing checks by method</h3>
          {Object.entries(m).sort((x, y) => y[1] - x[1]).map(([k, v]) => (
            <div key={k} style={{ marginBottom: 10 }}>
              <div className="row between"><span className="small"><b className="mono">{k}</b> {METHOD[k] || ""}</span><span className="mono small">{v}</span></div>
              <div className="bar"><span style={{ width: `${(100 * v) / max}%`, background: "var(--grpo)" }} /></div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
