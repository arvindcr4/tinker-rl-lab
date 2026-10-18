import React, { useMemo, useState } from "react";
import { DATA } from "../data.js";
import { grade, GRADING_SPEC_ID } from "../grade.js";

const EX = DATA.examples, CHECK = DATA.grader_check;
const CURRENT_CHECK = CHECK?.grading_spec_id === GRADING_SPEC_ID ? CHECK : null;
const COLS = [["base", "Base model"], ["grpo_g8", "After GRPO, G = 8"], ["ppo", "After PPO"]];
const mark = (c) => (c === "1" || c === 1 ? "✓" : "✗");

// Highlight the boxed answer the grader reads (the last \boxed{...}).
function Completion({ text, boxed }) {
  if (boxed === null) return <pre className="completion">{text}</pre>;
  const key = `\\boxed{${boxed}}`, at = text.lastIndexOf(key);
  if (at < 0) return <pre className="completion">{text}</pre>;
  return <pre className="completion">{text.slice(0, at)}<mark>{key}</mark>{text.slice(at + key.length)}</pre>;
}

function GraderProbe() {
  const [text, setText] = useState("so the answer is \\boxed{1234.00}");
  const [gold, setGold] = useState("1234");
  const g = grade(text, gold);
  return (
    <div className="panel finding">
      <h3>Try the grader</h3>
      <p className="small">The same strict rule scores every answer: take the last <span className="mono">\boxed{"{…}"}</span>, normalize supported formatting, then compare exact rational numbers with the gold answer. Invalid boxes, non-finite values and zero denominators receive no credit. No box, no credit.</p>
      <label className="small" htmlFor="probe-text">Model answer</label>
      <textarea id="probe-text" rows={3} value={text} onChange={(e) => setText(e.target.value)} />
      <div className="row">
        <label className="small" htmlFor="probe-gold">Gold</label>
        <input id="probe-gold" className="mono" value={gold} onChange={(e) => setGold(e.target.value)} style={{ width: 120 }} />
        <span className={`pill ${g.correct ? "good" : "bad"}`}>{g.correct ? "correct" : "wrong"}</span>
        <span className="small mono">{g.status}{g.boxed !== null ? `: read "${g.boxed}"` : ""}</span>
      </div>
    </div>
  );
}

export default function RealAnswers() {
  const patterns = Object.entries(EX.pattern_counts).sort((a, b) => b[1] - a[1]);
  const [pattern, setPattern] = useState(EX.items.find((it) => it.pattern === "010")?.pattern ?? patterns[0]?.[0] ?? "");
  const items = EX.items.filter((it) => it.pattern === pattern);
  const [pick, setPick] = useState(0);
  const item = items[Math.min(pick, items.length - 1)];
  const regraded = useMemo(() => EX.items.flatMap((it) => COLS.map(([k]) => grade(it[k].text, it.gold).correct === it[k].stored)), []);

  return (
    <section className="view">
      <div className="panel">
        <h2>Read the actual answers</h2>
        <p className="small">Stored greedy answers from seed {EX.seed} on the GSM8K test split: the base model, then after GRPO and the PPO-style variant. Historical examples are aligned by stored index; absent protocol/item hashes remain unverified and these examples do not establish a paired inferential cohort. Each answer is re-graded here, in your browser, by a port of the analysis grader. Pick an outcome pattern; the page shows the first {EX.per_pattern} test items of that pattern, so the examples are fixed by rule rather than chosen by hand.</p>
        <div className="row" style={{ marginTop: 10 }}>
          <span className={`pill wrap ${regraded.length === 0 ? "warn" : regraded.every(Boolean) ? "good" : "bad"}`}>{regraded.filter(Boolean).length}/{regraded.length} shown answers re-graded to their stored score</span>
          {CURRENT_CHECK ? <span className={`pill wrap ${CURRENT_CHECK.disagreements > 0 ? "bad" : CURRENT_CHECK.coverage_complete && CURRENT_CHECK.disagreements === 0 ? "good" : "warn"}`}>{CURRENT_CHECK.completions.toLocaleString("en")} stored answers from {CURRENT_CHECK.runs_checked ?? CURRENT_CHECK.runs} runs: {CURRENT_CHECK.disagreements ?? "unknown"} disagreements; coverage {CURRENT_CHECK.coverage_complete ? "complete" : "incomplete"}</span>
            : <span className="pill wrap warn">Full audit for the current grading specification unavailable; rebuild the export</span>}
        </div>
      </div>
      <div className="panel">
        <p className="small">Current grading specification: {GRADING_SPEC_ID}. Historical stored scores remain unchanged.</p>
        <h3>All {EX.n.toLocaleString("en")} test items by outcome (base · GRPO · PPO)</h3>
        <p className="small" id="pattern-help">Select a pattern to read its stored examples. A check means correct; a cross means wrong.</p>
        <div className="table-box"><table aria-describedby="pattern-help">
          <thead><tr><th>Base</th><th>GRPO</th><th>PPO</th><th className="num">Items</th><th>Reading</th></tr></thead>
          <tbody>
            {patterns.map(([k, n]) => (
              <tr key={k} className={`pick ${k === pattern ? "on" : ""}`} onClick={() => { setPattern(k); setPick(0); }}>
                {[...k].map((c, j) => <td key={j} className={`mark ${c === "1" ? "yes" : "no"}`}><span aria-hidden="true">{mark(c)}</span><span className="sr-only">{c === "1" ? "Correct" : "Wrong"}</span></td>)}
                <td className="num">{n}</td>
                <td><button type="button" className="pattern-choice" aria-pressed={k === pattern}
                  onClick={() => { setPattern(k); setPick(0); }}>{READING[k]}</button></td>
              </tr>
            ))}
          </tbody>
        </table></div>
      </div>
      {item ? <div className="panel">
        <div className="row between">
          <h3 style={{ margin: 0 }}>Test item #{item.i} · gold answer <span className="mono">{item.gold}</span></h3>
          <div className="seg" role="group" aria-label="Stored test items">{items.map((it, j) => <button key={it.i} type="button" aria-pressed={it === item} onClick={() => setPick(j)}>#{it.i}</button>)}</div>
        </div>
        <p style={{ margin: "10px 0 14px" }}>{item.question}</p>
        <div className="grid3">
          {COLS.map(([k, label]) => {
            const g = grade(item[k].text, item.gold);
            return (
              <div key={k} className="answer">
                <div className="row between">
                  <b>{label}</b>
                  <span className={`pill ${g.correct ? "good" : "bad"}`}>{g.correct ? "correct" : "wrong"}</span>
                </div>
                <div className="small mono">{g.status === "ok" ? `boxed: ${g.boxed}` : g.status}{item[k].trunc ? " · hit the 512-token cap" : ""} · stored {mark(item[k].stored)} {g.correct === item[k].stored ? "agrees" : "DISAGREES"}</div>
                <Completion text={item[k].text} boxed={g.boxed} />
              </div>
            );
          })}
        </div>
      </div>
      : <div className="panel">Stored answer examples unavailable for this export.</div>}
      <GraderProbe />
    </section>
  );
}

const READING = {
  "111": "all three right",
  "000": "all three wrong",
  "011": "both trained models fixed it",
  "100": "both trained models broke it",
  "010": "only GRPO fixed it",
  "110": "PPO broke it, GRPO kept it",
  "101": "GRPO broke it, PPO kept it",
  "001": "only PPO fixed it",
};
