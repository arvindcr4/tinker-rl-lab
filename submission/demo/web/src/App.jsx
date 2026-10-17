import React, { useEffect, useState } from "react";
import { DATA, SS, SD } from "./data.js";
import Overview from "./views/Overview.jsx";
import AdvantageLab from "./views/AdvantageLab.jsx";
import Zvf from "./views/Zvf.jsx";
import SameStack from "./views/SameStack.jsx";
import Platform from "./views/Platform.jsx";
import StackDiff from "./views/StackDiff.jsx";
import Audit from "./views/Audit.jsx";
import RealAnswers from "./views/RealAnswers.jsx";
import Viva from "./views/Viva.jsx";

// Order is the defense demo route.
const VIEWS = [
  ["overview", "Overview", Overview],
  ["advantage", "Advantage lab", AdvantageLab],
  ["zvf", "Zero-variance groups", Zvf],
  ["samestack", "GRPO vs PPO", SameStack],
  ["platform", "Platform calibration", Platform],
  ["answers", "Real answers", RealAnswers],
  ["stackdiff", "Stack diff", StackDiff],
  ["audit", "Evidence audit", Audit],
  ["viva", "Hard questions", Viva],
];

const current = () => {
  const h = location.hash.slice(1);
  return VIEWS.some(([id]) => id === h) ? h : "overview";
};
const go = (id) => { location.hash = id; };

export default function App() {
  const [view, setView] = useState(current);
  const [present, setPresent] = useState(false);

  useEffect(() => {
    const onHash = () => { setView(current()); window.scrollTo({ top: 0 }); };
    const onKey = (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey || /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
      const i = VIEWS.findIndex(([id]) => id === current());
      if (e.key === "ArrowRight" && i < VIEWS.length - 1) go(VIEWS[i + 1][0]);
      else if (e.key === "ArrowLeft" && i > 0) go(VIEWS[i - 1][0]);
      else if (/^[1-9]$/.test(e.key)) go(VIEWS[+e.key - 1][0]);
      else if (e.key === "p") setPresent((v) => !v);
    };
    window.addEventListener("hashchange", onHash);
    window.addEventListener("keydown", onKey);
    return () => { window.removeEventListener("hashchange", onHash); window.removeEventListener("keydown", onKey); };
  }, []);

  useEffect(() => { document.documentElement.classList.toggle("present", present); }, [present]);

  const idx = VIEWS.findIndex(([id]) => id === view);
  const View = VIEWS[idx][2], next = VIEWS[idx + 1];
  const sources = SS.sources.concat(
    [DATA.samestack_v1, DATA.platform, DATA.audit].filter(Boolean).map(({ path, sha256 }) => ({ path, sha256 })),
  );

  return (
    <>
      <header className={`top ${view === "overview" ? "" : "compact"}`}><div className="wrap">
        <div className="row between">
          <div className="eyebrow">M.Tech Project Phase 2 · PES University · Arvind C R</div>
          <button type="button" className="ghost" aria-pressed={present} onClick={() => setPresent(!present)} title="Larger text for projectors (P)">Present</button>
        </div>
        {view === "overview" ? <h1>TinkerRL Measurement Lab</h1> : <div className="brand">TinkerRL Measurement Lab</div>}
        {view === "overview" && (
          <p className="lede">An interactive walk through what the project measured about GRPO post-training: where the learning signal comes from, when it disappears, whether GRPO beats PPO on the same stack, and why a method label does not pin down a training stack. Every number here is recomputed in this page from stored run files.</p>
        )}
      </div></header>
      <nav className="tabs" aria-label="Sections"><div className="wrap">
        {VIEWS.map(([id, label], i) => (
          <a key={id} href={`#${id}`} aria-current={id === view ? "page" : undefined}><span className="key">{i + 1}</span>{label}</a>
        ))}
      </div></nav>
      <main className="wrap">
        <View />
        <div className="next-row">
          <span className="small">Keys: ← → or 1–9 to move, P to present</span>
          {next && <a className="next" href={`#${next[0]}`}>Next: {next[1]} →</a>}
        </div>
      </main>
      <footer><div className="wrap">
        <details>
          <summary>Sources ({sources.length} files, SHA-256)</summary>
          <div className="table-box"><table><tbody>
            {sources.map((s) => (
              <tr key={s.path}><td className="mono">{s.path}</td><td className="mono">{s.sha256.slice(0, 16)}…</td></tr>
            ))}
          </tbody></table></div>
        </details>
        <p style={{ marginTop: 10 }}>Built offline from stored results by <span className="mono">submission/demo/web/build_demo_web.py</span>. No model runs in this page, and nothing here is a new measurement. Registry: {SD.entries.length} stacks.</p>
      </div></footer>
    </>
  );
}
