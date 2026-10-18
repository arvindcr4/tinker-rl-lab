import React, { useEffect, useRef, useState } from "react";
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
import Submissions from "./views/Submissions.jsx";
import { Papers, PreviousSemester } from "./views/ResearchDocuments.jsx";

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
  ["submissions", "Submissions", Submissions],
  ["papers", "Papers & thesis", Papers],
  ["previous-semester", "Previous semester", PreviousSemester],
];

const current = () => {
  const h = location.hash.slice(1);
  return VIEWS.some(([id]) => id === h) ? h : "overview";
};
const go = (id) => { location.hash = id; };

export default function App() {
  const [view, setView] = useState(current);
  const [present, setPresent] = useState(false);
  const [shortcuts, setShortcuts] = useState(true);
  const mainRef = useRef(null), navRef = useRef(null), previousView = useRef(view);

  useEffect(() => {
    const onHash = () => { setView(current()); };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    const onKey = (e) => {
      if (!shortcuts || e.defaultPrevented || e.isComposing || e.repeat || e.metaKey || e.ctrlKey || e.altKey ||
          e.target.isContentEditable || e.target.closest("input, select, textarea, [role='slider']")) return;
      const i = VIEWS.findIndex(([id]) => id === current());
      if (e.key === "ArrowRight" && i < VIEWS.length - 1) go(VIEWS[i + 1][0]);
      else if (e.key === "ArrowLeft" && i > 0) go(VIEWS[i - 1][0]);
      else if (/^[1-9]$/.test(e.key)) go(VIEWS[+e.key - 1][0]);
      else if (e.key === "0") go("submissions");
      else if (e.key.toLowerCase() === "p") setPresent((v) => !v);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [shortcuts]);

  useEffect(() => { document.documentElement.classList.toggle("present", present); }, [present]);

  const idx = VIEWS.findIndex(([id]) => id === view);
  const View = VIEWS[idx][2], next = VIEWS[idx + 1], previous = VIEWS[idx - 1];
  useEffect(() => {
    document.title = `${VIEWS[idx][1]} · TinkerRL Measurement Lab`;
    if (previousView.current !== view) {
      // Native selects may change on each arrow press. Keep their focus so the
      // user can finish choosing a section without an interrupted interaction.
      if (document.activeElement?.id !== "section-select") mainRef.current?.focus({ preventScroll: true });
      window.scrollTo({ top: 0, behavior: "instant" });
      previousView.current = view;
    }
  }, [view, idx]);

  useEffect(() => {
    const rail = navRef.current;
    const revealCurrent = () => {
      const active = rail?.querySelector('[aria-current="page"]');
      if (active && rail.clientWidth) rail.scrollLeft = Math.max(0, active.offsetLeft - rail.offsetLeft - (rail.clientWidth - active.offsetWidth) / 2);
    };
    revealCurrent();
    if (!rail || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(revealCurrent);
    observer.observe(rail);
    return () => observer.disconnect();
  }, [view]);
  const sources = DATA.manifest?.sources || (SS.sources || []).concat(
    [DATA.samestack_v1, DATA.platform, DATA.audit].filter(Boolean).map(({ path, sha256 }) => ({ path, sha256 })),
  );
  const missing = DATA.manifest?.missing || ["Full provenance manifest unavailable in this legacy export"];

  return (
    <>
      <a className="skip-link" href="#main-content" onClick={(e) => {
        e.preventDefault();
        mainRef.current?.focus();
        mainRef.current?.scrollIntoView({ block: "start", behavior: "instant" });
      }}>Skip to content</a>
      <header className={`top ${view === "overview" ? "" : "compact"}`}><div className="wrap">
        <div className="row between">
          <div className="eyebrow">M.Tech Project Phase 2 · PES University · Arvind C R</div>
          <div className="row display-controls">
            <button type="button" className="ghost" aria-pressed={shortcuts} onClick={() => setShortcuts(!shortcuts)} title="Enable or disable single-key navigation">Shortcuts</button>
            <button type="button" className="ghost" aria-pressed={present} onClick={() => setPresent(!present)} title="Larger text for projectors (P)">Present</button>
          </div>
        </div>
        <h1 className={view === "overview" ? undefined : "brand"}>TinkerRL Measurement Lab</h1>
        {view === "overview" && (
          <p className="lede">An interactive walk through what the project measured about GRPO post-training: where the learning signal comes from, when it disappears, whether GRPO beats PPO on the same stack, and why a method label does not pin down a training stack. Validated inference comes from a canonical offline analysis. The browser computes descriptive arm means, ZVF summaries, advantage examples and answer re-grades; registry comparisons and audit summaries are exported offline.</p>
        )}
      </div></header>
      <nav className="tabs" aria-label="Sections">
        <div className="wrap section-picker">
          <label htmlFor="section-select">Section <span className="mono">{idx + 1}/{VIEWS.length}</span></label>
          <select id="section-select" value={view} onChange={(e) => go(e.target.value)}>
            {VIEWS.map(([id, label], i) => <option key={id} value={id}>{i + 1}. {label}</option>)}
          </select>
        </div>
        <div className="wrap section-links" ref={navRef}>
          {VIEWS.map(([id, label], i) => (
            <a key={id} href={`#${id}`} aria-current={id === view ? "page" : undefined}>{i < 10 && <span className="key" aria-hidden="true">{(i + 1) % 10}</span>}{label}</a>
          ))}
        </div>
      </nav>
      <main className="wrap" id="main-content" ref={mainRef} tabIndex={-1} aria-label={VIEWS[idx][1]}>
        <div className="section-context"><span>Section <span className="mono">{idx + 1} / {VIEWS.length}</span></span><span>{view === "submissions" ? "Submission history · Offline snapshot" : ["papers", "previous-semester"].includes(view) ? "Public documents · Curated index" : "Stored results · Offline demo"}</span></div>
        <View />
        <nav className="next-row" aria-label="Previous and next sections">
          {previous ? <a className="next previous" href={`#${previous[0]}`}>← Previous: {previous[1]}</a> : <span className="small">Choose any section to explore</span>}
          {next && <a className="next" href={`#${next[0]}`}>Next: {next[1]} →</a>}
        </nav>
        <p className="small keyboard-hint">{shortcuts ? "Keys: ← → or 1–9 to move, 0 for submissions, P to present. Turn shortcuts off in the header if needed." : "Keyboard shortcuts are off. Use the section links or Previous and Next to move."}</p>
      </main>
      <footer><div className="wrap">
        <details>
          <summary>Provenance ({sources.length} source files, SHA-256)</summary>
          <div className="table-box provenance"><table><caption className="sr-only">Export source files and their SHA-256 hashes</caption><thead><tr><th scope="col">Source file</th><th scope="col">Role</th><th scope="col">SHA-256</th></tr></thead><tbody>
            {sources.map((s) => (
              <tr key={s.path}><td className="mono">{s.path}</td><td>{s.roles?.length ? s.roles.join("; ") : (s.role || "source")}</td><td className="mono">{s.sha256 || "missing"}</td></tr>
            ))}
          </tbody></table></div>
          <p className="small">Missing or unverified coverage: {missing.length ? missing.join("; ") : "none reported"}</p>
          <p className="small">Hashes identify the files read by this export. They do not retrospectively prove an unrecorded training environment or protocol.</p>
        </details>
        <p style={{ marginTop: 10 }}>Experimental data exported offline from stored results by <span className="mono">submission/demo/web/build_demo_web.py</span>. Submission metadata and the public-document index are curated separately. No model runs in this page, and nothing here is a new measurement. Registry: {SD.entries.length} stacks.</p>
      </div></footer>
    </>
  );
}
