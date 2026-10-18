import React from "react";
import { DOCUMENTS, DOCUMENTS_VERIFIED_ON, PUBLIC_DOCUMENT_COMMIT, CONTINUATION_PAPERS, GROUP_MEMBERS, GROUP_GUIDES, SOLO_GUIDE, SEMESTER_BOUNDARY } from "../researchDocuments.js";
import { PRIMARY_SUBMISSIONS, submissionDate } from "../submissions.js";

function DocumentLink({ document, primary = false }) {
  return <a className={`document-link${primary ? " primary" : ""}`} href={document.href} target="_blank" rel="noopener noreferrer">
    <span>{document.label}<span className="sr-only"> (opens on GitHub in a new tab)</span></span>
    <span className="document-format">{document.format} <span aria-hidden="true">↗</span></span>
  </a>;
}

function SourceNote() {
  return <p className="small document-source-note">Public document index checked <time dateTime={DOCUMENTS_VERIFIED_ON}>{submissionDate(DOCUMENTS_VERIFIED_ON)}</time>, pinned to repository commit <span className="mono">{PUBLIC_DOCUMENT_COMMIT.slice(0, 8)}</span>. Document links open GitHub in a new tab and need internet access. This page itself stays offline; source files and archive dates do not certify results or submission.</p>;
}

export function ResearchOverview() {
  return <section className="research-overview" aria-labelledby="read-work-heading">
    <div className="row between"><h3 id="read-work-heading">Read the work</h3><span className="small">Paper record · Public documents · Academic history</span></div>
    <div className="grid2">
      <a className="panel research-entry" href="#papers">
        <span className="eyebrow">Paper & solo continuation</span>
        <h4>Papers & thesis</h4>
        <p className="small">The ICICC paper record, public review thesis, and selected research manuscripts with their evidence limits.</p>
        <span className="card-link">Explore papers & thesis <span aria-hidden="true">→</span></span>
      </a>
      <a className="panel research-entry" href="#previous-semester">
        <span className="eyebrow">Semester 3 · Group 6</span>
        <h4>Previous semester</h4>
        <p className="small">The six-student foundation, frozen capstone report, recorded contributions, and transition to solo work.</p>
        <span className="card-link">Explore the earlier work <span aria-hidden="true">→</span></span>
      </a>
    </div>
  </section>;
}

export function Papers() {
  const icicc = PRIMARY_SUBMISSIONS.find((record) => record.id === "icicc-2027-243");
  return <section className="view research-documents" aria-labelledby="papers-heading">
    <div className="document-heading"><span className="eyebrow">The research, in context</span><h2 id="papers-heading">Papers & thesis</h2><p className="lede">Read the paper record, then follow the public manuscripts and the evidence behind them.</p></div>
    <div className="publication-grid">
      <article className="panel publication-feature" aria-labelledby="featured-paper-heading">
        <div className="row between"><span className="eyebrow">{icicc.venue} · Paper {icicc.number}</span><span className="pill good">{icicc.decision}</span></div>
        <h3 id="featured-paper-heading">{icicc.title}</h3>
        <p className="small">Organizer acceptance notification: <time dateTime={icicc.decisionOn}>{submissionDate(icicc.decisionOn)}</time></p>
        <p>A paper record about receipt-bound held-out evaluation across agentic and reasoning suites. Its public manuscript version has not been verified, so the documents below are separately identified research context.</p>
        <div className="callout"><strong>Public paper version unverified.</strong> Camera-ready submission, registration and final publication remain unverified. Acceptance does not validate the experimental claims.</div>
        <a className="document-link" href="#submissions"><span>Read the decision & reviewer feedback</span><span aria-hidden="true">→</span></a>
      </article>
      <aside className="panel thesis-feature" aria-labelledby="thesis-heading">
        <span className="eyebrow">Available public document</span>
        <h3 id="thesis-heading">Phase 2 review thesis</h3>
        <div className="row"><span className="pill info">Public research review</span><span className="small">317 pages</span></div>
        <p>Corrected export from <time dateTime="2026-10-11">11 Oct 2026</time>, with editable sources and documented evidence gaps.</p>
        <DocumentLink document={DOCUMENTS.thesis} primary />
        <DocumentLink document={DOCUMENTS.thesisSources} />
        <DocumentLink document={DOCUMENTS.thesisReview} />
        <p className="small">Independent final-thesis review and institutional approval are still pending. This is not a verified copy of the accepted ICICC paper.</p>
      </aside>
    </div>
    <section className="document-section" aria-labelledby="contribution-heading">
      <div><h3 id="contribution-heading">What the research contributes</h3><p className="small">A reading guide to the wider project, not additional claims of acceptance or scientific completion.</p></div>
      <div className="grid3">
        <article className="panel contribution-card"><span className="eyebrow">01 · Measurement</span><h4>Make the learning signal visible</h4><p>Group-relative advantages, zero-variance diagnostics, and group-size and length-bias investigations.</p><a className="card-link" href="#advantage">Try the advantage lab →</a></article>
        <article className="panel contribution-card"><span className="eyebrow">02 · Reproducibility</span><h4>Describe the full training stack</h4><p>Reporting fields, registry comparisons, and same-stack analysis that distinguish a method label from its implementation.</p><a className="card-link" href="#stackdiff">Inspect stack differences →</a></article>
        <article className="panel contribution-card"><span className="eyebrow">03 · Evidence</span><h4>Keep claims traceable</h4><p>Source maps, number audits, and explicit separation of stored observations, missing inputs, and prospective tests.</p><a className="card-link" href="#audit">Open the evidence audit →</a></article>
      </div>
      <div className="callout document-caveat"><strong>Read alongside the limits.</strong> Public availability does not establish execution provenance, a multi-seed trained-versus-base gain, or completed original campaign contracts. Historical thesis statements also require the later runtime errata.</div>
      <div className="document-actions"><DocumentLink document={DOCUMENTS.completion} /><DocumentLink document={DOCUMENTS.errata} /><DocumentLink document={DOCUMENTS.currentStatus} /></div>
    </section>
    <section className="document-section" aria-labelledby="continuation-heading">
      <div className="row between"><div><span className="eyebrow">Semester 4 · Solo continuation</span><h3 id="continuation-heading">Selected continuation manuscripts</h3></div><span className="pill info">Public source files</span></div>
      <p>Arvind C R, with project guide {SOLO_GUIDE}. These selected P1–P7 records show the progression from diagnostics to reporting and controller audits. Labels below summarize the revised scope; source availability is not a venue decision.</p>
      <div className="manuscript-grid">{CONTINUATION_PAPERS.map((record) => <article className="panel manuscript-card" key={record.id} data-manuscript-id={record.id}>
        <div className="row"><span className="paper-number">{record.id}</span><h4>{record.title}</h4></div>
        <p>{record.focus}</p><p className="small"><strong>Scope:</strong> {record.boundary}</p><DocumentLink document={record.source} />
      </article>)}</div>
      <div className="document-actions"><DocumentLink document={DOCUMENTS.continuationMap} /><DocumentLink document={DOCUMENTS.roster} /></div>
      <details className="panel document-disclosure"><summary>Where does the earlier fraud paper fit?</summary><div className="document-disclosure-body"><p>The historical P08 fraud study is absorbed into the thesis and archived. It is not another independent venue paper or RL evidence. The current manuscript roster reuses P8 for a workshop/artifact note; those are different records.</p><DocumentLink document={DOCUMENTS.fraudArchive} /></div></details>
    </section>
    <SourceNote />
  </section>;
}

export function PreviousSemester() {
  return <section className="view research-documents" aria-labelledby="previous-semester-heading">
    <div className="document-heading"><span className="eyebrow">Academic history · Semester 3</span><h2 id="previous-semester-heading">Previous semester</h2><p className="lede">The Group 6 foundation, preserved with its own authorship, deliverables, and unresolved evidence limits.</p></div>
    <article className="panel semester-feature" aria-labelledby="group-heading">
      <div className="row between"><span className="eyebrow">Six students · Shared foundation</span><span className="pill info">Historical archive</span></div>
      <h3 id="group-heading">Semester 3 · Group 6</h3>
      <p>The group established the literature foundation, multi-framework RL post-training environment, initial experiments, evaluation scaffolding, capstone report, and historical paper package.</p>
      <dl className="stat"><dt>Original report date</dt><dd><time dateTime="2026-04-04">4 Apr 2026</time></dd><dt>Frozen boundary</dt><dd><span className="mono">{SEMESTER_BOUNDARY.tag}</span> · <span className="mono">{SEMESTER_BOUNDARY.commit.slice(0, 8)}</span></dd><dt>Project guides</dt><dd>{GROUP_GUIDES.join(" and ")}</dd></dl>
      <div className="document-actions"><DocumentLink document={DOCUMENTS.capstone} primary /><DocumentLink document={DOCUMENTS.groupProvenance} /></div>
    </article>
    <div className="callout document-caveat"><strong>Preserved history, with an erratum.</strong> The frozen report contains unresolved baseline-ranking, evaluation-unit, statistical-test and DPO-outcome contradictions. Its historical numbers are not promoted as current verified findings. <a href={DOCUMENTS.groupErratum.href} target="_blank" rel="noopener noreferrer">Read the frozen-report erratum<span className="sr-only"> (opens on GitHub in a new tab)</span> <span aria-hidden="true">↗</span></a></div>
    <section className="document-section" aria-labelledby="archive-heading">
      <div><h3 id="archive-heading">Explore the earlier deliverables</h3><p className="small">Each link targets a file present in the public repository snapshot.</p></div>
      <div className="grid2">
        <article className="panel archive-card"><span className="eyebrow">01 · Report</span><h4>Frozen capstone report</h4><p>The editable DOCX preserves the capstone deliverable. The original-report PDF is referenced in historical documentation but is absent from this public snapshot.</p><DocumentLink document={DOCUMENTS.capstone} /></article>
        <article className="panel archive-card"><span className="eyebrow">02 · Source</span><h4>Archival LaTeX snapshot</h4><p>Exact historical report source. A referenced dashboard PDF is missing from the tag, so this is not a standalone build bundle.</p><DocumentLink document={DOCUMENTS.capstoneSource} /></article>
        <article className="panel archive-card"><span className="eyebrow">03 · Prepared package</span><h4>Historical paper manifest</h4><p>The April 19 blind-review manifest records a prepared package. It does not prove submission, and it is not the later May main-track submission record. Its paper PDFs are absent from this public snapshot.</p><DocumentLink document={DOCUMENTS.preparedPackage} /></article>
        <article className="panel archive-card"><span className="eyebrow">04 · Attribution</span><h4>Authorship and contribution record</h4><p>The group citation and recorded division of work remain separate from the later solo continuation.</p><DocumentLink document={DOCUMENTS.groupCitation} /><DocumentLink document={DOCUMENTS.groupContributions} /></article>
      </div>
    </section>
    <section className="document-section" aria-labelledby="team-heading">
      <div><h3 id="team-heading">The team behind the foundation</h3><p className="small">Roles summarized from the historical contribution record. All six students are credited for conceptualization, literature review and the original draft.</p></div>
      <ul className="team-grid">{GROUP_MEMBERS.map((member) => <li key={member.name}><h4>{member.name}</h4><p className="small">{member.contribution}</p></li>)}</ul>
      <p className="small">Semester boundaries distinguish academic phases; they do not establish claim-level ownership or resolve conflicting experimental evidence.</p>
    </section>
    <section className="panel document-section" aria-labelledby="progression-heading">
      <div><span className="eyebrow">One shared codebase · Two academic phases</span><h3 id="progression-heading">How the work continued</h3></div>
      <ol className="research-progression">
        <li><span className="paper-number" aria-hidden="true">1</span><div><h4>Group foundation</h4><p>Semester 3: literature, integrations, early experiments and the frozen capstone deliverables.</p></div></li>
        <li><span className="paper-number" aria-hidden="true">2</span><div><h4>Individual continuation</h4><p>After the frozen capstone boundary, Arvind C R continued with guide {SOLO_GUIDE}, expanding diagnostics, manuscript work, audits and evidence mapping.</p></div></li>
        <li><span className="paper-number" aria-hidden="true">3</span><div><h4>Current public review</h4><p>The 11 Oct 2026 thesis export is a later research-review artifact. Independent review, institutional approval and scientific evidence gaps remain explicit.</p></div></li>
      </ol>
      <div className="document-actions"><a className="document-link" href="#papers"><span>Continue to papers & thesis</span><span aria-hidden="true">→</span></a><DocumentLink document={DOCUMENTS.history} /><DocumentLink document={DOCUMENTS.continuationProvenance} /></div>
    </section>
    <SourceNote />
  </section>;
}
