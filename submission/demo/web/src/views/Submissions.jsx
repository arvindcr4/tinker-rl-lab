import React, { useState } from "react";
import { PRIMARY_SUBMISSIONS, SUBMISSIONS_VERIFIED_ON, WORKSHOP_PAPERS, WORKSHOP_PROPOSAL, decisionSummary, submissionDate, workshopGroups } from "../submissions.js";
import { FEEDBACK_VERIFIED_ON, feedbackCounts, filterSubmissions } from "../submissionFeedback.js";
import SubmissionFeedback from "./SubmissionFeedback.jsx";

const ALL_SUBMISSIONS = [...PRIMARY_SUBMISSIONS, ...WORKSHOP_PAPERS, WORKSHOP_PROPOSAL];

function DateLabel({ date }) {
  return <time dateTime={date}>{submissionDate(date)}</time>;
}

export function DecisionBadge({ decision }) {
  const tone = decision === "Accepted" ? "good" : ["Rejected", "Desk-rejected"].includes(decision) ? "bad" : "warn";
  return <span className={`pill ${tone}`}>{decision}</span>;
}

function SubmissionCard({ submission }) {
  return (
    <article className="panel submission-card" aria-labelledby={`${submission.id}-title`}>
      <div className="row between">
        <span className="eyebrow">{submission.venue} · {submission.kind}</span>
        <DecisionBadge decision={submission.decision} />
      </div>
      <h3 id={`${submission.id}-title`}>{submission.title}</h3>
      <dl className="stat">
        <dt>{submission.kind === "Workshop organization proposal" ? "Proposal" : "Paper"}</dt><dd>#{submission.number}</dd>
        {submission.submittedOn && <><dt>Submission recorded</dt><dd><DateLabel date={submission.submittedOn} /></dd></>}
        <dt>Decision notification</dt><dd><DateLabel date={submission.decisionOn} /></dd>
        <dt>Evidence</dt><dd>{submission.evidence}</dd>
      </dl>
      <p className="small">{submission.note}</p>
      {submission.deadline && <div className="callout">
        <p>{submission.deadline.label}: <DateLabel date={submission.deadline.date} /></p>
        <p className="small">{submission.deadline.caveat}</p>
      </div>}
      <SubmissionFeedback submission={submission} />
    </article>
  );
}

export function SubmissionOverview() {
  return (
    <section className="panel submission-overview" aria-labelledby="submission-overview-heading">
      <div className="row between">
        <h3 id="submission-overview-heading">Conference submissions</h3>
        <span className="small">Verified <DateLabel date={SUBMISSIONS_VERIFIED_ON} /></span>
      </div>
      <p className="small">Submission decisions are separate from the experimental findings above.</p>
      <ul className="submission-summary">
        {PRIMARY_SUBMISSIONS.map((submission) => <li key={submission.id}>
          <span>{submission.venue} · {submission.kind} #{submission.number}</span>
          <DecisionBadge decision={submission.decision} />
        </li>)}
      </ul>
      <p className="small">ICICC camera-ready, registration and final publication remain unverified.</p>
      <a className="card-link" href="#submissions">View decisions and submission history <span aria-hidden="true">→</span></a>
    </section>
  );
}

export default function Submissions({ initialQuery = "", initialCoverage = "all" }) {
  const [query, setQuery] = useState(initialQuery);
  const [coverage, setCoverage] = useState(initialCoverage);
  const matches = filterSubmissions(ALL_SUBMISSIONS, { query, coverage });
  const visible = new Set(matches.map(({ id }) => id));
  const groups = workshopGroups(WORKSHOP_PAPERS.filter(({ id }) => visible.has(id)));
  const primary = PRIMARY_SUBMISSIONS.filter(({ id }) => visible.has(id));
  const showProposal = visible.has(WORKSHOP_PROPOSAL.id);
  const workshopCount = groups.reduce((sum, group) => sum + group.submissions.length, 0);
  const filtering = query.trim().length > 0 || coverage !== "all";
  const counts = feedbackCounts(ALL_SUBMISSIONS);
  const distinctTitles = new Set(WORKSHOP_PAPERS.map(({ title }) => title)).size;
  return (
    <section className="view" aria-labelledby="submissions-heading">
      <div>
        <h2 id="submissions-heading">Conference submissions</h2>
        <p className="small">Verified <DateLabel date={SUBMISSIONS_VERIFIED_ON} /> · Notification dates in UTC · Offline snapshot</p>
      </div>
      <p>Submission and decision status verified from organizer email notifications. These outcomes do not validate experimental claims or establish final publication.</p>
      <section className="panel feedback-controls" aria-labelledby="feedback-controls-heading">
        <div>
          <h3 id="feedback-controls-heading">Find feedback and responses</h3>
          <p className="small">{FEEDBACK_VERIFIED_ON ? <>Feedback checked <DateLabel date={FEEDBACK_VERIFIED_ON} />. </> : "Feedback verification is pending. "}{counts.withReviews}/{counts.total} submissions have retrieved reviewer feedback; {counts.withDecisionReasons} have committee / decision context; {counts.withResponses} have author responses.</p>
        </div>
        <div className="grid2">
          <div><label htmlFor="submission-search">Search titles, venues or feedback</label><input id="submission-search" type="search" value={query} onChange={(event) => setQuery(event.target.value)} autoComplete="off" aria-controls="submission-results" /></div>
          <div><label htmlFor="feedback-filter">Feedback coverage</label><select id="feedback-filter" value={coverage} onChange={(event) => setCoverage(event.target.value)} aria-controls="submission-results">
            <option value="all">All submissions</option>
            <option value="with_reviews">Reviewer feedback retrieved</option>
            <option value="with_reasons">Committee / decision feedback retrieved</option>
            <option value="with_responses">Author responses retrieved</option>
            <option value="without_reviews">Reviewer text not retrieved</option>
          </select></div>
        </div>
        <div className="row between">
          <p className="small" role="status" aria-live="polite" aria-atomic="true">Showing {matches.length} of {ALL_SUBMISSIONS.length} submissions</p>
          <button type="button" className="ghost" disabled={!filtering} onClick={() => { setQuery(""); setCoverage("all"); }}>Clear filters</button>
        </div>
        <p className="small">Missing text means it was not retrieved from the checked sources, not that no feedback or response exists. Author replies are reported positions, not proof of completed revisions.</p>
      </section>
      <div id="submission-results" className="submission-results">
      {primary.length > 0 && <div className="grid2 submission-primary">
        {primary.map((submission) => <SubmissionCard key={submission.id} submission={submission} />)}
      </div>}
      {matches.length === 0 && <div className="panel"><h3>No matching submissions</h3><p className="small">Try another term or clear the filters to see the full verified history.</p></div>}
      {(workshopCount > 0 || showProposal) && <details className="panel submission-history" open={filtering || undefined}>
        <summary>Additional NeurIPS submission history ({workshopCount} workshop submissions{showProposal ? " + 1 workshop proposal" : ""})</summary>
        <p className="small">Full history: {WORKSHOP_PAPERS.length} workshop-paper submissions of {distinctTitles} distinct titles across {workshopGroups().length} workshops. These decisions and the workshop-organization proposal are separate from the NeurIPS main track.</p>
        <p className="small">Titles are preserved as submitted, including historical claims; they are not endorsed findings of this results dashboard. Submission dates are notification dates, except the ICICC creation record.</p>
        {workshopCount > 0 && <section aria-labelledby="workshop-papers-heading">
          <h3 id="workshop-papers-heading">Workshop-paper decisions</h3>
          <p className="small">{decisionSummary(groups.flatMap(({ submissions }) => submissions))}</p>
          {groups.map(({ venue, submissions }) => <details key={venue} className="workshop-group" open={filtering || undefined}>
            <summary>{venue}<span className="small">{submissions.length} submissions · {decisionSummary(submissions)}</span></summary>
            <ul className="submission-list">
              {submissions.map((submission) => <li key={submission.id} data-submission-id={submission.id}>
                <div className="row between"><span className="mono">Paper #{submission.number}</span><DecisionBadge decision={submission.decision} /></div>
                <h4>{submission.title}</h4>
                <p className="small">Submission recorded <DateLabel date={submission.submittedOn} /> · {submission.decisionDateLabel || "Decision notification"} <DateLabel date={submission.decisionOn} /></p>
                {submission.note && <p className="small">{submission.note}</p>}
                <SubmissionFeedback submission={submission} headingLevel={5} />
              </li>)}
            </ul>
          </details>)}
        </section>}
        {showProposal && <section aria-labelledby="workshop-proposal-heading">
          <h3 id="workshop-proposal-heading">Workshop organization proposal</h3>
          <SubmissionCard submission={WORKSHOP_PROPOSAL} />
        </section>}
      </details>}
      </div>
      <p className="small">Status metadata and paraphrased feedback are included. Correspondence, reviewer identities, raw review text and private submission links are not embedded. This page does not poll an inbox or a submission portal.</p>
    </section>
  );
}
