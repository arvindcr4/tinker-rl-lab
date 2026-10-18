import React from "react";
import { COVERAGE_LABELS, feedbackFor, feedbackSummary } from "../submissionFeedback.js";
import { submissionDate } from "../submissions.js";

function EntryDate({ date }) {
  return date ? <span className="small"> · <time dateTime={date}>{submissionDate(date)}</time></span> : null;
}

function Points({ points, label }) {
  if (!points.length) return null;
  return <div className="feedback-points">
    {label && <p className="feedback-label">{label}</p>}
    <ul>{points.map((point, index) => <li key={index}>{point}</li>)}</ul>
  </div>;
}

function ReviewScores({ entry }) {
  if (!entry.scores?.length) return null;
  return <div className="feedback-scores">
    <p className="feedback-label">Recorded scores</p>
    <dl>{entry.scores.map((score, index) => {
      const asOf = score.asOf || entry.snapshotOn || entry.date;
      return <div key={index}><dt>{score.label}</dt><dd>
        <strong>{score.value}</strong>
        <span className="small"> · {score.scale ? `Scale: ${score.scale}` : "Scale not specified in checked source"}{asOf ? <> · As recorded <time dateTime={asOf}>{submissionDate(asOf)}</time></> : " · Score date unverified"}</span>
        {score.note && <p className="small">{score.note}</p>}
      </dd></div>;
    })}</dl>
  </div>;
}

export default function SubmissionFeedback({ submission, feedback = feedbackFor(submission.id), headingLevel = 4 }) {
  const Heading = `h${headingLevel}`;
  const EntryHeading = `h${Math.min(headingLevel + 1, 6)}`;
  const sections = [
    ["decision", "Committee and decision feedback", feedback.decisionReasons],
    ["reviews", "Reviewer feedback", feedback.reviews],
    ["responses", "Author responses", feedback.authorResponses],
  ];
  return <details className="submission-feedback" data-feedback-for={submission.id}>
    <summary>Feedback and responses<span className="small">{feedbackSummary(feedback)}</span></summary>
    <p className="small">{feedback.evidence} Summaries are paraphrased; review text and private correspondence are not reproduced.</p>
    {sections.map(([key, label, entries]) => <section key={key} aria-labelledby={`${submission.id}-${key}-heading`} className="feedback-section">
      <div className="row between">
        <Heading id={`${submission.id}-${key}-heading`}>{label}</Heading>
        <span className={`pill ${feedback.coverage[key] === "available" ? "info" : "warn"}`}>{COVERAGE_LABELS[feedback.coverage[key]] || COVERAGE_LABELS.not_verified}</span>
      </div>
      {feedback.coverageNotes[key] && <p className="small">{feedback.coverageNotes[key]}</p>}
      {key === "responses" && feedback.authorResponseWarning && <aside className="callout feedback-warning" aria-labelledby={`${submission.id}-response-warning`}>
        <EntryHeading id={`${submission.id}-response-warning`}>{feedback.authorResponseWarning.label}</EntryHeading>
        <p>{feedback.authorResponseWarning.text}</p>
        <p className="small">{feedback.authorResponseWarning.sourceBasis} · Source dated <time dateTime={feedback.authorResponseWarning.sourceDate}>{submissionDate(feedback.authorResponseWarning.sourceDate)}</time> · Checked <time dateTime={feedback.authorResponseWarning.checkedAt}>{submissionDate(feedback.authorResponseWarning.checkedAt)}</time></p>
      </aside>}
      {entries.map((entry, index) => <div key={index} className="feedback-entry">
        <EntryHeading>{entry.label}<EntryDate date={entry.date} /></EntryHeading>
        {entry.snapshotOn && <p className="small">Source snapshot: <time dateTime={entry.snapshotOn}>{submissionDate(entry.snapshotOn)}</time></p>}
        {key === "reviews" ? <>
          <ReviewScores entry={entry} />
          <Points points={entry.strengths} label="Strengths noted" />
          <Points points={entry.concerns} label="Concerns raised" />
          <Points points={entry.requests} label="Requested changes or questions" />
        </> : <Points points={entry.points} />}
      </div>)}
      {key === "responses" && (feedback.authorResponseReceipts || []).map((receipt, index) => <div key={`receipt-${index}`} className="feedback-entry receipt-only">
        <EntryHeading>{receipt.label}<EntryDate date={receipt.date} /></EntryHeading>
        <p className="small">Receipt only; response text not retrieved. {receipt.note}</p>
      </div>)}
    </section>)}
    {feedback.authorResponses.length > 0 && <p className="small feedback-caveat">Author responses describe the authors’ position. They do not verify that revisions were implemented or concerns resolved.</p>}
    {feedback.notes.length > 0 && <div className="small feedback-caveat"><Points points={feedback.notes} label="Coverage limits" /></div>}
  </details>;
}
