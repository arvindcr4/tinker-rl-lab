import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync, mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'esbuild';
import { COVERAGE_LABELS, FEEDBACK_VERIFIED_ON, SUBMISSION_FEEDBACK, feedbackFor, feedbackCounts, feedbackSummary, filterSubmissions } from '../src/submissionFeedback.js';
import { PRIMARY_SUBMISSIONS, WORKSHOP_PAPERS, WORKSHOP_PROPOSAL } from '../src/submissions.js';

const here = resolve(dirname(fileURLToPath(import.meta.url)), '..');

function assertFeedbackDate(value) {
  assert.match(value, /^\d{4}-\d{2}-\d{2}$/);
  const parsed = new Date(`${value}T00:00:00.000Z`);
  assert.ok(Number.isFinite(parsed.getTime()), `Invalid calendar date: ${value}`);
  assert.equal(parsed.toISOString().slice(0, 10), value, `Invalid calendar date: ${value}`);
  assert.ok(value <= FEEDBACK_VERIFIED_ON, `Date exceeds feedback verification: ${value}`);
}

const records = [
  { id: 'sample-reviewed', venue: 'Sample conference', kind: 'Main-track paper', title: 'Sample reasoning audit', number: '1', decision: 'Rejected' },
  { id: 'sample-reason', venue: 'Sample workshop', kind: 'Workshop paper', title: 'Other study', number: '2', decision: 'Desk-rejected' },
  { id: 'sample-receipt', venue: 'Sample workshop', kind: 'Workshop paper', title: 'Receipt-only study', number: '3', decision: 'Decision available; outcome unverified' },
];
const empty = () => structuredClone(feedbackFor('missing', {}));
const reviewed = {
  ...empty(), evidence: 'Archived review report and correspondence, checked 18 Oct 2026.',
  coverage: { reviews: 'partial', decision: 'partial', responses: 'available' },
  coverageNotes: { reviews: 'A historical snapshot is available; current portal content is unverified.', decision: 'Initial committee context only; final rationale unavailable.' },
  decisionReasons: [{ label: 'Initial committee context', date: '2026-07-24', points: ['The initial committee assessment requested narrower claims.'] }],
  reviews: [{
    label: 'Reviewer 1 · original report', kind: 'review', snapshotOn: '2026-07-24',
    strengths: ['Useful measurement tooling.'], concerns: ['Limited causal attribution.'], requests: ['Clarify the matched-budget baseline.'],
    scores: [{ label: 'Overall rating', value: 3, scale: '1–6, documented in archive header', asOf: '2026-07-24' },
      { label: 'Confidence', value: 2, note: 'No field scale was recorded in the checked source.' }],
  }],
  authorResponses: [{ label: 'Author response', date: '2026-07-25', points: ['The authors said they would narrow the claim.'] }],
  notes: ['This is historical feedback, not verification of any revision.'],
};
const reasonOnly = {
  ...empty(), coverage: { reviews: 'not_found', decision: 'available', responses: 'not_found' },
  decisionReasons: [{ label: 'Organizer reason', points: ['The paper did not meet the stated format requirement.'] }],
};
const receiptOnly = {
  ...empty(), coverage: { reviews: 'unavailable', decision: 'not_found', responses: 'partial' },
  authorResponseReceipts: [{ label: 'Author response receipt', date: '2026-07-24', note: 'The receipt does not include response text.' }],
};
const data = { 'sample-reviewed': reviewed, 'sample-reason': reasonOnly, 'sample-receipt': receiptOnly };

test('every known submission has explicit feedback coverage without inventing workshop reviews', () => {
  const all = [...PRIMARY_SUBMISSIONS, ...WORKSHOP_PAPERS, WORKSHOP_PROPOSAL];
  assert.equal(FEEDBACK_VERIFIED_ON, '2026-10-18');
  assert.equal(all.length, 34);
  assert.deepEqual(Object.keys(SUBMISSION_FEEDBACK).sort(), all.map(r => r.id).sort());
  assert.deepEqual(feedbackCounts(all), { total: 34, withReviews: 2, withDecisionReasons: 4, withResponses: 1, withoutReviews: 32 });
  for (const record of [...WORKSHOP_PAPERS, WORKSHOP_PROPOSAL]) {
    const feedback = feedbackFor(record.id);
    assert.equal(feedback.reviews.length, 0, record.id);
    assert.equal(feedback.authorResponses.length, 0, record.id);
    assert.equal(feedback.coverage.reviews, 'not_found', record.id);
    assert.equal(feedback.coverage.responses, 'not_found', record.id);
    assert.match(feedback.coverageNotes.responses, /does not establish that no portal response exists/);
  }
  for (const id of ['neurips-2026-proposal-488', 'ai4metascience-2026-6']) {
    const feedback = feedbackFor(id);
    assert.equal(feedback.coverage.decision, 'partial');
    assert.match(feedback.decisionReasons[0].label, /not a submission-specific rationale/);
    assert.match(feedback.decisionReasons[0].date, /^2026-/);
  }
});

test('main-track reviews preserve historical scores, archive dates and later discussion separately', () => {
  const feedback = feedbackFor('neurips-2026-main-36320');
  const originals = feedback.reviews.filter(r => r.kind === 'review');
  const followups = feedback.reviews.filter(r => r.kind === 'reviewer_followup');
  assert.equal(originals.length, 3);
  assert.equal(followups.length, 2);
  assert.deepEqual(originals.map(r => r.scores.find(s => s.label === 'Overall').value), [3, 3, 1]);
  for (const review of originals) {
    assert.equal(review.snapshotOn, '2026-07-24');
    assert.equal(review.date, undefined, 'Archive retrieval is not a review posting date');
    assert.equal(review.scores.length, 6);
    for (const score of review.scores) {
      assert.equal(score.asOf, '2026-07-24');
      assert.ok(Number.isFinite(score.value));
      const scale = score.label === 'Overall' ? '1–6' : score.label === 'Confidence' ? '1–5' : '1–4';
      assert.ok(score.scale.startsWith(scale), score.label);
    }
  }
  assert.deepEqual(followups.map(r => r.date), ['2026-07-30', '2026-08-02']);
  assert.ok(followups.every(r => !r.scores), 'Follow-up comments do not invent updated ratings');
  assert.equal(feedback.decisionReasons[2].snapshotOn, '2026-07-24');
  assert.equal(feedback.decisionReasons[2].date, undefined);
  assert.match(feedback.coverageNotes.decision, /final paper-specific rejection rationale is missing/);
  assert.equal(feedback.authorResponses.length, 3);
  assert.deepEqual(feedback.authorResponses.map(r => r.date), ['2026-07-30', '2026-08-02', '2026-08-02']);
  assert.equal(feedback.authorResponseReceipts.length, 2);
  assert.match(feedback.authorResponseReceipts[0].note, /drafts have not been verified as identical/);
  const icicc = feedbackFor('icicc-2027-243');
  assert.equal(icicc.reviews.length, 2);
  assert.ok(icicc.reviews.every(r => !r.scores && r.date === '2026-10-11'));
  assert.equal(icicc.authorResponses.length, 0);
});

test('feedback date validation rejects normalized impossible dates and respects leap years', () => {
  for (const value of ['2026-02-28', '2024-02-29', '2000-02-29', FEEDBACK_VERIFIED_ON]) {
    assert.doesNotThrow(() => assertFeedbackDate(value), value);
  }
  for (const value of ['2026-02-30', '2026-02-29', '2025-02-29', '1900-02-29', '2026-04-31', '2026-13-01', '2026-00-01', '2026-10-00']) {
    assert.throws(() => assertFeedbackDate(value), /Invalid calendar date/, value);
  }
  for (const value of ['2026-2-28', '2026-02-28T00:00:00Z', 'not-a-date']) {
    assert.throws(() => assertFeedbackDate(value), value);
  }
  const tomorrow = new Date(`${FEEDBACK_VERIFIED_ON}T00:00:00.000Z`);
  tomorrow.setUTCDate(tomorrow.getUTCDate() + 1);
  assert.throws(() => assertFeedbackDate(tomorrow.toISOString().slice(0, 10)), /exceeds feedback verification/);
});

test('curated feedback uses allowlisted anonymous fields with no raw correspondence or private identifiers', () => {
  const keys = (object, allowed) => assert.ok(Object.keys(object).every(key => allowed.includes(key)), Object.keys(object).join(', '));
  for (const feedback of Object.values(SUBMISSION_FEEDBACK)) {
    keys(feedback, ['evidence', 'coverage', 'coverageNotes', 'decisionReasons', 'reviews', 'authorResponses', 'authorResponseReceipts', 'notes', 'authorResponseWarning']);
    keys(feedback.coverage, ['reviews', 'decision', 'responses']);
    keys(feedback.coverageNotes, ['reviews', 'decision', 'responses']);
    for (const coverage of Object.values(feedback.coverage)) assert.ok(Object.hasOwn(COVERAGE_LABELS, coverage));
    for (const entry of [...feedback.decisionReasons, ...feedback.reviews, ...feedback.authorResponses, ...feedback.authorResponseReceipts]) {
      keys(entry, ['label', 'date', 'snapshotOn', 'kind', 'points', 'strengths', 'concerns', 'requests', 'scores', 'note']);
      for (const key of ['date', 'snapshotOn']) if (entry[key]) assertFeedbackDate(entry[key]);
    }
    for (const review of feedback.reviews) {
      assert.match(review.label, /^Reviewer [123]( follow-up)?$/);
      for (const score of review.scores || []) {
        keys(score, ['label', 'value', 'scale', 'asOf', 'note']);
        assertFeedbackDate(score.asOf);
      }
    }
    if (feedback.authorResponseWarning) {
      keys(feedback.authorResponseWarning, ['label', 'checkedAt', 'sourceDate', 'sourceBasis', 'text']);
      assertFeedbackDate(feedback.authorResponseWarning.checkedAt);
      assertFeedbackDate(feedback.authorResponseWarning.sourceDate);
    }
  }
  const serialized = JSON.stringify(SUBMISSION_FEEDBACK);
  assert.ok(!/https?:\/\/|[\w.+-]+@[\w.-]+|(?:message|thread|forum|invitation|reviewer)Id|api.?key|access.?token|Bearer |\/workspace\/|\/home\/|\/Users\//i.test(serialized));
  assert.ok(!/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/i.test(serialized));
});

test('missing feedback stays explicitly unverified, without invented empty-source conclusions', () => {
  const missing = feedbackFor('unknown', data);
  assert.deepEqual(missing.coverage, { reviews: 'not_verified', decision: 'not_verified', responses: 'not_verified' });
  assert.equal(missing.reviews.length, 0);
  assert.match(feedbackSummary(missing), /not yet verified/);
  assert.equal(COVERAGE_LABELS.not_found, 'Not found in checked sources');
});

test('coverage totals count retrieved content, not response receipts or decision availability', () => {
  assert.deepEqual(feedbackCounts(records, data), { total: 3, withReviews: 1, withDecisionReasons: 2, withResponses: 1, withoutReviews: 2 });
  assert.deepEqual(feedbackCounts([], data), { total: 0, withReviews: 0, withDecisionReasons: 0, withResponses: 0, withoutReviews: 0 });
  assert.match(feedbackSummary(reviewed), /1 review report \(partial coverage\)/);
  assert.match(feedbackSummary(receiptOnly), /Author response not retrieved/);
  const updated = { ...reviewed, reviews: [...reviewed.reviews, { label: 'Reviewer 1 follow-up', kind: 'reviewer_followup' }] };
  assert.match(feedbackSummary(updated), /2 reviewer feedback entries/);
  assert.ok(!feedbackSummary(updated).includes('2 review reports'));
});

test('search finds feedback, titles, decisions and scores without mutating source records', () => {
  const before = JSON.stringify({ records, data });
  for (const query of ['  MATCHED-BUDGET   ', 'causal sample', 'overall rating', 'narrow claim']) {
    assert.deepEqual(filterSubmissions(records, { query }, data).map(r => r.id), ['sample-reviewed']);
  }
  assert.deepEqual(filterSubmissions(records, { query: 'format requirement' }, data).map(r => r.id), ['sample-reason']);
  assert.deepEqual(filterSubmissions(records, { query: 'outcome unverified' }, data).map(r => r.id), ['sample-receipt']);
  assert.equal(filterSubmissions(records, { query: 'does-not-exist' }, data).length, 0);
  assert.equal(filterSubmissions(records, { query: '   ' }, data).length, 3);
  assert.equal(JSON.stringify({ records, data }), before);
});

test('coverage filters combine with search and never count missing response bodies as responses', () => {
  const expected = {
    all: ['sample-reviewed', 'sample-reason', 'sample-receipt'],
    with_reviews: ['sample-reviewed'], with_reasons: ['sample-reviewed', 'sample-reason'],
    with_responses: ['sample-reviewed'], without_reviews: ['sample-reason', 'sample-receipt'],
  };
  for (const [coverage, ids] of Object.entries(expected)) {
    assert.deepEqual(filterSubmissions(records, { coverage }, data).map(r => r.id), ids);
  }
  assert.equal(filterSubmissions(records, { query: 'receipt-only', coverage: 'with_responses' }, data).length, 0);
  assert.equal(filterSubmissions(records, { query: 'sample', coverage: 'without_reviews' }, {}).length, 3);
});

async function renderFeedback(feedback, headingLevel = 4) {
  const built = await build({
    stdin: { contents: `import React from 'react'; import {renderToStaticMarkup} from 'react-dom/server';
      import SubmissionFeedback from './src/views/SubmissionFeedback.jsx';
      export const html = renderToStaticMarkup(React.createElement(SubmissionFeedback, ${JSON.stringify({ submission: records[0], feedback, headingLevel })}));`,
      resolveDir: here, sourcefile: 'feedback-fixture.jsx', loader: 'jsx' },
    bundle: true, platform: 'node', format: 'cjs', write: false,
  });
  const dir = mkdtempSync(join(tmpdir(), 'demo-feedback-'));
  try {
    const file = join(dir, 'fixture.cjs'); writeFileSync(file, built.outputFiles[0].contents);
    return (await import(pathToFileURL(file).href)).html;
  } finally { rmSync(dir, { recursive: true, force: true }); }
}

test('feedback disclosures show archived dates, score provenance, attribution and unresolved coverage', async () => {
  const html = await renderFeedback(reviewed);
  assert.match(html, /<details class="submission-feedback" data-feedback-for="sample-reviewed">/);
  assert.match(html, /<summary>Feedback and responses/);
  for (const label of ['Committee and decision feedback', 'Reviewer feedback', 'Author responses', 'Strengths noted', 'Concerns raised', 'Requested changes or questions']) assert.ok(html.includes(label), label);
  assert.match(html, /Source snapshot: <time dateTime="2026-07-24">24 Jul 2026/);
  assert.match(html, /<strong>3<\/strong>/);
  assert.match(html, /Scale: 1–6, documented in archive header/);
  assert.match(html, /Scale not specified in checked source/);
  assert.match(html, /initial committee assessment/);
  assert.match(html, /final rationale unavailable/);
  assert.match(html, /do not verify that revisions were implemented or concerns resolved/);
  assert.ok(!/href=|<script|\bNaN\b|\bInfinity\b/.test(html));
});

test('receipt-only responses are labeled and workshop heading depth remains accessible', async () => {
  const html = await renderFeedback(receiptOnly, 5);
  assert.match(html, /<h5 id="sample-reviewed-responses-heading">Author responses<\/h5>/);
  assert.match(html, /<h6>Author response receipt/);
  assert.match(html, /Receipt only; response text not retrieved/);
  assert.match(html, /Source unavailable/);
  assert.ok(!html.includes('1 author response'));
});

test('provenance warning precedes historical author claims without inventing a posted withdrawal', async () => {
  const feedback = feedbackFor('neurips-2026-main-36320');
  assert.equal(feedback.authorResponseWarning.sourceDate, '2026-07-28');
  assert.equal(feedback.authorResponseWarning.checkedAt, '2026-10-18');
  const html = await renderFeedback(feedback);
  assert.match(html, /<aside class="callout feedback-warning" aria-labelledby="sample-reviewed-response-warning">/);
  assert.match(html, /three of the five claimed seed pairs were backfilled summaries/);
  assert.match(html, /posting of that revision is not verified/);
  assert.match(html, /August 2 comments below are genuine historical author statements/);
  assert.match(html, /Source dated <time dateTime="2026-07-28">28 Jul 2026/);
  assert.match(html, /Checked <time dateTime="2026-10-18">18 Oct 2026/);
  assert.ok(html.indexOf('feedback-warning') < html.indexOf('Author discussion comment'));
  assert.ok(!/later (?:audit|published correction)|posted withdrawal/i.test(html));
  assert.deepEqual(filterSubmissions(PRIMARY_SUBMISSIONS, { query: 'backfilled upstream-run' }).map(r => r.id), ['neurips-2026-main-36320']);
});

test('feedback text remains escaped and the module never reads or changes experimental data', async () => {
  const html = await renderFeedback({ ...reasonOnly, decisionReasons: [{ label: '<script>label</script>', points: ['<img src=x onerror=alert(1)>'] }] });
  assert.match(html, /&lt;script&gt;label&lt;\/script&gt;/);
  assert.ok(!/<script|<img/.test(html));
  const source = readFileSync(join(here, 'src/submissionFeedback.js'), 'utf8');
  assert.ok(!/\b(?:fetch|XMLHttpRequest|WebSocket|localStorage|sessionStorage)\b/.test(source));
  assert.ok(!/from ["'].*(?:data|analysis|stats)\./.test(source));
});

test('presenter mode enlarges recorded scores to match feedback text and score provenance', () => {
  const css = readFileSync(join(here, 'src/styles.css'), 'utf8');
  const fontSizeFor = selector => {
    let fontSize;
    for (const [, selectors, declarations] of css.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
      if (selectors.split(',').map(value => value.trim()).includes(selector)) {
        const match = declarations.match(/(?:^|;)\s*font-size:\s*([^;}]+)/);
        if (match) fontSize = match[1].trim();
      }
    }
    return fontSize;
  };
  assert.equal(fontSizeFor('.feedback-scores'), '14px');
  assert.equal(fontSizeFor(':root.present .feedback-scores'), '17px');
  assert.equal(fontSizeFor(':root.present .feedback-scores'), fontSizeFor(':root.present .feedback-points'));
  assert.equal(fontSizeFor(':root.present .feedback-scores'), fontSizeFor(':root.present .small'));
});
