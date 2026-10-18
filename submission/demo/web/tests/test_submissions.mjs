import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync, mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'esbuild';
import { PRIMARY_SUBMISSIONS, SUBMISSIONS_VERIFIED_ON, WORKSHOP_PAPERS, WORKSHOP_PROPOSAL, decisionSummary, submissionDate, workshopGroups } from '../src/submissions.js';

const here = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const records = [...PRIMARY_SUBMISSIONS, ...WORKSHOP_PAPERS, WORKSHOP_PROPOSAL];

test('primary decisions preserve venue, title, dates and unresolved publication status', () => {
  assert.equal(SUBMISSIONS_VERIFIED_ON, '2026-10-18');
  assert.equal(PRIMARY_SUBMISSIONS.length, 2);
  const [icicc, neurips] = PRIMARY_SUBMISSIONS;
  assert.deepEqual([icicc.venue, icicc.number, icicc.decision, icicc.submittedOn, icicc.decisionOn],
    ['ICICC 2027', '243', 'Accepted', '2026-09-24', '2026-10-11']);
  assert.equal(icicc.title, 'Receipt-Bound Held-Out Evaluation of a GRPO-Trained Language Model Across Fourteen Agentic and Reasoning Suites');
  assert.equal(icicc.deadline.date, '2026-10-30');
  assert.match(icicc.deadline.label, /Organizer-stated/);
  assert.match(icicc.deadline.caveat, /Timezone unspecified; completion unverified/);
  assert.match(icicc.note, /Camera-ready submission, registration and final publication are unverified/);
  assert.deepEqual([neurips.venue, neurips.kind, neurips.number, neurips.decision, neurips.submittedOn, neurips.decisionOn],
    ['NeurIPS 2026', 'Main-track paper', '36320', 'Rejected', '2026-05-05', '2026-09-24']);
  assert.equal(neurips.title, 'Reward Contrast, Not Algorithm Labels: A Diagnostic Audit of Critic-Free Group-Relative RL for LLMs');
  assert.match(neurips.note, /Public availability and the selected public-release option are unverified/);
});

test('workshop history counts submissions separately from unique titles and organization proposals', () => {
  assert.equal(WORKSHOP_PAPERS.length, 31);
  assert.equal(new Set(WORKSHOP_PAPERS.map(r => r.title)).size, 11);
  assert.equal(new Set(records.map(r => r.id)).size, 34);
  const groups = workshopGroups();
  const titles = [
    'Does Reinforcement Finetuning Actually Pay Off? A Systematic Evidence Map of LLM RFT Claims at ICLR/ICML/ACL 2026',
    'Group Relative Policy Optimization Under the Microscope: When GRPO Beats PPO for LLM Reasoning',
    'Verifiable Rewards vs Preferences: A Head-to-Head Agenda for Reasoning Post-Training',
    'DPO After the Hype: Robustness, Geometry, and Failure Modes of Direct Preference Optimization',
    'Process Beats Outcome? Credit Assignment with Step-Level Rewards in LLM Reasoning',
    'SFT Is Not Solved: A Critical Review of Instruction-Tuning Recipes and Matched-Compute Baselines',
    'Reward Models Are the Bottleneck: Ranking Quality, Overoptimization, and Downstream Policy Impact',
    'RLHF Revisited: Isolating Human Feedback Signal from Optimization Theater',
    'Beyond Vanilla DPO: A Comparative Field Guide to Preference Objectives for LLM Alignment',
    'Beyond Named Algorithms: What Else Is Moving the Needle in LLM Post-Training?',
    'Ten Causal Hypotheses from Indexing 1526 LLM Post-Training Papers',
  ];
  // Index order matches the submission IDs below, not an alphabetical title sort.
  const titleOrder = {
    'FLLMPT NeurIPS 2026': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Pre-to-Post NeurIPS 2026': [0, 1, 5, 2, 3, 6, 7, 8, 4, 9, 10],
    'NeurIPS 2026 Workshop MATH-AI': [1, 2, 4, 0],
    'JUDGe 2026': [6, 8, 3],
    'NeurIPS 2026 AutoMLR Workshop': [10],
    'AI4MetaScience': [10],
  };
  const expected = {
    'FLLMPT NeurIPS 2026': { ids: Array.from({ length: 11 }, (_, i) => String(i + 3)), decision: 'Rejected', on: '2026-09-30' },
    'Pre-to-Post NeurIPS 2026': { ids: Array.from({ length: 11 }, (_, i) => String(i + 1)), decision: 'Desk-rejected', on: '2026-09-07' },
    'NeurIPS 2026 Workshop MATH-AI': { ids: ['3', '4', '5', '6'], decision: 'Desk-rejected', on: '2026-09-07' },
    'JUDGe 2026': { ids: ['1', '2', '3'], decision: 'Desk-rejected', on: '2026-09-08' },
    'NeurIPS 2026 AutoMLR Workshop': { ids: ['2'], decision: 'Desk-rejected', on: '2026-09-06' },
    'AI4MetaScience': { ids: ['6'], decision: 'Decision available; outcome unverified', on: '2026-09-30' },
  };
  assert.equal(groups.length, Object.keys(expected).length);
  for (const { venue, submissions } of groups) {
    assert.deepEqual(submissions.map(r => r.number), expected[venue].ids);
    assert.deepEqual(submissions.map(r => r.title), titleOrder[venue].map(i => titles[i]));
    for (const record of submissions) {
      assert.equal(record.decision, expected[venue].decision);
      assert.equal(record.decisionOn, expected[venue].on);
      assert.equal(record.submittedOn, '2026-08-02');
      assert.equal(record.kind, 'Workshop paper');
    }
  }
  assert.equal(decisionSummary(WORKSHOP_PAPERS), '11 rejected · 19 desk-rejected · 1 decision available; outcome unverified');
  assert.deepEqual([WORKSHOP_PROPOSAL.number, WORKSHOP_PROPOSAL.kind, WORKSHOP_PROPOSAL.decision, WORKSHOP_PROPOSAL.submittedOn, WORKSHOP_PROPOSAL.decisionOn],
    ['488', 'Workshop organization proposal', 'Rejected', '2026-06-07', '2026-07-12']);
  assert.ok(!WORKSHOP_PAPERS.includes(WORKSHOP_PROPOSAL));
  assert.equal(WORKSHOP_PROPOSAL.title, 'Diagnosing RL Post-Training: Measurement, Reproducibility, and Failure Modes in RL Fine-Tuning of LLMs');
  assert.match(WORKSHOP_PROPOSAL.note, /not a research-paper submission/);
});

test('grouping and decision counts are derived without mutating or fabricating records', () => {
  const before = JSON.stringify(WORKSHOP_PAPERS);
  assert.deepEqual(workshopGroups([]), []);
  assert.equal(decisionSummary([]), '');
  const sample = [
    { venue: 'One', decision: 'Accepted' },
    { venue: 'One', decision: 'Rejected' },
    { venue: 'Two', decision: 'Desk-rejected' },
  ];
  assert.deepEqual(workshopGroups(sample).map(g => [g.venue, g.submissions.length]), [['One', 2], ['Two', 1]]);
  assert.equal(decisionSummary(sample), '1 accepted · 1 rejected · 1 desk-rejected');
  assert.equal(JSON.stringify(WORKSHOP_PAPERS), before);
});

test('curated records expose only allowlisted status metadata and no private sources', () => {
  const allowed = new Set(['id', 'venue', 'kind', 'number', 'title', 'submittedOn', 'decision', 'decisionOn', 'decisionDateLabel', 'evidence', 'note', 'deadline']);
  for (const record of records) {
    assert.ok(Object.keys(record).every(key => allowed.has(key)), record.id);
    assert.match(record.number, /^\d+$/);
    assert.ok(['Accepted', 'Rejected', 'Desk-rejected', 'Decision available; outcome unverified'].includes(record.decision));
    for (const key of ['submittedOn', 'decisionOn']) {
      assert.match(record[key], /^\d{4}-\d{2}-\d{2}$/);
      assert.ok(record[key] <= SUBMISSIONS_VERIFIED_ON);
      assert.ok(!Number.isNaN(Date.parse(record[key])));
    }
    assert.ok(record.submittedOn <= record.decisionOn);
    if (record.deadline) assert.deepEqual(Object.keys(record.deadline).sort(), ['caveat', 'date', 'label']);
  }
  const serialized = JSON.stringify(records);
  assert.ok(!/https?:\/\/|[\w.+-]+@[\w.-]+|gmail|openreview|reviewer|messageId|threadId|api.?key/i.test(serialized));
  const source = readFileSync(join(here, 'src/submissions.js'), 'utf8');
  assert.ok(!/\b(?:fetch|XMLHttpRequest|WebSocket|localStorage|sessionStorage)\b/.test(source));
  assert.ok(!/\bimport\b/.test(source), 'Submission facts do not consume experimental results');
});

test('notification dates are stable across local timezones and never inferred', () => {
  const original = process.env.TZ;
  try {
    for (const tz of ['UTC', 'America/Los_Angeles', 'Asia/Kolkata']) {
      process.env.TZ = tz;
      assert.equal(submissionDate('2026-10-11'), '11 Oct 2026');
      assert.equal(submissionDate('2026-10-30'), '30 Oct 2026');
    }
    assert.throws(() => submissionDate(undefined), RangeError);
  } finally {
    if (original === undefined) delete process.env.TZ;
    else process.env.TZ = original;
  }
});

test('submission view renders all records, caveats and native disclosures separately from measurements', async () => {
  const built = await build({
    stdin: { contents: `import React from 'react'; import {renderToStaticMarkup} from 'react-dom/server';
      import Submissions, {SubmissionOverview} from './src/views/Submissions.jsx';
      export const html = renderToStaticMarkup(React.createElement(Submissions));
      export const overview = renderToStaticMarkup(React.createElement(SubmissionOverview));
      export const filtered = Object.fromEntries([
        ['responses', { initialCoverage: 'with_responses' }],
        ['workshop', { initialQuery: 'AI4MetaScience' }],
        ['warning', { initialQuery: 'backfilled upstream-run' }],
        ['empty', { initialQuery: 'no-such-feedback' }],
      ].map(([key, props]) => [key, renderToStaticMarkup(React.createElement(Submissions, props))]));`,
      resolveDir: here, sourcefile: 'submissions-fixture.jsx', loader: 'jsx' },
    bundle: true, platform: 'node', format: 'cjs', write: false,
  });
  const dir = mkdtempSync(join(tmpdir(), 'demo-submissions-'));
  try {
    const file = join(dir, 'fixture.cjs');
    writeFileSync(file, built.outputFiles[0].contents);
    const { html, overview, filtered } = await import(pathToFileURL(file).href);
    assert.match(html, /<section class="view" aria-labelledby="submissions-heading">/);
    assert.equal((html.match(/data-submission-id=/g) || []).length, 31);
    assert.equal((html.match(/<details class="workshop-group">/g) || []).length, 6);
    assert.equal((html.match(/<article /g) || []).length, 3);
    assert.match(html, /31 workshop-paper submissions of 11 distinct titles across 6 workshops/);
    assert.match(html, /11 rejected · 19 desk-rejected/);
    assert.match(html, /not endorsed findings/);
    assert.match(html, /verified from organizer email notifications/);
    assert.match(html, /do not validate experimental claims or establish final publication/);
    assert.match(html, /<time dateTime="2026-10-18">18 Oct 2026<\/time>/);
    assert.equal((html.match(/data-feedback-for=/g) || []).length, 34);
    assert.match(html, /2\/34 submissions have retrieved reviewer feedback; 4 have committee \/ decision context; 1 have author responses/);
    assert.match(html, /<label for="submission-search">Search titles, venues or feedback<\/label>/);
    assert.match(html, /<label for="feedback-filter">Feedback coverage<\/label>/);
    assert.match(html, /role="status" aria-live="polite" aria-atomic="true"/);
    const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
    assert.equal(new Set(ids).size, ids.length, 'Disclosure headings and controls have unique IDs');
    assert.ok(!/href=|\b(?:NaN|Infinity)\b|\bunder review\b/.test(html));
    assert.match(overview, /href="#submissions"/);
    assert.match(overview, /separate from the experimental findings/);
    assert.match(overview, /final publication remain unverified/);
    assert.equal((overview.match(/class="pill /g) || []).length, 2);
    for (const name of ['responses', 'warning']) {
      assert.match(filtered[name], /Showing 1 of 34 submissions/);
      assert.match(filtered[name], /data-feedback-for="neurips-2026-main-36320"/);
      assert.ok(!filtered[name].includes('data-feedback-for="icicc-2027-243"'));
      assert.ok(!filtered[name].includes('Additional NeurIPS submission history'));
    }
    assert.match(filtered.workshop, /<details class="panel submission-history" open="">/);
    assert.match(filtered.workshop, /<details class="workshop-group" open="">/);
    assert.equal((filtered.workshop.match(/data-submission-id=/g) || []).length, 1);
    assert.match(filtered.workshop, /Decision available; outcome unverified/);
    assert.match(filtered.empty, /Showing 0 of 34 submissions/);
    assert.match(filtered.empty, /No matching submissions/);
    assert.ok(!filtered.empty.includes('data-feedback-for='));
  } finally { rmSync(dir, { recursive: true, force: true }); }
});
