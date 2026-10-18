import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync, mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'esbuild';
import { DOCUMENTS, DOCUMENTS_VERIFIED_ON, PUBLIC_DOCUMENT_COMMIT, CONTINUATION_PAPERS, GROUP_MEMBERS, GROUP_GUIDES, SOLO_GUIDE, SEMESTER_BOUNDARY } from '../src/researchDocuments.js';
import { PRIMARY_SUBMISSIONS } from '../src/submissions.js';

const here = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const root = resolve(here, '../../..');
const documents = [...Object.values(DOCUMENTS), ...CONTINUATION_PAPERS.map(p => p.source)];
const publicObjects = JSON.parse(readFileSync(join(here, 'tests/fixtures/public_document_objects.json'), 'utf8'));
const gitBlobId = (bytes) => createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex');
let rendered;
async function markup() {
  if (rendered) return rendered;
  const result = await build({
    stdin: { contents: `import React from 'react'; import {renderToStaticMarkup} from 'react-dom/server';
      import {Papers, PreviousSemester, ResearchOverview} from './src/views/ResearchDocuments.jsx';
      export const pages = Object.fromEntries(Object.entries({papers: Papers, previous: PreviousSemester, overview: ResearchOverview})
        .map(([id, View]) => [id, renderToStaticMarkup(React.createElement(View))]));`,
    resolveDir: here, sourcefile: 'documents-fixture.jsx', loader: 'jsx' },
    bundle: true, platform: 'node', format: 'cjs', write: false,
  });
  const dir = mkdtempSync(join(tmpdir(), 'demo-documents-'));
  try {
    const file = join(dir, 'fixture.cjs'); writeFileSync(file, result.outputFiles[0].contents);
    rendered = (await import(pathToFileURL(file).href)).pages;
    return rendered;
  } finally { rmSync(dir, { recursive: true, force: true }); }
}

test('every public document URL resolves to a tracked object at the pinned public commit', () => {
  assert.equal(DOCUMENTS_VERIFIED_ON, '2026-10-18');
  assert.equal(PUBLIC_DOCUMENT_COMMIT, 'eb9eb4927ddf1c5b24017f1fd913c96744bb6dd4');
  assert.equal(publicObjects.commit, PUBLIC_DOCUMENT_COMMIT);
  assert.deepEqual(Object.keys(publicObjects.blobs).sort(), documents.map(d => d.path).sort());
  for (const document of documents) {
    const url = new URL(document.href);
    assert.equal(url.origin, 'https://github.com');
    assert.equal(decodeURIComponent(url.pathname), `/arvindcr4/tinker-rl-lab/${document.kind}/${PUBLIC_DOCUMENT_COMMIT}/${document.path}`);
    assert.ok(!/[?#\s]/.test(document.href));
    assert.equal(document.kind, 'blob');
    // Snapshot captured with git ls-tree at the verified public commit. Hash the
    // local bytes so a shallow checkout or source archive needs no Git history.
    assert.equal(gitBlobId(readFileSync(join(root, document.path))), publicObjects.blobs[document.path], document.path);
    assert.ok(document.label && document.format);
  }
  assert.equal(documents.filter(d => d.path.endsWith('.pdf')).length, 1, 'Only the verified public thesis PDF is linked');
  assert.equal(DOCUMENTS.capstone.format, 'DOCX');
  assert.equal(DOCUMENTS.capstoneSource.format, 'LaTeX');
});

test('metadata preserves phase ownership and distinguishes selected manuscripts from the archived fraud study', () => {
  assert.equal(GROUP_MEMBERS.length, 6);
  assert.deepEqual(GROUP_MEMBERS.map(member => member.name), ['Arvind C R', 'Sandhya Jeyaraj', 'Madhu Kumara L', 'Mohammad Rafi', 'Dhruva N Murthy', 'Arumugam Chetty K']);
  assert.deepEqual(GROUP_GUIDES, ['Anwesh Reddy Paduri', 'Narayana Darapaneni']);
  assert.equal(SOLO_GUIDE, 'Ramesh Prakash Guledgudd');
  assert.equal(SEMESTER_BOUNDARY.tag, 'capstone-final-2026-04-25');
  assert.equal(SEMESTER_BOUNDARY.commit, '21a99ef766543a1ca86bfbad188445d3596ea73c');
  assert.deepEqual(CONTINUATION_PAPERS.map(p => p.id), ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']);
  assert.ok(CONTINUATION_PAPERS.every(p => p.boundary && p.focus && p.source.format === 'LaTeX'));
  assert.match(DOCUMENTS.fraudArchive.path, /archive\/absorbed\/P08_fraud\/paper_P8_fraud\.tex$/);
  assert.match(CONTINUATION_PAPERS[0].boundary, /not an established positive scaling law/);
  assert.match(CONTINUATION_PAPERS[6].boundary, /do not establish closed-loop learning benefits/);
});

test('paper view features the actual title and discloses manuscript, review and evidence boundaries', async () => {
  const { papers, overview } = await markup();
  const icicc = PRIMARY_SUBMISSIONS.find(p => p.id === 'icicc-2027-243');
  assert.match(papers, /<section class="view research-documents" aria-labelledby="papers-heading">/);
  assert.ok(papers.includes(icicc.title));
  assert.match(papers, /Public paper version unverified/);
  assert.match(papers, /Camera-ready submission, registration and final publication remain unverified/);
  assert.match(papers, /not a verified copy of the accepted ICICC paper/);
  assert.match(papers, /Independent final-thesis review and institutional approval are still pending/);
  assert.match(papers, /not establish execution provenance, a multi-seed trained-versus-base gain/);
  assert.match(papers, /current manuscript roster reuses P8 for a workshop\/artifact note/);
  assert.equal((papers.match(/data-manuscript-id=/g) || []).length, 7);
  assert.match(papers, /<details class="panel document-disclosure"><summary>Where does the earlier fraud paper fit\?/);
  assert.match(overview, /href="#papers"/);
  assert.match(overview, /href="#previous-semester"/);
});

test('previous-semester view keeps archive limits beside deliverables and credits every group member', async () => {
  const { previous } = await markup();
  assert.match(previous, /aria-labelledby="previous-semester-heading"/);
  for (const person of [...GROUP_MEMBERS.map(m => m.name), ...GROUP_GUIDES, SOLO_GUIDE]) assert.ok(previous.includes(person));
  assert.match(previous, /<time dateTime="2026-04-04">4 Apr 2026<\/time>/);
  assert.match(previous, /not a standalone build bundle/);
  assert.match(previous, /does not prove submission/);
  assert.match(previous, /not the later May main-track submission record/);
  assert.match(previous, /PDFs are absent from this public snapshot/);
  assert.match(previous, /unresolved baseline-ranking, evaluation-unit, statistical-test and DPO-outcome contradictions/);
  assert.match(previous, /do not establish claim-level ownership/);
  assert.match(previous, /<ol class="research-progression">/);
  assert.match(previous, /href="#papers"/);
});

test('document routes keep unique headings, explicit external behavior and no undisclosed destinations', async () => {
  const pages = await markup();
  const allowed = new Set(documents.map(d => d.href));
  for (const html of Object.values(pages)) {
    const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]);
    assert.equal(new Set(ids).size, ids.length);
    for (const [, href] of html.matchAll(/href="([^"]+)"/g)) assert.ok(href.startsWith('#') || allowed.has(href), href);
    for (const [tag] of html.matchAll(/<a [^>]*href="https:[^>]*>/g)) {
      assert.match(tag, /target="_blank"/);
      assert.match(tag, /rel="noopener noreferrer"/);
    }
    assert.ok(!/\b(?:NaN|Infinity)\b|camera.ready.candidate|gmail|openreview\.net|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}/.test(html));
  }
  for (const page of ['papers', 'previous']) {
    assert.match(pages[page], /links open GitHub in a new tab and need internet access/);
    assert.match(pages[page], /<time dateTime="2026-10-18">18 Oct 2026<\/time>/);
  }
  for (const filename of ['researchDocuments.js', 'views/ResearchDocuments.jsx']) {
    const source = readFileSync(join(here, 'src', filename), 'utf8');
    assert.ok(!/\b(?:fetch|XMLHttpRequest|WebSocket|localStorage|sessionStorage)\b/.test(source));
    assert.ok(!/from ["'][^"']*data(?:\.js|\.json)/.test(source), 'Document curation does not derive experimental findings');
  }
});
