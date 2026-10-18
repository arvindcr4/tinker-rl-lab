import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { transform } from 'esbuild';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { demoDataInput } from '../vite.config.js';
import { GRADING_SPEC_ID, grade } from '../src/grade.js';
const require = createRequire(import.meta.url);
let version=0;
async function renderAnswers(data) {
  globalThis.__reviewData = data;
  let code = await readFile(new URL('../src/views/RealAnswers.jsx', import.meta.url), 'utf8');
  code = code.replace('import { DATA } from "../data.js";', 'const DATA = globalThis.__reviewData;');
  code = code.replace('"react"', JSON.stringify(pathToFileURL(require.resolve('react')).href));
  code = code.replace('"../grade.js"', JSON.stringify(new URL('../src/grade.js', import.meta.url).href));
  code = (await transform(code, {loader:'jsx', format:'esm'})).code;
  const component = await import('data:text/javascript;base64,' + Buffer.from(code + `\n// ${version++}`).toString('base64'));
  return renderToStaticMarkup(React.createElement(component.default));
}
test('empty examples are warning and nonzero audited disagreement stays bad', async () => {
  const data={ examples:{pattern_counts:{}, items:[], n:0, seed:42, per_pattern:1}, grader_check:{grading_spec_id:GRADING_SPEC_ID, completions:3,runs:1,disagreements:2,coverage_complete:false}};
  const html=await renderAnswers(data);
  assert.match(html, /class="pill wrap warn">0\/0 shown/);
  assert.match(html, /class="pill wrap bad">3 stored answers/);
  data.grader_check.disagreements=0;
  assert.match(await renderAnswers(data), /class="pill wrap warn">3 stored answers/);
  data.grader_check.coverage_complete=true;
  assert.match(await renderAnswers(data), /class="pill wrap good">3 stored answers/);
});
test('explicit demo data cannot silently fall back', () => {
  assert.throws(()=>demoDataInput({DEMO_DATA:''}), /nonempty/);
  const plugin=demoDataInput({DEMO_DATA:'/tmp/fresh-review.json'});
  const context={error(message){throw new Error(message);}};
  assert.throws(()=>plugin.buildEnd.call(context), /not applied/);
  assert.equal(plugin.resolveId('./data.json',fileURLToPath(new URL('../src/data.js', import.meta.url))), '/tmp/fresh-review.json');
  plugin.buildEnd.call(context);
});
test('historical demo data remains explicit in logging', () => {
  const plugin=demoDataInput({}); let message;
  const original=console.info;
  try { console.info=(value)=>{message=value;}; plugin.buildStart(); } finally {console.info=original;}
  assert.match(message, /data.json \(historical\)/);
});
test('Unicode digits rejected consistently with Python', () => {
  for (const digits of ['١٢','１２','1e٢','१२']) assert.equal(grade(`\\boxed{${digits}}`,'12').correct,0);
});
test('overview, source roles, cohort selection and margin disclosures stay fail-closed', async () => {
  const src=async(p)=>readFile(new URL('../src/'+p,import.meta.url),'utf8');
  assert.match(await src('views/Overview.jsx'), /MODEL.completion.selected.*MODEL.completion.planned.*MODEL.completion.missing/);
  assert.match(await src('App.jsx'), /s.roles\?\.length \? s.roles.join/);
  assert.match(await src('views/SameStack.jsx'), /MODEL.selectionValid && MODEL.selectedRuns.includes\(r\)/);
  assert.doesNotMatch(await src('views/Viva.jsx'), /The 2 pp equivalence margin/);
});
