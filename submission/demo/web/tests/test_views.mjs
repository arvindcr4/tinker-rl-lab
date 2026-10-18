import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync, mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'esbuild';

const here = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const historical = JSON.parse(readFileSync(join(here, 'src/data.json')));
async function render(data) {
  const built = await build({
    stdin: { contents: `import React from 'react'; import {renderToStaticMarkup} from 'react-dom/server';
      import SameStack from './src/views/SameStack.jsx'; import Overview from './src/views/Overview.jsx'; import Viva from './src/views/Viva.jsx'; import RealAnswers from './src/views/RealAnswers.jsx';
      export const html = [SameStack, Overview, Viva, RealAnswers].map((View) => renderToStaticMarkup(React.createElement(View))).join('');`,
      resolveDir: here, sourcefile: 'fixture.jsx', loader: 'jsx' },
    bundle: true, platform: 'node', format: 'cjs', write: false,
    plugins: [{ name: 'fixture-data', setup(b) {
      b.onLoad({ filter: /\/src\/data\.json$/ }, () => ({ contents: JSON.stringify(data), loader: 'json' }));
    } }],
  });
  const dir = mkdtempSync(join(tmpdir(), 'demo-render-'));
  try {
    const file = join(dir, 'fixture.cjs'); writeFileSync(file, built.outputFiles[0].contents);
    return (await import(pathToFileURL(file).href)).html;
  } finally { rmSync(dir, { recursive: true, force: true }); }
}

test('missing arms, zero contrasts, one pair and disjoint historical seeds render without final claims', async () => {
  const variants = [
    [],
    [{ arm: 'grpo_g8', seed: 1, pre: .4, post: .5, steps: [] }],
    [{ arm: 'grpo_g8', seed: 1, pre: .4, post: .5, steps: [] }, { arm: 'ppo', seed: 2, pre: .6, post: .5, steps: [] }],
    [{ arm: 'grpo_g8', seed: 1, pre: .4, post: .5, steps: [] }, { arm: 'ppo', seed: 1, pre: .6, post: .5, steps: [] }],
  ];
  for (const runs of variants) {
    const data = structuredClone(historical); data.samestack.runs = runs;
    delete data.analysis;
    const html = await render(data);
    assert.ok(html.includes('Inferential statistics unavailable'));
    assert.ok(html.includes('Pending'));
    assert.ok(!html.includes('Equivalence established'));
    assert.ok(!html.includes('every arm improves'));
    assert.ok(!/\b(?:NaN|Infinity)\b/.test(html));
  }
});

test('exported historical data never becomes canonical by numeric seed overlap', async () => {
  const html = await render(historical);
  assert.ok(html.includes('no confirmatory verdict'));
  assert.ok(html.includes('own mean pre-training score'));
  assert.ok(!html.includes('Training works for both methods'));
});

test('missing examples render an explicit empty state without trusting a stale grader audit', async () => {
  const data = structuredClone(historical);
  data.examples = { n: 0, seed: null, items: [], pattern_counts: {}, per_pattern: 4 };
  const html = await render(data);
  assert.ok(html.includes('Stored answer examples unavailable'));
  assert.ok(html.includes('Full audit for the current grading specification unavailable'));
});
