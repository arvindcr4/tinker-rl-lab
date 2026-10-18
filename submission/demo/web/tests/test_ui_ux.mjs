import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync, mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'esbuild';

const here = resolve(dirname(fileURLToPath(import.meta.url)), '..');
let rendered;
async function markup() {
  if (rendered) return rendered;
  const result = await build({
    stdin: { contents: `import React from 'react'; import {renderToStaticMarkup} from 'react-dom/server';
      import App from './src/App.jsx';
      export const pages = Object.fromEntries(['overview', 'advantage', 'answers', 'viva'].map(id => {
        globalThis.location = {hash: '#' + id};
        return [id, renderToStaticMarkup(React.createElement(App))];
      }));`, resolveDir: here, sourcefile: 'ui-fixture.jsx', loader: 'jsx' },
    bundle: true, platform: 'node', format: 'cjs', write: false,
  });
  const directory = mkdtempSync(join(tmpdir(), 'demo-ui-'));
  try {
    const file = join(directory, 'fixture.cjs');
    writeFileSync(file, result.outputFiles[0].contents);
    rendered = (await import(pathToFileURL(file).href)).pages;
    return rendered;
  } finally { rmSync(directory, { recursive: true, force: true }); }
}

test('every route keeps a named main landmark, one h1 and a skip destination', async () => {
  for (const [id, html] of Object.entries(await markup())) {
    assert.equal((html.match(/<h1(?:\s|>)/g) || []).length, 1, id);
    assert.match(html, /class="skip-link" href="#main-content"/);
    assert.match(html, /<main[^>]*id="main-content"[^>]*tabindex="-1"[^>]*aria-label="[^"]+"/);
    assert.match(html, /<button[^>]*aria-pressed="true"[^>]*>Shortcuts on<\/button>/);
    assert.match(html, /<caption class="sr-only">Export source files and their SHA-256 hashes<\/caption>/);
  }
});

test('mobile and desktop section navigation retain all nine deep links and the active state', async () => {
  const pages = await markup();
  for (const [id, html] of Object.entries(pages)) {
    assert.equal((html.match(/<option /g) || []).length, 9);
    assert.match(html, new RegExp(`<option value="${id}" selected=""`));
    assert.match(html, new RegExp(`href="#${id}" aria-current="page"`));
  }
  assert.ok(!pages.overview.includes('Previous:'));
  assert.ok(pages.overview.includes('Next: Advantage lab'));
  assert.ok(pages.viva.includes('Previous: Evidence audit'));
  assert.ok(!pages.viva.includes('Next:'));
});

test('answer pattern selection and reward controls use named native controls', async () => {
  const { answers, advantage } = await markup();
  assert.match(answers, /<table aria-describedby="pattern-help"/);
  assert.match(answers, /<button type="button" class="pattern-choice" aria-pressed="true"/);
  assert.ok(!answers.includes('aria-selected='));
  assert.ok(!/<tr[^>]*tabindex=/.test(answers));
  assert.match(answers, /role="group" aria-label="Stored test items"/);
  assert.match(advantage, /role="group" aria-label="Group size"/);
  assert.match(advantage, /role="group" aria-label="Answer rewards"/);
});

test('offline shell has no remote stylesheet or script dependencies', () => {
  const html = readFileSync(join(here, 'index.html'), 'utf8');
  assert.ok(!/(?:src|href)=["'](?:https?:)?\/\//i.test(html));
});

function luminance(hex) {
  const values = hex.match(/[a-f\d]{2}/gi).map(v => parseInt(v, 16) / 255)
    .map(v => v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4);
  return values[0] * .2126 + values[1] * .7152 + values[2] * .0722;
}

test('theme text and status colors meet 4.5:1 against their surfaces', () => {
  const css = readFileSync(join(here, 'src/styles.css'), 'utf8');
  for (const selector of [':root', ':root[data-theme="dark"]']) {
    const declaration = css.slice(css.indexOf(selector + ' {')).split('}')[0];
    const colors = Object.fromEntries([...declaration.matchAll(/--([\w-]+):\s*(#[a-f\d]{6})/gi)]
      .map(([, name, value]) => [name, value]));
    for (const foreground of ['ink', 'muted', 'grpo', 'ppo', 'good', 'warn', 'bad']) {
      for (const background of ['bg', 'panel']) {
        const values = [luminance(colors[foreground]), luminance(colors[background])].sort((a, b) => b - a);
        const ratio = (values[0] + .05) / (values[1] + .05);
        assert.ok(ratio >= 4.5, `${selector} ${foreground} on ${background}: ${ratio.toFixed(2)}`);
      }
    }
  }
});
