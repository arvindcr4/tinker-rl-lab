// 16 October 2026 revision: render every slide and print layout inspection for the edited deck.
import fs from 'node:fs/promises';
import path from 'node:path';
import {FileBlob, PresentationFile} from '@oai/artifact-tool';
const deck = 'reports/project_finish_2026-10-11/defense/TinkerRL_PES_Defense_2026-10-11.pptx';
const out = process.argv[2];
await fs.mkdir(out, {recursive: true});
const p = await PresentationFile.importPptx(await FileBlob.load(deck));
for (let i = 0; i < p.slides.items.length; i++) {
  const png = await p.export({slide: p.slides.items[i], format: 'png', scale: 1});
  await fs.writeFile(path.join(out, `rev16-${String(i + 1).padStart(2, '0')}.png`), new Uint8Array(await png.arrayBuffer()));
}
console.log((await p.inspect({kind: 'layout', maxChars: 20000})).ndjson);
